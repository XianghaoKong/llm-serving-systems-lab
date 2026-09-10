"""Independent Triton fused kernels with explicit FP32 arithmetic contracts.

RMSNorm: cast once after FP32 normalization and affine multiplication.
SwiGLU: cast once after FP32 SiLU(gate) * up. GPU-only, inference and autograd.
"""
import torch
import triton
import triton.language as tl


def validate(x, w=None):
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("expected [rows, positive width]")
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("expected CUDA floating tensor")
    if w is not None and (w.shape != (x.shape[1],) or w.device != x.device or w.dtype != x.dtype):
        raise ValueError("weight must match width, device and dtype")


def rms_reference(x, w, eps=1e-6):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def swiglu_reference(g, u):
    return (torch.nn.functional.silu(g.float()) * u.float()).to(g.dtype)


@triton.jit
def _rms_fwd(X, W, Y, R, SX0: tl.constexpr, SX1: tl.constexpr, SW: tl.constexpr,
             H: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + row * SX0 + c * SX1, c < H, 0).to(tl.float32)
    w = tl.load(W + c * SW, c < H, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    tl.store(Y + row * H + c, x * r * w, c < H)
    tl.store(R + row, r)


@triton.jit
def _rms_dx(X, W, DY, R, DX, SX0: tl.constexpr, SX1: tl.constexpr,
            SW: tl.constexpr, SD0: tl.constexpr, SD1: tl.constexpr,
            H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + row * SX0 + c * SX1, c < H, 0).to(tl.float32)
    w = tl.load(W + c * SW, c < H, 0).to(tl.float32)
    dy = tl.load(DY + row * SD0 + c * SD1, c < H, 0).to(tl.float32)
    r = tl.load(R + row)
    a = dy * w
    dx = r * (a - x * (tl.sum(a * x, 0) / H) * r * r)
    tl.store(DX + row * H + c, dx, c < H)


@triton.jit
def _rms_dw(X, DY, R, DW, M: tl.constexpr, H: tl.constexpr,
            SX0: tl.constexpr, SX1: tl.constexpr, SD0: tl.constexpr,
            SD1: tl.constexpr, RM: tl.constexpr, CN: tl.constexpr):
    # Deterministic column tiles: no atomics and no full [rows,width] scratch.
    c = tl.program_id(0) * CN + tl.arange(0, CN)
    rows = tl.arange(0, RM)
    acc = tl.full((RM, CN), 0, tl.float32)
    for base in range(tl.cdiv(M, RM)):
        m = base * RM + rows
        mask = (m[:, None] < M) & (c[None, :] < H)
        x = tl.load(X + m[:, None] * SX0 + c[None, :] * SX1, mask, 0).to(tl.float32)
        dy = tl.load(DY + m[:, None] * SD0 + c[None, :] * SD1, mask, 0).to(tl.float32)
        r = tl.load(R + m, m < M, 0)
        acc += x * dy * r[:, None]
    tl.store(DW + c, tl.sum(acc, 0), c < H)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        validate(x, w)
        if eps <= 0:
            raise ValueError("eps must be positive")
        m, h = x.shape
        y = torch.empty((m, h), device=x.device, dtype=x.dtype)
        r = torch.empty(m, device=x.device, dtype=torch.float32)
        if m:
            _rms_fwd[(m,)](x, w, y, r, *x.stride(), w.stride(0), h, eps,
                            triton.next_power_of_2(h), num_warps=4)
        ctx.save_for_backward(x, w, r)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, r = ctx.saved_tensors
        m, h = x.shape
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        dw = torch.zeros_like(w)
        if m:
            _rms_dx[(m,)](x, w, dy, r, dx, *x.stride(), w.stride(0), *dy.stride(),
                           h, triton.next_power_of_2(h), num_warps=4)
            _rms_dw[(triton.cdiv(h, 64),)](x, dy, r, dw, m, h, *x.stride(),
                                           *dy.stride(), 32, 64, num_warps=4)
        return dx, dw, None


def rms_norm(x, w, eps=1e-6):
    return RMSNorm.apply(x, w, eps)


@triton.jit
def _swiglu(G, U, Y, DG, DU, DY, H: tl.constexpr, N: tl.constexpr,
            SG0: tl.constexpr, SG1: tl.constexpr, SU0: tl.constexpr,
            SU1: tl.constexpr, SD0: tl.constexpr, SD1: tl.constexpr,
            BACK: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    row, col = idx // H, idx % H
    g = tl.load(G + row * SG0 + col * SG1, idx < N, 0).to(tl.float32)
    u = tl.load(U + row * SU0 + col * SU1, idx < N, 0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    if BACK:
        dy = tl.load(DY + row * SD0 + col * SD1, idx < N, 0).to(tl.float32)
        tl.store(DG + idx, dy * u * s * (1.0 + g * (1.0 - s)), idx < N)
        tl.store(DU + idx, dy * g * s, idx < N)
    else:
        tl.store(Y + idx, g * s * u, idx < N)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        validate(g)
        if u.shape != g.shape or u.dtype != g.dtype or u.device != g.device:
            raise ValueError("gate and up must have matching shape, dtype, device")
        y = torch.empty_like(g, memory_format=torch.contiguous_format)
        if g.numel():
            _swiglu[(triton.cdiv(g.numel(), 256),)](g, u, y, y, y, y, g.shape[1],
                g.numel(), *g.stride(), *u.stride(), 0, 0, False, 256)
        ctx.save_for_backward(g, u)
        return y

    @staticmethod
    def backward(ctx, dy):
        g, u = ctx.saved_tensors
        dg = torch.empty_like(g, memory_format=torch.contiguous_format)
        du = torch.empty_like(u, memory_format=torch.contiguous_format)
        if g.numel():
            _swiglu[(triton.cdiv(g.numel(), 256),)](g, u, dg, dg, du, dy, g.shape[1],
                g.numel(), *g.stride(), *u.stride(), *dy.stride(), True, 256)
        return dg, du


def swiglu(g, u):
    return SwiGLU.apply(g, u)


@triton.jit
def _w4(A, Q, S, Z, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        a = tl.load(A + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        q = tl.load(Q + n[None, :] * (K // 2) + k[:, None] // 2,
                    (k[:, None] < K) & (n[None, :] < N), 0).to(tl.int32)
        s = tl.load(S + n[None, :] * (K // GROUP) + k[:, None] // GROUP,
                    (k[:, None] < K) & (n[None, :] < N), 0).to(tl.float32)
        z = tl.load(Z + n[None, :] * (K // GROUP) + k[:, None] // GROUP,
                    (k[:, None] < K) & (n[None, :] < N), 0).to(tl.float32)
        w = ((((q >> ((k[:, None] % 2) * 4)) & 15).to(tl.float32) - z) * s).to(a.dtype)
        acc = tl.dot(a, w, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def validate_w4(a, q, s, z, group):
    validate(a)
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("W4A16 activation must be FP16 or BF16")
    if group <= 0 or group % 2 or a.shape[1] % group:
        raise ValueError("K must be divisible by positive even group size")
    if q.ndim != 2 or q.shape[1] * 2 != a.shape[1] or q.dtype != torch.uint8:
        raise ValueError("packed weight must be uint8 [N,K/2]")
    if s.shape != (q.shape[0], a.shape[1] // group) or z.shape != s.shape:
        raise ValueError("scales and zeros must be [N,K/group]")
    if s.dtype != a.dtype or z.dtype != torch.uint8:
        raise ValueError("scale dtype must match activation; zero points must be uint8")
    if any(t.device != a.device or not t.is_contiguous() for t in (a, q, s, z)):
        raise ValueError("W4A16 requires same-device contiguous tensors")


def dequantize(q, s, z, group=128):
    values = torch.stack((q & 15, q >> 4), dim=-1).flatten(-2).float()
    return ((values.reshape(q.shape[0], -1, group) - z.float().unsqueeze(-1))
            * s.float().unsqueeze(-1)).reshape(q.shape[0], -1)


def w4a16(a, q, s, z, group=128):
    validate_w4(a, q, s, z, group)
    m, k = a.shape
    n = q.shape[0]
    y = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if m and n:
        _w4[(triton.cdiv(m, 16), triton.cdiv(n, 64))](a, q, s, z, y, m, n, k,
            group, 16, 64, 32, num_warps=4, num_stages=3)
    return y
