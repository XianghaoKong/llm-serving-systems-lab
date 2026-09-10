"""Independent TileLang kernels; wrapper copies noncontiguous inputs explicitly."""
import functools
import torch
import tilelang
import tilelang.language as T
from s9_kernels import validate, validate_w4


@functools.lru_cache(None)
def rms_factory(m, h, dtype, eps, backward=False):
    b = 1 << (h - 1).bit_length()

    @T.prim_func
    def forward(X: T.Tensor((m, h), dtype), W: T.Tensor((h,), dtype),
                Y: T.Tensor((m, h), dtype), R: T.Tensor((m,), "float32")):
        with T.Kernel(m, threads=128) as row:
            x = T.alloc_fragment((1, b), "float32")
            xx = T.alloc_fragment((1, b), "float32")
            total = T.alloc_fragment((1,), "float32")
            for _, j in T.Parallel(1, b):
                x[0, j] = T.if_then_else(j < h, X[row, j], 0)
                xx[0, j] = x[0, j] * x[0, j]
            T.reduce_sum(xx, total, dim=1)
            total[0] = T.rsqrt(total[0] / h + eps)
            for j in T.Parallel(b):
                if j < h:
                    Y[row, j] = x[0, j] * total[0] * T.cast(W[j], "float32")
            R[row] = total[0]

    @T.prim_func
    def dx_kernel(X: T.Tensor((m, h), dtype), W: T.Tensor((h,), dtype),
                  DY: T.Tensor((m, h), dtype), R: T.Tensor((m,), "float32"),
                  DX: T.Tensor((m, h), dtype)):
        with T.Kernel(m, threads=128) as row:
            x = T.alloc_fragment((1, b), "float32")
            a = T.alloc_fragment((1, b), "float32")
            dot = T.alloc_fragment((1, b), "float32")
            total = T.alloc_fragment((1,), "float32")
            for _, j in T.Parallel(1, b):
                x[0, j] = T.if_then_else(j < h, X[row, j], 0)
                a[0, j] = T.if_then_else(j < h, T.cast(DY[row, j], "float32") * T.cast(W[j], "float32"), 0)
                dot[0, j] = x[0, j] * a[0, j]
            T.reduce_sum(dot, total, dim=1)
            for j in T.Parallel(b):
                if j < h:
                    DX[row, j] = R[row] * (a[0, j] - x[0, j] * total[0] / h * R[row] * R[row])

    return tilelang.compile(dx_kernel if backward else forward,
        out_idx=[4] if backward else [2, 3], target="cuda")


@functools.lru_cache(None)
def dw_factory(m, h, dtype):
    @T.prim_func
    def main(X: T.Tensor((m, h), dtype), DY: T.Tensor((m, h), dtype),
             R: T.Tensor((m,), "float32"), DW: T.Tensor((h,), dtype)):
        with T.Kernel(T.ceildiv(h, 64), threads=128) as block:
            accum = T.alloc_fragment((32, 64), "float32")
            result = T.alloc_fragment((64,), "float32")
            T.clear(accum)
            for base in T.serial(T.ceildiv(m, 32)):
                for i, j in T.Parallel(32, 64):
                    if base * 32 + i < m and block * 64 + j < h:
                        accum[i, j] += T.cast(X[base * 32 + i, block * 64 + j], "float32") * T.cast(DY[base * 32 + i, block * 64 + j], "float32") * R[base * 32 + i]
            T.reduce_sum(accum, result, dim=0)
            for j in T.Parallel(64):
                if block * 64 + j < h:
                    DW[block * 64 + j] = result[j]
    return tilelang.compile(main, out_idx=[3], target="cuda")


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        validate(x, w)
        if eps <= 0:
            raise ValueError("eps must be positive")
        x, w = x.contiguous(), w.contiguous()
        if not x.shape[0]:
            r = torch.empty(0, device=x.device, dtype=torch.float32)
            y = torch.empty_like(x)
        else:
            y, r = rms_factory(*x.shape, str(x.dtype).split('.')[-1], eps)(x, w)
        ctx.save_for_backward(x, w, r)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, r = ctx.saved_tensors
        dy = dy.contiguous()
        if not x.shape[0]:
            return torch.empty_like(x), torch.zeros_like(w), None
        dtype = str(x.dtype).split('.')[-1]
        dx = rms_factory(*x.shape, dtype, ctx.eps, True)(x, w, dy, r)
        dw = dw_factory(*x.shape, dtype)(x, dy, r)
        return dx, dw, None


def rms_norm(x, w, eps=1e-6):
    return RMSNorm.apply(x, w, eps)


@functools.lru_cache(None)
def swiglu_factory(m, h, dtype, back=False):
    @T.prim_func
    def forward(G: T.Tensor((m, h), dtype), U: T.Tensor((m, h), dtype),
                Y: T.Tensor((m, h), dtype)):
        with T.Kernel(T.ceildiv(m * h, 256), threads=128) as block:
            for j in T.Parallel(256):
                index = block * 256 + j
                if index < m * h:
                    g = T.cast(G[index // h, index % h], "float32")
                    u = T.cast(U[index // h, index % h], "float32")
                    Y[index // h, index % h] = g / (1 + T.exp(-g)) * u

    @T.prim_func
    def backward(G: T.Tensor((m, h), dtype), U: T.Tensor((m, h), dtype),
                 DY: T.Tensor((m, h), dtype), DG: T.Tensor((m, h), dtype),
                 DU: T.Tensor((m, h), dtype)):
        with T.Kernel(T.ceildiv(m * h, 256), threads=128) as block:
            for j in T.Parallel(256):
                index = block * 256 + j
                if index < m * h:
                    g = T.cast(G[index // h, index % h], "float32")
                    u = T.cast(U[index // h, index % h], "float32")
                    dy = T.cast(DY[index // h, index % h], "float32")
                    s = 1 / (1 + T.exp(-g))
                    DG[index // h, index % h] = dy * u * s * (1 + g * (1 - s))
                    DU[index // h, index % h] = dy * g * s
    return tilelang.compile(backward if back else forward,
        out_idx=[3, 4] if back else [2], target="cuda")


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        validate(g)
        if u.shape != g.shape or u.dtype != g.dtype or u.device != g.device:
            raise ValueError("gate and up must match")
        g, u = g.contiguous(), u.contiguous()
        ctx.save_for_backward(g, u)
        if not g.numel():
            return torch.empty_like(g)
        return swiglu_factory(*g.shape, str(g.dtype).split('.')[-1])(g, u)

    @staticmethod
    def backward(ctx, dy):
        g, u = ctx.saved_tensors
        if not g.numel():
            return torch.empty_like(g), torch.empty_like(u)
        return tuple(swiglu_factory(*g.shape, str(g.dtype).split('.')[-1], True)(g, u, dy.contiguous()))


def swiglu(g, u):
    return SwiGLU.apply(g, u)


@functools.lru_cache(None)
def w4_factory(m, n, k, dtype, group):
    bm, bn, bk = 16, 64, 32
    @T.prim_func
    def main(A: T.Tensor((m, k), dtype), Q: T.Tensor((n, k // 2), "uint8"),
             S: T.Tensor((n, k // group), dtype), Z: T.Tensor((n, k // group), "uint8"),
             Y: T.Tensor((m, n), dtype)):
        with T.Kernel(T.ceildiv(m, bm), T.ceildiv(n, bn), threads=128) as (bx, by):
            a = T.alloc_shared((bm, bk), dtype)
            w = T.alloc_shared((bn, bk), dtype)
            acc = T.alloc_fragment((bm, bn), "float32")
            T.clear(acc)
            for block in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[bx * bm, block * bk], a)
                for i, j in T.Parallel(bn, bk):
                    nn = by * bn + i
                    kk = block * bk + j
                    if nn < n and kk < k:
                        q = T.cast(Q[nn, kk // 2], "int32")
                        val = T.bitwise_and(T.shift_right(q, (kk % 2) * 4), 15)
                        w[i, j] = (T.cast(val, "float32") - T.cast(Z[nn, kk // group], "float32")) * T.cast(S[nn, kk // group], "float32")
                    else:
                        w[i, j] = 0
                T.gemm(a, w, acc, transpose_B=True)
            T.copy(acc, Y[bx * bm, by * bn])
    return tilelang.compile(main, out_idx=[4], target="cuda")


def w4a16(a, q, s, z, group=128):
    validate_w4(a, q, s, z, group)
    if not a.shape[0] or not q.shape[0]:
        return torch.empty((a.shape[0], q.shape[0]), device=a.device, dtype=a.dtype)
    return w4_factory(a.shape[0], q.shape[0], a.shape[1], str(a.dtype).split('.')[-1], group)(a, q, s, z)
