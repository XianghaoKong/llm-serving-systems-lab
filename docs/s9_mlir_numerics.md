# S9 RMSNorm 数值兼容性：审核说明

发布入口：[S9 扩展汇总与复现](../results/s9/extensions/mlir-decode-20261102/)。

2026-11-01；基线提交 `45a5bb5eddd333a336752aec7c4380ff6eff123d`。

## 2026-11-02 更新：主机路径与真实 decode 图复用

`MLIRRMSNorm.__call__` 现在只做一次参数检查，并缓存库路径、外部库参数和
启动包装器。正常路径不再分配/写回均值诊断缓冲；显式 `.run()` 继续返回均值
供数值检查。每次调用的输出 tensor 仍独立分配，避免覆盖调用者保留的结果。
归约 bitcode、求和顺序、BF16 舍入均未改变。

新增 `ExactDecodeGraph`：对每个提示长度桶准备 31 个精确位置的 decode 图，
每个请求先执行新的 prefill，并将 KV 复制到双缓冲。图内 `torch.cat(...,out=)`
将 KV 写入预分配的另一组连续布局缓冲，保留原 DynamicCache 的真实长度和
stride；不引入容量长度的补零注意力。逐步传入新 token、更新 KV 并重放对应
图，支持同桶不同提示的真实 greedy 续写。内存随最大上下文长度增长，不随
图数量重复保存整份 KV。

运行范围：当前固定模型/Qwen2 SDPA、BF16、batch=1、固定提示长度与 decode
上限、同一 CUDA stream、单个活动请求，调用需在 `torch.inference_mode()`
下且模型保持 eval。`begin/step/reset` 检查生命周期；
修改模型存储、执行后端或长度桶后应重新准备。返回的图 logits 是借用缓冲，
再次使用对应图时会被覆盖。准备耗时与预热请求延迟分开报告。

十轮 Williams 平衡顺序、同一次五后端比较，150 次模型请求。各路径都使用
相同输入及 eager 续写前缀，32 次 forward，不含 HTTP/tokenizer/采样；图路径
的初始 KV 复制和每步 token 复制计入耗时。表内为墙钟中位数，降幅按同轮
配对比值取中位数：

| 提示长度 | 优化前并行 MLIR ms | 精简适配器 ms | 精简 + decode 图 ms | 图相对优化前 |
|---|---:|---:|---:|---:|
| 33 | 409.286 | 366.737 | 249.551 | -39.52% |
| 346 | 411.918 | 366.421 | 263.126 | -36.19% |
| 4096 | 633.053 | 587.923 | 507.869 | -19.79% |

适配器精简和图重放在三个长度的全部十轮中均带来改善。单步图仍执行
769 个 GPU kernel（其中 57 个 RMSNorm），但只需一次 `cudaGraphLaunch`
提交，另有一次 token 复制。CUDA-event 阶段时间仍不能当作纯 kernel 活跃
时间；完整模型墙钟才是主要指标。

每桶图准备耗时约 3.39–3.48 秒（模型/算子已经预热），按本次相对精简路径
的节省估计，需要约 30–45 次同桶请求才能摊平。4096-token 桶的持久 KV
双缓冲为 225.7 MiB，另有图池、logits 和驱动开销；完整分配记录随报告保留。
这仍是 RTX 4070 的本地推理循环测量，未部署为服务，也未在 A100 上复测。

验证：55 项 pytest 通过；三个长度五条检查路径的 480 个 logits 位置逐位
匹配 eager；同一批图换用第二个提示并使用实际 greedy token 反馈也逐位
匹配，32/32 预测一致；零 fallback。优化前适配器源码作为冻结对照重新测量，
不使用历史运行延迟直接比较。

```bash
python src/s9_benchmark_host.py --libraries ../libraries-warp-v1 \
  --before-adapter results/s9/extensions/mlir-decode-20261102/baseline/s9_mlir_adapter_before.py \
  --model /path/to/pinned-model --output ../runs/host-graph-new
```

以下章节保留各阶段历史记录；其中“主机路径尚未优化”的描述属于前一阶段。

## 2026-11-02 后续定位：346-token 延迟并非特殊形状回退

上一轮相对旧 MLIR 的 +1.49% 来自五轮配对比值的中位数；两条路径各自的
墙钟中位数实际为 477.315/477.196 ms。统计口径不同会给出不同方向，少量
样本不足以证明实际退化。原来的循环轮换还让新版本在五轮中的四轮先于旧版
执行，并未充分平衡两者的先后顺序。

追加 10 个 ABBA/BAAB 块，每个长度每路径 20 次请求（五个长度共 200 次）：
346-token 旧/新平均为 419.22/419.42 ms，配对平均变化约 +0.05%；单轮实验
按测试块重采样的平均差区间为 [-1.41,+1.73] ms，跨过零。345/347-token
对照也未证实独有的退化。4096-token 的改善则在 10 块中均出现，约 -10.66%。
这些区间只描述该轮样本，不包含跨时段系统漂移；不能比较不同运行的绝对延迟。

原因定位：346-token prefill 平均减少 6.80 ms，但后续 decode 的 CUDA-event
累计时间增加约 6.52 ms。保持 decode 版本不变，仅替换 prefill 的混合实验
也出现相同现象。异步 CPU/GPU 流水中，旧版较长的 prefill 使提交队列积累；
新版更快消耗队列，等待主机提交的时间随后计入 decode。它不是 decode 的
GPU 算术变慢。固定位置的完整模型 CUDA Graph 验证：prefill 从 29.71 降到
22.70 ms，decode 从 9.53 降到 7.66 ms，输出均逐位一致。

单独的主机提交探针显示整轮主机端执行/提交占约 414–419 ms，最终 GPU
收尾约 1 ms。这里的提交墙钟不是纯 CPU 计算时间，也包含框架/驱动调用及
其中的等待。性能限制已转移到执行与提交路径；单靠减少 GPU 归约工作不再
保证模型循环加速。新增对齐检查约 0.053 µs/次、1824 次约 0.10 ms，无法
解释此前约 7 ms 的方向变化。未将剩余小波动强行归因到某个具体 CPU API。

346 是提示的行数，归约宽度仍为 1536。16/33/346/4096 行的已生成均值
bitcode 在各自版本内 SHA256 相同，没有 346 特有分支；345/347 使用经该
调度等价性验证的诊断映射，逐位比较通过且零回退，正式库支持范围未扩大。
346 之所以反复出现，是因为它一直作为历史请求 req_0994 的重点复现长度；
之前的数值偏差也在 33 和 4096 token 出现过。

基准脚本已改用 Williams 平衡顺序：同时平衡位置、相对先后顺序及相邻
后端组合，默认使用完整周期（四路径 8 块、五路径 10 块）；显式指定不足
周期时在环境记录中标明。新增 19 项顺序性质测试，连同 3 项分析测试共
22 项通过，并完成五后端完整 10 块的小规模集成检查。Pass 和适配器未变。

## 2026-11-02 更新：保序并行归约

新增 `s9.aten_mean_warp` lowering，通过 `build_libraries.py --parallel` 生成。
旧标量 lowering 保留作为 CPU oracle 和性能对照。每行仅启动一个完整 warp，
每个物理 lane 处理 48 个 BF16 元素；不再由每个线程读取并计算整行。

对 ATen 的逻辑 lane `g*32+lane`，保留四个累加器、步长 `blockWidth*4`
和局部左结合。原来跨 warp 的合并变为 lane 内不同 `g` 的寄存器合并，
仍使用同一棵树；最后保持 `1,2,4,8,16` 的 down-shuffle，并广播 lane 0。
这样改变任务映射，不重结合浮点加法。最终缩放、epsilon、rsqrt、BF16 舍入
和权重乘法保持不变；均值与尾部仍链接在同一个 Triton kernel 中。

GPU op 的调用契约要求：一个 warp 的全部 32 lanes 使用相同的、8 字节对齐
的行地址进入 collective；库清单要求 `implementation=warp32,num_warps=1`。
适配器拒绝不匹配的启动配置，并为未对齐但连续的输入复制到对齐分配。
正常对齐输入不需要这次复制。未覆盖的形状/类型继续走原有兼容路径。
shuffle 使用 side-effect inline asm，外部调用为 impure，函数标注
`convergent,noduplicate`。这是显式 GPU contract lowering，仍不是任意 TTIR
归约的自动识别/替换。

同机、同次五后端轮换测试：346 行的 GPU P50 为 2.770 µs，旧标量原型
113.181 µs、ATen 兼容路径 12.210 µs、原全融合 2.378 µs；4096 行分别为
23.774、1256.136、194.438、23.437 µs。所有新 kernel 为 78 个寄存器/线程、
0 字节本地内存，仍为一次 kernel 启动。当前保留 mean 调试输出，以便对照。

3 个合成提示（33/346/4096 tokens）各检查 32 个 logits 位置，逐位一致且零
fallback。相对兼容路径，5 轮配对的模型循环延迟中位变化为 -1.53%、-1.86%、
-3.84%。它不是纯 GPU 活跃时间、生产服务延迟或 A100 结果；不能把算子
40–53 倍于旧原型的加速当作模型加速。共享桌面 GPU 未锁频，短提示收益较小。

验证包括原有 17 组 CPU oracle/GPU 均值及输出对照、6 个 IR 测试方法（含
shape 边界及非法 contract 子用例）、24 个 pytest 用例。新增未对齐输入、
逻辑 warp 边界、FP32 次正规平方、大有限平方、正负零和错误启动契约检查。

```bash
python compiler/s9-numerics/build_libraries.py --tool /path/to/s9-opt \
  --output ../libraries-warp --parallel --parallel --rows 1 2 4 8 16 33 346 4096
python src/s9_benchmark_mlir.py --libraries ../libraries-warp \
  --scalar-libraries ../libraries-v3 --fixture /path/to/captured_input.pt \
  --model /path/to/pinned-model --output ../runs/performance-warp
```

以下内容保留第一阶段数值定位的历史记录；其中“原型尚无性能证明”指当时的
标量版本，不代表上述并行版本的测量结果。

## 本次完成的结果

在本机 RTX 4070 / WSL2 上复现了 S9 全融合 RMSNorm 相对 Qwen 原始路径的数值偏移，定位到求和归约顺序，并实现了一个真正执行 IR Transformation 的 C++/MLIR Pass。Pass 生成的 LLVM bitcode 已链接进实际执行的 Triton kernel。

同一请求、同一续写前缀下，32 个位置的最大 logits NRMSE 从 **3.760032% 降到 0**。修复后的全部 logits 逐位一致，32/32 个 greedy 预测一致；1,824 次 RMSNorm 调用全部使用生成代码，回退次数为 0。

这是数值兼容性原型。尚未证明性能收益，也未在 A100 上复测；不建议直接替换正式性能基线。

## 复现范围与历史结果的区别

软件：PyTorch 2.8.0+cu128、Triton 3.4.0、Transformers 4.57.1、LLVM/MLIR 19.1.1。模型为 Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，BF16、SDPA。

请求采用历史首个失败请求 `req_0994` 的长度 346，CUDA Generator seed=2026，生成 [100,10000) 范围的合成 token IDs；它是原协议的长度匹配回放，不是语义质量评估。FMA contraction 关闭。

| 检查 | 原全融合路径 | MLIR 生成代码路径 |
|---|---:|---:|
| 本机 32 个位置全部检查的最大 logits NRMSE | 3.760032%（零起始位置 21） | 0 |
| 同一次 32 位置回放，仅检查首末位置 | 1.410748% | 0 |
| greedy 预测与参考一致 | 32/32 | 32/32 |

本机最初的 4 位置实验最大 NRMSE 为 1.479692%，其首末位置门槛值为 1.154141%。因此，**没有在 RTX 4070 上重现历史 A100 首个 4 位置请求的 3.005801% 门槛失败**。本次复现的是数值偏移及其传播；扩展检查全部 32 个位置后发现超过 2% 的位置。不能把增强后的检查与历史仅首末检查混为一谈。

## 定位证据

在原模型前向中捕获全部 57 个 RMSNorm 的相同输入、权重和 epsilon，然后分别计算 ATen、原 Triton 和插入观测点的 Triton 版本。

1. 57 层的 FP32 平方全部逐位一致。
2. 57 层的第一次数值差异均位于求和归约。额外 `torch.sum` 诊断与 ATen mean 使用相应的 CUDA reduction schedule；每层都验证 `ATen mean == ATen sum × FP32(1/1536)`，Triton 侧同样成立，排除了此用例的后续缩放差异。
3. 只替换均值为 ATen 结果后，所有层从加 epsilon 到最终输出全部逐位一致。注入实验中的独立 sum 观测仍来自 Triton，不作为注入后的结果。
4. 所有层的仪器化 kernel 输出都与未修改的 S9 kernel 逐位一致，降低了插入观测点改变计算的风险。
5. 五个层的低精度输出发生变化：`layers.1.input_layernorm` 10 个元素、`layers.7.input_layernorm` 20 个、`layers.8.input_layernorm` 6 个、`layers.19.post_attention_layernorm` 6 个、`layers.27.input_layernorm` 17 个。

保存的反例来自 `model.layers.1.input_layernorm`，第一个输出差异在第 236 行（零起始）。此层的 sum 有 113/346 行不同，最大 2 ULP；mean 有 94/346 行不同，最大 3 ULP。经过 BF16 舍入，10 个输出元素不同，输出最大绝对误差 0.0078125。

反例保留完整 `[346,1536]` 输入，而非直接切成一个独立行：ATen 的归约调度依赖总行数，切成单行会改变参考计算顺序。

## Pass 的语义与实际接入

入口是显式注册的自定义操作：

```mlir
%v = "s9.aten_mean"(%squared_f32) {
  rows = 346 : i64,
  contract = "pytorch-2.8.0-cuda-f32-mean-v1"
} : (vector<1536xf32>) -> f32
```

`s9-lower-aten-mean` 将其替换为 `vector.extract`、`arith.addf` 和 `arith.mulf`：四个独立累加器、按原顺序合并累加器、需要时进行跨 warp 合并，最后按 1/2/4/8/16 的偏移顺序实现 warp 内求和，再乘 FP32 缩放因子。没有启用 reassoc、contract 或其他 fast-math 标志。

Verifier 只接受固定 `vector<1536xf32> -> f32`、1–4096 行及指定数值约定。普通 `vector.reduction` 不会被改写；必要的 BF16 截断/扩展保留。这里是**显式数值约定的 lowering**，不是声称对任意浮点加法重排都语义等价。

实际执行链路：

```text
真实 Triton IR 和捕获输入用于诊断
  → 将已定位的 mean 区域表达为带约定的 MLIR 输入
  → C++ Pass 展开 FP32 运算顺序
  → 上游 MLIR lowering / LLVM scalarizer
  → LLVM bitcode
  → Triton extern_libs 链接
  → 原 Qwen 的 rsqrt / BF16 舍入 / 权重乘法
```

当前区域包装由 `build_libraries.py` 显式生成，没有自动从任意 TTIR 中识别并替换 `tt.reduce`，也没有把本 Pass 插入 Triton 自身的 PassManager。LLVM bitcode 是实际执行的编译产物，GPU 验证未使用另一份手写 Triton 求和算法代替 Pass。

使用 bitcode 边界避免把 LLVM/MLIR 19.1.1 的 C++ 库直接装入 Triton 3.4.0 的不同 LLVM 工具链。此次组合已实测可用，但仍应固定版本。LLVM scalarizer 在链接前消除大向量，使 NVPTX 编译可行；它不改变 FP32 加法树。

这一版是展开运算图的正确性实现，未设计高效的协作式 GPU 归约。后续需要在保持计算顺序的条件下，将逻辑 lane 映射到 GPU 线程，并评估寄存器、访存和模型延迟。

## 验证结果

| 验证 | 结果 |
|---|---|
| C++ 编译、链接 | 通过 |
| CTest / FileCheck | 1 套件通过，含 5 个测试方法、11 个行数子例和 9 个非法输入子例 |
| IR 合法性、幂等性、无 Pass 对照 | 通过 |
| 舍入边界保留、普通归约不改写 | 通过 |
| CPU/GPU mean 对 ATen | 17 组用例全部逐位一致 |
| GPU RMSNorm 输出、重复执行 | 17 组用例全部逐位一致 |
| 算子用例总规模 | 9,358 行，14,373,888 个输出元素 |
| 行数覆盖 | 1、2、4、8、16、33、346、4096 |
| 数据覆盖 | 真实捕获输入、固定种子随机输入、零及较大动态范围 |
| 接口及原 S9 分析测试 | 9/9 通过（6 个新增 GPU 测试，3 个已有测试） |
| 模型 | 一个请求 × 32 个位置，全部 logits 逐位一致 |

未覆盖：A100/AMD GPU、200 请求正式回放、其他宽度、训练反向、全部 BF16 位模式及 NaN/Inf/subnormal 特殊语义。逐位一致只适用于上述已测范围。

## 审核入口

- `compiler/s9-numerics/s9-opt.cpp`：操作约定、Verifier、真正的 IR Rewrite。
- `compiler/s9-numerics/build_libraries.py`：生成前后 IR、LLVM bitcode 和 CPU 动态库。
- `src/s9_numeric_probe.py`：原误差复现、观测点一致性和均值注入实验。
- `src/s9_mlir_adapter.py`：生成代码的 Triton 接入、校验和及回退。
- `src/s9_validate_mlir.py`：CPU/GPU 数值验收及模型回放。
- `tests/test_s9_mlir_adapter.py`：非连续输入、FP16/未编译形状回退、空输入和参数拒绝。

建议先审核数值约定是否明确、展开顺序是否匹配来源、参考与增强门槛是否分开报告，再决定是否投入 A100 复测和并行化优化。这是第一阶段的历史审核记录；后续并行化与主机路径成果见本文顶部及 S9 扩展目录，原 A100 数据保持不变。

## 复现命令

使用 Linux / WSL2、可用的 NVIDIA CUDA GPU，安装 LLVM/MLIR 19.1.1、CMake、Ninja 和 Clang 19。在仓库根目录执行；输出目录须为新目录。

```bash
python3 -m venv .venv-s9-numerics
source .venv-s9-numerics/bin/activate
pip install -r compiler/s9-numerics/requirements.txt

cmake -S compiler/s9-numerics -B work/s9-mlir-build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-19
cmake --build work/s9-mlir-build -j2
ctest --test-dir work/s9-mlir-build --output-on-failure

export TRITON_CACHE_DIR="$PWD/work/s9-triton-cache"
# MODEL 指向指定 revision 的完整本地模型；脚本不自动下载或更换 revision。
MODEL=/path/to/Qwen2.5-1.5B-Instruct
python src/s9_numeric_probe.py --model "$MODEL" \
  --output work/s9-probe --length 346 --steps 32

python compiler/s9-numerics/build_libraries.py \
  --tool work/s9-mlir-build/s9-opt --output work/s9-libraries \
  --parallel --rows 1 2 4 8 16 33 346 4096
python src/s9_validate_mlir.py --libraries work/s9-libraries \
  --fixture work/s9-probe/captured_input.pt --output work/s9-validation \
  --model "$MODEL" --replay-dir work/s9-probe
S9_MLIR_LIBRARIES="$PWD/work/s9-libraries" pytest -q \
  tests/test_s9_mlir_adapter.py tests/test_s9_analyze.py
```

若只审核 CPU/IR，可先运行 CMake 和 CTest；这些检查不能替代 GPU 数值验收。

## 依据

- [S9 Qwen 适配器](https://github.com/XianghaoKong/llm-serving-systems-lab/blob/45a5bb5eddd333a336752aec7c4380ff6eff123d/src/s9_qwen_adapter.py)
- [S9 原实验结果](https://github.com/XianghaoKong/llm-serving-systems-lab/blob/45a5bb5eddd333a336752aec7c4380ff6eff123d/docs/s9_results.md)
- [PyTorch 2.8 Reduce.cuh](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/Reduce.cuh)
- [PyTorch 2.8 MeanOps](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/SharedReduceOps.h)
- [PyTorch 2.8 mean kernel](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/ReduceMomentKernel.cu)
- [Triton 3.4 NVIDIA 编译管线](https://github.com/triton-lang/triton/blob/v3.4.0/third_party/nvidia/backend/compiler.py)
