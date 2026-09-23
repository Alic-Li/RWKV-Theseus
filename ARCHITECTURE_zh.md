# RWKV-Theseus：同卡分支蒸馏

当前实现使用每卡一份模型、每卡一个本地 teacher 分支、全 rank TimeMix DDP。它替代原来的 teacher/student GPU 1:1 配对，保留外围 JSONL 转换和 manifest 格式；默认使用新增的完整 epoch reader。

## 模型与迁移顺序

HF 原生 Qwen3.5 文本实现加载本地 Qwen3.8-27B 权重，负责冻结的 embedding、GDN、Attention、RoPE、norm 与 MLP。64 个 decoder block 中的 16 个 full-attention 按 `3,7,...,63` 迁移。每阶段只训练一个新的 RWKV7 TimeMix，其余参数冻结。

TimeMix 和 CUDA recurrence 代码直接位于 `theseus/timemix.py`、`theseus/wkv7.py` 与 `kernels/`，上游来源与适配说明保留在文件头，许可证见 LICENSE-RWKV7。不依赖上游 Lightning/DeepSpeed trainer。

## 单卡每个 chunk 的路径

```text
固定数据集 tokens → 冻结 HF 前缀 → input norm → x
                                            ├→ 原 FullAttention → y（无梯度）
                                            └→ DDP(TimeMix) → prediction（有梯度）
                                                        ↓
                                             逐 token NMSE / 可选 cosine
```

当前层包装器同时持有原始 attention 与新增 TimeMix。训练前缀 forward 临时切入 teacher 分支，捕获点在 input norm 之后、attention residual 之前，目标包括 Attention 内部 gate/o_proj。然后独立调用 DDP(TimeMix)，输入就是捕获到的同一份 x。

训练时在目标 attention 安装临时 forward hook，获得输出后退出 HF forward，并在 finally 移除 hook；当前 block 的 residual/MLP、后续 decoder block 和最终 norm 均不执行。此前完成的 KV/GDN cache 更新保留，无需改写 HF 内部 forward。TimeMix 输出之后仅计算 loss/backward，不运行 student 后缀。

按从浅到深迁移，已经迁移的 TimeMix 都在目标层之前，作为冻结前缀参与计算，其 recurrent state 随 chunk 延续。Teacher 是本阶段替换前的混合模型；两个分支共用这份前缀产生的同一个 x，并非另跑原始全 Attention 前缀。局部损失不依赖模型后缀，故可共享此前的全部计算。完整 student 路径仍在验证时执行。

## 状态与损失

每个 rank 保留三类训练状态：冻结前缀的 HF KV/GDN cache、目标 FullAttention 的 KV cache、当前 TimeMix 的 previous-input shift 与 FP32 recurrent state。HF cache 和 TimeMix state 各自更新。同一窗口内连续传递，切换窗口时统一 reset。

一次 forward 处理一个 chunk，默认 256 token。每个 token 在相同位置计算 NMSE；chunk 内 TimeMix 做 BPTT，chunk 边界 detach state。BF16 forward，当前 TimeMix 参数/梯度与 AdamW 状态为 FP32，loss 为 FP32。

```text
NMSE_t = mean((prediction_t - y_t)^2) / max(mean(y_t^2), epsilon)
loss_t = NMSE_t + cosine_weight * (1 - cosine_t)
RRMS = sqrt(mean(NMSE_t))
```

所有 rank 累积 loss 的 token 和后执行 backward。DDP 默认平均梯度，再乘 `world_size / 全局有效 token 数` 得到全局逐 token 平均梯度。不等长 chunk 不会被等权平均。梯度累积只在最后一个 micro-step 同步。

## rank 与数据

每张 GPU 一个 torchrun rank，各自持有完整冻结模型和当前 TimeMix，所有 rank 都训练。支持单卡、奇数卡、偶数卡；多节点要求每节点相同进程数。

一个 Gloo world group 用于控制、配置一致性、提交 checkpoint 和阶段权重广播；一个覆盖全部 rank 的 NCCL group 用于 CUDA DDP 与指标归约。只有 TimeMix core 进入 DDP。没有 pair group、跨卡 hidden 发送或 ACK。

EpochTokenStream 通过 `pair_id=rank, pairs=world_size` 兼容其既有分片接口。这些字段只代表数据分片编号，不再表示 teacher/student 配对。每阶段仍从同一个 seed、manifest 的完整遍历起点开始。改变卡数会改变每步有效 token 数和数据分片；中途精确恢复要求 world size 不变。

## 阶段与 checkpoint

每阶段初始化一个 FP32 TimeMix、DDP、AdamW 和 warmup scheduler。阶段末验证并保存 complete checkpoint，然后释放旧 DDP/optimizer，将该层转成冻结 BF16 TimeMix，移除原 attention，广播权重并进入更深层。16 个阶段自动连续运行。

checkpoint 的迁移权重仅包含各 TimeMix core；原 attention 从不可变基座加载。每 rank 保存 reader cursor、HF cache、TimeMix state 和 RNG；rank 0 保存 optimizer/scheduler 与全体迁移层权重。先写临时目录，所有 rank 完成后写 COMMITTED，最后原子更新 latest。

manifest 使用 `training_topology=local_branch_v1` 区分拓扑。旧配对版的阶段中途 checkpoint 无法精确恢复，因此明确拒绝；只有迁移顺序一致的旧 complete checkpoint 才可以从下一阶段开始。配置、基座和数据指纹继续校验。

## 验证与导出

验证不修改训练 reader 或 cache。每 rank 从独立验证 reader 取样，用同一份模型先运行当前原 attention 分支，再以独立状态重放完整 student。只将有限个验证 chunk 的边界 x/y 暂存在 CPU，报告 teacher-forced 与实际 student forward 的 NMSE/RRMS/cosine。`debug_input` 在验证中额外核对边界输入。

可选最终 logits KL：从磁盘临时恢复所有原 attention 得到 M0，把验证隐藏状态暂存 CPU；恢复 student 后重放相同 tokens，分块计算 `KL(p_M0 || p_student)`。不同时驻留第二份完整模型。所有状态在退出验证时恢复，指标跨全 rank 汇总。

最终 export 只包含迁移 delta 和基座引用；推理时由项目 loader 在 HF 模型中安装 TimeMix。具体启动、数据格式、超参数和 W&B 配置见 README.md。

## 完整 epoch 模式（当前默认）

`EpochTokenStream` 只读现有 manifest/sidecar。所有文档切成不重叠 context 窗口，短尾保留；窗口确定性打乱后按 rank 分片。每个 stage 每个 token 恰好参与一次训练，source 权重不用于重复抽样。每阶段复用相同顺序。

以各 rank 最大 chunk 数除以 grad_accum_steps 向上取整计算 optimizer step 总数。提前耗尽的 rank 用零损失 TimeMix forward/backward 匹配 DDP collective，不修改 recurrent state、不记入 token 数。每步仍按全局有效 token 数归一化。stage_steps 仅在显式 sampled 模式下有效。

checkpoint_interval 默认 3000，阶段结束也保存。reader 保存 epoch_v1 模式、窗口/offset/消费 token 数；旧 sampled 中途 checkpoint 无法恢复成完整 epoch。默认运行目录改为 runs/theseus_shallow_to_deep，保留旧训练结果。

由浅入深训练使用新输出目录 `runs/theseus_shallow_to_deep`。此前由深到浅的 checkpoint 迁移顺序不同，不能续训到新流程；请不带 `--resume` 启动新训练。旧 runs 和 data 保留不动。
