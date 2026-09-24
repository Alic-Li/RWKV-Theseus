# RWKV-Theseus：Parallel Layer Migration

每个 rank 拥有一份始终冻结的原始 Qwen，以及 16 个独立 DDP TimeMix。沿用原有 manifest、数据 reader、HF Runner、NMSE、原子 checkpoint 和 W&B 接口。原始 Qwen3.8-27B 的替换位置是 decoder 层 `3,7,...,63`；其余 Gated DeltaNet、MLP、embedding、norm 和输出权重不变。

## 每个微批次

1. reader 提供 token chunk；窗口切换时同时重置原始 HF cache 和所有 TMix state。
2. 原始 Qwen 完整 forward 一次，16 个 Capture 同时收集各自 input norm 之后、attention residual 之前的原始 input/target，全部 detach。teacher 无任何 TMix 子模块。
3. 按层执行 `DDP(TimeMix)(teacher_input, own_state)`，独立计算该层逐 token NMSE，单独 backward，立即丢弃所有本层 autograd 引用。state detach 后跨 chunk 延续。
4. 每层独立进行有效 token 归一化、梯度裁剪、AdamW step、scheduler step、zero_grad。`grad_accum_steps > 1` 时仅累计梯度，最后一个微批次逐层更新；计算图仍在每次 backward 后释放。

没有把 16 层 loss 合并后反向，没有已经训练好的 TMix 参与 teacher forward。Parallel 表示所有层共享一次原始 teacher pass 并在同一训练遍历中迁移，TMix 计算按层串行以限制激活显存。

## 损失与分布式

`NMSE_t = mean((prediction_t-target_t)^2) / max(mean(target_t^2), epsilon)`。每层 loss 使用 token sum 反向；DDP 平均梯度后乘 `world_size / global_valid_tokens`，得到全局 token 均值梯度。随后只裁剪本层参数，记录裁剪前范数。cosine 仅作监控，`cosine_weight` 必须为 0。

所有层共享 reader 和更新步数，但分别拥有 optimizer、scheduler 和 recurrent state。耗尽 rank 每层运行零损失 dummy 来保持 collective 顺序，不推进本地 recurrent state；有效 token 计数不会重复乘层数。

BF16 forward；TMix 参数、梯度、Adam 状态和损失为 FP32。原始 HF cache 以及全部 detached input/target 同时驻留，16 份 TMix 参数和 optimizer state 也同时驻留；只有当前层的反向激活存活。其显存需求不能套用旧单层 stage 的测量。

## 日志与验证

CLI 与 W&B 顶层训练指标仅有 `sum_loss = Σ NMSE_i` 和 `mean_cos = mean(cos_i)`。W&B 按实际 decoder 编号分组，如 `layer_03/{nmse,cosine,rrms,grad_norm,lr}`；RRMS 为 `sqrt(NMSE)`。`progress/global_step` 是共同更新步数。验证以 `val/` 为前缀。

训练期间验证使用隔离的原始 teacher cache 和各层临时 TMix state，只比较各层原始 input/target，不组装混合模型、不更改训练 reader/state。完整模型的 composition validation 只在所有层完成后进行。

## Checkpoint 与恢复

格式 2 使用 `training_topology=parallel_layers_v1`，目录名 `parallel_step_XXXXXXXX[_complete]`。

- `migrated.safetensors`：全部 16 层 TMix 权重，键为 `<decoder_layer>.<parameter>`，不包含任何 Qwen/base 参数。
- `optimizer.pt`：按层保存 AdamW 和 scheduler state。
- `ranks/rank_XXXXX.pt`：每 rank reader cursor、RNG、原始 HF cache/cursor，以及全部 TMix detached recurrent state。
- `manifest.json`：配置、层顺序、公共 step/complete、模型和数据指纹、world size 和软件版本。

所有层更新完成后统一原子提交 `COMMITTED` 和 `latest`。中途恢复要求原始模型、数据与 rank 拓扑一致，同时恢复全部层权重、optimizer/scheduler 和流状态。旧 sequential checkpoint 不能继续 parallel 训练，避免把混合 teacher 历史误作原始 teacher 数据。

## 最终组装和独立加载

全部训练完成后，先提交最终 delta checkpoint，释放 teacher、DDP、Adam 和 cache，再重新加载原始 Qwen，将所有目标 attention 一次性替换为训练好的 TMix。执行完整组装 forward 的有限值检查，以及配置启用时的原始 teacher KL。其他 Qwen 权重不变。

仅在此阶段向 `output/converted/` 原子导出完整 safetensors 权重、config、TimeMix 元数据和可用的 tokenizer 文件。使用 `theseus.inference.load_export(path, device=...)` 加载；不需要原始 Qwen 目录。自定义 TMix/cache 仍需本项目 runtime，不使用 stock AutoModel。最终 delta 已提交而导出未完成时，可从 complete checkpoint 恢复来完成组装导出。

TimeMix/WKV7 来源和许可证见 `theseus/timemix.py`、`theseus/wkv7.py`、`kernels/` 及 `LICENSE-RWKV7`。
