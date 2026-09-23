# RWKV-Theseus

用原生 PyTorch/torchrun 将 Qwen 的 16 个 full-attention 分支从浅到深替换为独立 RWKV7 TimeMix。HF 负责模型加载、Qwen forward、GDN、RoPE、norm 与 MLP；仅当前阶段新层可训练。无 Lightning、DeepSpeed、Accelerate、Trainer、TP、PP。

本地 `Qwen3.8-27B` 的 config 实际类型是 `qwen3_5`，64 个 block 中迁移顺序为 `3,7,...,63`，其余 48 个 GDN 保留。训练仅运行目标层之前的冻结前缀和目标 Attention/TimeMix；目标 block 的 FFN 与后续 block 不执行。详见 [架构设计](ARCHITECTURE_zh.md) 和 [实现验证记录](VERIFICATION.md)。


## 安装

以下命令均从 `RWKV-Theseus/` 执行。需要 Linux、支持 BF16 的 NVIDIA GPU，以及 CUDA toolkit/nvcc。正式训练支持一张或多张 GPU，每张 GPU 放一份完整文本模型；27B 权重实测约 53.8 GB，另需优化器、缓存、激活和 CUDA workspace。

```bash
# 云服务器：使用已有 UV 环境
source /home/rwkv/alic-li/python_env/py312/bin/activate
uv pip install --index-url https://mirrors.ustc.edu.cn/pypi/simple -e '.[test,tracking]'
export CUDA_HOME=/usr/local/cuda  # 按本机实际路径设置
```

启动脚本使用当前激活环境的 Python，不依赖项目 `.venv`。其他机器可以自行使用 UV 创建环境。HF 缓存适配目前显式支持 Transformers 5.8.0 和 5.17.0；版本范围用于避免安装时降级已有 Torch，其他 HF 版本仍需要兼容性测试。`requirements-lock.txt` 保留最初 Torch 2.10 / Transformers 5.8 的本地测试环境记录，不用于升级现有共享环境。没有安装 FLA/causal-conv1d 时，HF 使用自身 PyTorch GDN 实现，可以运行但吞吐较低。

CUDA recurrence 首次使用时通过 `torch.utils.cpp_extension.load` 编译。建议多卡启动前先编译一次：

```bash
python -c 'from theseus.wkv7 import extension; extension()'
```

Qwen3.5 的 48 个 GatedDeltaNet 层依赖 `causal-conv1d` 才能使用优化卷积；缺少它时 Transformers 会回退到明显更慢的 PyTorch 实现。RTX PRO 6000 Blackwell 可在已有环境中安装：

```bash
TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 \
  /home/rwkv/.local/bin/uv pip install \
  --python /home/rwkv/alic-li/python_env/py312/bin/python \
  --index-url https://mirrors.ustc.edu.cn/pypi/simple \
  --no-build-isolation 'causal-conv1d==1.7.0'
```

安装后运行 `python -c 'import causal_conv1d; print(causal_conv1d.__version__)'` 确认扩展可加载。`TORCH_CUDA_ARCH_LIST` 应按机器算力调整；上面的 `12.0` 只针对当前 Blackwell 服务器。

本机测试使用 CUDA toolkit 13.3、PyTorch cu128、GCC 15。可用 `CC=gcc-15 CXX=g++-15` 指定本机已有的编译器；不要求其他机器也使用这个组合。CUDA 翻译单元不包含 PyTorch C++ 头文件，减少 nvcc/宿主编译器兼容问题。

## 数据输入

本项目不包含数据下载、字段猜测或清洗代码。原始数据由使用者在项目外清洗为下面的固定 JSONL；仓库中的 `jsonl_to_manifest.py` 只负责严格校验、套用基座 chat template、分词并生成训练器读取的 manifest。

### 人工清洗 JSONL 的约定

建议把原始数据清洗成 UTF-8 JSONL，一行是一条完整、独立的多轮对话：

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"问题"},{"role":"assistant","content":"回答"}]}
{"messages":[{"role":"user","content":"下一条独立样本"},{"role":"assistant","content":"回答"}]}
```

- 顶层必须是 JSON object，`messages` 必须是非空数组。
- 每个 message 必须有 `role` 和非空字符串 `content`；`role` 只允许 `system`、`user`、`assistant`、`tool`。
- assistant 可额外包含字符串 `reasoning_content`，它会原样交给 Qwen chat template；其他额外 message 字段会被拒绝。顶层除 `messages` 外的元数据会被忽略。
- `system` 最多一条且只能位于首条。其后的首条必须为 user；普通轮次按 user → assistant 交替，assistant 后可以接一条或多条 tool，再接 assistant。完整 document 必须以 assistant 结束。
- 每条 document 至少包含一个 user 和一个 assistant。
- 一行就是一个 document，不跨行延续会话，不把两条无关会话拼在一行。
- 结构化 `tool_calls` 不属于这个固定 schema；如需保留工具调用，应在项目外先按基座所需的文本表示清洗到 `content`。
- 训练集和验证集应在项目外完成去重与划分，避免同一任务或相同 prompt 同时出现在两个分区。

这个 JSONL 是**人工清洗交换格式**，训练器本身不直接读取。分别转换训练集和验证集：

```bash
python jsonl_to_manifest.py \
  --input /path/to/clean/train.jsonl \
  --tokenizer ../Qwen3.8-27B \
  --output data/train \
  --name qwen38-train \
  --workers 128

python jsonl_to_manifest.py \
  --input /path/to/clean/val.jsonl \
  --tokenizer ../Qwen3.8-27B \
  --output data/val \
  --name qwen38-val \
  --workers 128
```

`--tokenizer` 必须是本地 HF 模型/tokenizer 目录，脚本使用 `local_files_only=True`，并直接调用该 tokenizer 的 `apply_chat_template(..., add_generation_prompt=False)`；不复制或改写模板。`--output` 是一个尚不存在的新目录，不是 manifest 文件名。转换成功后其中包含 `manifest.json` 和两个二进制 sidecar；失败时删除临时输出，不留下半成品。

转换器默认使用机器全部逻辑 CPU，也可以通过 `--workers` 显式指定线程数。工作队列最多保留约 `2 × workers` 条记录，主线程始终按原 JSONL 顺序写入，因此不同线程数生成的 token 和 offsets 顺序一致。`tqdm` 显示 document 进度、document/s 和累计 token 数；`--progress-interval` 控制最少多少条 document 更新一次，设为 `0` 可关闭。线程越多不保证越快，建议用实际数据比较 32、64、128；超长对话较多时也要观察内存占用。`--weight` 默认为 1.0，仅写入这个 source 的采样权重。

转换器不会下载、清洗、截断、拼接或划分数据。每行渲染后的全部 token 都参与蒸馏，不支持 assistant-only mask；超长 document 在 epoch 模式下按 `context_tokens` 分窗完整遍历。

### 训练器实际读取的格式

训练集默认入口是 `data/train/manifest.json`，验证集默认入口是 `data/val/manifest.json`。两个分区格式相同且必须分别提供。上述转换命令会生成：

```text
data/
├── train/
│   ├── manifest.json
│   ├── corpus.bin
│   └── corpus.offsets.npy
└── val/
    ├── manifest.json
    ├── corpus.bin
    └── corpus.offsets.npy
```

最小 manifest 示例：

```json
{
  "format": 1,
  "vocab_size": 248077,
  "tokenizer_files": {
    "tokenizer.json": "<sha256>",
    "tokenizer_config.json": "<sha256>",
    "chat_template.jinja": "<sha256>"
  },
  "sources": [
    {
      "tokens": "corpus.bin",
      "offsets": "corpus.offsets.npy",
      "weight": 1.0,
      "tokens_sha256": "<sha256>",
      "offsets_sha256": "<sha256>"
    }
  ]
}
```

文件契约：

- `tokens` 是无 header 的一维 little-endian `uint32` 数组，即 NumPy dtype `<u4`；每个值是一个 token ID。
- `offsets` 是标准 `.npy` 文件，内容是一维 `int64` 数组。N 条 document 必须有 N+1 个 offset。
- 必须满足 `offsets[0] == 0`、严格递增、`offsets[-1] == tokens 数量`；空 document 不合法。第 i 条 document 是 `tokens[offsets[i]:offsets[i+1]]`。
- `tokens`、`offsets` 路径相对 manifest 所在目录解析。启动时会计算并核对两个文件的 SHA256。
- `vocab_size` 必须不大于模型 embedding 的词表大小。
- `tokenizer_files` 可省略；建议提供。提供后，key 是相对 HF 模型目录的文件名，value 是 SHA256，启动时会逐项验证，防止数据与模型 tokenizer/chat template 不一致。
- `sources` 可以有多项；`weight` 必须为正有限数。权重归一化后表示每次选择来源的概率，不是精确 token 比例。
- `documents`、`token_count`、来源名称等字段可以作为额外元数据写入，但当前 reader 不依赖它们。

训练启动前会完整校验 manifest 引用文件的 SHA256，训练期间通过只读 NumPy memmap 按需读取，不把全部 token 加载进内存。

### 每阶段完整遍历与状态

默认 `training_mode: "epoch"`：每个 stage 将训练 manifest 中所有 source 的全部 token 恰好训练一次，然后才进入下一层。不会按 source weight 有放回抽样；weight 只在旧 `sampled` 模式生效。无需重新转换数据。

每篇文档从头到尾切为最多 `context_tokens` 的窗口，窗口再切为最多 `chunk_tokens` 的连续 chunk，所有短尾保留，不 padding、不跨文档。窗口按 seed 确定性打乱并分配给不同 rank；每个 stage 重放相同完整遍历顺序。窗口内保留 recurrent/cache 状态，窗口切换 reset。

step 数由完整遍历的实际 chunk 数、world size、梯度累积次数计算；epoch 模式忽略 `stage_steps`。tqdm 总数是本阶段实际 optimizer step 数。先耗尽的 rank 通过零损失 dummy 保持 DDP 同步，不增加有效 token，也不更新自己的 recurrent state。最终不完整的累积批次按实际有效 token 归一化。

每 3000 个 optimizer step 保存 checkpoint，阶段结束额外保存 complete checkpoint。中途恢复保存每 rank 的窗口编号、chunk offset、已消费 token 数和模型状态。旧 sampled 模式 checkpoint 不能直接作为 epoch 游标恢复；默认配置使用新的 `runs/theseus_shallow_to_deep`，旧 runs 保留。

`training_mode: "sampled"` 仅供旧行为/短测试使用，此时 `stage_steps` 控制采样训练步数。

## 启动与恢复

双卡机器先按 [NCCL 检查说明](NCCL_REVIEW.md) 运行 `scripts/check_nccl.py`，它不加载 27B 权重。

修改 [configs/train.json](configs/train.json) 中模型、数据、输出路径。相对路径按**启动时的工作目录**解析。建议始终从仓库根目录执行：

```bash
source /home/rwkv/alic-li/python_env/py312/bin/activate
cd /path/to/RWKV-Theseus
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# 首次编译/加载 CUDA recurrence。
python -c 'from theseus.wkv7 import extension; extension()'

# 先做通信检查，不加载 27B。
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 scripts/check_nccl.py

# 正式训练。GPUS 必须等于 CUDA_VISIBLE_DEVICES 中的设备数。
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash scripts/torchrun.sh
```

每张 GPU 都运行一份冻结模型和一个当前可训练 TimeMix。冻结前缀只计算一次，当前 FullAttention 和 TimeMix 共享输入；只有 TimeMix 参与所有 rank 的 DDP。支持 1、2、3、4、8 等 GPU 数，不再区分 teacher/student 卡。四卡每步有四份独立样本，旧架构只有两份；相同步数下有效 token 预算随之增加。

默认每个 stage 完整遍历训练集一次；16 个 stage 会在一个命令中自动从浅到深连续运行，不需要手工逐阶段启动。每阶段结束时框架会验证、保存 complete checkpoint、冻结当前 TimeMix、广播权重并自动进入下一阶段。

先跑一个 optimizer update 的 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 \
bash scripts/torchrun.sh --stop-after 1
```

`--stop-after` 是本次进程累计执行的 optimizer update 数，只用于短测试。它会保存一个未完成阶段 checkpoint 后正常退出。

中断或 smoke test 后恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 \
bash scripts/torchrun.sh --resume runs/theseus_shallow_to_deep/latest
```

不要对已有 `latest` 的 output 再次无 `--resume` 启动；框架会拒绝覆盖。阶段中恢复要求 world size 和 rank 对应的设备映射不变，阶段边界 complete checkpoint 可以更换拓扑并从该阶段的数据起点继续。

所有进程由同一个 torchrun 启动。Gloo world group 用于配置检查、checkpoint 和阶段广播；一个覆盖全部 rank 的 NCCL group 用于 TimeMix DDP。没有跨卡 token/hidden 传输，也没有配对 header/ACK。单卡也通过 `torchrun --nproc_per_node=1` 启动。

多节点需要每节点相同的 GPU 数，以及共享的模型、数据和 checkpoint 路径。例如每节点运行：

```bash
torchrun --nnodes=2 --node_rank="$NODE_RANK" --nproc_per_node=8 \
  --master_addr="$MASTER_ADDR" --master_port=29500 \
  train.py --config configs/train.json
```

所有节点的全部 rank 共同同步 TimeMix 梯度。

## 训练配置与超参数

[configs/train.json](configs/train.json) 现在显式列出全部可配置项；未填写的字段仍会使用 `theseus/config.py` 中的默认值。配置中不能出现未知字段。

模型、路径和运行后端：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `model` | `../Qwen3.8-27B` | 本地 HF 基座目录；每个 rank 都加载一份完整文本模型 |
| `train_manifest` | `data/train/manifest.json` | 训练 token manifest |
| `validation_manifest` | `data/val/manifest.json` | 验证 token manifest |
| `output` | `runs/theseus_shallow_to_deep` | 日志和 checkpoint 根目录；已有 `latest` 时必须恢复或更换目录 |
| `expected_attention_layers` | `16` | 必须检测到的 full-attention 层数，也就是 stage 数 |
| `head_size` | `64` | 新 TimeMix 的 head size；CUDA 后端允许 2–128，且必须整除 hidden size |
| `backend` | `cuda` | TimeMix recurrence；正式训练用 `cuda`，`reference` 只用于 CPU 测试 |
| `distributed_backend` | `nccl` | 正式多 GPU 用 `nccl`；`gloo` 只配合 CPU reference 测试 |
| `seed` | `42` | 初始化和确定性数据窗口采样 seed |
| `timeout_minutes` | `60` | torch.distributed process-group 超时 |

序列和阶段：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `chunk_tokens` | `256` | 单次 recurrent forward/backward 的最大 token 数；短尾不会 padding |
| `context_tokens` | `4096` | 一个状态连续窗口的最大长度；超过该边界后重置 HF cache 和 TimeMix state |
| `grad_accum_steps` | `1` | 每次 optimizer update 累积的 chunk 数；只有最后一个 micro-step 做 student DDP 同步 |
| `training_mode` | `epoch` | 每阶段完整遍历全部 token；`sampled` 为旧的定步数采样模式 |
| `stage_steps` | `1000` | 仅 sampled 模式生效，epoch 模式自动计算并忽略此值 |

仅 sampled 模式的 `stage_steps` 数组示例：

```json
"stage_steps": [1000, 1000, 1200, 1200, 1500, 1500, 1800, 1800,
                2000, 2000, 2500, 2500, 3000, 3000, 4000, 4000]
```

一个 update 的标称 token 数约为 `GPU 数 × grad_accum_steps × chunk_tokens`；document 尾块可能更短，因此框架最终按所有 student 的实际有效 token 数归一化梯度。`context_tokens` 控制状态连续范围，`chunk_tokens` 控制单次反向显存，两者含义不同。

优化器和损失：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `lr` | `1e-4` | 当前 stage 新 TimeMix 的 AdamW 学习率 |
| `betas` | `[0.9, 0.999]` | AdamW beta1/beta2 |
| `adam_eps` | `1e-8` | AdamW epsilon |
| `weight_decay` | `0.01` | 仅用于二维、名称以 `.weight` 结尾的矩阵；norm、bias 和 mixing 参数不衰减 |
| `warmup_steps` | `300` | 每个 stage 的线性 warmup 更新次数；`0` 表示关闭 |
| `constant_steps` | `3000` | warmup 后保持峰值 `lr` 的更新次数 |
| `min_lr` | `3e-6` | cosine 衰减终点学习率 |
| `clip_norm` | `1.0` | 全局梯度范数裁剪阈值；W&B/日志记录裁剪前范数 |
| `cosine_weight` | `0.0` | 总损失中 `1-cosine` 的权重；设为 `0.05` 可启用需求中的可选项 |
| `loss_epsilon` | `1e-6` | NMSE 分母下限，防止 teacher 能量接近零时数值爆炸 |

主损失按 token 计算：

```text
NMSE = mean((student - teacher)^2) / max(mean(teacher^2), loss_epsilon)
loss = NMSE + cosine_weight * (1 - cosine_similarity)
```

保存、日志和验证：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `log_interval` | `10` | 每多少 optimizer step 输出训练指标；每阶段第 1 步固定记录 |
| `checkpoint_interval` | `3000` | 阶段内保存间隔；`0` 关闭定期保存，但阶段结束仍保存 |
| `validation_interval` | `100` | 阶段内验证间隔；`0` 关闭定期验证，但阶段结束仍验证 |
| `validation_chunks` | `8` | 每次验证消费的 chunk 数，不是 document 数 |
| `original_teacher_kl` | `false` | 是否额外计算原始未迁移 M0 对完整 student 的最终 logits KL；开销较大 |
| `kl_token_block` | `16` | 计算 logits KL 时 lm_head 的 token 分块大小，用于限制峰值显存 |
| `debug_input` | `false` | 验证时检查 student 输入与本地 teacher 捕获输入是否接近 |

W&B 字段 `wandb_mode`、`wandb_project`、`wandb_entity`、`wandb_name` 见下一节。修改 `output`、日志/保存/验证间隔、超时和 W&B 字段可以恢复；其他会改变训练语义的字段在阶段中恢复时必须与 checkpoint 一致。

如果要使用另一份配置文件，不经过固定为 `configs/train.json` 的辅助脚本，可直接执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py --config configs/my_train.json
```

## 进度条与多卡效率

rank 0 的 tqdm 每个 optimizer step 更新全局 `nmse`、`rrms`、`cosine`，并显示阶段、step/s 和 ETA；屏幕刷新最多约每 0.5 秒一次，其他 rank 不重复显示。恢复时从 checkpoint 的 step 开始。JSON 日志与 W&B 仍按 `log_interval` 写入，`TQDM_DISABLE=1` 可以关闭进度条。

HF 5.17 且已安装 FLA 时，冻结前缀的 BF16、batch=1、长度不超过 512 的 GDN 使用 HF 的 fused recurrent 内核分发，包含 512 token 的完整 chunk 和 257～511 token 的短尾；较长序列、CPU 与需要梯度的运算仍使用原 chunk 路径。这减少新序列长度触发的 Triton 编译和慢 rank 等待。计算顺序不同，允许 BF16 舍入差异；基座权重和数学递推不变。`THESEUS_GDN_KERNEL=chunk` 可回退到原内核作对照。更新后需要重启训练进程才能生效；可从已有 checkpoint 恢复，不需要重新转换数据。

训练在目标 attention 输出处通过临时 hook 返回，不再执行该 block 的 residual/FFN、后续 block 或最终 norm。完整验证仍执行所有层。CUDA AdamW 使用 fused 更新；全 rank 的指标归约不再在前后额外执行 device-wide synchronize。

不同 rank 的序列长度、上下文长度和窗口重置位置仍可不同。GPU 利用率包含 NCCL 等待，不能仅凭瞬时 100% 判断有效计算。需要定位时临时启用：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 GPUS=7 THESEUS_PROFILE=1 \
bash scripts/torchrun.sh --resume runs/theseus_shallow_to_deep/latest
```

`rank_profile` 在日志 step 记录各 rank 的 teacher 耗时、token 长度、上下文位置与 reset 标记。该诊断会额外同步计时，正常训练保持默认关闭。启动命令中的七个设备代表七张卡。数据采样、文件和 manifest 不受上述优化影响。

2026-09-22 云端 7 张 RTX PRO 6000 的对照：保持 `chunk_tokens=512`、`context_tokens=16384`、同一数据顺序和训练超参，分别运行 40 步。排除第 1 步，第 2～40 步均处理 120,493 个有效 token；旧版约 2,115 token/s，保留编译缓存重跑旧版约 2,113 token/s，512-token recurrent 分发约 15,027 token/s。单步超过 1 秒的次数从 16 次降到 0 次；第 40 步 NMSE 差约 `4.4e-5`。这是包含不同长度首次调用开销的短程测试，不能当作整个 epoch 的固定加速倍数。诊断日志位于云端 `/tmp/theseus-perf512-4vv3ythv/`，验证和定期保存在对照中关闭，生产配置未修改。

## W&B 指标记录

可选依赖：`uv pip install --index-url https://mirrors.ustc.edu.cn/pypi/simple 'wandb>=0.19'`，或安装项目的 `.[tracking]`。默认关闭；在训练 JSON 中设置：

```json
{
  "wandb_mode": "online",
  "wandb_project": "RWKV-Theseus",
  "wandb_entity": null,
  "wandb_name": "qwen38-migration"
}
```

在线模式先在已激活环境中运行 `wandb login`，或自行设置 `WANDB_API_KEY`；不要把 API key 写入训练配置。`wandb_entity` 可填个人/团队名称，null 使用账号默认值。无网络时设置 `wandb_mode: "offline"`，记录位于训练 output 下的 `wandb/`，之后可用 `wandb sync <offline-run目录>` 上传。`disabled` 不导入 W&B，不要求安装依赖。模式以训练配置为准。

仅 rank 0 初始化 W&B，训练指标已在 student group 汇总，避免每张卡重复写 run。训练日志同时记录 `seconds_per_step` 和所有 rank 合计的 `tokens_per_second`。只主动记录以下标量：

- `train/loss`：NMSE + cosine_weight × (1 − cosine)，以及 `train/nmse`、`train/cosine`。
- `train/lr`、`train/grad_norm`（裁剪前梯度范数）。
- `train/seconds_per_step`、`train/tokens_per_second`：日志区间平均耗时和所有 rank 的合计有效 token 吞吐。
- `val/nmse`、`val/rrms`、`val/cosine`：实际 student forward 的验证结果；启用 KL 后再记录 `val/original_teacher_kl`。
- `progress/stage`、`progress/stage_step`，图表横轴为跨阶段累计的 `progress/global_step`。

训练日志遵循 `log_interval`（每阶段首步也记录），验证日志遵循原有验证频率。不记录参数/梯度直方图，不上传 checkpoint 或数据；完整 teacher-forced 等指标仍在 `metrics.jsonl`。W&B 初始化失败会报错，需离线时应显式选择 offline。

同一 output 内 `--resume` 在线恢复会读取 `wandb_run.json` 接续同一 run；变更 project/entity 或 output 会建立新记录。恢复较早 checkpoint 可能重复记录对应优化步数。offline 每次启动单独生成 run 文件，不承诺自动拼接。W&B 配置可在恢复时调整，不改变训练数据和模型恢复逻辑。

## 训练语义

- 第 s 阶段 teacher 是前 s−1 个替换已完成的模型；只新增第 s 个替换。
- 每个 rank 使用原 TokenStream 读取自己的数据；在卡内共享同一个隐藏输入，保证两个分支按 token 位置对齐。
- 捕获点在 input norm 之后、attention residual 之前；包括 teacher attention 自身 gate/o_proj。
- 训练时 HF 原生 forward 只运行到目标 decoder 层，原 FullAttention 产生目标，新 TimeMix 独立建图；不执行后续 decoder 层。完整 student forward 只用于验证。
- BF16 forward；当前新层保留 FP32 参数、梯度和 AdamW 一二阶状态；loss 和 WKV state 为 FP32。阶段完成后该层转为冻结 BF16 权重，广播给所有 rank。
- TimeMix 的 previous-input shift 与矩阵状态跨 chunk 连续；HF KV/GDN 状态也连续保留。每 chunk detach，不跨 chunk 回传梯度。
- 上下文窗口结束统一重置所有状态，不无限增长 attention KV。
- TimeMix 不使用跨层 v_first；使用原 block 深度的初始化，所需代码直接放在 `theseus/timemix.py`、`theseus/wkv7.py` 和 `kernels/`，来源 commit 与适配说明写在文件头；上游许可证保留在 [LICENSE-RWKV7](LICENSE-RWKV7)。

主损失为每 token 的 `mean((student-teacher)^2) / max(mean(teacher^2), epsilon)`，再对有效 token 平均；`cosine_weight=0.05` 可启用余弦损失。日志 RRMS 明确定义为 `sqrt(NMSE)`。

配置支持 LR、betas、Adam eps、weight decay、clip、warmup、梯度累积、阶段步数和各类间隔。sampled 模式的 `stage_steps` 可为整数或 16 项数组；epoch 模式自动计算步数。调度每阶段重新开始：第 1～300 步线性 warmup 到 `1e-4`，第 301～3300 步保持 `1e-4`，第 3301 步到阶段最后一步按 cosine 衰减到 `3e-6`。epoch 模式使用完整数据遍历算出的实际阶段步数，不使用 `stage_steps=1000`。短测试阶段不足 3301 步时，只执行走到的 warmup/constant 部分，不压缩调度。步数均指 optimizer step。续训保留 Adam 状态和当前阶段 step；允许修改上述学习率配置，并按已完成 step 计算下一次更新的 LR，旧恒定学习率 checkpoint 也可采用新调度，不重新 warmup。`checkpoint_interval=0`/`validation_interval=0` 关闭中间定期操作，但每阶段末始终保存和验证。

## Checkpoint、验证与导出

每次 checkpoint 含全部迁移层 delta、当前 optimizer/scheduler、每 rank RNG、数据 reader cursor、HF 缓存、TimeMix 状态和位置。只在完整 optimizer step、通信清空后保存。共享目录的所有 shard 写完后提交 `COMMITTED` 并原子更新 `latest`；未提交目录不参与恢复。

checkpoint 标记 `training_topology=local_branch_v1`。旧版配对训练的中途 checkpoint 会拒绝恢复；已完成阶段的 checkpoint 可用于开始下一阶段。

中途精确恢复要求相同配置、基座、数据和 rank 拓扑；允许调整输出路径和日志/保存/验证间隔。阶段边界可改变拓扑并从数据起点重放。基座指纹使用 config/index 内容哈希与权重分片大小/mtime；基座目录必须保持不变，这不等于全量权重内容哈希。不同 CUDA kernel/硬件不保证位级复现。

验证隔离训练缓存和 reader，报告 teacher-forced 与实际 student forward 的 NMSE/RRMS/cosine。开启 `original_teacher_kl` 后，teacher 临时从磁盘恢复原 attention，计算固定原模型 M0 对完整 student 的 `KL(p_M0 || p_student)`，不额外驻留第二个完整模型。lm_head 按 `kl_token_block` 个 token 分块；词表较大，此验证很昂贵。

```bash
torchrun --standalone --nproc_per_node=2 validate.py \
  --config configs/train.json --checkpoint runs/theseus_shallow_to_deep/latest
python export.py --checkpoint runs/theseus_shallow_to_deep/latest --output runs/export
```

只有全部阶段完成后才能最终导出。导出包含 `theseus.json` 和 `migrated.safetensors`，依赖不可变 HF 基座及本项目 loader：

```python
import torch
from theseus.inference import load_export
runner = load_export("runs/export", device="cuda")
# 同一个 stream 的后续 chunk 继续复用 runner；新文档先 runner.reset()
ids = torch.tensor([[100, 101, 102]], device="cuda")
hidden = runner.forward(ids)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = runner.model.lm_head(hidden)
```

不是无需定制就能由 stock AutoModel 加载的纯 HF checkpoint，也未接入现有 C++ 推理工程的权重格式。

## 测试和探针

```bash
pytest -q
python scripts/probe_model.py --model ../Qwen3.8-27B --tokens 16 --backward
# 仅需 CPU，即可跑原生 torchrun 小模型端到端：
python tests/make_fixture.py /tmp/theseus-demo --stages 16
torchrun --standalone --nproc_per_node=4 train.py --config /tmp/theseus-demo/config.json
```

测试覆盖非零初态、CUDA 前后向、短尾、current stream、整段/分块对齐、FP32 优化器、冻结边界、缓存恢复、DDP 加权梯度、中途/阶段边界恢复、全部 16 阶段和导出加载。

当前 CUDA 代码是便于检查的基线：训练保存每个 timestep 的 FP32 state，显存约 `4*B*H*(T+1)*N*N` 字节；5120 宽、64 head size、256 token 时约 337 MB。反向用原子归约，未做上游 fused kernels 的性能优化；冻结推理不保存该历史。GDN 可使用 HF 原生 PyTorch fallback，长序列/27B 正式训练吞吐仍需实际多卡测量。

由浅入深训练使用新输出目录 `runs/theseus_shallow_to_deep`。此前由深到浅的 checkpoint 迁移顺序不同，不能续训到新流程；请不带 `--resume` 启动新训练。旧 runs 和 data 保留不动。
