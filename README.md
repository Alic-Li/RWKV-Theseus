# RWKV-Theseus

RWKV-Theseus 使用 Parallel Layer Migration，将 Qwen3.8-27B 中的 16 个 Full Attention 层独立迁移为 RWKV7 TimeMix。每个 batch 只运行一次完整冻结原始 Qwen，捕获全部层的 detached input/target，随后逐层独立反向、裁剪和更新。项目不依赖 Lightning、DeepSpeed、Accelerate、Trainer、张量并行或流水线并行。

对于当前 Qwen3.8-27B 权重，模型包含 64 个 decoder block，其中 16 个为 Full Attention，迁移顺序为 `3, 7, ..., 63`；其余 48 个 Gated DeltaNet 层保持不变。训练期间不把任何 TimeMix 装入 teacher；完整 assembled model 仅在全部训练完成后组装、验证和导出。详细设计参见 [架构说明](ARCHITECTURE_zh.md)，验证信息保存在仓库的测试和 `verification.json` 中。

本项目面向研究和工程验证，当前接口、性能和 checkpoint 格式可能随实验迭代调整。


## 安装与环境要求

以下命令均从 `RWKV-Theseus/` 执行。需要 Linux、支持 BF16 的 NVIDIA GPU，以及 CUDA toolkit/nvcc。正式训练支持一张或多张 GPU，每张 GPU 放一份完整文本模型；27B 权重实测约 53.8 GB，另需优化器、缓存、激活和 CUDA workspace。

```bash
# 云服务器：使用已有 UV 环境
source /home/rwkv/alic-li/python_env/py312/bin/activate
uv pip install --index-url https://mirrors.ustc.edu.cn/pypi/simple -e '.[test,tracking]'
export CUDA_HOME=/usr/local/cuda  # 按本机实际路径设置
```

启动脚本使用当前激活环境中的 Python，不依赖项目 `.venv`。其他机器可以使用 UV 创建独立环境。当前明确支持 Transformers 5.8.0 和 5.17.0；其他版本需要自行进行兼容性验证。`requirements-lock.txt` 仅保留初始测试环境记录，不用于强制升级共享环境。未安装 FLA 或 `causal-conv1d` 时，HF 会回退到 PyTorch GDN 实现，功能仍可运行，但吞吐会降低。

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

本项目已在 CUDA toolkit 13.3、PyTorch cu128 和 GCC 15 环境中验证。其他环境可使用兼容的 CUDA、PyTorch 和宿主编译器组合；不要求与该测试环境完全一致。

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

训练启动前会完整校验 manifest 引用文件的 SHA256。训练期间通过只读 NumPy memmap 按需读取数据，不会将全部 token 加载到内存。

### 并行迁移的完整遍历与状态

默认 `training_mode: "epoch"`：所有层共享一次训练 manifest 的完整遍历，每个 token 同时为全部层提供训练样本。不会按 source weight 有放回抽样；weight 只在旧 `sampled` 模式生效。无需重新转换数据。

每篇文档从头到尾切为最多 `context_tokens` 的窗口，窗口再切为最多 `chunk_tokens` 的连续 chunk，所有短尾保留，不 padding、不跨文档。窗口按 seed 确定性打乱并分配给不同 rank；全部层共享同一遍历顺序。窗口内保留 recurrent/cache 状态，窗口切换 reset。

`micro_batch_size > 1` 时，每个 rank 将等长窗口打包成一个 batch，窗口各占一条独立 lane。整组窗口从位置零同步前进，组结束后一起重置 cache 和 TimeMix state。短尾按实际长度分组，最后不足一个 batch 的组直接以较小 batch 运行；所有 token 仍只遍历一次。为了保持独立上下文和避免 padding，打包会改变原先单样本窗口的处理顺序。此模式目前用于 `epoch`，`sampled` 模式保持单样本。

step 数由完整遍历的实际 chunk 数、world size、梯度累积次数计算；epoch 模式忽略 `stage_steps`。tqdm 总数是全部层共同的 optimizer step 数。先耗尽的 rank 通过零损失 dummy 保持 DDP 同步，不增加有效 token，也不更新自己的 recurrent state。最终不完整的累积批次按实际有效 token 归一化。

每 3000 个 optimizer step 保存 checkpoint，训练结束额外保存 complete checkpoint。中途恢复保存每 rank 的窗口编号、chunk offset、已消费 token 数和模型状态。旧 sampled 模式 checkpoint 不能直接作为 epoch 游标恢复；默认配置使用新的 `runs/theseus_parallel`，旧 runs 保留。

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

每张 GPU 运行一份原始冻结 Qwen 和全部独立 TMix。每个微批次只执行一次 teacher forward，各层 TMix 各自参与所有 rank 的 DDP。支持 1、2、3、4、8 等 GPU 数；各 rank 读取独立样本。

默认完整遍历训练集一次，同时训练全部 16 层；每层有独立 AdamW、scheduler 和 recurrent state。完成后自动保存最终 delta checkpoint，重新加载原始 Qwen，一次性替换全部目标层，完成 composition validation 后导出到 `output/converted/`。

先跑一个 optimizer update 的 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 \
bash scripts/torchrun.sh --stop-after 1
```

`--stop-after` 是本次进程累计执行的 optimizer update 数，只用于短测试。它会保存一个未完成的 parallel checkpoint 后正常退出。

中断或 smoke test 后恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 \
bash scripts/torchrun.sh --resume runs/theseus_parallel/latest
```

如果 output 中已经存在 `latest`，必须使用 `--resume` 或指定新的 output 目录；框架会拒绝直接覆盖已有运行。中途恢复要求 world size 和 rank 到设备的映射保持不变；complete checkpoint 仅用于完成或读取最终导出，不会再开始新的训练阶段。

所有进程由同一个 torchrun 启动。Gloo world group 用于配置检查、checkpoint 同步；一个覆盖全部 rank 的 NCCL group 用于 TimeMix DDP。没有跨卡 token/hidden 传输，也没有配对 header/ACK。单卡也通过 `torchrun --nproc_per_node=1` 启动。

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
| `output` | `runs/theseus_parallel` | 日志和 checkpoint 根目录；已有 `latest` 时必须恢复或更换目录 |
| `expected_attention_layers` | `16` | 必须检测到的 full-attention 层数，独立 student 数 |
| `head_size` | `64` | 新 TimeMix 的 head size；CUDA 后端允许 2–128，且必须整除 hidden size |
| `backend` | `cuda` | TimeMix recurrence；正式训练用 `cuda`，`reference` 只用于 CPU 测试 |
| `distributed_backend` | `nccl` | 正式多 GPU 用 `nccl`；`gloo` 只配合 CPU reference 测试 |
| `seed` | `42` | 初始化和确定性数据窗口采样 seed |
| `timeout_minutes` | `60` | torch.distributed process-group 超时 |

序列和训练步数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `chunk_tokens` | `512` | 单次 recurrent forward/backward 的最大 token 数；短尾不会 padding |
| `context_tokens` | `16384` | 一个状态连续窗口的最大长度；超过该边界后重置 HF cache 和 TimeMix state |
| `grad_accum_steps` | `1` | 每次 optimizer update 累积的 chunk 数；只有最后一个 micro-step 做 student DDP 同步 |
| `micro_batch_size` | `1` | 每张卡每次 forward 的独立窗口数；`epoch` 模式可设为 2、4 等，实际最后一组可能较小。每步有效 token 数约为 `micro_batch_size × chunk_tokens × grad_accum_steps × GPU 数` |

示例使用 `micro_batch_size: 8`。并行迁移同时保留 16 层 FP32 参数和 Adam 状态，显存需求与旧逐阶段实现不同；请按实际设备选择 batch/context。

| `training_mode` | `epoch` | 所有层共享一次完整 token 遍历；`sampled` 为旧的定步数采样模式 |
| `stage_steps` | `1000` | 仅 sampled 模式生效，epoch 模式自动计算并忽略此值 |

`stage_steps` 保留旧字段名，表示所有层共同的更新次数；数组形式仅接受各项相等。

一个 update 的标称 token 数约为 `GPU 数 × micro_batch_size × grad_accum_steps × chunk_tokens`；document 尾块可能更短，因此框架最终按所有 student 的实际有效 token 数归一化梯度。`context_tokens` 控制状态连续范围，`chunk_tokens` 控制单次反向显存，两者含义不同。

优化器和损失：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `lr` | `1e-4` | 每层 TimeMix 的 AdamW 学习率 |
| `betas` | `[0.9, 0.999]` | AdamW beta1/beta2 |
| `adam_eps` | `1e-8` | AdamW epsilon |
| `weight_decay` | `0.01` | 仅用于二维、名称以 `.weight` 结尾的矩阵；norm、bias 和 mixing 参数不衰减 |
| `warmup_steps` | `300` | 每层独立的线性 warmup 更新次数；`0` 表示关闭 |
| `constant_steps` | `3000` | warmup 后保持峰值 `lr` 的更新次数 |
| `min_lr` | `3e-6` | cosine 衰减终点学习率 |
| `clip_norm` | `1.0` | 每层梯度范数裁剪阈值；W&B/日志记录裁剪前范数 |
| `cosine_weight` | `0.0` | 兼容字段，必须为 0；只优化 NMSE |
| `loss_epsilon` | `1e-6` | NMSE 分母下限，防止 teacher 能量接近零时数值爆炸 |

主损失按 token 计算：

```text
NMSE = mean((student - teacher)^2) / max(mean(teacher^2), loss_epsilon)
loss_i = mean_token(NMSE_i)
```

保存、日志和验证：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `log_interval` | `10` | 每多少 optimizer step 输出训练指标；第 1 步固定记录 |
| `checkpoint_interval` | `3000` | 训练保存间隔；`0` 关闭定期保存，但训练结束仍保存 |
| `validation_interval` | `100` | 训练验证间隔；`0` 关闭定期验证，但训练结束仍验证 |
| `validation_chunks` | `8` | 每次验证消费的 chunk 数，不是 document 数 |
| `original_teacher_kl` | `false` | 仅最终组装验证时计算原始 M0 对完整 student 的 logits KL；开销较大 |
| `kl_token_block` | `16` | 计算 logits KL 时 lm_head 的 token 分块大小，用于限制峰值显存 |
| `debug_input` | `false` | 保留兼容；parallel 模式直接使用 detached teacher 捕获输入 |

W&B 字段 `wandb_mode`、`wandb_project`、`wandb_entity`、`wandb_name` 见下一节。修改 `output`、日志/保存/验证间隔、超时和 W&B 字段可以恢复；其他会改变训练语义的字段在中途恢复时必须与 checkpoint 一致。

如果要使用另一份配置文件，不经过固定为 `configs/train.json` 的辅助脚本，可直接执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py --config configs/my_train.json
```

## 进度条与多卡效率

rank 0 的 tqdm 每个 optimizer step 只展示 `sum_loss = Σ NMSE_i` 和 `mean_cos = mean(cos_i)` 两个训练指标，附带进度和 ETA；屏幕刷新最多约每 0.5 秒一次，其他 rank 不重复显示。恢复时从 checkpoint 的 step 开始。JSON 日志与 W&B 仍按 `log_interval` 写入，`TQDM_DISABLE=1` 可以关闭进度条。

HF 5.17 且已安装 FLA 时，冻结前缀的 BF16、长度不超过 512 的 GDN 使用 HF 的 fused recurrent 内核分发，支持单样本和批量窗口，包含 512 token 的完整 chunk 和 257～511 token 的短尾；较长序列、CPU 与需要梯度的运算仍使用原 chunk 路径。这减少新序列长度触发的 Triton 编译和慢 rank 等待。计算顺序不同，允许 BF16 舍入差异；基座权重和数学递推不变。`THESEUS_GDN_KERNEL=chunk` 可回退到原内核作对照。更新后需要重启训练进程才能生效；可从已有 checkpoint 恢复，不需要重新转换数据。

训练始终执行完整原始 Qwen forward，再逐层运行 TimeMix 和 fused AdamW。不同 rank 的序列长度、上下文长度和窗口重置位置可以不同；每层按全局有效 token 归一化。

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

仅 rank 0 初始化 W&B。顶层仅记录 `sum_loss` 和 `mean_cos`；每层按实际 decoder 编号记录 `layer_03/nmse`、`layer_03/cosine`、`layer_03/rrms`、`layer_03/grad_norm`、`layer_03/lr`，共 16 组。横轴是 `progress/global_step`。验证指标使用 `val/` 前缀，同样分层。

日志遵循 `log_interval`（首步也记录），验证遵循 `validation_interval`。不上传 checkpoint 或数据。`metrics.jsonl` 同时保留全部分层结果。

同一 output 内 `--resume` 在线恢复会读取 `wandb_run.json` 接续同一 run；变更 project/entity 或 output 会建立新记录。恢复较早 checkpoint 可能重复记录对应优化步数。offline 每次启动单独生成 run 文件，不承诺自动拼接。W&B 配置可在恢复时调整，不改变训练数据和模型恢复逻辑。

## 训练语义

- teacher 始终是完整的原始冻结 Qwen，每个微批次只 forward 一次；16 个捕获点都在 input norm 之后、attention residual 之前，input/target 全部 detach。
- TMix 独立于 teacher 模型注册树。每层执行 forward → 独立 NMSE backward → 本层梯度裁剪 → 本层 optimizer/scheduler step；不会合并 16 层 loss backward。计算图在本层 backward 后释放。
- `grad_accum_steps > 1` 时，每个微批次逐层 backward，只累计梯度，最后一个微批次逐层更新；不保留此前计算图。DDP 平均后按全局有效 token 数归一化，耗尽 rank 用零损失 dummy 保持 collective 顺序。
- 每层拥有 FP32 参数、梯度、Adam 状态；BF16 forward、FP32 loss/recurrent state。TimeMix 和原始 HF cache 各自跨 chunk 延续，state 每块 detach，窗口切换一起 reset。
- 每层独立采用 warmup → constant → cosine 学习率调度。epoch 模式共享一次完整数据遍历，sampled 模式共享 `stage_steps` 次更新。
- 训练期间验证只计算原始 teacher input 上各个 TMix 的独立指标，不运行 assembled model。

NMSE 是每 token 的 `mean((prediction-target)^2) / max(mean(target^2), epsilon)`，随后对全局有效 token 平均；RRMS 为 `sqrt(NMSE)`。cosine 仅用于监控。

## Checkpoint、验证与导出

中间 checkpoint 格式为 `format=2, training_topology=parallel_layers_v1`：`migrated.safetensors` 仅含全部 TMix 权重，`optimizer.pt` 按层保存独立 optimizer/scheduler，`ranks/` 保存各 rank 的 RNG、reader、原始 HF cache 和全部 TMix recurrent state。不会重复写入 Qwen/base 权重。只在所有层完成一次更新后原子提交 `COMMITTED` 和 `latest`。

中途精确恢复要求相同基座、数据和 rank 拓扑；旧 sequential/stage checkpoint 不能作为 parallel 训练的续训起点。允许修改输出、日志间隔和学习率配置，保留 Adam 动量及已完成步数。

全部训练完成后释放训练模型和优化器，重新加载原始 Qwen，仅一次性替换指定层，其他权重不变。最终执行 composition validation（包括有限值检查；`original_teacher_kl` 开启时计算原模型 KL），然后只导出一份完整模型到 `output/converted/`。中途 standalone validation 仍是独立层验证，完成后的 standalone validation 才验证组装模型。

```bash
torchrun --standalone --nproc_per_node=2 validate.py \
  --config configs/train.json --checkpoint runs/theseus_parallel/latest
python export.py --checkpoint runs/theseus_parallel/latest --output runs/export
```

只有全部层训练完成后才能最终导出。训练入口自动导出；`export.py` 用于另行指定最终输出路径。完整导出包含 `config.json`、`theseus.json`、完整模型 safetensors 分片及可用的 tokenizer 文件；通过本项目 loader 直接加载，不再依赖原始 Qwen 目录：

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

该导出结果不是可由标准 Hugging Face `AutoModel` 直接加载的纯 HF checkpoint，也不兼容现有 C++ 推理工程的权重格式。

### 将阶段权重写入旧 GDN 转换产物

如果已有 `convert_qwen2rwkv_lightning.py` 生成的 hybrid `.pth`，可用独立脚本把下载的 Theseus `migrated.safetensors` 写入指定 Full Attention 层。层号从 0 开始，第一阶段是第 3 层。脚本递归扫描权重目录中的 `.safetensors`，按路径中的数字自然排序；同一个参数出现多次时，后面的文件覆盖前面的版本，并在输出中报告覆盖次数。每个指定层必须具有完整的 TimeMix 参数。

```bash
python replace_timemix_lightning.py \
  --base ../converted/Qwen3.8-27B-RWKV7-Hybrid.pth \
  --weights-dir /path/to/downloaded-checkpoints \
  --layers 3 \
  --out ../converted/Qwen3.8-27B-stage01-shared-wkv.pth \
  --dry-run

# 检查通过后去掉 --dry-run；--verify 可额外回读并逐字节校验整个输出。
python replace_timemix_lightning.py \
  --base ../converted/Qwen3.8-27B-RWKV7-Hybrid.pth \
  --weights-dir /path/to/downloaded-checkpoints \
  --layers 3 \
  --out ../converted/Qwen3.8-27B-stage01-shared-wkv.pth
```

可用 `--layers 3,7,11` 一次替换多层。默认将训练时 FP32 的 TimeMix 权重转换为基座 `.pth` 的 dtype；`--dtype source` 保留 safetensors 原始 dtype。输出包括新的 `.pth`、`.config.json` 和 `.manifest.json`；基座文件不会被修改。

输出沿用 GDN 转换产物的 `rwkv_lightning_qwen_hybrid_v1` 格式、plain state_dict、`blocks.<层号>.att.*` 命名和 `[out,in]` 连续矩阵布局。`w1/w2/a1/a2/g1/g2` 从 Theseus 的 `[in,out]` 转置，`[1,1,C]` 广播参数压平为 `[C]`，`r_k` 保留 `[heads,head_size]`；manifest 记录变换。GDN 和 TimeMix 是不同算子，其各自专有参数和形状必须保留，不能把 TimeMix 的 23 个参数冒充 GDN 的卷积、decay/beta 参数。共同投影沿用 `receptance.weight/key.weight/value.weight/output.weight/ln_x.weight` 名称。

`--weights ../migrated.safetensors` 可直接指定单个文件，不必整理目录。可以继续以已替换的 canonical hybrid 为基座替换后续层；也允许将 GDN 层替换为同层号的完整 TimeMix 权重。旧脚本的 `rwkv_lightning_qwen_hybrid_timemix_v1` 产物需从原 hybrid 基座重新导出。

更新后的 `rwkv_lightning_cuda` hybrid 后端按每层实际权重特征选择 GDN、TimeMix 或 Attention；`geometry.layer_types` 用于记录，不强制周期或比例。支持纯 Attention、纯 TimeMix、连续 TimeMix 以及任意三类混排。TimeMix 使用独立 shift 与 FP32 WKV 状态，没有跨层 `v_first`，保留 Qwen RMSNorm 和 SwiGLU FFN。GDN 与 TimeMix 的状态更新统一调用 `rwkv_wkv_effective_fp32_launch`，底层为同一份 D64/D128 模板内核；支持两者都用 128，也支持不同 head size 混用。GDN 保留 conv4，Attention 使用原有 Qwen GQA/RoPE 配置。

转换脚本默认从每层 `r_k` 自动推断 head size，不再默认写死 64。`--head-size 128` 用于校验输入确实为 D128 权重，不会把 D64 权重重新分头。配置会记录 `contract.wkv_kernel=shared_dplr_fp32_v1` 和每个 recurrent 层的 `wkv_layers`。新训练若需要 D128，请在训练配置设置 `"head_size": 128`。现有 `migrated.safetensors` 是 D64，其原始分组会保留。

本机导出及验证命令（工作目录为 `qwen2rwkv`）：

```bash
/home/alic-li/python_env/py312/bin/python RWKV-Theseus/replace_timemix_lightning.py \
  --base converted/Qwen3.8-27B-RWKV7-Hybrid.pth \
  --weights migrated.safetensors --layers 3 \
  --out converted/Qwen3.8-27B-stage01-shared-wkv.pth --verify

rwkv_lightning_cuda/build-hybrid/rwkv_lighting_cuda \
  --model-path converted/Qwen3.8-27B-stage01-shared-wkv.pth \
  --vocab-path Qwen3.8-27B/tokenizer.json --host 127.0.0.1 --port 8000
```

## 测试和探针

```bash
pytest -q
python scripts/probe_model.py --model ../Qwen3.8-27B --tokens 16 --backward
# 仅需 CPU，即可跑原生 torchrun 小模型端到端：
python tests/make_fixture.py /tmp/theseus-demo --stages 16
torchrun --standalone --nproc_per_node=4 train.py --config /tmp/theseus-demo/config.json
```

测试覆盖非零初态、CUDA 前后向、短尾、current stream、整段/分块对齐、FP32 优化器、冻结边界、缓存恢复、DDP 加权梯度、中途/最终提交后恢复、全部 16 层和导出加载。

当前 CUDA 代码是便于检查的基线：训练保存每个 timestep 的 FP32 state，显存约 `4*B*H*(T+1)*N*N` 字节；5120 宽、64 head size、256 token 时约 337 MB。反向用原子归约，未做上游 fused kernels 的性能优化；冻结推理不保存该历史。GDN 可使用 HF 原生 PyTorch fallback，长序列/27B 正式训练吞吐仍需实际多卡测量。

由浅入深训练使用新输出目录 `runs/theseus_parallel`。此前由深到浅的 checkpoint 迁移顺序不同，不能续训到新流程；请不带 `--resume` 启动新训练。旧 runs 和 data 保留不动。
