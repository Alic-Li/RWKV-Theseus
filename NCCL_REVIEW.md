# 全 rank DDP 检查

当前训练采用同卡 teacher/TimeMix 分支，全部 rank 参与 TimeMix DDP。旧版 1:1 配对通信已删除；历史配对版的性能数字不能作为新版结果。

- 一张 GPU 一个 rank，启动前检查 BF16、设备数量和 UUID，拒绝重复设备。
- Gloo world 做控制；全 rank NCCL group 做梯度和指标归约，加载模型前预热。
- 只有 TimeMix core 包装 DDP，冻结模型不归约梯度。
- 不等长样本按全局 token 数缩放梯度；累积时 no_sync 同时覆盖 forward/backward。
- 阶段切换释放旧 DDP/optimizer 后广播冻结权重，核对 SHA256。
- checkpoint 在完整 optimizer step 后提交，保存每个 rank 的 reader 和独立 recurrent/cache 状态。
- 单节点硬件测试与跨节点测试分开记录；跨节点尚未实测。

```bash
source /home/rwkv/alic-li/python_env/py312/bin/activate
# CPU：四个训练 rank、不同序列长度、梯度累积和两个阶段。
torchrun --standalone --nproc_per_node=4 scripts/check_nccl.py --backend gloo
# GPU：当前本地蒸馏拓扑下的全 rank DDP。
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 scripts/check_nccl.py
pytest -q tests/test_distributed.py tests/test_resume.py tests/test_nccl_hardware.py
```

`check_nccl.py` 不加载 27B，检查 BF16 CUDA TimeMix、FP32 Adam 状态、每次更新参数一致性与阶段广播。pytest 另用串行基准核对全 rank 加权梯度，并运行真实 HF 小模型的单卡、16 阶段训练和精确恢复。真实 27B 吞吐需单独测量。
