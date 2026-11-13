# VI Bot Planner

基于 VbotNav-0 公开描述重实现的 System 1 局部导航。S1 使用仿真导航数据训练，视觉端使用冻结的 DINOv2-Small 和 Depth Anything V2 Small。融合 RGB 与 LiDAR BEV 后，16 层条件 Transformer 在 32 维 VAE 潜空间生成 10 个局部轨迹点，每点为 `[x, y, sin(yaw), cos(yaw)]`。

[下载 v0.1.0 权重及仿真视频](https://github.com/jiojol/vi_bot_planner/releases/tag/v0.1.0) · [论文架构对照](docs/paper_alignment.md) · [验证结果](docs/validation.md)

## 运行一次推理

以下为已核验的 Linux x86-64 桌面环境。Python 3.11，CUDA 12.8 版 PyTorch。需要 CPU 运行时，可将安装命令的 PyTorch 索引改为 `https://download.pytorch.org/whl/cpu`，并在运行示例时使用 `--device cpu`。

```bash
git clone https://github.com/jiojol/vi_bot_planner.git
cd vi_bot_planner
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m scripts.fetch_weights
python -m scripts.demo
python -m scripts.demo --mode s1_guided_flow_25d_full
```

下载脚本把本项目的推理权重放入 `artifacts/revision3/model.pt`，从官方仓库拉取两套固定版本的视觉权重，并逐文件校验 SHA-256。下载总量约 300 MB。视觉骨干来自 [DINOv2-Small](https://huggingface.co/facebook/dinov2-small) 和 [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)。

示例读取一帧校园会车仿真的 RGB、点云、深度、位姿与局部目标，输出速度与 10 点轨迹。推理入口为 `models.runtime.load_policy`，组合规划入口为 `navigation.runtime.Navigator`。输入字段及坐标定义见 [默认模式](configs/navigation_input_contract.json) 和 [2.5D 候选](configs/revision5_input_contract.json)。

## 当前范围

- 默认 `s1_astar_v4` 使用 S1 生成轨迹候选，并结合传感器驱动的局部规划。
- 实验模式 `s1_guided_flow_25d_full` 使用同一 S1 权重，加入 2.5D 地形代价、动态目标跟踪和流匹配引导。定向仿真中 17/17 绕行及交通回合干净到达，20/20 高差回合在边缘前安全停止。
- 本仓库提供导航推理和仿真记录。Go2 实机、爬楼执行、Orin NX 延迟以及候选版本的 242 回合全量回归仍待验证。云端 S2 未实现，局部目标由调用方提供。
