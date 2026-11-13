# 第二轮：论文对齐与改进边界

依据用户给出的 `vbotnav.pdf` 图1、§4.2、§5.1–5.2重新核对。原始论文将S1作为端侧执行系统；Stage 2以真实机器人局部轨迹为主，VAE与flow连接细节未完全公开。本工程仍是纯仿真重实现，不声称等同作者训练流程。

## 这次实际修改

| 项目 | 论文明确描述 | 本次实现与边界 |
|---|---|---|
| RGB骨干 | DINOv3语义、DA2-Small几何 | 官方DINOv3仍401。明确命名的DINOv2-Small替代 + 官方DA2-Small，真实预训练冻结特征；不把DINOv2叫作DINOv3 |
| DA2输入 | 几何特征 | 改为DA2训练过的backbone末层384维patch特征，包含空间信息；不再只用单通道深度输出，不当米制深度 |
| 融合 | 双向FiLM、BEV queries cross-attention、gated fusion、Q-Former | 保留这些结构，Q-Former采用2层query Transformer（层数和具体结构是本项目选择） |
| 生成器 | 16层flow-matching Transformer，environment/goal/noise/time条件 | 16层独立初始化的AdaLN条件Transformer，目标点每层显式注入；AdaLN-Zero、EMA、余弦调度是工程选择，非论文已公开参数 |
| 位置编码 | 细节未公开 | 修正旧版整数2**arange在高维溢出和极高空间频率，采用浮点1…32频率；是实现修正，不是论文原设定 |
| 标签 | 10×(x,y,sinψ,cosψ) | 不变；固定中心[0,0,0,1]、尺度[1,1,.5,.5]，在归一化空间保持标准直线Flow路径与Y−ε目标，最终才归一化朝向 |
| 纠正数据 | §5.2描述偏航、错误转向、overshoot等纠正；Stage 1明确描述DAgger | 本次仅训练布局生成3200个受扰恢复样本，一半临近目标，标签为真实执行未来；这是针对S1的仿真扩展，不声称复现作者S1 DAgger |
| 采样 | 细节不足 | 训练标签分层采样：普通、转弯、停止及恢复样本；不使用验证/测试补训练 |
| 选择模型 | 未给可直接复现参数 | 只根据验证集ADE、朝向和0.4秒控制前缀误差选检查点；旧测试已查看，最终旧清单只作为固定对比，不宣称首次盲测 |

冻结视觉特征来自原始320×240RGB，采用官方processor，预处理内参保存在 `artifacts/revision2/preprocessing.json`。DINOv2 resize256×341、center crop224×224；DA2 resize392×518。特征缓存按split、manifest、权重校验与处理版本绑定。LiDAR仍只使用当前射线命中/有效mask，不引入深度真值、完整地图或未来观察。

`evaluation/oracle_tracker.py` 是完整地图与未来执行轨迹特权诊断，仅验证轨迹跟踪器是否能执行正确监督，不能当作学生成绩。原始权重、600回合结果和旧源码均保留。

## 来源

- 本地论文 `../vbotnav.pdf`，图1和§4.2/§5。
- https://arxiv.org/abs/2610.14634
- https://github.com/facebookresearch/dinov3
- https://huggingface.co/facebook/dinov2-small
- https://github.com/DepthAnything/Depth-Anything-V2
- https://github.com/facebookresearch/flow_matching
- https://github.com/facebookresearch/DiT/blob/main/models.py （AdaLN-Zero设计参考，不代表VbotNav论文的确定细节）

## 实际学生状态纠正轮

旧学生策略真实驱动6个训练布局回合（expert mixing=0），获得491个合格纠正窗口，5个教师未来碰撞窗口被丢弃。该批数据与3200个扰动恢复样本聚合后，直接Flow额外训练3000步。它是单独候选 `direct_flow_dagger`。最终选中的latent Flow使用原训练集+3200恢复样本，**没有使用这491个额外窗口**；不得把最终提升直接归因于这轮DAgger。
