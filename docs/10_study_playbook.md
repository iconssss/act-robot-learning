# 项目一学习与复现实战手册

本项目的价值不在于背出一个成功率，而在于能从数据到闭环执行，讲清楚
一个机器人模仿学习系统为何这样设计、如何验证、以及哪些结论仍有边界。
本手册把仓库中的材料组织成可执行的学习路径。

## 1. 先建立全局心智模型

先不读实现，手画下面这条链路，并能对每个箭头说出输入、输出和失败模式：

```text
demonstration episodes
  -> LeRobotDataset samples
  -> current RGB + 14-D proprioception
  -> ACT predicts a future 14-D action chunk
  -> simulator executes selected actions
  -> new observation
  -> replan, evaluate success, inspect videos
```

这里的核心不是普通的 `image -> label` 分类：动作会改变机器人的下一帧
观测，因此训练中用 demonstration 的未来动作作监督，而部署时只能依靠新
观测不断修正。把这句话用自己的话解释清楚，是项目叙述的主线。

## 2. 建议的学习顺序（7 次 60--90 分钟）

### 第 1 次：任务与数据

阅读 `README.md`、`docs/01_dataset.md`，运行或阅读
`scripts/inspect_dataset.py` 与 `scripts/visualize_episode.py`。

必须能回答：一个 episode 是什么；50 个 episode / 20,000 帧意味着什么；
本任务的 observation 为什么是 top RGB 加 14-D state；action 14 个维度
为何要与 state 对齐。对物理维度的含义，以环境定义和数据元数据为准，
不要从张量形状猜测。

练习：任选一个时刻写出 `o_t`、`s_t`、`a_t` 的含义，并说明该帧如何产生
监督对 `(o_t, a_{t:t+H-1})`。

### 第 2 次：ACT 的训练/推理差别

阅读 `docs/02_act_architecture.md`，在 LeRobot 0.6.0 源码中定位 ACT config、
preprocessing、model、loss、`select_action` 与 `predict_action_chunk`。

必须画出：RGB 先由 ResNet-18 编码为视觉特征；14-D proprioception 被投影；
两者与 CVAE latent 一起条件化 Transformer；模型产生未来动作序列。训练时
编码器能看 ground-truth future chunk 来学习 latent，同时有 action loss 和
KL 正则；推理时未来真值不存在，只能用 prior/decoder。

### 第 3 次：action chunk 与闭环

阅读 `docs/04_rollout.md`。用符号区分预测长度 `chunk_size=H` 和实际连续执行
长度 `n_action_steps=N`：前者决定模型一次学习/预测多远，后者决定多久重新
读取观测并规划。在本实验中两者一同变化，因此结论是对 action-chunk
configuration 的比较，不能拆成对单个字段的因果结论。

练习：解释 `(100,100)` 在 50 Hz 下是 2 秒才 replan；为什么这仍是 closed loop，
又为什么其局部执行阶段会累积 open-loop error。

### 第 4 次：训练工程

阅读 `docs/03_training.md`、`configs/` 和 `scripts/train_baseline.sh`。把每个
可复现字段填到自己的实验卡片：dataset revision、LeRobot/PyTorch/Python、GPU、
seed、batch size、steps、learning rate、chunk fields、renderer、checkpoint 路径。

重点理解：CPU smoke test 只为提前暴露 schema、依赖和 shape 问题，不是正式
训练；GPU 训练与 evaluation 必须放 Linux RTX 4090；headless MuJoCo 需要 EGL。

### 第 5 次：指标不等于 loss

阅读 `docs/04_rollout.md` 和 `docs/06_failure_analysis.md`。训练 loss 只衡量
示范分布上的动作拟合；success rate 衡量 policy 自己造成的观测分布中是否完成
任务。行为克隆会有 distribution shift：一个小误差让状态偏离演示，后续观测
更陌生，误差可能累积。

练习：用失败 contact sheet 分别说明 grasp timing、trajectory drift 或
insufficient correction 如何在视频中出现；不要只说“泛化能力不足”。

### 第 6 次：实验与统计边界

阅读 `docs/05_experiments.md`、`docs/08_multiseed_confirmation.md` 和
`results/tables/multiseed_summary.csv`。先复述单次 seed 1000：short 86%、
baseline 70%、long 82%；再复述最终更可信的三种子比较：baseline
`72.67% +/- 10.26`，short `76.00% +/- 8.72`。

正确结论是 short 的观察均值高 3.33 个百分点、值得继续研究更频繁 replanning，
但两个样本波动区间重叠，不能说已证明 short 统计显著更优或对所有任务都更好。
long 仍只有一个种子，是下一步最合理的补实验。

### 第 7 次：口头答辩

先不看答案，完成 `docs/07_interview_prep.md` 的问答，再对照
`docs/08_multiseed_confirmation.md` 修正其中任何仅基于单种子的表述。最后用
30 秒、90 秒、5 分钟三个版本讲述项目；每个版本都要包含任务、数据、ACT、
closed-loop evaluation、真实结果和一个限制。

## 3. 从零重跑的规范方法

原始 checkpoint、数据集缓存和 MP4 已按成本控制要求从云端删除，详见
`docs/09_cloud_cleanup.md`；因此重跑不是“点击已有 checkpoint”，而是一个
有意的可复现实验。

1. 在 Windows 本机维护 Git 和文档，在 `robot-cloud` 执行 Linux/CUDA 命令。
2. 从 GitHub clone 本仓库到云端持久共享盘，而不是容器系统盘。
3. 依据 `environment/` 和 `docs/00_environment.md` 重建隔离环境，固定
   LeRobot 0.6.0、Python 3.12.13、PyTorch 2.11.0+cu130 与数据 revision。
4. 将 Hugging Face、Torch、dataset、checkpoint、日志与视频缓存显式指向共享盘；
   不把密钥、密码或 token 写入仓库、shell history 或实验配置。
5. 先跑 dataset inspection 与 1--3 batch smoke test；记录 observation keys、
   state/action shape 和 loss 是否可算。
6. 用 frozen config 启动 baseline。训练过程保存 config、seed、stdout 日志、
   checkpoint 和 milestone evaluation。
7. 每个最终 checkpoint 至少做 50 个独立 rollout，保存 success count、平均回报、
   成功/失败视频；训练 loss 不能替代该步骤。
8. 仅改变事先声明的实验变量，其他字段保持不变；用多个 seed 后再给出强结论。
9. 将轻量 CSV、config、图表和 contact sheet 提交 Git；大 checkpoint 和 MP4
   留在共享盘或对象存储，并在 README 说明其位置/保留策略。

## 4. 训练与调试技巧

| 现象 | 软件层 | 首先检查 | 处理原则 |
| --- | --- | --- | --- |
| `python`/package import 失败 | Python / dependency | 解释器路径、隔离环境、锁定版本 | 不在 base 环境随意补包 |
| CUDA unavailable | CUDA / PyTorch | `nvidia-smi`、`torch.cuda.is_available()`、torch CUDA build | 本机 WSL CPU-only 正常；GPU 任务转 cloud |
| dataset key 或 shape 不匹配 | dataset / config | inspection 输出、feature keys、state/action dims | 按 metadata/environment 定义修 config，不猜维度 |
| MuJoCo 无显示或初始化失败 | simulation | EGL、renderer env、驱动 | 云端 headless 使用 EGL，再重新跑短 rollout |
| loss 降但 success 低 | model / evaluation | rollout 视频、失败时刻、replanning | 不凭 loss 宣布成功，检查 distribution shift |
| 下载或缓存反复丢失 | storage / network | 持久卷路径、临时文件、校验 | 使用可恢复下载和原子写入，避免把数据留系统盘 |

调参时一次只改变一个问题相关的变量，并用命名清晰的目录记录：
`task / policy / chunk / seed / step / timestamp`。先确认能复现 baseline，才做
ablation；先短跑定位错误，才花 GPU 时间训练 100k steps。

## 5. 面试时的项目叙述模板

“我做的是 LeRobot + ACT 的 ALOHA transfer-cube 视觉模仿学习闭环仿真项目。
我把 top RGB 和 14 维双臂 proprioception 输入 ACT，训练它预测未来 14 维动作
chunk；rollout 时只执行选定的一段动作、读回新观测，再重新规划。我没有只报告
training loss，而是在 50 个独立 closed-loop episode 上计算 success rate，并
保留成功/失败证据。我比较了 `(50,50)`、`(100,100)`、`(150,150)` 的 chunk
configuration。单 seed 时 short 为 86%、baseline 为 70%，后续 baseline/short
各补到 3 个 seed 后，均值为 76.00% 与 72.67%，但波动重叠，所以我的结论是
短 chunk 是有前景的 replanning 设置，而不是已被统计证明的普适最优方案。”

随后主动补一句局限：这是模拟器结果，long 还只有一个 seed，真实机器人迁移还
需要处理 sim-to-real、传感延迟和安全控制。
