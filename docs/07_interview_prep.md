# 面试准备：LeRobot + ACT 视觉模仿学习闭环仿真项目

> 使用方式：先用自己的话回答，再对照本文件。不要逐字背诵；应掌握逻辑、边界和项目中的真实证据。

## 1. 项目一句话与自我介绍版本

### 30 秒版本

我完成了一个基于 LeRobot 和 ACT 的视觉模仿学习闭环仿真项目，任务是 ALOHA simulated transfer cube。我从数据集结构、ACT 模型数据流开始，训练了视觉加 proprioception 的 ACT policy，并在 MuJoCo 仿真中做了 50-episode closed-loop rollout evaluation。100k-step baseline 达到 70% success；我又做了 action-chunk configuration ablation，短 chunk `(50,50)` 达到 86%，baseline `(100,100)` 为 70%，长 chunk `(150,150)` 为 82%。项目还包含可复现配置、视频、失败案例和 GitHub 工程化文档。

### 90 秒版本

这个项目的目标不是跑通官方 demo，而是完整理解机器人 imitation learning 的闭环链路：`observation -> dataset -> ACT -> action chunk -> environment -> rollout -> evaluation -> ablation`。

我使用 LeRobot 0.6.0 的 ALOHA simulated transfer-cube 数据集。每个时间步的输入包括顶视 RGB 图像和 14 维双臂 proprioception，监督信号是 14 维 robot action。ACT 不只预测下一步动作，而是预测未来动作 chunk；推理时环境执行 chunk 中选定的动作，获得新的观测后再重新规划。

工程上，我先完成 dataset inspection、episode visualization 和一批 CPU/GPU smoke test，再在 RTX 4090 上训练 100k steps。baseline 在 50 个独立 closed-loop rollout 中达到 70% success。我随后固定数据、模型、seed、batch size 和训练步数，只比较 `(chunk_size, n_action_steps)` 为 `(50,50)`、`(100,100)`、`(150,150)` 的 action-chunk configuration，结果为 86%、70%、82%。

我还处理了两个真实部署问题：headless 云容器中的 MuJoCo 必须使用 EGL 渲染后端；LeRobot 0.6.0 resume 需要使用等号形式的 `--config_path=/...`，否则无法从 checkpoint 恢复。最终我用真实 rollout 视频做了 failure analysis，而不是仅根据 training loss 下结论。

### 面试中必须主动说明的边界

- 消融结果使用固定 seed 1000；它是工程上的对照证据，不是统计显著性结论。
- short 的 86% 高于 baseline 的 70%，说明在此 benchmark/seed/协议下更频繁 replanning 有利；不能说“短 chunk 在所有任务上都更好”。
- 所有结果来自仿真，不等同于直接部署到真实机器人。

## 2. 项目事实速查

| 项目项 | 实际值 |
| --- | --- |
| 数据集 | `lerobot/aloha_sim_transfer_cube_human` |
| 数据集 revision | `6a43d500f101255823a9d2b9dc244eeb01a2cd31` |
| 环境 | `AlohaTransferCube-v0`，50 Hz |
| 数据规模 | 50 episodes，20,000 frames |
| 观测 | 顶视 RGB + 14-D robot state |
| 动作 | 14-D dual-arm action |
| 模型 | LeRobot 0.6.0 ACT，ResNet-18，CVAE latent dim 32 |
| 训练 | RTX 4090，batch size 8，100k steps，seed 1000 |
| baseline | `(chunk, execute)=(100,100)`，70.0% = 35/50 |
| short | `(50,50)`，86.0% = 43/50 |
| long | `(150,150)`，82.0% = 41/50 |
| 最终评估 | 每个条件 50 closed-loop episodes，最多 400 env steps/episode |

## 3. 核心问答

### Q1. ACT 是什么？

ACT 是 **Action Chunking with Transformers**，一种面向机器人模仿学习的视觉控制 policy。它使用视觉和机器人 proprioception 作为条件，通过 Transformer 一次预测未来的一段连续机器人动作，而不是只输出下一步动作。原始 ACT 还使用 CVAE 表示同一观测下可能存在的多模态动作策略，例如不同但都可行的抓取轨迹。

在我的项目中，ACT 接收顶视图像和 14 维双臂状态，输出 14 维动作组成的未来 action chunk。

### Q2. 为什么机器人 policy 要预测 action chunk？

单步 policy 每一步都要决定一个动作，容易缺少短时轨迹的一致性。action chunk 让 policy 同时表达“接下来如何接近、抓取、移动”的局部行为计划，因此更容易学习平滑、协调的 manipulation trajectory。

但预测 chunk 不代表永远 open loop 执行到底。系统仍要不断从环境获得新观测并重新规划；chunk 是局部计划，而 closed-loop feedback 用来修正误差。

### Q3. action chunk 和 single-step policy 有什么区别？

single-step policy 建模的是 `a_t = pi(o_t)`；action-chunk policy 建模的是 `a_{t:t+H-1} = pi(o_t)`。后者输出未来 H 步的动作序列，监督目标也从单个 action 变为未来 action window。

single-step 的反馈频率高，但难以显式保证多个动作之间的时序一致性。chunk policy 更有轨迹先验，但 chunk 过长时预测末端更容易偏离真实环境，产生 open-loop error。

### Q4. observation、state、action、episode 分别是什么？

- **observation**：policy 在时刻 `t` 可获得的信息；本项目中包括 top RGB image 和 robot state。
- **state / proprioception**：机器人自身传感状态，例如左右臂关节位置和夹爪位置；这里是 14 维。
- **action**：policy 发送给环境/控制器的控制命令；这里是对应的 14 维双臂动作。
- **episode**：从 `env.reset()` 开始，到任务成功、失败或时间上限的一条完整轨迹。数据集有 50 条 demonstration episodes；评估也使用独立 episode。

### Q5. 为什么 proprioception 很重要？

图像能告诉 policy 物体和机器人在画面中的相对位置，但不能精确、直接地提供当前关节角、夹爪开合等内部状态。proprioception 让 policy 知道“机器人此刻实际处于什么姿态”，因此能把视觉目标转换为可执行的关节动作。视觉和 proprioception 是互补的：前者主要提供外界语义和几何线索，后者主要提供精确的机器人自身状态。

### Q6. image encoder 为什么必要？

transfer cube 是视觉操作任务。仅凭 robot state 无法知道方块在桌面上的位置、是否被抓住或是否已放置。image encoder 将 RGB 图像压缩为语义特征/visual tokens，使 Transformer 能够把物体位置和场景关系与当前 robot state 结合。

本项目使用 ResNet-18 作为视觉 backbone。它不是直接输出动作，而是先提取图像特征，再由 ACT 融合状态和视觉信息。

### Q7. ACT 的 CVAE latent 在训练和推理阶段分别怎么工作？

机器人示范常有多模态性：面对相似观测，可能存在多种可行的未来轨迹。ACT 用 CVAE latent 表示这种潜在的行为模式。

- **训练时**：编码器可以看到 ground-truth future action chunk，并据此产生 latent distribution；decoder 在 observation、state 和 latent 条件下预测 action chunk。训练目标包含 reconstruction/action loss 与 KL regularization。
- **推理时**：没有 ground-truth future action，因此不能使用训练编码器从未来动作推断 latent。policy 使用先验（常见实现中为标准正态/确定性先验行为）来生成动作。

核心点是：训练时的 future actions 只能用于学习 latent 和监督，推理时不可获得。

### Q8. 为什么 inference 时没有 ground-truth future action？

ground-truth future action 只存在于 demonstration dataset 中。真实 rollout 时 policy 正在自己控制机器人，未来环境状态和未来动作都尚未发生。若推理时依赖 future ground truth，就发生了 information leakage，模型无法在真实机器人运行。因此 inference 只能使用当前可观测的 RGB、state 和 policy 的 latent prior。

### Q9. Transformer 如何生成未来动作序列？

视觉 backbone 先将图像编码为 visual features，robot state 被投影到同一特征空间；ACT Transformer 对这些条件信息和 action-query/decoder tokens 做 attention，输出多个时间位置对应的 action embedding，最后映射为连续的 14 维动作。每个输出位置对应未来 chunk 中的一个时间步，因此一次 forward 得到一个未来序列。

### Q10. training 与 inference 的数据流有什么差异？

训练：

```text
current observation + ground-truth future action chunk
-> ACT/CVAE -> predicted action chunk
-> action reconstruction loss + KL term -> backpropagation
```

推理：

```text
current observation -> ACT prior/decoder -> predicted action chunk
-> execute selected actions -> new observation -> replan
```

最大差异是训练有 future action label，推理没有；推理必须依赖环境反馈闭环纠正。

### Q11. chunk_size 和 n_action_steps 分别影响什么？

- `chunk_size`：model 预测和训练监督的未来 action sequence 长度，即预测 horizon。
- `n_action_steps`：推理时从一次预测 chunk 中连续选择多少动作后再次调用 policy，即 replanning interval。

本项目将它们成对设置为 `(50,50)`、`(100,100)`、`(150,150)`，所以实验测量的是 joint action-chunk configuration。50 Hz 下，它们分别对应约 1、2、3 秒的预测/执行范围。

### Q12. 为什么需要 replanning？

机器人执行动作会改变环境，也会受到状态估计误差、接触不确定性、抓取滑动和视觉误差影响。一次长 open-loop sequence 的后半段是基于旧 observation 预测的，误差会逐步积累。replanning 在取得新图像和 state 后重新预测，使 policy 可以根据“方块是否真的抓住、手臂是否偏移”进行纠正。

### Q13. chunk 执行完才 replan 和 receding-horizon execution 的区别？

如果整个 chunk 执行完才 replan，那么 interval 等于完整 chunk，open-loop 时间最长。receding-horizon execution 则每次只执行预测 chunk 的前 `N` 步（`N` 可小于 chunk size），随后用新观测重新预测一个重叠的未来 chunk。

在本项目中 `n_action_steps=chunk_size`，因此每个条件执行完整 chunk 后再 replan。这让 action chunk 配置影响更明显，但也意味着它不是“固定预测 horizon、仅变执行步数”的纯控制实验。

### Q14. normalization 为什么对机器人 action 很重要？

不同关节、夹爪和图像特征的数值范围可能不同。若不归一化，幅度大的维度会主导 loss 和梯度，导致学习不稳定；网络输出的动作尺度也可能不合理。LeRobot 使用 dataset statistics 对 observation/action 做 normalization，并在输出 action 前做对应的反归一化，使训练数值稳定且环境收到正确物理尺度的控制量。

### Q15. rollout 为什么是 closed-loop？

因为动作影响环境，环境产生的新 observation 又成为 policy 下一次预测的输入：

```text
o_t -> policy -> action -> env.step -> o_(t+1) -> policy
```

它不同于 offline dataset replay。rollout 会暴露模型自己造成的偏离，例如抓取失败后看到的画面不再属于 demonstration distribution。

### Q16. 为什么 training loss 低不代表 task success rate 高？

loss 通常在 demonstration distribution 上衡量动作重建误差，而 task success 取决于闭环执行。一个小的早期动作误差可以让夹爪错过物体；此后 policy 进入训练未覆盖的 observation，误差会累积。任务成功还受接触、时序、末端放置等离散事件影响，所以 loss 和 success rate 不必单调对应。

### Q17. behavior cloning 的 distribution shift 是什么？

behavior cloning 在 expert demonstrations 的 state distribution 上训练；部署时 policy 的小错误会把机器人带到 expert 很少访问的状态。这些新 observation 不在训练分布内，policy 更容易再犯错，形成 distribution shift。

### Q18. compounding error 是什么？

第一个动作的小误差改变下一帧 observation，第二个动作基于一个偏离的状态作出预测，之后误差继续积累。在长时序 manipulation 中，这种逐步偏离就是 compounding error。闭环 replanning 可以缓解，但不能自动消除，因为 policy 仍可能没有见过偏离后的状态。

### Q19. 你的 chunk ablation 得到了什么结论？

固定 dataset、ACT architecture、seed、batch size、100k training steps 和 50-episode evaluation protocol 后：short `(50,50)` 是 86%（43/50），baseline `(100,100)` 是 70%（35/50），long `(150,150)` 是 82%（41/50）。

在该固定 seed 和该任务上，更频繁的反馈/replanning 与更高 success rate 一致。一个合理解释是短 interval 能更快纠正抓取和传输中的偏差；但是 long 也高于 baseline，说明关系不单调。结论必须限定为单 seed 工程实验，不能说成普遍规律。

### Q20. 你的失败案例说明了什么？

我没有只写“泛化能力不足”，而是从 LeRobot 的 per-episode success labels 中索引 56 个录制视频，识别出 20 个失败视频，并人工复核了 6 个 contact sheet。可观察到的失败包括：接近后未形成稳定抓取、方块到两臂之间但未完成 handoff、部分 transport 后未到达最终 placement。

long chunk 的部分失败中出现“方块偏移后动作继续而没有及时恢复”的现象，这与长 open-loop interval 的假设一致，但单个视频不能证明因果；我把它标为 hypothesis，而不是已证实机制。

### Q21. 你遇到过哪些真实工程问题？

1. 云容器没有桌面显示。MuJoCo/dm_control 在首次 evaluation 时因 OpenGL context 失败。通过设置 `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl` 使用 GPU-backed EGL headless backend 修复。
2. LeRobot 0.6.0 resume 失败。源码显示 resume resolver 重新读取 `sys.argv`，只识别 `--config_path=/path`，而不是空格形式。改为等号形式后成功从 20k checkpoint 恢复，并保留 optimizer、RNG 和 data order。
3. Hugging Face 直连受限。将 HF cache 放在共享存储，并使用命令作用域的镜像变量；没有把 cache 放进 Git 或容器临时盘。

### Q22. 为什么 checkpoint、dataset、cache 要放共享存储？

云容器可能释放或重启，系统盘不适合作为唯一副本。共享存储保证训练 checkpoint、dataset、torch weights、HF cache、evaluation videos 和 logs 可在容器生命周期变化后保留。代码本身通过 Git/GitHub 版本控制；运行产物放 shared volume，二者职责不同。

### Q23. 如果换成 Diffusion Policy，数据流有什么变化？

系统外层基本不变：dataset 仍提供 observation/action trajectories，environment 仍做 closed-loop rollout，evaluation 仍看 success rate 和视频。

变化在 policy head 和训练目标：ACT 用 Transformer/CVAE 直接预测 action chunk；Diffusion Policy 通常从噪声 action trajectory 出发，条件于 observation，通过多步 denoising 生成未来动作序列。推理时间可能更高，因为每个 action chunk 需要多次 denoise。action horizon、replanning 和闭环评估仍然需要。

### Q24. 如果未来换成 π0.5，哪些部分不变，哪些变化？

不变的部分：数据收集/版本管理、observation/action schema 校验、训练/验证划分、checkpoint 管理、rollout loop、success-rate evaluation、视频与 failure analysis。

变化的部分：模型输入可能增加 language instruction、多相机或其他 modality；processor/tokenizer、预训练权重、显存需求、fine-tuning strategy、action representation 和 inference latency 都可能变化。也就是说，policy 模型可替换，但机器人学习工程闭环仍保留。

### Q25. 你为什么没有立即做 RGB + state vs state-only 消融？

该问题有价值，但当前 LeRobot ACT 配置不一定支持“关闭视觉”而不改变 processor、feature schema 或 model path。为了避免把 framework 改动和 modality effect 混在一起，我优先完成了对现有实现直接支持、且控制变量清晰的 action-chunk ablation。后续若实现 state-only，需要验证 action/state preprocessing、network input path 和 evaluation protocol 都保持公平，然后重新训练与评估。

### Q26. 如果继续改进这个项目，你会怎么做？

优先级是：

1. 对 baseline/short/long 增加随机种子，报告均值和方差或置信区间；
2. 对最终 checkpoint 做 100-episode evaluation，减少 50-episode 测量噪声；
3. 做公平的 visual/state modality ablation；
4. 扩大 object initial state 或 camera condition，测试 robustness；
5. 若转向真实机器人，增加安全限幅、action rate、延迟测量和 sim-to-real calibration。

## 4. 追问与回答策略

### “86% 比 70% 是否显著？”

不能仅凭一个 seed 断言统计显著。50 个 episode 下差异有工程意义，但 evaluation episodes 不是独立训练重启。应通过多个 training seeds 重新训练，报告每个配置跨 seed 的均值和方差/置信区间。

### “为什么 long 82%，仍高于 baseline？”

说明性能不只由 replanning interval 决定。chunk horizon 也改变了学习目标，且训练本身有随机性；在本实验中 long 的局部轨迹先验可能仍有效。正确结论是 short 最好、关系非单调，而不是简单宣称“越短越好”。

### “为什么 baseline 20k 没有结果？”

20k checkpoint 已保存，但首次 environment evaluation 暴露了 headless OpenGL/EGL 配置问题，在 aggregate metric 产生前中断。我修复后从 checkpoint 恢复；40k 到 100k 使用一致的 EGL 协议。20k 被显式标记为 missing，不纳入趋势比较。

### “怎样保证实验可复现？”

我固定了 LeRobot release、dataset revision、seed、batch size、steps、policy config、GPU/driver/PyTorch/MuJoCo 版本；将 YAML config、训练日志、checkpoint、evaluation results、视频索引和系统记录分别保存；代码与文档推送 GitHub，重数据放持久共享存储。

## 5. 面试回答检查清单

每次回答技术问题时，尽量覆盖：

1. **定义**：这个概念是什么；
2. **机制**：在数据流或代码中如何发生；
3. **取舍**：为什么这么设计，有什么代价；
4. **证据**：我的项目中哪个配置、指标或视频支持它；
5. **边界**：哪些结论尚需多 seed、真实机器人或更大评估验证。

## 6. 建议的模拟面试顺序

1. 用 90 秒版本介绍项目；
2. 回答 Q1、Q2、Q10、Q15，证明理解训练/推理闭环；
3. 回答 Q11、Q12、Q19，解释自己的 ablation；
4. 回答 Q16--Q18、Q20，说明为什么需要 rollout 和 failure analysis；
5. 回答 Q21、Q22，展示机器人软件工程能力；
6. 回答 Q23、Q24、Q26，展示迁移与研究判断能力。

项目证据、路径和结果见仓库根目录 README、`docs/04_rollout.md`、
`docs/05_experiments.md` 与 `docs/06_failure_analysis.md`。
