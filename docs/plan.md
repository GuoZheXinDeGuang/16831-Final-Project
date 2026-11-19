# RASR Course Project：Claude Code 单人执行计划

> 项目：Reachability-Aware Subgoal Repair for Hierarchical JEPA Planning  
> 目标：**2026 年 12 月 1 日前**准备好可展示的结果（暂按 12 月 1 日倒排，实际 presentation 日期以课程通知为准）。  
> 执行：**一个人 + Claude Code**。这是课程项目，不按论文规模扩张。

## 一、项目目标与边界

在现有 H-JEPA 分层规划器上做三项改动：

1. **Reachability Estimator**：预测在给定低层 MPC horizon `H` 和搜索预算 `B` 下，当前状态能否到达某个 latent subgoal。
2. **Reachability-aware Selection**：选择既能推进最终目标、又有较高可执行概率的 subgoal。
3. **Local Subgoal Repair**：执行停滞或可达性下降时局部替换失败 subgoal；修复失败才回退到完整高层重规划。

**核心研究问题**：在相同计算预算下，Reachability + Local Repair 是否比普通 H-JEPA 和 Full Replanning 更有效？

**第一版不做**：重新预训练 JEPA、第二个仿真环境、Push-T transfer、大规模 dynamics shift、复杂超参搜索。

## 二、给 Claude Code 的总指令

1. **先检查仓库再修改。** 阅读 `AGENTS.md`、README、配置、测试、现有环境、模型 checkpoint 和 H-JEPA planner 接口。不要假设某个文件或 API 一定存在。
2. 第一件事创建 `docs/IMPLEMENTATION_STATUS.md`，记录仓库现状、可用 checkpoint、可运行命令、已完成工作、阻塞项、单 episode 耗时。每完成一个阶段都更新。
3. **一次只执行一个阶段**：每阶段必须有最小可运行测试、日志、验收结果，然后再进入下一阶段。
4. 保持预训练 H-JEPA backbone 冻结。推理期间只可使用图像、latent、候选动作、预测和历史；仿真 GT 状态只能用于标注和评测。
5. 训练、validation、test 必须按 episode 或 start-goal 配置隔离；不能用 test 结果选 reachability 阈值。
6. 日志记录：seed、环境、checkpoint、git commit、`H`、`B`、候选数、模型 forward 次数、planning wall-clock、成功与否和失败原因。
7. 若关键外部依赖 **约一天内无法复现**，停下来说明阻塞，不要自行重写完整世界模型或偷偷换方法。
8. 下文 `scripts/`、`rasr/` 等目录是**建议结构**，应根据真实仓库适配。

## 三、时间表

| 时间 | 阶段 | 必须产出 |
|---|---|---|
| 11/14–11/17 | P0：H-JEPA 复现 | 基线 episode 跑通、日志和最小评测 |
| 11/18–11/21 | P1：可达性标签与预测 | rollout 数据、kNN、MLP、校准图 |
| 11/22–11/24 | P2：Subgoal Reranking | Baseline 与 Reachability 选择对比 |
| 11/25–11/27 | P3：Local Repair | 失败检测、替换、全局回退机制 |
| 11/28–11/30 | P4：实验与 Slides | 主对比、2 个消融、案例视频、展示 |
| 12/1 起 | Buffer | 只修 bug、补图和排练，不引入新方法 |

## 四、实施清单

### P0：复现 H-JEPA（最高优先级）

- [ ] 定位官方代码、预训练 checkpoint、环境和依赖，验证是否真实可获取。
- [ ] 在 **Visual AntMaze** 或 **FourRoom Distractors** 中选择更容易复现的**一个**。
- [ ] 运行原版 Hierarchical MPC 至少一个完整 episode。
- [ ] 保存 observation、candidate subgoals、actions、latent progress、环境成功标志、耗时和 world-model forward calls。
- [ ] 固定一个小型评估集合（建议 10–20 个 start-goal pairs），为所有方法复用同一集合和 seeds。
- [ ] 建立统一评测命令与结构化日志（JSONL/CSV）。
- [ ] 环境成功标准必须来自环境本身，而非仅凭 latent distance 减小。

**验收**：给定相同配置和 seed，可重复运行一条完整轨迹，并自动导出 baseline 成功率/运行成本。

### P1：Reachability 数据集与模型

- [ ] 定义一个尝试记录：`episode_id, split, seed, z_t, g, H, B, Y, min_latent_distance, steps, runtime, forward_calls`。大 tensor 单独存储，不直接塞进 JSON。
- [ ] 从真实低层 MPC 尝试中收集正负标签；优先复用已有 rollout logs。
- [ ] 定义 `Y=1`：低层 MPC 在 `H` 步、预算 `B` 内，到达经 training/validation 校准的 latent subgoal 阈值；否则为 `0`。
- [ ] **先做 kNN baseline**，再实现轻量 MLP：`R_phi(z_t,g,H,B)`，BCE loss。
- [ ] 数据集先分 train/validation/test，记录正负样本数，避免测试信息泄漏。
- [ ] 评估 AUROC、PR-AUC、Brier Score、reliability diagram；报告采集 labels 的耗时与算力。
- [ ] 阈值只在 validation 上确定；保存模型、归一化参数和配置。

**验收**：一条命令复现 kNN vs MLP 评估图/表，说明数据量和收集成本。如果数据几乎只有一种标签，先修正采样方案。

### P2：Reachability-aware Subgoal Selection

- [ ] 保持原有 H-JEPA 生成的候选 subgoal 集合不变。
- [ ] 对每个候选计算：`goal_progress_cost - lambda * log(reachability + delta)`，选择最低 cost。
- [ ] 配置支持启用/关闭 reranking，设置 `lambda`、预测器类型和 normalization。
- [ ] 保存所有候选的 progress score、reachability score、最终选择。
- [ ] 与原版 H-JEPA 使用相同 start-goal、`H/B` 和尽可能相同候选池；统计额外预测开销。

**验收**：关闭功能可恢复原版行为；打开后至少有一个可解释的候选选择变化，并有初步对比结果。

### P3：Failure Monitoring + Local Repair

- [ ] 设计触发条件：连续 `m` 次进展停滞，或 reachability 低于 validation 选出的阈值。
- [ ] 加 debounce / cooldown，避免单次噪声导致频繁触发。
- [ ] 在失败 subgoal 附近生成有限个替代候选，候选来自真实轨迹/现有模型支持的 latent subgoals，不能任意随机插值假装可执行。
- [ ] 用 downstream progress、reachability 和与原目标的偏离惩罚对替代候选打分。
- [ ] 尽量保留后续 subgoals；局部候选均不可行则在固定预算内回退到 **Full Replanning**。
- [ ] 日志记录触发原因、原目标、新目标、搜索开销、是否 fallback、是否恢复成功。
- [ ] 输出至少一个成功 repair 和一个失败 repair 的 rollout。

**验收**：可复现的停滞情形能触发 repair；超出 repair budget 时正确 fallback，且不会死循环。

### P4：实验对比、消融与 Presentation

#### 主要比较方法

| 方法 | 含义 | 优先级 |
|---|---|---|
| B1 | 原版 H-JEPA | 必须 |
| B2 | 更频繁的普通 Replanning | 必须 |
| B3 | Reachability Reranking，但不 repair | 必须 |
| B4 | 相同 failure trigger 下的 Full Replanning | 必须 |
| Ours | Reachability Reranking + Local Repair | 必须 |
| Flat JEPA | 非分层参考 | 可选（已有代码能跑才做） |

#### 指标

- **Primary**：环境 Task Success Rate。
- **Secondary**：Recovery Success、Subgoal Success、修复次数、到达步数、world-model forward calls、planning wall-clock。
- **Estimator**：kNN/MLP 的 AUROC、PR-AUC、Brier、calibration 图。
- 尽可能采用 **3 个 seeds 起步，5 个更好**；报告 episode 数与 uncertainty intervals。样本少时诚实说明，不夸大显著性。
- **Matched Compute**：优先按 world-model forward calls 限额比较；同时报告 wall-clock，包含预测器、触发检测和修复的计算开销。

#### 消融（至少两个）

- [ ] Ours 去掉 Reachability Scoring。
- [ ] Ours 去掉 Local Repair（相当于 B3）。
- [ ] 可选：MLP vs kNN 做完整 planner 版本对比。
- [ ] 可选：Repair neighborhood 大小。

#### 展示案例

- [ ] Repair 成功的 rollout。
- [ ] Repair 失败的 rollout。
- [ ] 对比同一初始条件下 local vs full replanning（如果差异有意义）。
- [ ] 所有 plot 从真实日志生成，带坐标轴、单位、样本数和 seeds。

**验收**：统一命令生成对比表和图；可以明确解释 repair **什么时候有效、什么时候无效**。

## 五、建议代码结构（以实际仓库为准）

```text
configs/rasr/                # 方法/环境/预算配置
rasr/reachability.py         # kNN 与 MLP
rasr/collect_attempts.py     # MPC 执行标签采集
rasr/subgoal_ranker.py       # 可达性重排序
rasr/failure_monitor.py      # 失败检测
rasr/local_repair.py         # 局部修复与回退
scripts/run_rasr_eval.py     # 统一实验入口
scripts/plot_rasr_results.py # 图表
tests/                      # 最小单测
docs/IMPLEMENTATION_STATUS.md
outputs/rasr/               # 轨迹、日志、模型和图片
```

## 六、赶不上时如何缩减

按重要性排序：**复现 baseline → 采集标签/kNN → reranking → local repair → 公平对比/Slides → MLP 精调与扩展**。

1. **Checkpoint/官方环境不可用**：立即报告实际阻塞，未经确认不要替换成另一套模型后仍称作 H-JEPA。
2. **Rollout 很耗时**：减小评估规模、缓存结果、复用日志；优先保证干净的 train/test 划分。
3. **MLP 效果差**：保留 kNN 为可复现的诊断 baseline，清楚报告 learned head 未达到预期。
4. **Local Repair 不稳定**：如实展示 filtering-only 结果和 repair proof of concept，不伪称完整验证。
5. **最后只剩两三天**：停止加入新功能；优先稳定 baseline/B3/B4/Ours、算力统计、案例和展示。

## 七、Presentation 建议（7–9 页）

1. 问题：High-level subgoal 不一定能被 low-level MPC 执行。
2. 原始 H-JEPA pipeline 与失败例子。
3. Reachability labels 与 kNN/MLP。
4. Reranking + Local Repair 机制。
5. 环境、baselines 和 matched-compute 设置。
6. 成功率、恢复率、规划开销。
7. 可达性预测性能/校准图。
8. 成功/失败修复案例。
9. Limitations 与结论。

## 八、最终完成标准

- [ ] 有清晰的安装、checkpoint 和执行说明，别人可复现。
- [ ] Frozen backbone，推理不使用 privileged state。
- [ ] B1–B4 与 Ours 的结果均有真实运行记录；缺失结果明确标注。
- [ ] 报告 kNN 和 MLP 可达性预测性能（如数据允许）。
- [ ] 至少两个 ablations、合理的 seeds/不确定性分析和失败案例。
- [ ] 给出计算预算与 wall-clock；不捏造数据。
- [ ] 12 月初可以现场讲清问题、方法、实验和局限。

## 九、直接复制给 Claude Code 的第一条指令

```text
请完整阅读 RASR_Claude_Code_执行计划_中文版.md。
先检查当前仓库结构、README、AGENTS.md、H-JEPA planner 接口、预训练 checkpoint、
仿真环境、已有评测脚本与测试。只执行 P0，不要提前实现 P1–P4。

创建 docs/IMPLEMENTATION_STATUS.md，记录已验证的依赖、实际命令、已完成事项、
阻塞项、每个 episode 的耗时以及后续 TODO。
然后跑通最小 H-JEPA baseline smoke test，保存真实日志。

如果上游代码、checkpoint 或环境无法访问，请明确说明阻塞点和备选方案，
不要自行假设已经复现。

完成 P0 后，报告修改文件、实际执行命令、测试结果、测得的耗时和下一步建议。
```
