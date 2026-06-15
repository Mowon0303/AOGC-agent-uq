# 研究计划：LLM Agent 的步骤级不确定性"传播"与失败定位

> 面向 AAAI main track | 算力约束：RTX 3060 单卡(12GB) + API | 免训练为主

---

## 0. 一页纸总览(TL;DR)

**一句话**：现有 Agent 不确定性量化(UQ)几乎都把不确定性当成"某一步要不要问人/转交/弃答"的单点开关，且直接套用单轮 QA 的 UQ 信号。本计划改为研究**不确定性如何沿轨迹传播、早期不确定性如何决定最终成败**，并交付一个能定位"决定性失败步"的细粒度评测，配一个训练-free 的轨迹级估计器。

**核心论点(要去证明的"反直觉发现")**

1. 单轮 UQ 信号(token 熵、语义熵、verbalized confidence)在 agent 轨迹上**校准很差**，因为它们忽略了"环境 grounding"(动作是否与工具返回一致)。
2. **早期步骤的不确定性比末期更能预测最终失败**(误差复合效应)——这意味着应该"早停/重规划"，而不是等末步才问人。
3. 用我们的轨迹级估计器做"选择性 agent"，能在**相同人工询问预算下显著提高任务成功率**。

**三个贡献(AAAI 卖点)**

- **方法**：训练-free 的多信号步骤级 UQ + 传播聚合，含一个新信号——**动作-观察 grounding 一致性**(单轮 UQ 完全捕捉不到)。
- **评测/数据**：首个**步骤级失败定位**诊断集(标注"决定性失败步")——直接回应综述点名的"缺细粒度 benchmark"。
- **实证洞见**：误差复合规律 + 单轮信号失效的系统证据。

**为什么适合 3060**：信号计算轻量；开源小模型(7B/3B，4-bit)只用来取 logit 算熵；agent rollout 主要靠 API。无需任何训练。

---

## 1. 背景与精确 gap

### 已有工作(必须在 related work 里区分清楚)

- **纲领综述**：*Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities* (arXiv 2602.05073) 点名四大难题：① 估计器选择 ② 异质实体的不确定性 ③ **交互系统中的不确定性动态** ④ **缺细粒度 benchmark**。本计划主攻 ③ + ④(占用最少)。
- **澄清/问人**：*Ask or Assume?*(2603.26233)、*Structured Uncertainty guided Clarification*(2511.08798，用 EVPI 区分 specification vs model uncertainty)。→ 已占"问人"。
- **转交**：ReDAct(小模型不确定→转交大模型)。→ 已占"deferral"。
- **弃答**：I-CALM(prompt 激励弃答)、*Teaching LLMs to Abstain via Semantic Confidence Reward*(2510.24020，需训练)。→ 已占"abstain"。
- **工具调用决策**：*LLM Agents Already Know When to Call Tools*(2605.09252)。
- **单轮 UQ 基础**：语义熵(Kuhn 2023 / Farquhar Nature 2024)、verbalized confidence(Tian 2023; Lin 2022)、P(True) 自评(Kadavath 2022)、self-consistency(Wang 2022)。

### 我们的差异化(没被占的缝)

| 维度 | 现有工作 | 本计划 |
|---|---|---|
| 时间粒度 | 单步开关 | **跨步传播/动态** |
| 用途 | 问人/转交/弃答 | **早停 + 重规划 + 失败步定位** |
| 信号 | 套用单轮信号 | **新增动作-观察 grounding 一致性** |
| 评测 | 看最终成功率 | **步骤级失败定位 + 风险-覆盖曲线** |

---

## 2. 核心假设(可证伪)

- **H1(信号失效)**：单轮 UQ 信号对"该步是否出错"的 AUROC 在 agent 设定下显著低于其在单轮 QA 上的表现。
- **H2(误差复合)**：把同等大小的不确定性"注入"不同步位，越早的步对最终失败的预测力越强(早期步 AUROC > 末期步)。
- **H3(可用性)**：基于轨迹级 UQ 的选择性策略，在固定"人工询问/重试预算"下，任务成功率优于所有单轮基线与无 UQ 的 baseline(更优的风险-覆盖曲线)。
- **H4(grounding 增益)**：加入"动作-观察一致性"信号后，错误检测 AUROC 与校准 ECE 均改善(消融可验证)。

---

## 3. 方法(训练-free)

### 3.1 每步的多信号不确定性

在轨迹的第 t 步(状态=历史+最新观察，输出=思考+动作):

1. **动作自一致性**：同一状态采样 k 个候选动作(k=5~8)，按"动作语义等价"聚类，用聚类熵/最大簇占比当不确定性(把语义熵迁移到动作空间)。
2. **Verbalized confidence**：让模型对自己这步动作打 0–100 置信(black-box 也能用，覆盖 API 模型)。
3. **Token 级**(仅开源模型，有 logit)：动作 token 的平均/最小 log-prob、熵。
4. **★ 动作-观察 grounding 一致性(新信号)**：让模型先复述/抽取上一步工具观察的关键事实，比对其与真实观察的 NLI/字面一致性；不一致→"观察被忽略/被幻读"，是 agent 专有的不确定性来源。可再加"工具返回自身可靠性"(报错码、空结果、低检索分)。

### 3.2 轨迹级聚合(传播)

- **基线聚合**：min / mean / last 几种简单池化。
- **传播聚合(主打)**：用一个轻量、可解释的折扣累积——`R_t = max(u_t, γ·R_{t-1})` 或对早期步加权(对应 H2 的早期权重),得到截至第 t 步的轨迹风险 `R_t`。γ 与权重用一个小验证集做无梯度搜索(网格/CMA-ES)。
- 这层**不训练大模型**，只拟合几个标量超参,完全在 3060/CPU 上完成。

### 3.3 选择性 Agent 决策策略

给定 `R_t` 与阈值，输出动作集合 {继续 / 重采样该步 / 触发重规划 / 询问人类 / 提前放弃}。
- 用风险-覆盖框架评估:不同阈值 → (放弃率/询问预算, 剩余任务成功率)。
- 对照"无 UQ"和"单轮信号"策略画曲线。

---

## 4. 实验设计

### 4.1 模型

- **开源(本地 3060，取 logit)**：Qwen2.5-7B-Instruct 与 Llama-3.1-8B-Instruct(4-bit,~5–6GB);长上下文吃紧时降到 Qwen2.5-3B。用 vLLM/HF 起服务。
- **API(验证普适性,黑盒信号)**：GPT-4o-mini、Claude Haiku、DeepSeek 之一两个。
- 说明:token/语义熵需开源模型;verbalized + 自一致性 + grounding 信号开源与 API 都能用——以此证明方法对黑盒也成立。

### 4.2 Benchmark(选轻量、确定性评分、抗泄漏)

- **ALFWorld**:纯文本具身任务,本地轻量,经典 agent 评测。
- **τ-bench (tau-bench)**:工具型客服 agent,确定性成功判定。
- **BFCL / API-Bank**:函数调用正确性,步骤可判。
- **GAIA 子集**(可选,偏难):综合助理任务。
建议先 ALFWorld + τ-bench 跑通主结果,再加 BFCL 做工具调用切片。

### 4.3 基线(把单轮 UQ 迁到步骤/轨迹级)

- token 平均/最小 log-prob、熵
- 语义熵(动作空间版)
- verbalized confidence
- P(True) 自评
- self-consistency 一致度
- 以上 × {min/mean/last 聚合} 作为对照,凸显"传播聚合 + grounding"的增益。

### 4.4 评测指标

- **错误检测**:步骤级与轨迹级 AUROC / AUPRC(不确定性能否预测出错)。
- **校准**:ECE、Brier、可靠性图。
- **选择性**:风险-覆盖曲线、AURC(越低越好)、成功率 vs 人工询问预算曲线。
- **★ 失败定位**:在标注了"决定性失败步"的子集上,报 step-localization 准确率/Top-k 命中(本计划独有指标)。

### 4.5 关键消融

- 去掉 grounding 信号(验 H4);
- 传播聚合 vs 简单池化(验 H2 的早期权重);
- k(采样数)对精度-成本的影响;
- 开源 vs API(普适性);
- 跨 benchmark 迁移(超参在 A 调、在 B 测)。

---

## 5. 数据/标注(可控成本)

- **步骤级失败定位标注**:对失败轨迹,用强 LLM-judge 半自动标"哪一步是决定性失败步"(给出整条轨迹 + 成功标准),再人工抽检 100–200 条校准 judge 质量(报 judge 与人工的一致性 κ)。
- 规模:每 benchmark 标 300–500 条轨迹即可支撑诊断结论;这部分本身就是可复用的贡献。
- 全部基于公开 benchmark 的 rollout,无隐私/版权风险。

---

## 6. 预期贡献与 AAAI 定位

1. 训练-free、对黑盒也成立的**轨迹级 Agent UQ 方法**(含新 grounding 信号)。
2. **步骤级失败定位诊断集 + 指标**(填综述点名空白)。
3. **误差复合 / 单轮信号失效**的系统实证。
> AAAI main track 吃"方法 + 严谨实证 + 清晰洞见"这一套;本计划不靠规模,靠想法清晰度与实验可信度——契合小算力定位。

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| H2/H1 不成立(没有反直觉发现) | 即便为"否",系统比较 + 诊断集仍是可发表的实证贡献;预留"负结果也成文"的叙事 |
| 与某篇在投/新挂 arXiv 撞车 | grounding 信号 + 失败定位指标是组合差异点;持续 arXiv 监控,必要时加 benchmark 或加黑盒普适性切片 |
| API 成本超支 | 先用 ALFWorld(本地免费)出主结论,API 只做普适性验证;k 自适应 |
| LLM-judge 标注不可靠 | 人工抽检报一致性;失败定位只在高一致子集上下结论 |
| 3060 显存不足 | 降到 3B / 缩上下文 / 4-bit;logit 信号本就不需大模型 |

---

## 8. 12 周里程碑(可压缩)

1. **W1–2** 跑通 ALFWorld + 1 个开源模型的 agent rollout 与日志管线。
2. **W3–4** 实现 5 个单轮基线信号 + 步骤级评测(出 H1 初步结论)。
3. **W5–6** 实现 grounding 信号 + 传播聚合(H4/H2)。
4. **W7–8** 选择性策略 + 风险-覆盖/询问预算曲线(H3)。
5. **W9** 失败定位标注 + 指标。
6. **W10** 加 τ-bench / BFCL + API 普适性。
7. **W11** 消融 + 跨 benchmark 迁移。
8. **W12** 写作 + 复现包整理。

---

## 9. 算力/成本可行性核对

- **本地**:Qwen2.5-7B/Llama-3.1-8B @4-bit 推理,3060 12GB 可跑(留 KV cache 余量,必要时 3B)。无训练。
- **API**:主结论走本地;API 仅普适性,采样 k≈5、数百 episode,量级约数十美元。
- **存储/算法层**:传播聚合只拟合 2–3 个标量,CPU 秒级。

---

## 10. 关键参考

- UQ in LLM Agents（综述，gap 来源）— arXiv 2602.05073
- Semantic Uncertainty / Detecting hallucinations with semantic entropy — Kuhn et al. 2023; Farquhar et al. Nature 2024
- Just Ask for Calibration（verbalized）— Tian et al. 2023
- Language Models (Mostly) Know What They Know（P(True)）— Kadavath et al. 2022
- Self-Consistency — Wang et al. 2022
- 选择性预测 / 风险-覆盖 — Kamath et al. 2020；Geifman & El-Yaniv
- Agent benchmark — ALFWorld (Shridhar 2021)、τ-bench (2024)、BFCL、API-Bank、GAIA (Mialon 2023)
- 邻近工作（需区分）— Ask or Assume? (2603.26233)、Structured Uncertainty Clarification (2511.08798)、Abstain via Semantic Confidence Reward (2510.24020)
