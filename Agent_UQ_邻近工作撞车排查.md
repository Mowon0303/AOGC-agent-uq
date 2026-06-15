# Agent UQ 邻近工作撞车排查(2025-09 ～ 2026-06)

> 目的:核对前一版研究计划三个抓手是否被近期 arXiv 工作占用。
> 注:arXiv 编号来自检索结果,投稿前请逐一核对准确编号与发表 venue。

---

## 0. 结论先行

**三个抓手里两个已被占,一个存活。**

| 原抓手 | 状态 | 谁占了 |
|---|---|---|
| ① 跨步"传播 / 早期误差更致命" | ❌ **已被占(强)** | Agentic UQ(Salesforce, 2601.15703)、TRACER(2602.11409)、Uncertainty Propagation(2604.23505)、Where Agents Fail(2509.25370) |
| ② 步骤级"失败定位"评测集 | ❌ **已被占(强)** | AgentErrorBench(2509.25370)、Seeing the Whole Elephant(2604.22708)、VerifyMAS(2605.17467)等一批失败归因 benchmark |
| ③ **动作-观察 grounding 一致性信号** | ⚠️ **基本存活,但缝比原以为的窄(见 §4b)** | 电信域 entity-faithfulness(2601.07342)、MIRAGE-Bench 测现象(2507.21017);**且 2026-06-15 复检新发现 FRANQ(2505.21072)机制几乎同构(限 RAG)、综述 2602.05073 附录 E.1 已 sketch 同思想** |

**建议:不要再以"轨迹 UQ / 传播 / 失败定位"为卖点(会被审稿人指为 incremental)。改以抓手 ③ 重定位——见 §4。但卖点不能写成"提出 grounding 当 UQ"(已是旧轴),必须钉在"agent 特有失败 + 与现有 UQ 正交可融合"的实证——见 §4b。**

> **2026-06-15 web 复检备注**:本文档原 12 个 arXiv 编号经逐一 WebFetch 核实**全部真实存在、编号准确**,三篇红线均未做掉 AOGC。但独立新颖性检索挖出本文档原先遗漏的**两条更危险的撞车线**(FRANQ、综述 E.1),已补入 §4b。撞车判断结论需据此收紧。

---

## 1. A 组｜威胁抓手①(传播 / 早期误差 / 选择性控制)

| 论文 | 时间 | 做了什么 | 与我们重叠 |
|---|---|---|---|
| **Agentic Uncertainty Quantification**(Salesforce)`2601.15703` | 2026-01 | 提出"Spiral of Hallucination":早期错误(含**误读工具输出**)污染上下文不可逆;Dual-Process AUQ 把 verbalized 置信变成主动控制(记忆传播 + 反思触发)。ALFWorld/WebShop SOTA | **高度重叠**:我们的"早期更致命 + 传播聚合 + 按不确定性触发解决"几乎全被它覆盖,甚至点到了"误读工具输出" |
| **TRACER: Trajectory Risk Aggregation**`2602.11409` | 2026-02 | 从跨步不确定性轨迹聚合出 run 级风险,处理关键 episode | **高度重叠**:就是我们的"轨迹级聚合" |
| **Uncertainty Propagation in LLM-Based Systems**`2604.23505` | 2026-04 | 系统级不确定性传播建模 | 重叠:传播这条线 |
| **Where LLM Agents Fail & Learn From Failures**`2509.25370` | 2025-09 | 论证"误差复合是 agent 可靠性主瓶颈";出 AgentErrorTaxonomy + AgentErrorBench | 重叠:误差复合论点 + 失败标注集 |
| Self-Verification Dilemma `2602.03485` / SELAUR `2602.21158` | 2026-02 | 自验证过度/不足;用不确定性奖励自进化 | 邻近:自验证、用不确定性做控制 |

---

## 2. B 组｜威胁抓手②(失败定位 / 归因 benchmark)

| 论文 | 时间 | 做了什么 | 与我们重叠 |
|---|---|---|---|
| **AgentErrorBench**(含于 `2509.25370`) | 2025-09 | **首个**从 ALFWorld/GAIA/WebShop 系统标注的失败轨迹集 | **直接撞**:我们想做的"失败步标注集"已存在 |
| **Seeing the Whole Elephant**`2604.22708` | 2026-04 | 多智能体失败归因 benchmark | 直接撞:失败定位评测 |
| **VerifyMAS**`2605.17467` | 2026-05 | 把失败归因当"假设验证",在长轨迹上定位错误 agent/步 | 直接撞:步级定位 |
| **From Flat Logs to Causal Graphs**`2602.23701` | 2026-02 | 层次化失败归因(因果图) | 撞:定位 |
| **Early Diagnosis of Wasted Computation**`2606.01365` | 2026-06 | 用在线 trace 信号(工具可靠性/证据可得性/预算压力等)早诊断无效计算 | 撞:在线信号 + 早诊断 |

---

## 3. C 组｜威胁"公平比较 / 校准"角度

| 论文 | 时间 | 做了什么 | 与我们重叠 |
|---|---|---|---|
| **Agentic Uncertainty Reveals Agentic Overconfidence**`2602.06948` | 2026-02 | 在 SWE-bench Pro 上比多模型多方法的 UQ,AUROC/ECE/Brier;发现普遍过度自信 | 撞:我们想做的"head-to-head + 风险指标比较"已有雏形 |
| Benchmarking UQ Calibration(long-form QA)`2602.00279` / EACL'26 | 2026-02 | UQ 校准 benchmark(非 agent) | 邻近:方法学 |
| 黑盒 UQ:Multi-Agent black-box `2412.09572`、Generating with Confidence `2305.19187` | 2024-25 | 仅文本可得时的 UQ(verbalized/一致性) | 邻近:黑盒信号已被系统研究 |

---

## 4. D 组｜抓手③最接近的工作(我们的存活区,需精读)

| 论文 | 时间 | 做了什么 | 留给我们的缝 |
|---|---|---|---|
| **MIRAGE-Bench**(LLM Agent is Hallucinating)`2507.21017` | 2025-07 | 系统刻画 agent **误读观察**(点不存在的按钮、读错目录、臆造状态转移)并建 benchmark | 它**只测现象、不做 UQ 信号**;可作为我们 grounding 信号的评测床与动机 |
| **Agentic Diagnostic Reasoning over Telecom**`2601.07342` | 2026-01 | 静态分析器做 entity-faithfulness:抽 agent 提到的实体 ID,核对是否来自工具返回,不可溯=幻觉 | **最接近我们的 grounding 信号**,但**限电信/数据中心窄域、是检查器不是不确定性信号**、未融入选择性控制 |
| Agent 幻觉/忠实性综述 `2510.24476` `2510.06265` | 2025-10 | 综述层面提"给检索信息打置信" | 仅综述,无具体通用方法 |

---

## 4b. D' 组｜2026-06-15 独立复检新增的撞车(原文档遗漏,最关键)

> 这一组不是作者已知论文,是独立新颖性检索 + PDF 原文核验挖出来的。**审稿人最可能拿来毙稿的就在这里。**

| 论文 | 时间 | 做了什么 | 与 AOGC 的关系 / 缝 |
|---|---|---|---|
| 🔴 **FRANQ: Faithfulness-Aware UQ for Fact-Checking RAG**`2505.21072` | 2026-04(v5) | 把输出拆 atomic claims(带 token span)→ **NLI 判每个 claim 是否被检索证据 entail(faithfulness)**→ 按 faithful 与否分流 UQ 方法估 factuality → 服务 selective fact-checking | **机制几乎与 AOGC 同构**(claim 抽取 + NLI 对证据核验 + 不可溯→不确定 + 融合 + 选择性)。唯一差别:**锁死 RAG 单轮静态检索**,只在 future work 提一句"可扩展到 agent"。**审稿人最可能说"这不就是 FRANQ 搬到 agent 上吗"——必须正面区分(见 §5 防守 + v2 §3 漏检实验)** |
| 🔴 **UQ 综述 `2602.05073` 附录 E.1**("Evidentiality Classification") | 2026-02 | 综述附录给了个 prompt 原型,逐条问"工具调用参数是否可溯源到先前工具结果?陈述的事实是否被工具结果确认?"标 evidential/non_evidential/uncertain 喂进 UQ | **AOGC 思想的口语版**;更麻烦:综述把 evidentiality/grounding 称作"已有方法早已考虑的旧轴",把 interactivity 当新前沿。缓解:E.1 只是未实现的 LLM-judge sketch、只在 τ²-bench retail 比划、没系统评测、没 cite FRANQ |
| 🟡 **GSAR: Typed Grounding for Hallucination Detection in Multi-Agent**`2604.23366` | 2026-04(Oracle) | 抽 claim→四分类(grounded/ungrounded/contradicted/complementary)+证据加权→连续 groundedness 分→三档动作{proceed,regenerate,replan} | 机制邻近,但**明确不把 groundedness 当 UQ**(不与 verbalized/语义熵/P(true) 比)、**不 late-fusion**、LLM-judge(非 NLI+string match)、只在 FEVER 评、AIOps 窄场景。定位与我方不同 |
| 🟢 **The Confidence Dichotomy(Tool-Use Agents 误校准)**`2601.07264` | 2026-01 | 实证"evidence tools(web 搜索)带噪→系统性过度自信;verification tools(代码解释器)→纠偏",解法是 RL 微调 | **不是撞车,是绝佳 motivation 引文**:实证证明"agent 读了带噪 observation 后置信反而虚高"=AOGC 要堵的盲点真实存在 |

**独立审稿人对 AOGC 的新颖性存活打分:6/10。** 活着,但缝窄;核心机制与 FRANQ 高度同构是最大扣分项。要拉到 7.5+,必须:(a) 给 agent 场景**特有**的失败模式与设计(跨步 observation 归因、agent 误读 vs 工具本身错的区分);(b) 强 UQ 基线对照 + 风险-覆盖曲线证明正交可加;(c) 显式 cite 并实验区分 FRANQ 与综述 E.1。

---

## 5. 推荐重定位(pivot):从"造轮子"改为"补盲点"

**新的一句话**:现有 agent UQ(verbalized 系如 Agentic UQ、熵系、TRACER)对"**观察-grounding 失败**"是盲的——当 agent 误读/无视工具返回时,它的 verbalized 置信仍然很高;我们提出一个**通用、训练-free 的动作-观察一致性信号**,**正交补强**现有 SOTA 的错误检测与校准。

**为什么这条还活着且安全**

- 它**不与 A/B 组冲突**,反而**把它们当基线来补强**(你不再声称发明传播/定位)。
- 沾边的 2601.07342 是窄域检查器;你做"通用 + 当不确定性信号 + 接选择性控制 + 多 benchmark"即拉开。

**针对 §4b 两条新撞车的强制防守(否则会被一句话毙)**

- **vs FRANQ(2505.21072)**:不能只说"换个域"。要论证 **RAG 单轮静态证据 ≠ agent 多轮动态观察**——agent 的 observation 是会被它自己误读/无视的异质动态证据源;并**实测**把 FRANQ 式做法直接套到 agent 轨迹会漏掉哪类失败(见 v2 §3 的"FRANQ 漏检对比实验")。
- **vs 综述 E.1**:别把贡献写成"提出用 grounding 当 UQ"(综述已把它当旧轴、只是没实现)。把 AOGC 定位为该综述"observation/异质实体不确定性开放挑战"的**首个落地实例**:训练-free + 黑盒 + 实体/数值级 string-match + 轻量 NLI 溯源的**可复现 pipeline + 系统 UQ 评测**(对比 verbalized/语义熵/P(true) + 风险-覆盖曲线),而非一个 LLM-judge prompt。

**修订后的贡献(三点)**

1. **诊断**:用实验证明"observation-grounding 失败"是现有 agent UQ 的系统盲区(高置信却误读)——一个干净的反直觉发现。
2. **方法**:训练-free 的 grounding 一致性信号(抽取 agent 对观察的复述/引用的实体与断言,核对是否可溯到真实工具返回;不可溯→不确定),并给出与 verbalized/熵信号的**融合**方式。
3. **收益**:在 SOTA 之上叠加该信号,改善步级/轨迹级错误检测 AUROC、校准 ECE、风险-覆盖 AURC。

**基线(改为"在其之上加我们的信号")**:Agentic UQ(2601.15703)、TRACER(2602.11409)、semantic entropy、verbalized confidence、P(True)。

**评测床(全部复用,不自建)**:MIRAGE-Bench(测 grounding 失败)、AgentErrorBench(测失败定位)、ALFWorld / τ-bench / BFCL。

**算力**:依旧 3060 友好——只运行/复用现有方法 + 计算轻量信号,**不训练**。grounding 信号用一次额外 LLM 调用或本地小模型 NLI 即可。

---

## 6. 投稿前必须手动做的三件事

1. **精读最危险的(更新版,FRANQ 升为头号)**:**FRANQ `2505.21072`**(机制同构,最危险)、UQ 综述附录 E.1 `2602.05073`、Agentic UQ `2601.15703`、MIRAGE-Bench `2507.21017`、Telecom entity-faithfulness `2601.07342`——确认 grounding 信号没被它们的附录/实验偷偷做掉。
2. **核对编号与 venue**:上表 ID 来自检索,需到 arXiv/OpenReview 逐一核实(尤其是否已被某会录用,影响"撞车"严重度)。
3. **持续监控**:这方向月更很快,设 arXiv/Semantic Scholar 订阅(关键词:agent uncertainty / observation faithfulness / trajectory risk)。

---

## 附:全部相关条目(便于建库)

- 综述:UQ in LLM Agents `2602.05073`
- 传播/控制:Agentic UQ `2601.15703`、TRACER `2602.11409`、Uncertainty Propagation `2604.23505`、SELAUR `2602.21158`、Self-Verification Dilemma `2602.03485`
- 失败定位/归因:AgentErrorBench/`2509.25370`、Seeing the Whole Elephant `2604.22708`、VerifyMAS `2605.17467`、Flat Logs→Causal Graphs `2602.23701`、Early Diagnosis `2606.01365`
- 自愈/重规划:Self-Healing Framework `2605.06737`、Self-Healing Orchestrators `2606.01416`、Hierarchical Error-Corrective Graph `2603.08388`
- 比较/校准/黑盒:Agentic Overconfidence `2602.06948`、Benchmarking UQ Calibration `2602.00279`、Black-box UQ `2412.09572` `2305.19187`
- grounding/忠实:MIRAGE-Bench `2507.21017`、Telecom entity-faithfulness `2601.07342`、幻觉综述 `2510.24476` `2510.06265`
- **🔴 06-15 新增撞车(最近邻)**:FRANQ(faithfulness-aware UQ,RAG)`2505.21072`、UQ 综述附录 E.1 evidentiality `2602.05073`、GSAR(typed grounding,multi-agent)`2604.23366`
- **motivation 引文**:The Confidence Dichotomy(tool-use agent 误校准)`2601.07264`
- 旁系(澄清/转交/弃答):Ask or Assume? `2603.26233`、Structured Uncertainty Clarification `2511.08798`、ReDAct `2604.07036`、Abstain via Semantic Confidence Reward `2510.24020`
