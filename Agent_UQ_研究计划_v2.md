# 研究计划 v2：补上 Agent 不确定性量化的"观察-grounding 盲点"

> 面向 AAAI main track | 算力：RTX 3060 单卡(12GB) + API | 全程免训练
> v2 变更：原 v1 的抓手①(传播/早期误差)与②(失败定位评测)已被近半年工作占用(见《邻近工作撞车排查》),本版把它们**降级为基线/评测床**,核心收敛到唯一存活且最少被占的抓手③——**动作-观察 grounding 一致性**。

---

## 0. 一页纸总览(TL;DR)

**一句话**：现有 agent UQ(verbalized 置信系、熵系、轨迹风险系)有一个共同盲点——**它们不检查 agent 是否真的"读对了"工具/环境的观察**。当 agent 误读或无视观察(observation-grounding 失败)时,它的 verbalized 置信和 token 熵仍然很高,于是这些 UQ 信号对一整类失败"看不见"。我们提出一个**通用、训练-free、黑盒可用**的动作-观察一致性信号(AOGC),它与现有信号**正交**,叠加到 SOTA 之上能显著改善错误检测/校准/风险-覆盖,尤其在 grounding 引发的失败上。

**核心论点(反直觉、可证伪)**
- **H1(盲点存在)**：在 grounding 引发的失败上,现有 UQ 信号(verbalized / 语义熵 / AUQ / TRACER)的错误检测 AUROC 接近随机,而置信往往偏高(系统性过度自信)。
- **H2(正交且增益)**：AOGC 与现有信号相关性低;late-fusion 后整体 AUROC/ECE/AURC 改善。
- **H3(黑盒可迁移)**：仅靠文本(无 logit)即可计算 AOGC,在闭源 API 上同样有效。
- **H4(可用性)**：把融合信号接入选择性控制(早停/重读观察/问人),在固定预算下提升任务成功率。

**三点贡献(AAAI 卖点)**
1. **诊断**:首次系统量化"观察-grounding 失败"是现有 agent UQ 的盲区(高置信却读错)。
2. **方法**:训练-free 的 AOGC 信号 + 与现有信号/聚合器的融合配方。
3. **收益**:在 SOTA(AUQ、TRACER 等)之上即插即用地提升错误检测与校准,并给出选择性控制收益。

**为什么安全**：不再声称发明"轨迹 UQ/传播/失败定位"——那些被占了;我们**把它们当基线和评测床来补强**,冲突面最小。
**为什么 3060 够**：不训练任何大模型;AOGC 仅需每步一次轻量校验(本地 NLI 小模型即可,离线近免费)。

> **★ 卖点钉死(2026-06-15 web 复检后修订,必读)**
> 独立检索发现两条原计划没设防的撞车:**FRANQ(`2505.21072`)**——claim 抽取+NLI 溯源+不可溯→UQ+融合+selective,机制几乎与 AOGC 同构,但锁死 RAG 单轮;**UQ 综述 `2602.05073` 附录 E.1** 已 sketch "工具结果可溯源性当不确定性"的思想,且综述把 grounding/evidentiality 称作"旧轴"。
> **因此卖点不能写成"我们提出用 grounding 一致性做 UQ"**(会被一句话毙)。改钉在两点:
> 1. **agent 特有失败模式**:agent 在多轮工具/环境交互中会**误读/无视 observation 而置信不降**——这是现有 UQ(含 FRANQ 式 RAG faithfulness)在 agent 域**系统性漏掉**的失败。
> 2. **正交性实证**:AOGC 是首个 training-free、黑盒、与现有 UQ **正交可融合**、并**驱动选择性控制**的通用域信号(融合后 AUROC/风险-覆盖实打实涨)。
> 独立审稿人新颖性存活打分 **6/10**,守住上述两点 + 做 FRANQ 漏检实验(§3.5)可拉到 7.5+。

---

## 1. 精确定位与差异化

### 1.1 已被占,降级使用(不作为卖点)
- 传播/早期误差/选择性控制:Agentic UQ(AUQ,`2601.15703`)、TRACER(`2602.11409`)、Uncertainty Propagation(`2604.23505`)→ **当聚合骨干/基线**。
- 失败步定位与标注:AgentErrorBench(`2509.25370`)、失败归因 benchmark(`2604.22708` 等)→ **当现成评测床**,不自建。
- 单轮 UQ:语义熵、verbalized、P(True)、self-consistency → **当对照信号**。

### 1.2 我们占的缝(抓手③)
- **🔴 最近邻 FRANQ(`2505.21072`)**:claim 抽取 + NLI 对证据核验 + 不可溯→UQ + 融合 + selective——**机制几乎与 AOGC 同构**,但**锁死 RAG 单轮静态检索**,不涉及 agent 多轮 / 工具调用 / 轨迹 / 选择性控制闭环。→ 我方差异 = **agent 多轮动态观察(可被 agent 误读)+ 跨步归因 + 选择性控制**,并用 §3.5 漏检实验实证 FRANQ 式做法直接套 agent 会漏哪类失败。
- **🔴 综述 `2602.05073` 附录 E.1**:已用 LLM-judge prompt sketch "工具结果可溯源性当不确定性"。→ 我方差异 = **可复现 string-match + 轻量 NLI pipeline + 系统 UQ 评测(对比基线 + 风险-覆盖)**,定位为该综述开放挑战的"首个落地实例"。
- 电信域 entity-faithfulness 静态检查器(`2601.07342`)——**窄域、是检查器不是不确定性信号、未融合未接控制**。
- MIRAGE-Bench(`2507.21017`)——**只测"agent 在幻觉"的现象,不提供 UQ 信号**(可当评测床+motivation)。
- 差异化五连:**通用域 × 当成不确定性信号 × 与现有 UQ 融合 × 接选择性控制 × 多 benchmark**;对 FRANQ/E.1 额外加 **agent 特有失败 + 正交性实证** 两道护城河。

> 投稿前红线:必须先精读 `2505.21072`(FRANQ,头号)、`2602.05073` 附录 E.1、`2601.07342`、`2601.15703`(尤其附录)、`2507.21017`,确认 AOGC 没被它们悄悄做掉(见 §8)。

---

## 2. 方法

### 2.1 AOGC：动作-观察 grounding 一致性信号

设第 t 步:输入含最新观察 o_t(工具返回/环境反馈),agent 产出 reasoning r_t 与动作 a_t。

1. **抽取 agent 对观察的"主张"**:从 r_t/a_t 中抽出归因于 o_t 的实体、数值、断言(如"文件里显示 X""检索返回了 Y""存在按钮 Z")。
2. **可溯性校验**:每个主张必须能在真实 o_t(或更早工具返回)中溯源——
   - 实体/数值:字符串与实体匹配;
   - 释义性断言:用轻量 NLI(本地 DeBERTa-MNLI)判 o_t 是否 entail 该断言;
   - 动作合法性(可选):a_t 引用的对象(按钮/字段/路径)是否真存在于 o_t。
3. **打分**:`u_t^g = 不可溯主张占比 (+ λ·非法动作指示)`。越高=越可能"读错/臆造观察"。

> 关键:这是 agent 专有、且单轮 UQ 结构上拿不到的信号——单轮信号衡量"模型对自己输出多确定",AOGC 衡量"输出是否忠于环境证据"。

### 2.2 与现有信号/聚合器的融合
- **步级 late-fusion**:把 `u_t^g` 与 verbalized v_t、语义熵 e_t 做逻辑回归/秩融合——**只拟合 2–4 个标量**,在小 dev 集上无梯度搜索,非训练大模型。
- **聚合器无关性**:把融合后的步级不确定性分别灌进 (a) 简单 min/mean/last、(b) TRACER 风格聚合、(c) AUQ 框架——证明 AOGC 对任意聚合器都带来增益(强卖点)。

### 2.3 选择性控制(验 H4)
按融合不确定性触发 {继续 / 重读观察并复述 / 重采样该步 / 问人 / 早停};用风险-覆盖与"成功率 vs 预算"评估。

---

## 3. 实验设计

### 3.1 模型
- 本地(3060,取 logit 用):Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct(4-bit,~5–6GB);上下文吃紧降 Qwen2.5-3B。vLLM/HF 起服务。
- API(黑盒普适,验 H3):GPT-4o-mini、Claude Haiku 之一两个。
- 校验器:本地 DeBERTa-v3-MNLI(几百 MB,CPU/GPU 皆可),AOGC 近免费、可离线。

### 3.2 评测床(全部复用)
- **MIRAGE-Bench**:grounding 失败富集 → H1/H2 主战场。
- **AgentErrorBench**:带失败步标注 → 错误检测/定位评测。
- **ALFWorld / τ-bench / BFCL**:通用 agent 任务,确定性成功判定。

### 3.3 基线(策略=在其之上加 AOGC)
verbalized confidence、semantic entropy、P(True)、self-consistency、**AUQ(`2601.15703`)**、**TRACER(`2602.11409`)**。报"基线"与"基线+AOGC"两组。

### 3.4 指标
- 错误检测:步级/轨迹级 AUROC、AUPRC。
- 校准:ECE、Brier、可靠性图、过度自信率。
- 选择性:AURC、成功率 vs(问人/重试)预算。
- **★ 盲点切片指标**:仅在"grounding 引发的失败"子集上报上述指标——这里我们最强、基线最弱。

### 3.5 关键分析/消融
- **盲点演示**:现有信号在 grounding 失败上的 AUROC ≈ 随机 vs AOGC 高(H1)。
- **正交性**:AOGC 与各信号的相关/互信息低;融合增益来源分解(H2)。
- **黑盒迁移**:无 logit 下 AOGC 仍有效(H3)。
- **消融**:实体溯源-only vs +NLI vs +动作合法性;融合方式;两种聚合器下的增益;λ、k 敏感性。

### 3.5b ★ FRANQ 漏检对比实验(防 FRANQ 撞车的命门,W3–4 起做)
**目的**:实证"把 FRANQ 式 RAG-faithfulness 直接搬到 agent 轨迹会系统性漏检 agent 特有的 grounding 失败",从而把 AOGC 的贡献从"换域"提升为"agent 域必需的新失败模式 + 新信号"。

- **FRANQ-as-agent 基线(我方复现)**:在每个 agent 步,把"当前观察 o_t"当作 FRANQ 的"检索证据",把 agent 输出当作"待核验答案",跑 FRANQ 原样的 claim 抽取 + NLI faithfulness + 其 UQ 分流。**这是最强、最贴脸的 baseline**——审稿人想到的"这不就是 FRANQ 吗"我方先做掉。
- **AOGC 的增量来自三类 agent 特有失败**(FRANQ 设定下结构性看不到):
  1. **跨步 observation 归因**:agent 把"第 t−3 步工具返回"误当成"第 t 步观察",或把早期 observation 当作仍然成立——FRANQ 只比单轮 query↔证据,无跨步证据漂移概念。
  2. **agent 误读 vs 工具本身错的区分**:工具返回正确但 agent 复述错(grounding 失败)vs 工具返回本身为空/报错(aleatoric)——AOGC 显式分离,FRANQ 把两者都当"证据不足"。
  3. **动作-观察一致性(非仅断言)**:a_t 引用的对象(按钮/字段/路径)是否真存在于 o_t——FRANQ 无动作概念。
- **怎么报**:在 MIRAGE-Bench 的 grounding 失败子集上,报 **(i)** FRANQ-as-agent 单独 AUROC、**(ii)** AOGC 单独 AUROC、**(iii)** 二者融合 AUROC;并按上面三类失败切片,展示 FRANQ-as-agent 在哪一类接近随机、AOGC 把它补回来。**核心数字 = AOGC 相对 FRANQ-as-agent 在 agent 特有失败切片上的 ΔAUROC。**
- **退路**:若 FRANQ-as-agent 已经很强、AOGC 增量小 → 把 AOGC 重定位为"FRANQ 的 agent 化 + 跨步/动作扩展 + 选择性控制落地",仍是清晰 delta(但需更强的选择性控制收益来撑)。

---

## 4. 预期结论形态(无论正负都能成文)
- 若 H1/H2 成立:得到"现有 agent UQ 有 grounding 盲点 + 一个便宜信号即可补"的清晰故事。
- 若 AOGC 与现有信号相关(非正交):退一步定位为"**单次调用的廉价代理**",讲效率(用一个信号近似多采样),仍可发表。

---

## 5. 算力/成本核对
- 本地 4-bit 7B/3B 推理,3060 可跑;AOGC 用本地 NLI,**零 API 成本、可离线**。
- API 仅 H3 普适性,k≈5、数百 episode,约数十美元。
- 融合层只拟合标量,CPU 秒级,**无大模型训练**。

---

## 6. 10–12 周里程碑(早期就 de-risk)
1. **W1–2** 跑通 MIRAGE-Bench + 1 本地模型 rollout 与日志。
2. **W2–3 ★关键先验**：先只做"盲点演示"(基线信号在 grounding 失败上的 AUROC)——若 H1 不成立,尽早转向。
3. **W3–4** 实现 AOGC(实体溯源 + NLI)+ **复现 FRANQ-as-agent 基线并跑 §3.5b 漏检对比**(防 FRANQ 撞车的第二个早 de-risk:若 AOGC 相对 FRANQ-as-agent 在 agent 特有失败切片上无可见 ΔAUROC,尽早走 §3.5b 退路)。
4. **W5–6** 融合 + 两个聚合器(AUQ/TRACER)对比(H2)。
5. **W7** 黑盒 API 迁移(H3)。
6. **W8** 选择性控制 + 风险-覆盖(H4)。
7. **W9–10** AgentErrorBench/τ-bench/BFCL 扩展 + 消融。
8. **W11–12** 写作 + 复现包。

---

## 7. 风险与对策
| 风险 | 对策 |
|---|---|
| `2601.07342` 已做实体溯源 | 我们=通用域+UQ 信号+融合+选择性控制+多 benchmark;并把它列为最近邻并实验对比 |
| AUQ 附录已含类似信号 | 先精读;若有,转为"系统化+正交性+黑盒+聚合器无关"的更强实证,并明确 delta |
| H1 不成立(无盲点) | W2–3 就能看出;预案见 §4 退路(效率叙事) |
| NLI 校验不准 | 人工抽检 100–200 条报一致性;盲点结论只在高一致子集下 |
| 3060 显存 | 降 3B/缩上下文/4-bit;AOGC 不依赖大模型 |

---

## 8. 投稿前必须手动做
1. 精读最危险的五篇:**FRANQ `2505.21072`(头号,机制同构)**、UQ 综述附录 E.1 `2602.05073`、AUQ `2601.15703`(连附录)、MIRAGE-Bench `2507.21017`、Telecom entity-faithfulness `2601.07342`。
2. ~~核对所有 arXiv 编号~~ **(2026-06-15 已 web 核实:12 个原编号全部真实且准确)**;仍需核对各篇是否已被会议录用(影响撞车严重度)。
3. 设关键词订阅持续监控:agent uncertainty / observation faithfulness / trajectory risk / grounding / faithfulness-aware UQ。

---

## 9. 关键参考
- 综述:UQ in LLM Agents `2602.05073`
- 作骨干/基线:AUQ `2601.15703`、TRACER `2602.11409`、Uncertainty Propagation `2604.23505`
- 评测床:AgentErrorBench `2509.25370`、MIRAGE-Bench `2507.21017`、Seeing the Whole Elephant `2604.22708`
- 最近邻(grounding):**FRANQ(faithfulness-aware UQ,RAG)`2505.21072`**、**UQ 综述附录 E.1 evidentiality `2602.05073`**、**GSAR(typed grounding,multi-agent)`2604.23366`**、Telecom entity-faithfulness `2601.07342`
- motivation(已实证盲点真实):The Confidence Dichotomy(tool-use agent 误校准)`2601.07264`
- 单轮 UQ:Semantic Entropy(Kuhn 2023 / Farquhar Nature 2024)、Verbalized(Tian 2023)、P(True)(Kadavath 2022)、Self-Consistency(Wang 2022)、Black-box UQ `2305.19187`
- 校准/比较:Agentic Overconfidence `2602.06948`
- Agent benchmark:ALFWorld(Shridhar 2021)、τ-bench(2024)、BFCL、GAIA(Mialon 2023)
