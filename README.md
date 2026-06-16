# AOGC · Agent UQ 的观察-grounding 盲点

> 🤝 **接手/继续这个项目的 agent：先读 [`HANDOFF.md`](HANDOFF.md)**（自包含的状态、瓶颈、下一步、教训），再回来读本文件。

LLM Agent 不确定性量化(UQ)研究项目。目标会场:**AAAI main track**。算力约束:**RTX 3060 单卡(12GB) + API,全程免训练**。

## 一句话

现有 agent UQ(verbalized 置信系、熵系、轨迹风险系如 AUQ/TRACER)有一个共同盲点——**它们不检查 agent 是否真的"读对了"工具/环境返回**。agent 误读观察时,其置信仍然很高,于是对这一整类失败"看不见"。本项目提出训练-free 的 **AOGC(动作-观察一致性)信号**,与现有信号正交,叠加到 SOTA 之上提升错误检测/校准/风险-覆盖。

## 文件索引(建议阅读顺序)

| 文件 | 内容 |
|---|---|
| `Agent_UQ_研究计划_v2.md` | **当前主计划**(grounding 盲点版)。假设 H1–H4、AOGC 方法、实验/基线/指标、10–12 周里程碑、风险对策 |
| `Agent_UQ_邻近工作撞车排查.md` | 近半年(2025-09～2026-06)邻近工作对照表;说明为何要从 v1 转向 v2 |
| `Agent_UQ_研究计划_v1.md` | 初版(通用 agent UQ 计划)。其抓手①传播、②失败定位已被占,**仅作历史参考** |

## 当前状态

- 方向探索:推理 / Agent / 效率 / 安全 → 选定 **Agent UQ**。
- 撞车排查:v1 的传播、失败定位被占;**grounding 信号(AOGC)存活**。
- **2026-06-15 web 深度复检**:12 个原 arXiv 编号全真;但新挖出两条撞车 **FRANQ `2505.21072`(机制同构)+ 综述 E.1 `2602.05073`**(已补入撞车排查 §4b、v2 §1.2/§3.5b)。新颖性存活 **6/10**,缝窄需主动守。
- 卖点已重写:**不写"提出 grounding 当 UQ"**(旧轴),改钉 **"agent 特有失败(误读观察而置信不降)+ 与现有 UQ 正交可融合"**。
- 已收敛到 **v2**。下一步 = 实现 + 早期 de-risk(含 §3.5b FRANQ 漏检对比)。

## 投稿前红线(动手前必做)

1. 精读最危险的,确认 AOGC 没被偷做掉(**2026-06-15 复检后头号威胁变了**):
   - 🔴 **FRANQ(faithfulness-aware UQ for RAG)`arXiv 2505.21072`** —— 机制几乎同构,锁死 RAG;**头号撞车,必读**
   - 🔴 **UQ 综述附录 E.1(evidentiality)`arXiv 2602.05073`** —— 已 sketch 同思想,且把 grounding 当"旧轴"
   - Agentic Uncertainty Quantification(AUQ)`arXiv 2601.15703`(连附录)
   - MIRAGE-Bench `arXiv 2507.21017`、Telecom entity-faithfulness `arXiv 2601.07342`
2. ~~核对所有 arXiv 编号~~ **(已 web 核实:12 个原编号全部真实且准确)**;仍需核对各篇是否已被会议录用。
3. 设关键词订阅持续监控:agent uncertainty / observation faithfulness / trajectory risk / grounding / **faithfulness-aware UQ**。

## 早期 de-risk(里程碑 W2–3)

先只做"**盲点演示**":在 MIRAGE-Bench 的 grounding 失败子集上,测现有信号(verbalized/语义熵)的错误检测 AUROC。若接近随机 → H1 成立,继续;若不成立 → 尽早转向(退路:把 AOGC 重定位为"单次调用的廉价代理",讲效率)。

## 关键设定速查

- 本地模型:Qwen2.5-7B / Llama-3.1-8B(4-bit);校验器用本地 DeBERTa-MNLI(AOGC 零 API、可离线)。**(可行性已核实:DeBERTa-v3-MNLI ~434M 可 CPU 离线;7B/8B 4-bit 约 4.5–5.5GB 可跑 3060。)**
- 评测床(全部复用,不自建):MIRAGE-Bench、AgentErrorBench、ALFWorld、τ-bench、BFCL。**易加载=ALFWorld + BFCL 静态集;较重=τ-bench(调付费 API 模拟 user)、AgentErrorBench 上游 GAIA/WebShop(受限)。MIRAGE-Bench 有 4 个同名项目,引用务必带 `arXiv 2507.21017` 锁定。**
- 时间线(**2026-06-15 核实修正**):AAAI-27 **abstract 2026-07-21(硬前置)/ 全文 2026-07-28**(非 07-27,且只剩 ~5 周,太赶);**NeurIPS 2026 已截止(5/6)出局**;现实退路 = **ICLR 2027(约 9 月,最对口)** / ACL-ARR / AAAI-28。

## 待办(next)

- [ ] W1–3 代码骨架:MIRAGE-Bench 加载 + 本地模型 rollout + 基线信号 + 盲点演示脚本
- [ ] AOGC 算法伪代码 + 指标实现(AUROC/ECE/AURC + 盲点切片)
- [ ] **W3–4 复现 FRANQ-as-agent 基线 + 跑 §3.5b 漏检对比(防 FRANQ 撞车的命门)**
- [ ] 精读五篇红线论文(FRANQ/E.1/AUQ/MIRAGE/Telecom)并记 delta
