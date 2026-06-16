# 项目交接 · AOGC（给接手的 AI agent）

> 你正在接手一个**冲 AAAI 的研究项目**。这份文件让你不依赖任何历史对话就能继续。
> 先读本文件（5 分钟），再按 §1 顺序读其余文档，然后从 §5「下一步」动手。
> 写于 2026-06-15，对应 commit `1481e5a`（main）。

---

## 0. 一分钟速览

- **是什么**：LLM Agent 不确定性量化（UQ）的新信号 **AOGC（Action-Observation Grounding Consistency，动作-观察一致性）**。检查 agent 是否"读对了"工具/环境返回的观察；误读时现有 UQ（verbalized 置信、语义熵）看不见，AOGC 能看见。训练-free、黑盒可用、与现有信号正交、可叠加。
- **目标会场**：AAAI main track（现实退路 ICLR 2027 / AAAI-28，见 README）。
- **算力**：全程免训练。本地 = Mac（开发+CPU/MPS 自测）；GPU = **Colab T4**（rollout+judge）或 RTX 3060。
- **现在到哪**：整条流水线已在**真 MIRAGE-Bench 数据**上端到端跑通；代码层全绿（6 个测试文件，40 测试）。**机制已验证活了**（AOGC 的差异化在真数据上会触发）。
- **唯一瓶颈**：**判定标签（is_error）不可信**——无 API key 时本地小模型当 judge 吐不出可靠标签。所以现在的 AUROC 数字（AOGC 0.512 vs FRANQ 0.602）**是噪声、不能下结论**。
- **下一步**：用 `experiments/label_gold.py` 人工标一小批 gold 标签（计划里的"人工抽检"），先确认"AOGC 的缝在真数据上为正"，再 ablate + fusion + 扩样本。见 §5。

---

## 1. 阅读顺序（都在仓库根目录）

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | **本文件 `HANDOFF.md`** | 活状态 + 下一步 + 教训（你在读） |
| 2 | `README.md` | 研究项目总览：一句话、H1–H4、贡献、红线、关键设定、时间线 |
| 3 | `CODE.md` | 代码开发说明：结构、已做/待做、环境坑、MIRAGE 关键认知、Colab 工作流 |
| 4 | `Agent_UQ_研究计划_v2.md` | **主研究计划**：方法、实验设计、§3.5b FRANQ 漏检对比、里程碑、风险 |
| 5 | `Agent_UQ_邻近工作撞车排查.md` | 撞车排查（含 §4b 最危险的两条线 FRANQ / 综述 E.1） |
| - | `Agent_UQ_研究计划_v1.md` | 历史参考（已被 v2 取代，别照 v1 做） |

---

## 2. 项目是什么（研究内核）

**核心论点（可证伪）**：现有 agent UQ 有一个共同盲点——不检查 agent 是否忠于观察。agent 误读/无视工具返回时，verbalized 置信和 token 熵仍然很高，于是这类失败"看不见"。

- **H1（盲点存在）**：grounding 引发的失败上，现有 UQ 的错误检测 AUROC ≈ 随机且过度自信。
- **H2（正交+增益）**：AOGC 与现有信号相关性低；late-fusion 后 AUROC/ECE/AURC 改善。
- **H3（黑盒可迁移）**：仅靠文本（无 logit）即可算 AOGC，闭源 API 上也有效。
- **H4（可用性）**：把融合信号接入选择性控制（早停/重读/问人），固定预算下提升成功率。

**AOGC 怎么算**（`aogc_uq/aogc/signal.py`）：从 agent 的 **reasoning** 抽出对观察的"主张"（实体/数值=字面匹配；释义断言=轻量 NLI 溯源），算"不可溯比例"；再加**动作合法性**项（动作引用的元素是否真存在于观察）。`u_t^g ∈ [0,1]`，越高越可能误读/臆造。

**⚠️ 卖点必须钉死（防被审稿人一句话毙，见撞车排查 §4b / v2 TL;DR）**：
- **不要**写成"我们提出用 grounding 一致性做 UQ"——综述 `2602.05073` 附录 E.1 已把它当"旧轴"、**FRANQ（`2505.21072`）已做其 RAG 版（机制几乎同构）**。
- **要**钉在两点：① **agent 特有失败**（多轮交互中误读/无视观察而置信不降，FRANQ 式 RAG faithfulness 在 agent 域系统性漏掉）；② **正交性实证**（AOGC 与现有 UQ 融合后实打实涨）。
- 防 FRANQ 的关键实验：`experiments/franq_vs_aogc.py` 实现的 **§3.5b 漏检对比**（FRANQ-as-agent 是一等基线，不是稻草人）。

---

## 3. 仓库与环境

- **GitHub**：`https://github.com/Mowon0303/AOGC-agent-uq`（**public**，main 分支）。本地：`<新机器路径>/AOGC-agent-uq`。
- **commit 红线：不要加 Co-Authored-By**（用户个人仓库习惯）。
- **Python**：纯 Python（非 TS）。开发机 Mac 上 Python 3.14 + torch 2.9（MPS）。依赖见 `requirements.txt`。
- **依赖坑**：`sentencepiece`（DeBERTa-v3 tokenizer 需要）在 Mac 的 homebrew Python 上被 PEP-668 拦；**Colab 能正常装**。所以真 DeBERTa-v3 NLI 在 Colab 跑；Mac 上用 `LexicalNLI`（离线 token 重叠代理）或 BPE 系 NLI 兜底。
- **跑测试**（CPU，全离线，应全绿 = 40）：
  ```bash
  for t in tests/test_*.py; do python3 "$t"; done
  ```
- **跑 demo**（合成数据，证明各层组合正确）：
  ```bash
  python3 experiments/blindspot_demo.py     # H1 盲点 harness
  python3 experiments/franq_vs_aogc.py      # §3.5b FRANQ 漏检对比
  python3 experiments/fusion_demo.py        # H2 融合 + H4 选择性控制
  python3 experiments/mirage_smoke.py       # 真 MIRAGE 观察 -> AOGC
  ```
- **Colab 工作流**（GPU 侧）：`notebooks/colab_rollout.ipynb`。零密钥可跑（`JUDGE_BACKEND="local"` 复用 rollout 模型当 judge；repo public 免 token）。打分 jsonl 存 Drive，拉回任意机器分析。详见 CODE.md「Colab 工作流」。

### 数据获取（不在仓库里）
- **MIRAGE-Bench**：`git clone https://github.com/sunblaze-ucb/mirage-bench`，设 `MIRAGE_ROOT=.../dataset_all`。**认准 arXiv 2507.21017**（有多个同名 benchmark）。`tests/fixtures/mirage/` 只放 3 个小快照做解析测试。

---

## 4. 现在到哪了（代码状态：基本齐，全绿）

| 层 | 文件 | 状态 |
|---|---|---|
| 数据契约 | `aogc_uq/data/schema.py` | ✅ Step/Trajectory/ErrorType，JSON 可序列化 |
| MIRAGE 加载器 | `aogc_uq/data/mirage.py` | ✅ 解析 GUI 快照 + tool-call 对话两种格式；risk→ErrorType 映射；`attach_response()` 解析模型回答 |
| 度量 | `aogc_uq/metrics/` | ✅ AUROC/AUPRC、ECE/Brier/overconfidence、AURC/risk-coverage、盲点切片 |
| AOGC 信号 | `aogc_uq/aogc/` | ✅ claim 抽取 + 字面/NLI 溯源 + `u_t^g`；NLI 后端 Lexical(离线)/Transformer(DeBERTa) |
| 基线 | `aogc_uq/baselines/` | ✅ verbalized；**FRANQ-as-agent**（防撞车一等基线） |
| 融合+控制 | `aogc_uq/fusion/` | ✅ rank/mean/LogReg 融合；success-vs-budget + decide_action |
| rollout+judge | `aogc_uq/rollout/` | ✅ HFGenerator(4bit)/EchoGenerator + LLMJudge/StubJudge + run_rollout(断点续跑) |
| Colab | `notebooks/colab_rollout.ipynb` | ✅ 零密钥可跑 |
| 诊断工具 | `experiments/inspect_scored.py` | ✅ 诊断 judge 率 + 为什么 AOGC vs FRANQ |
| 人工标注 | `experiments/label_gold.py` | ✅ 人工 gold 集 + 评估 |

**首次真跑（Colab T4, Qwen2.5-3B, popup+unexpected_transition × webarena+osworld, 100 步）结果**：
- ✅ 机制活了（`inspect_scored.py` 证实）：reasoning 解析 100/100、动作指向元素 16/100、跨步 5/100、**AOGC≠FRANQ 85/100**。
- ❌ **数字不可信**：judge 标签全是 `unparsed; heuristic fallback`（本地小模型吐不出可靠判定）。AOGC 0.512 vs FRANQ 0.602 在 14 个噪声标签上 = 随机波动，**不能据此说 AOGC 不如 FRANQ**。

---

## 5. 下一步（按优先级；从这里动手）

### 第一步：拿到可信标签（当前唯一瓶颈）
本地 judge 不可信。无 API key 下的可信路径 = **人工标一小批 gold**（计划本就要"人工抽检 100–200 条"）：
```bash
# 用已有的打分 jsonl（模型回答已存，无需重跑 GPU）
python3 experiments/label_gold.py <scored_*.jsonl> [gold.json] --n 40
```
逐条显示 goal/reasoning/action +「点的元素在不在观察里」提示，按 y/n/s/q 标，当场算 AOGC vs FRANQ。标的时候重点看 `⚠ action targets ... NOT among observed elements` 那些步——这是 AOGC 动作合法性发力、FRANQ 看不到的地方。
- **若有 API key**：把 judge 换成强模型（`LLMJudge(complete_fn)`，见 notebook 第 8 格的接法），自动标全量 + 人工抽检报 κ 一致性。这是论文级标签的正路。

### 第二步：确认缝为正 + 消融
有可信标签后：
- 跑 `AOGCConfig` 消融：entity-only vs +number vs +NLI vs +动作合法性（改开关，不改代码）——定位 AOGC 的增益来自哪一项。
- 跑 `LogRegFusion`（`aogc_uq/fusion/`）：证明 AOGC 与现有信号**正交可加**（融合 AUROC > 单信号）。这是 H2、也是防"已知积木工程组合"质疑的硬证据。

### 第三步：补齐 AOGC 的另一半卖点（跨步）
**MIRAGE GUI 快照是单决策点，`prior_observations` 恒空 → AOGC 的跨步归因项天生用不上**（见 §6）。要展示跨步优势，需要**多步轨迹**：
- MIRAGE 的 **tool-call 设置**（theagentcompany/taubench，有多轮 API output 历史 → prior_observations 非空），或
- 自建 **ALFWorld / τ-bench** 的 agent rollout loop（多步）。
- verbalized 基线接真数据还需"**会自报置信**"的生成变体（现在 MIRAGE prompt 不要置信）——做 H1/H2 对比要补。

### 第四步：扩规模 + 多模型/benchmark + 写作（见 v2 §6 里程碑）

---

## 6. 关键认知与教训（别重复踩）

1. **MIRAGE 是"决策点快照 prompt"，不是带标签轨迹**。目录名=被诱发的失败类别（非本次真错）；`is_error` 必须靠 rollout+judge。GROUNDING 切片 = `popup / unexpected_transition / misleading / error_feedback`；其余设置当非 grounding 对照（验 AOGC 不误报）。
2. **MIRAGE GUI 快照压制了 AOGC 一半卖点**（无跨步）。这是数据局限，不是 AOGC 的问题；用 tool-call/多步数据才能显跨步项。
3. **judge 是核心难点**。本地小模型判长 AXTree 既不准也吐不出 JSON。已改成 YES/NO 极简格式提升可解析率，但准确性仍有限——可信数字最终要强 judge + 人工校准。
4. **swebench 是最差的 grounding 测试床**（自由文本、无 `<think>/<action>`、无 AXTree、prompt 超长慢）。优先 webarena/osworld 的 popup/unexpected_transition。
5. **已修的坑（防回归 / 防困惑，都在 git 历史里）**：
   - `apply_chat_template` 跨 transformers 版本返回类型不一 → 用 `return_dict=True` 取 `input_ids`（commit 7c2d0d6）。
   - 长 prompt CUDA OOM → 收紧 `max_input_tokens=4096`/`max_new_tokens=256` + `empty_cache`；notebook 默认 3B（c11dced）。
   - **`parse_agent_response` 自由文本时 reasoning 留空 → AOGC/FRANQ 都读空 → 退化相等**（1d4f36c，曾导致 AOGC==FRANQ 的假象）。
   - 本地 judge 吐不出 JSON → 改 YES/NO（2fde2f9）。
   - `run_rollout(resume=False)` 现在截断而非追加（1481e5a）。
6. **不要凭记忆相信 arXiv 编号**：文档里的编号已 web 核实过一轮（README/撞车排查里有标注），但新论文月更很快，动手前再核一遍最危险的几篇（FRANQ 2505.21072、综述 2602.05073、AUQ 2601.15703、MIRAGE 2507.21017）。

---

## 7. 红线与约束（必须遵守）

- **训练-free**：不微调任何大模型；AOGC 只做轻量校验（本地 NLI 小模型）。融合层只拟几个标量。
- **算力**：Mac 开发/CPU 自测；GPU 用 Colab T4 / 3060。别假设有大显存。
- **git**：commit **不加 coauthor**。
- **结果不夸大**：合成 demo 标注清楚"SYNTHETIC, not paper results"；小样本只当方向性；judge 噪声时明说数字不可信。这是研究诚信，也是这个项目一路的做法。
- **信号不许读标签**：`is_error`/`error_type` 只给评测；任何 UQ 信号不得读（schema 里有注释）。

---

## 8. 一句话给接手 agent

代码地基和整条流水线都通了、机制也验证活了；你接手后**第一刀切在"拿可信标签"**（`label_gold.py` 人工标 40 条，或上强 judge），确认 AOGC 的动作合法性缝在真数据上为正，再做消融+融合+补跨步数据。别被"AOGC 0.512 < FRANQ 0.602"误导——那是噪声标签下的假象，不是结论。
