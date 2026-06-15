# AOGC 代码骨架(实现说明)

研究计划见 `Agent_UQ_研究计划_v2.md`。本文件只讲代码:结构、已落地/未落地、怎么跑、到真实结果的路径。

## 快速开始

```bash
# 依赖(metrics + AOGC 信号只需 CPU/MPS,不要 GPU)
pip install -r requirements.txt          # 见下方"环境"关于 sentencepiece 的坑

python3 tests/test_metrics.py            # 7 个 metrics 测试
python3 tests/test_aogc.py               # 10 个 AOGC 核心测试(离线,LexicalNLI)
python3 experiments/blindspot_demo.py    # W2-3 盲点演示(合成数据,验证管线)
```

全部 17 个测试 + demo 已在本机(macOS, Python 3.14, torch 2.9 MPS)跑通。

## 结构

```
aogc_uq/
  data/schema.py        # ★契约: Step / Trajectory / ErrorType,JSON 可序列化
  data/mirage.py        # MIRAGE-Bench 加载器(GUI 快照 + tool-call 对话两种格式)
  metrics/              # 测量层(已自测)
    detection.py        #   AUROC / AUPRC  (score=不确定性,正类=is_error)
    calibration.py      #   ECE / Brier / overconfidence / reliability_bins
    selective.py        #   risk-coverage 曲线 / AURC
    slicing.py          #   ★盲点切片: 仅 {grounding 错误 vs 正确} 上算 AUROC
  aogc/                 # ★核心信号(已自测,真 NLI 已在 MPS 验证)
    claims.py           #   抽 entities / numbers / assertions
    nli.py              #   NLI 后端: LexicalNLI(离线) | TransformerNLI(DeBERTa-MNLI)
    traceability.py     #   entity/number 字面匹配 + assertion 的 NLI 溯源
    signal.py           #   u_t^g = 不可溯主张占比(+λ·非法动作),带 breakdown
    claims.py           #   ...含 extract_action_target(只对 click/fill 等元素动作判合法性)
  baselines/
    verbalized.py       #   verbalized confidence -> 不确定性(黑盒,已可用)
    franq.py            #   FRANQ-as-agent 最强基线(reasoning-claim vs 当前观察)
  rollout/              # ★ GPU 侧推理 harness(Colab T4 / 3060;stub 后端 CPU 可测)
    generate.py         #   ResponseGenerator: EchoGenerator(stub) | HFGenerator(4-bit,惰性)
    judge.py            #   Judge: StubJudge | LLMJudge(complete 回调,SDK 无关);apply_verdict
    run.py              #   run_rollout(可断点续跑,逐行 flush jsonl) / load_scored
  fusion/               # (待) late-fusion 逻辑回归
experiments/
  blindspot_demo.py     # 端到端 H1 演示(合成数据)
  franq_vs_aogc.py      # §3.5b FRANQ 漏检对比(合成数据)
  mirage_smoke.py       # 真 MIRAGE 观察 -> AOGC(loader 闭环冒烟)
notebooks/
  colab_rollout.ipynb   # ★ Colab: clone MIRAGE -> 4-bit rollout -> judge -> 打分 jsonl 存 Drive
tests/                  # metrics/aogc/franq/mirage/rollout(共 32);全 CPU 可跑
  fixtures/mirage/      # 3 个真 MIRAGE 快照(Apache-2.0,见 SOURCE.md)
```

## 数据契约(最重要)

一切围绕 `Trajectory(steps=[Step,...])`。三层互不耦合,只靠 schema 对接:
1. **rollout 生产者**(MIRAGE/ALFWorld/τ-bench 加载器,或本地模型 agent loop)→ 产出 `Trajectory`。
2. **UQ 信号**(AOGC / verbalized / 语义熵 ...)→ 读 `Step`,产出每步标量(存进 `step.signals[name]`)。
3. **评测**(metrics / fusion)→ 吃标量 + 标签。

关键约定:**信号 = 不确定性(越高越可能出错)**;标签字段 `is_error` / `error_type` 只给评测用,信号不许读。`ErrorType.GROUNDING` 是本项目主攻的盲点类。

rollout 可 `Trajectory.to_dict()` dump 成 jsonl(在 3060 上跑),再在任意机器 `from_dict` 分析。

## 已落地 vs 待办

| 模块 | 状态 |
|---|---|
| schema / metrics / 盲点切片 | ✅ 实现 + 自测 |
| AOGC: claim 抽取 + 字面溯源 + NLI + u_t^g | ✅ 实现 + 自测(LexicalNLI 离线) |
| TransformerNLI(DeBERTa-MNLI on MPS) | ✅ 插管验证:P(entail\|支持)=0.99 vs P(entail\|臆造)=0.01 |
| verbalized 基线 | ✅ 可用(黑盒) |
| 盲点演示 harness | ✅ 合成数据跑通;换上真 rollout 即出 H1 真证据 |
| **MIRAGE-Bench 加载器** | ✅ 实现 + 自测(`data/mirage.py`,对真快照 fixtures)。两种格式都解析(GUI AXTree / tool-call 对话);risk 设置→ErrorType 映射(popup/unexpected_transition/misleading/error_feedback=GROUNDING);`attach_response()` 解析 `<think>/<action>` 与 `Thought/Action`;真观察上 AOGC 区分 grounded(0.00) vs hallucinated(0.50) |
| **本地模型 rollout loop** | ✅ harness 实现 + 自测(`aogc_uq/rollout/`,stub 后端 CPU 跑通)。`HFGenerator`(4-bit)+`LLMJudge`(API)真后端等 Colab/3060 跑;`run_rollout` 可断点续跑、逐行 flush。Colab notebook 已就绪(`notebooks/colab_rollout.ipynb`)。**注:真 H1 数字仍需在 Colab/3060 上实跑 rollout+judge** |
| verbalized 基线接真数据 | ⬜ 需"会自报置信"的生成变体(MIRAGE prompt 不要置信);AOGC vs FRANQ 不需要,可先出 |
| 语义熵 / P(True) / self-consistency 基线 | ⬜ 需模型采样,rollout 阶段做 |
| late-fusion(逻辑回归拟标量) | ⬜ 待做(`aogc_uq/fusion/`) |
| **FRANQ-as-agent 基线 + §3.5b 漏检对比** | ✅ 实现 + 自测(`baselines/franq.py`、`experiments/franq_vs_aogc.py`、`tests/test_franq.py`)。合成对比:单观察 grounding 上 FRANQ≈AOGC(公平强基线),动作合法性 grounding 上 FRANQ 盲(AUROC 0.50 vs AOGC 1.00),跨步正确引用上 FRANQ 假阳 100% vs AOGC 0% |

## Colab 工作流(GPU 侧)

3060 之前用 Colab(免费 T4 16GB ≥ 3060;sentencepiece 在 Colab 能装→可跑真 DeBERTa-v3)。
1. 把本仓库 push 到 GitHub(或 zip 上传 Drive),打开 `notebooks/colab_rollout.ipynb`,Runtime 选 GPU。
2. notebook 里设 `AOGC_REPO_URL`、`SETTINGS`(默认 grounding 四类)、`JUDGE_BACKEND`+API key(Colab Secrets)。
3. 跑 rollout → 打分 jsonl 存到 `Drive/aogc_runs/`(断线只丢一条,重跑续上)。
4. 把 jsonl 拉回 Mac,用 `aogc_uq.rollout.load_scored()` + 现有 metrics/AOGC/FRANQ 分析(无需 GPU)。

数据流不变:GPU 只管"生成回答+judge 标签",分析层全在 Mac。`run_rollout` 默认丢掉 `meta['messages']` 让 jsonl 小。

## MIRAGE-Bench 关键认知(核实自真仓库,非 README)

- 每个 `dataset_all/<setting>/<env>/*.json` 是一个**决策点快照(prompt)**,不是带标签的完整轨迹。**目录名=被诱发的失败类别**(ground-truth risk type),**不代表本次 rollout 真的错了**。
- 因此 `is_error` 必须靠 **跑模型 + LLM-judge** 得到(3060/API 阶段);loader 只给 `error_type=被诱发类型` 且 `is_error=None`。
- 两种磁盘格式:GUI(webarena/workarena/osworld,观察是 AXTree/a11y 文本)、tool-call(taubench/theagentcompany,完整 Thought/Action↔API output 对话)。
- 我们主攻的 **GROUNDING 切片 = popup / unexpected_transition / misleading / error_feedback**;其余设置(repetitive/unachievable/underspecified/users_questions)留作非 grounding 错误,用来验 AOGC 的特异性(不该误报)。
- 数据没 vendoring,要 `git clone https://github.com/sunblaze-ucb/mirage-bench data/raw/mirage-bench` 后设 `MIRAGE_ROOT=.../dataset_all`。`tests/fixtures/mirage/` 只放 3 个小快照做解析测试。

## 环境坑

- **sentencepiece**:DeBERTa-**v3** 的 tokenizer 需要它;本机 homebrew Python 是 PEP-668 externally-managed,`pip install sentencepiece` 被拦。两条路:(a) 建项目 venv;(b) `--break-system-packages`(不推荐)。当前 `TransformerNLI` 已用 BPE 系 NLI 模型(`cross-encoder/nli-distilroberta-base`,无需 sentencepiece)验证可用;真实评测建议在 3060 的 venv 里用 DeBERTa-v3。
- **DEFAULT_NLI_MODEL** 设为 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`;离线/无 sentencepiece 时用 `get_nli(None)` 走 LexicalNLI。
- 算力:metrics + AOGC(含 NLI)在 Mac MPS/CPU 即可开发;只有大模型 agent rollout 必须上 3060。

## 设计取舍(写论文时要 ablate 的点)

- claim 抽取是**透明启发式 v0**:正则抽 entities/numbers + 句子切分抽 assertions。论文消融 entity-only vs +NLI vs +动作合法性 = 在 `AOGCConfig` 里改开关/权重,不用改代码。
- `u_t^g` 只在**激活的项**上加权归一,所以"关掉 NLI"≠ 引入偏置。
- 短 entity(≤3 字符)溯源要求词边界,降假阴性误判。
