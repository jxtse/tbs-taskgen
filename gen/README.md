# tbsgen：TB-Science 出题管线 v2

按 `../docs/DESIGN.md` 的 MVP 搭好的出题管线：从检索结果开始，经过全文获取、可行性检查、规格、生成、机械门、修复、校准调度，最后打包成 Harbor 任务。**目前只做过调通和干跑，没有量产，也没有生成真实题目。** 量产要等定预算（见文末）。

## 1. 架构

```
src/tbsgen/
  config.py        全部配置与默认值（待定项都在这里）；config.toml 可覆盖
  budget.py        sqlite 预算账本：按 模型×阶段×条目 记 token/美元/次数；日预算、调用上限、单条目上限；熔断
  models.py        模型调用层：OpusClient（本机网关，同旧 llm.py）/ FakeClient（脚本化）/ RecordingClient（录制）
  state.py         每条目一个状态目录 + state.json 状态机；events.jsonl
  pipeline.py      编排：逐条推进、断点续跑、预算/熔断暂停、resume
  intake.py        S1 读检索层输出；S2 规则筛选；S3 全文；S4 可行性（规则部分 + 官方题去重）
  prompts.py       S5/S6A/S6C/S8 提示词（版本号写进 provenance）
  spec.py          S5 规格校验（依赖图规则）；S6B 按实测定阈值
  runner.py        Docker 沙箱里跑 生成器/参考解/基线/measure/verifier
  sandbox.py       docker run（--network none、2 CPU、4 GB、pids 512，与旧 validate.py 相同）
  gates.py         S7 机械门
  leaves.py, leakgate.py, leakcheck.py, introspect_probe.py   4 层泄漏门（L1–L4a）
  repair.py        S8 局部修复的保工审查（机械 diff）
  package.py       PKG：打包为 Harbor 任务 + 结构校验
  calib.py         S9：GLM 顺序校准 + Opus 教师分支的调度规则、Harbor 作业配置、结果收集
  report.py        日报（Markdown）
  stages_a.py, stages_b.py   各阶段处理函数
  cli.py           命令行 tbsgen
  fakes.py, dryrun.py        fake 模型脚本与端到端干跑
  backtest.py      泄漏门回测
  templates_metrics.py       verifier 公共部分（复制自旧 templates_metrics.py v0.3）
docker/Dockerfile.base       沙箱镜像 tbsgen-base:py311-sci-v2（同旧 package.py 的科学栈版本）
tests/                       单测 + 集成测试；tests/fixtures/toy_task 为人造最小题
eval/                        泄漏门回测、旧教师作业 UnknownApiError 归因
```

运行目录（`--run DIR`）：

```
ledger.sqlite  events.jsonl  reports/daily-<日期>.md  release/manifest.jsonl
items/<item_id>/
  state.json                 当前阶段、状态、历史、各阶段尝试次数
  S1_candidate.json S2_screen.json S3_fulltext.json fulltext.txt S4_probe.json
  _spec/{spec,depgraph,coverage,provenance}.json      出题侧，不进 Harbor
  task/                      工作树：generator/ solution/ baselines/ tests/ instruction.md
  runs/                      沙箱运行产物
  S6A_measure.json thresholds.json gates/attempt-<n>.json repair/round-<n>/
  pkg/                       Harbor 任务（只有这个目录交给 Harbor）
  calib/runs.json + 作业配置  S9
```

条目状态：`active`（还有下一阶段）、`waiting`（S9 等校准结果）、`paused`（预算或熔断暂停，`resume` 后从原阶段继续）、`done`、`dropped`（带原因码）、`parked`（需要人工处理）。每个阶段先写完产物，再原子地更新 `state.json`；进程在阶段中途被杀，重跑时从该阶段重新开始。

## 2. 各阶段输入输出

| 阶段 | 输入 | 输出 | 模型 | 失败去向 |
|---|---|---|---|---|
| S1 | 检索层输出（`tbsretr` 的 `papers` 或 `eval/raw/*.json` 的 `selected`）或候选 JSONL | `S1_candidate.json` | 无 | — |
| S2 | 题录 + 摘要 | `S2_screen.json`（分数、正负线索、是否黑盒形态） | 规则（廉价模型待定） | `dropped: low_screen_score` |
| S3 | DOI、OA 链接 | `fulltext.txt` + `S3_fulltext.json` | 无 | `parked: fulltext_unavailable`，并写 `S3_queue.json` |
| S4 | 全文 + 筛选记录 | `S4_probe.json`（计算核心线索、与官方 70 题去重、形态禁令） | 规则（沙箱探针待定） | `dropped: no_compute_core / tbs_duplicate / interact_form_banned_until_isolation` |
| S5 | 全文摘录 + 探针卡 | `_spec/spec.json`、`depgraph.json`、`coverage.json`、`provenance.json` | Opus | 校验不过重试 1 次，仍不过 `dropped: spec_invalid` |
| S6A | 规格 | `task/generator/`、`solution/`、`baselines/{mid,naive}.py`、`tests/measure.py` | Opus | 缺文件 `dropped` |
| S6A_run | 工作树 | `S6A_measure.json`：两个种子下 ref/mid/naive 的实测指标 | 无（Docker） | 生成器或参考解失败 → S8 |
| S6B | 实测指标 | `thresholds.json`：分离型指标取 ref 与 mid 的对数中点，非分离型取 ref 放宽 50%，取整一律向宽松方向，且保证 mid 仍不过 | 无 | mid 拉不开 → 回 S5 换 `mid_drop`（限 1 次），再不行 `dropped`；ref 不过 → S8 |
| S6C | 规格 + 阈值 + 实测 | `tests/test_outputs.py`、`instruction.md`、`baselines/exploit_*.py` | Opus | 缺文件 `dropped` |
| S7 | 完整工作树 | `gates/attempt-n.json` | 无（Docker） | 阻塞级 → S8；只有“阈值未写明”时由程序追加判据表 |
| S8 | 门日志 + 依赖图 + 相关文件 | `repair/round-n/`（补丁 + 审查结论） | Opus 修复；审查为机械 diff | 审查不过算用掉 1 轮；2 轮后 `dropped: repair_limit` |
| PKG | 通过 S7 的题 + 种子 0 产物 | `pkg/` + `PKG_check.json` | 无 | `parked: package_invalid` |
| S9 | `pkg/` | `calib/*.json` 作业配置 + 启动命令；`runs.json` | 不调用（只调度） | 见 §5 |
| S10 | 标签为 hard/unstable 的题 | `release/manifest.jsonl` | 无 | — |

**S7 机械门**：determinism（同种子两次哈希一致）；oracle（参考解在每个种子上都通过）；oracle_margin（参考解至少留出一半余量）；nop（空交付不过）；naive（每个种子至少 2 项不过）；mid（1–2 项不过，且全部判据都被评估）；exploit（作弊尝试全不过）；hidden_shift（只对 PROG）；thresholds_stated（每个阈值都在题面中出现）；leak；isolation（`environment/` 下没有 tests/truth/generator）。

**泄漏门（`leakcheck.py`）**：

- L1a：依赖图中 derive/internal/hidden_truth 节点的值或 `secret_names` 出现在题面或公开文本里。
- L1b：题面用规则保护公开文件（“不许读 / opaque / 内部状态 / 黑盒”）。agent 是 root，这种保护无效，判阻塞。
- L1c：公开文件与真值文件内容相同。如果该文件被题面声明为禁读，判阻塞，否则只记警告；文件名带 truth/answer 等字样也判阻塞。
- L2：真值数值出现在公开文件中，包括包含关系，以及置换、取负、缩放后相等。同名回显或题面写明的输入降为警告。
- L3：在 Docker 里导入公开 `.py`，按每个题号实例化，遍历私有属性和闭包，看能否拿到公开文件里没有的真值。
- L4a：把真值表和公开表按键连接，检查每个真值列与公开列及其两两乘积/比值的秩相关（≥0.995 且为辅助量组合时判阻塞）。
- L4b：廉价 agent 反解攻击，模型角色待定，当前跳过。

## 3. 怎么跑

```bash
cd gen
uv sync
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t tbsgen-base:py311-sci-v2 -f docker/Dockerfile.base docker/     # 沙箱镜像（已建好）
uv run pytest -q                                                    # 49 个测试（集成测试需要 Docker）
uv run python -m tbsgen.dryrun --out runs/dryrun-X                  # fake 模型端到端干跑
uv run python -m tbsgen.backtest --out eval/leak_backtest_X         # 泄漏门回测（--set holdout81 跑留出集）

# 真实运行（必须先在 config.toml 里设预算；不设则真实模型调用全部拒绝）
uv run tbsgen ingest --run runs/R --candidates <tbsretr 输出.json> [--top N] --config config.toml
uv run tbsgen run    --run runs/R --config config.toml [--max-steps N] [--only ID]
uv run tbsgen status --run runs/R ; uv run tbsgen report --run runs/R ; uv run tbsgen ledger --run runs/R
uv run tbsgen resume --run runs/R --config config.toml              # 恢复因预算/熔断暂停的条目
uv run tbsgen breaker --run runs/R --model claude-opus-5-5 --reset  # 人工解除熔断
uv run tbsgen calib-ingest --run runs/R --item ID [--job-dir DIR]   # 回填校准结果（默认取该次提交的作业目录；作业未结束则拒绝）
uv run tbsgen harbor-local --run runs/R [--item ID]                  # 本机 Harbor oracle/nop（换本地基础镜像）
uv run tbsgen reset  --run runs/R --item ID --stage S6B [--regen]    # 管线修复后让条目从某阶段重来（记入 history）
uv run tbsgen ping --run state/ping --calls 1 --budget-usd 1        # Opus 连通性（真实调用）
```

`config.toml` 示例（只需写要覆盖的项）：

```toml
[budget]
daily_usd_total = 30.0
daily_calls_per_model = { "claude-opus-5-5" = 40 }
[calib]
allow_submit = false
```

凭据：出题阶段的 Opus 走本机 Anthropic 兼容网关（`models.opus.url`，可用 `TBS_OPUS_URL` 覆盖），与旧 `llm.py` 一样，本进程不持有 key。如果改走需要 key 的端点，设置 `models.opus.key_env`，再用 `uv run tbsgen ...` 注入；key 只放在请求头里，不写日志或文件，错误信息会先脱敏。

## 4. 配置项与待定项默认值

| 配置 | 默认 | 说明 |
|---|---|---|
| `budget.daily_usd_total` | **None** | 未设置时拒绝一切真实模型调用（预算暂不定） |
| `budget.daily_usd_per_model / per_stage / daily_calls_per_model` | 空 | 可选的细分上限 |
| `budget.max_calls_per_item` | 24 | 单条目出题调用（S5–S8，所有尝试）总上限；S9 不计入 |
| `budget.run_usd_total / run_calls_per_model` | None / 空 | 整个运行目录（跨天）的总上限；S9 每次按 `calib.est_usd_per_run` 预留 |
| `budget.breaker_open_on` | quota, auth | 出现即熔断；另有连续 4 次错误也熔断 |
| `budget.est_usd_per_call` | 1.5 | 调用前按此金额预占，用于判断是否会超预算 |
| `models.prices` | Opus $5/$25 每百万 token | **占位**，同 DESIGN §3.1，需用账单替换 |
| `models.cheap` / `roles.screen/probe/review/invert` | None / rules | 廉价模型角色**待定**；S2/S4/S8 审查先用规则，L4b 跳过 |
| `spec.allowed_forms` | ANS, PROG | DOMF 需要逐格式读写库；INTERACT 在隔离方案落地前禁止 |
| `spec.prog_quota_frac` | 0.5 | PROG 占比上限（**待定**；当前只记录，未在调度中强制） |
| `thresholds.frac / min_ref_margin / loose_margin` | 0.5 / 0.10 / 0.50 | 阈值位置（**待定**）；非区分指标若 naive 明显更差，阈值收到参考与 naive 的对数中点（不紧于参考×(1±margin)） |
| `thresholds.sentinel_abs` | 1e5 | "≤"指标上参考值达到此量级视为缺输出（判分脚本的哨兵值） |
| `gen.s6a_max_tokens / s6c_max_tokens / continue_calls` | 64000 / 48000 / 1 | 多文件输出上限；被截断时最多 1 次续写，只要缺的文件 |
| `gen.mid_rework_max` | 1 | mid 不可区分时先由 S6M 只重写 `baselines/mid.py`，再不行才回 S5 |
| `gates.max_repair_rounds` | 2 | 沿用旧上限 |
| `calib.quick_pass_minutes` | None | GLM 首次通过且用时短时直接记 easy（DESIGN 建议 30 min，**待定**，默认关） |
| `calib.keep_labels` | hard, unstable | decision 2026-11-05 |
| `calib.variant_depth` | 0 | S9b 更难变体（**待定**，MVP 关闭） |
| `calib.infra_retries` | 2 | API/沙箱异常不计入校准次数 |
| `calib.allow_submit` | **False** | True 时：账本预留 → tmux 后台运行启动脚本；预留超总额则拒绝 |
| `calib.est_usd_per_run` | 20 | 每次校准作业的预留额，`calib-ingest` 用 Harbor `cost_usd` 冲销 |
| `calib.teacher_available` | True | False：GLM 首次失败直接记 `parked:teacher_unavailable` |
| `calib.prepared_tasks_dir` | None | 远端沙箱已注册模板的任务副本目录；设置后缺模板的条目不提交 |
| `fulltext.mode` | 旧文献缓存（只读）→ OA PDF 链接 | 浏览器通道（旧 `browser_fetch.py`）**未接入**，取不到全文的条目进 `S3_queue.json` 并暂存 |
| `package.dockerfile_mode` | selfcontained | Dockerfile 里用 pip 安装固定版本；`local_base` 只用于本机检查 |

## 5. S9 校准规则（decision 2026-11-05）

GLM-5.3 + Claude Code 2.1.280 先跑第 1 次：

- 失败 → Opus 教师。教师通过记 **hard**；教师失败再跑 1 次，仍失败则暂存（`parked:teacher_failed`）。
- 通过 → GLM 第 2 次。失败记 **unstable**，通过记 **easy**。

hard 和 unstable 保留。reward=1 一律算通过，即使之后报了 API 错误。没有 reward 的 API/沙箱异常算 `infra`，不计入次数，连续超过 2 次则暂存。`allow_submit=false` 时只写出作业配置和启动命令，由人执行；`true` 时先在账本预留、再后台启动。结果用 `calib-ingest` 回填。

远端沙箱不执行 Dockerfile：提交前要为每题建好模板（打包出的 Dockerfile 与旧批次相同，可用"共享基础镜像 + 数据层"的薄镜像方式），并把准备好的任务副本目录配到 `calib.prepared_tasks_dir`。

## 6. 已验证 / 未验证

**已验证（实际跑过）**

- `uv run pytest -q`：36 passed（单测 28 + 集成 8，集成测试在本机 Docker 里执行任务代码）。
- fake 模型端到端干跑（`runs/dryrun-final/`）：4 条候选（1 条人造 + 3 条来自检索层录制结果、全文取自旧缓存）全部走完 S1→S9。人造条目再用模拟的校准结果走到 S10（GLM 失败 + 教师通过 → hard）。机械门全过：nop、mid（2 项不过）、naive、exploit 都不过，参考解通过。
- 集成测试覆盖的情况：断点续跑（中途停止，换新进程继续，已完成阶段不重复调用模型）；预算上限触发暂停，`resume` 后继续；quota 错误触发熔断并暂停；参考解崩溃 → S8 修复 → 恢复；注入泄漏 → L2 阻塞 → 修复后通过；修不好 → 2 轮后丢弃；筛选丢弃；取不到全文 → 暂存。
- 打包出的 Harbor 任务：Harbor 0.23.0 能解析 `task.toml`。用本机 Docker 实际跑了 Harbor：oracle reward=1，nop reward=0（`state/harbor-local/jobs/`）。注意：本机检查时把 Dockerfile 换成了本地基础镜像，因为通过代理从 PyPI 下载超时。
- 泄漏门回测：见 `eval/LEAK_BACKTEST.md`（召回 7/7；样本内误报 0/231；留出误报 1/81）。
- Opus 连通性：2 次真实调用，非流式和流式（SSE）都成功，均记入账本，共 $0.0013（占位价）。

**未验证**

- 任何真实论文的 S5/S6A/S6C/S8：提示词没有在真实 Opus 上试过，Opus 产出能否通过解析和各道门都未知。
- `selfcontained` Dockerfile 在阿里云沙箱/自建服务器 上的构建；Harbor 在阿里云沙箱环境下加载 tbsgen 任务。
- S9 实际提交与 `calib-ingest` 对真实作业目录的解析（只用旧作业的 `result.json` 格式做过单测）。
- OA PDF 下载分支（单测只走了缓存分支；网络分支的代码与旧管线的标题校验逻辑相同，但没有实测）。
- L4b 反解攻击、廉价模型各角色、S9b 变体、DOMF/CODE/MULTI 形态。
- 价格是占位值，美元数字不是账单。

## 6b. e2e-iter1（2026-11-06）新增门与检查

- `truth_isolation`（PROG，阻断）：判分时把 `solution_cli.py` 换成探针，探针尝试读 `/tests` 下除隐藏输入外的所有文件。要求探针经 `_metrics.run_solution` 执行，且什么都读不到。本地判分步骤与 Harbor 一致，以 root 运行在容器内副本上（含 `/app/data`）。
- `cli_contract`（PROG，阻断）：指令必须原样包含 `solution_cli.py <input_dir> <output_dir>`。
- S6B：
  - mid "只差 1–2 项"按判据计数，同名的 public/hidden 指标算一个；
  - 参考在某指标上明显差于 naive 时记 `ref_worse_than_naive`，送去修复。

真实论文结果与失败模式见 `../docs/E2E_LOG.md`。

## 7. 真实 1 篇端到端烟测方案（已执行，见 docs/E2E_LOG.md）

- 选 1 篇已有全文缓存、非黑盒类的旧论文（例如 `10.1007/s10182-025-00550-5`）。只跑 S1–PKG，S9 只生成配置，不提交。
- 预计 Opus 调用 3–5 次：S5 1 次、S6A 1 次、S6C 1 次，S8 0–2 次。按占位价估算约 $2–3（S5 约 4 万输入/1 万输出，S6A 约 3 万/2.5 万，S6C 约 4.5 万/1.5 万，S8 每次约 3.5 万/0.6 万）。
- 配置：`daily_usd_total=5`、`daily_calls_per_model={claude-opus-5-5=6}`，超出即停。
- 可选：再跑 1 次 GLM（CC）校准，按旧数据约 $11/次，需要单独批准；教师作业还要先确认教师模型端点的额度。
