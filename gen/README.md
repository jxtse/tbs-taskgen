# tbsgen：TB-Science 出题管线 v2

出题管线 MVP：从检索结果开始，经过全文获取、可行性检查、规格、生成、机械门、修复、校准调度，最后打包成 Harbor 任务。**已在少量真实论文上端到端跑通（见 §6），还没有量产。**

## 1. 架构

```
src/tbsgen/
  config.py        全部配置与默认值（待定项都在这里）；config.toml 可覆盖
  budget.py        sqlite 预算账本：按 模型×阶段×条目 记 token/美元/次数；日预算、调用上限、单条目上限；熔断
  models.py        模型调用层：OpusClient（Anthropic 兼容端点）/ FakeClient（脚本化）/ RecordingClient（录制）
  state.py         每条目一个状态目录 + state.json 状态机；events.jsonl
  pipeline.py      编排：逐条推进、断点续跑、预算/熔断暂停、resume
  intake.py        S1 读检索层输出；S2 规则筛选（rules-v1/v2）；S3 全文（调 fulltext.py）；S4 可行性（规则部分 + 官方题去重）
  blocklist.py     S2 前置黑名单：按 DOI 前缀（可加发表年份）直接丢弃拿不到全文的论文，配置 screen.blocklist
  fulltext.py      S3 下载级联（cache/oa/elsevier/wiley/inst/browser）、校验、写入缓存、全文队列
  dlledger.py      S3 下载账本：每日/每小时上限、同站点间隔、冷却、熔断
  browser_ego.py   ego-browser 通道（运行 browser_fetch.mjs 脚本模板）
  browser_agent.py 浏览器批次：出口门控、按出口窗口定批次规模、脚本优先、Hermes CLI agent 兜底（提示模板 browser_agent_prompt.md，agent 唯一的浏览器工具调用受审动作脚本 browser_agent_action.mjs）
  remote.py        远程模式：HTTP 通道经 ssh 在另一台主机上运行，账本以本机为准
  screen_jev.py    S2 的 Jev 模式：问题原文、阈值、组合规则、HTTP 客户端（重试/记账/熔断）、并发预取
  prompts.py       S5/S6A/S6C/S8 提示词（版本号写进 provenance）
  spec.py          S5 规格校验（依赖图规则）；S6B 按实测定阈值
  runner.py        Docker 沙箱里跑 生成器/参考解/基线/measure/verifier
  sandbox.py       docker run（--network none、2 CPU、4 GB、pids 512）
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
  templates_metrics.py       verifier 公共部分，复制进每道题的 tests/_metrics.py
docker/Dockerfile.base       沙箱镜像 tbsgen-base:py311-sci-v2（python 3.11 + 固定版本的科学计算库）
tests/                       单测 + 集成测试；tests/fixtures/toy_task 为人造最小题
eval/                        泄漏门回测结果
```

运行目录（`--run DIR`）：

```
ledger.sqlite  events.jsonl  reports/daily-<日期>.md  release/manifest.jsonl
items/<item_id>/
  state.json                 当前阶段、状态、历史、各阶段尝试次数
  S1_candidate.json S2_screen.json S2_jev.json S3_fulltext.json fulltext.txt S4_probe.json
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
| S2 | 题录 + 摘要 | `S2_screen.json`（`score`、`keep`、`form_hint`、`method`、`reason`、`deferred`；Jev 模式另有各问题概率 `probabilities`、`decision`（keep/drop/uncertain）、`uncertain`、`jev_model`、`input_tokens`）；`S2_jev.json` 为按请求键缓存的 Jev 结果 | `roles.screen`：`rules`（默认）或 `jev`（TypeSafe Jev，失败回退规则） | `dropped: low_screen_score`（规则）/ `jev_*`（Jev）；无摘要时 `deferred`，S4 用全文开头重判 |
| S3 | DOI、OA 链接 | `fulltext.txt` + `S3_fulltext.json` | 无 | `parked: fulltext_unavailable`，写 `S3_queue.json` 并加入全文队列；`tbsgen fetch` 取到后写入 `fulltext.txt` 并推进到 S4 |
| S4 | 全文 + 筛选记录 | `S4_probe.json`（计算核心线索、与官方 70 题去重、形态禁令） | 规则（沙箱探针待定） | `dropped: no_compute_core / tbs_duplicate / interact_form_banned_until_isolation` |
| S5 | 全文摘录 + 探针卡 | `_spec/spec.json`、`depgraph.json`、`coverage.json`、`provenance.json` | Opus | 校验不过重试 1 次，仍不过 `dropped: spec_invalid` |
| S6A | 规格 | `task/generator/`、`solution/`、`baselines/{mid,naive}.py`、`tests/measure.py` | Opus | 缺文件 `dropped` |
| S6A_run | 工作树 | `S6A_measure.json`：两个种子下 ref/mid/naive 的实测指标 | 无（Docker） | 生成器或参考解失败 → S8 |
| S6B | 实测指标 | `thresholds.json`：分离型指标取 ref 与 mid 的对数中点，非分离型取 ref 放宽 50%，取整一律向宽松方向，且保证 mid 仍不过 | 无 | mid 拉不开 → 先由 S6M 只重写 mid（限 1 次），再回 S5 换 `mid_drop`（限 1 次），再不行 `dropped`；ref 不过 → S8 |
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
docker build -t tbsgen-base:py311-sci-v2 -f docker/Dockerfile.base docker/   # 沙箱镜像；可加 --build-arg PIP_INDEX_URL=<镜像源>
uv run pytest -q                                                    # 69 个测试：53 单测 + 16 集成（集成测试需要 Docker 和沙箱镜像，缺了就跳过）
uv run python -m tbsgen.dryrun --out runs/dryrun-X                  # fake 模型端到端干跑
uv run python -m tbsgen.backtest --out eval/leak_backtest_X         # 泄漏门回测（需要旧批次题目数据，不在仓库里；--set holdout81 跑留出集）

# 真实运行（必须先在 config.toml 里设预算；不设则真实模型调用全部拒绝）
uv run tbsgen ingest --run runs/R --candidates <tbsretr 输出.json> [--top N] --config config.toml
uv run tbsgen run    --run runs/R --config config.toml [--max-steps N] [--only ID]
uv run tbsgen status --run runs/R ; uv run tbsgen report --run runs/R ; uv run tbsgen ledger --run runs/R
uv run tbsgen resume --run runs/R --config config.toml              # 恢复因预算/熔断暂停的条目
uv run tbsgen breaker --run runs/R --model claude-opus-5-5 --reset  # 人工解除熔断
uv run tbsgen fetch --run runs/R --config config.toml [--max N] [--channels oa,browser] [--no-browser] [--dry-run]
                                                                    # 清全文队列（含 ego-browser 通道），取到全文后交给等待的条目并推进到 S4
uv run tbsgen fetch --run runs/R --status                          # 今日下载用量、冷却、熔断
uv run tbsgen fetch --run runs/R --finish-browser                  # 关闭本工具的 ego TaskSpace
uv run tbsgen fetch --run runs/R --browser --wait-for-egress         # 等出口进入 allow_cidrs 后跑一个浏览器批次
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

凭据：出题阶段的 Opus 走 Anthropic 兼容端点（`models.opus.url` 或环境变量 `TBS_OPUS_URL`，没有默认值；未设置时模型阶段以 `model:config` 暂停）。不设 `key_env` 时本进程不发送 key（适用于持有凭据的本机网关）。如果改走需要 key 的端点，设置 `models.opus.key_env`，再用 `uv run tbsgen ...` 注入；key 只放在请求头里，不写日志或文件，错误信息会先脱敏。

## 4. 配置项与待定项默认值

| 配置 | 默认 | 说明 |
|---|---|---|
| `budget.daily_usd_total` | **None** | 未设置时拒绝一切真实模型调用（预算暂不定） |
| `budget.daily_usd_per_model / per_stage / daily_calls_per_model` | 空 | 可选的细分上限 |
| `budget.max_calls_per_item` | 24 | 单条目出题调用（S5–S8，所有尝试）总上限；S9 不计入 |
| `budget.run_usd_total / run_calls_per_model` | None / 空 | 整个运行目录（跨天）的总上限；S9 每次按 `calib.est_usd_per_run` 预留 |
| `budget.breaker_open_on` | quota, auth | 出现即熔断；另有连续 4 次错误也熔断 |
| `budget.est_usd_per_call` | 1.5 | 调用前按此金额预占，用于判断是否会超预算 |
| `models.prices` | Opus $5/$25 每百万 token | **占位**，需按实际价格替换 |
| `models.cheap` / `roles.screen/probe/review/invert` | None / rules | `roles.screen = "jev"` 启用 Jev 筛选；其余廉价模型角色**待定**，S4/S8 审查用规则，L4b 跳过 |
| `screen.rules_version / min_score` | v2 / 1 | 规则筛选：v2 每个词干只计一次、“survey”只在文献综述语境扣分；v1（`min_score_v1`=2）保留作对照 |
| `screen.jev.model` | jev-1.14.0 | 固定版本；响应里的 `model` 不一致时报错并熔断（`jev-latest` 不检查） |
| `screen.jev.key_env` | TYPESAFE_API_KEY | 存放密钥的环境变量名 |
| `screen.jev.uncertain` | keep | 不确定时放行并标记 `uncertain=true`；`drop` 则丢弃 |
| `screen.jev.on_unavailable` | rules | 无密钥/熔断/API 不可用：回退规则（`method=rules-v2-fallback`）；`park` 则暂停条目 |
| `screen.jev.concurrency / timeout / max_retries` | 4 / 30 s / 4 | 并发预取数；单次超时；429/529/5xx 指数退避（遵守 `retry-after`）的重试上限，其他 4xx 不重试 |
| `screen.jev.thresholds` | 空 | 覆盖 `screen_jev.THRESHOLDS` |
| `models.prices["jev-1.14.0"]` | in $0.042 / 百万 | 只按输入计费 |
| `spec.allowed_forms` | ANS, PROG | DOMF 需要逐格式读写库；INTERACT 在隔离方案落地前禁止 |
| `spec.prog_quota_frac` | 0.5 | PROG 占比上限（**待定**；当前只记录，未在调度中强制） |
| `thresholds.frac / min_ref_margin / loose_margin` | 0.5 / 0.10 / 0.50 | 阈值位置（**待定**）；非区分指标若 naive 明显更差，阈值收到参考与 naive 的对数中点（不紧于参考×(1±margin)） |
| `thresholds.sentinel_abs` | 1e5 | "≤"指标上参考值达到此量级视为缺输出（判分脚本的哨兵值） |
| `gen.s6a_max_tokens / s6c_max_tokens / continue_calls` | 64000 / 48000 / 1 | 多文件输出上限；被截断时最多 1 次续写，只要缺的文件 |
| `gen.mid_rework_max` | 1 | mid 不可区分时先由 S6M 只重写 `baselines/mid.py`，再不行才回 S5 |
| `gates.max_repair_rounds` | 2 | S8 修复轮数上限 |
| `calib.quick_pass_minutes` | None | GLM 首次通过且用时短时直接记 easy（**待定**，默认关） |
| `calib.keep_labels` | hard, unstable | 保留的难度标签 |
| `calib.variant_depth` | 0 | S9b 更难变体（**待定**，MVP 关闭） |
| `calib.infra_retries` | 2 | API/沙箱异常不计入校准次数 |
| `calib.allow_submit` | **False** | True 时：账本预留 → tmux 后台运行启动脚本；预留超总额则拒绝；没配 `harbor_root` 和启动脚本时拒绝提交 |
| `calib.harbor_root / student_launcher / teacher_launcher` | None | 启动脚本目录与脚本名（`./<launcher> run <作业配置>`）；未设置时只打印 `harbor run --config ...` |
| `calib.student_agent / teacher_agent / environment` | None | Harbor 自定义 agent/环境的 `module:Class`；未设置用内置 `claude-code` agent 和 `docker` 环境 |
| `calib.est_usd_per_run` | 20 | 每次校准作业的预留额，`calib-ingest` 用 Harbor `cost_usd` 冲销 |
| `calib.teacher_available` | True | False：GLM 首次失败直接记 `parked:teacher_unavailable` |
| `calib.prepared_tasks_dir` | None | 远端沙箱已注册模板的任务副本目录；设置后缺模板的条目不提交 |
| `fulltext.cascade` | cache → oa → elsevier → wiley → inst → browser | 下载级联顺序；每个通道有自己的 `enabled`，默认只开 cache 和 oa（见根 README「Full text (S3)」） |
| `fulltext.write_cache_dir / ledger_path / queue_path` | `<run>/fulltext_cache`、`<run>/fulltext_ledger.sqlite`、`<run>/fulltext_queue.jsonl` | 写入缓存（含来源 json）、下载账本、浏览器队列 |
| `fulltext.limits / global_limits / host_caps` | `dlledger.py` 中的默认值 | 每通道每天/每小时成功与尝试上限、同站点最小间隔、全局上限；429 冷却 6 小时，验证页冷却 24 小时 |
| `fulltext.egress_check` | 关闭 | 开启后，出口 IP 不在 `allow_cidrs` 内则拒绝所有网络通道 |
| `fulltext.use_env_proxy` | False | 下载不读 `HTTP(S)_PROXY` |
| `fulltext.browser.*` | 关闭；并发 2；验证页等 40 秒 | ego-browser 通道；S3 只入队，`tbsgen fetch` 执行；同一出版商间隔 60–120 秒，不点击任何验证 |
| `fulltext.browser.batch` / `.agent` | 窗口 600 s、余量 60 s；agent 关闭；每篇 ≤25 次工具调用、≤12 次页面加载，同出版商两次加载间隔 ≥2.5 s | `tbsgen fetch --browser [--wait-for-egress]`：出口门控的限时批次；agent 为 Hermes CLI，每篇独立 TaskSpace，唯一工具 `tbsgen-browser`（look/open/click/click_xy/hover/scroll/select/type/screenshot/tabs/switch/back/wait_for/save_pdf/sso_login/handoff_status/finish）；`type` 只允许 `agent.institution_names` 里的机构名；agent 的 stderr 尾部（凭据样式字符串打码）写入批次目录 `agent-stderr.log` 和 `agent_end` 日志 |
| `fulltext.browser.handoff` | 关闭；等 120 s | 验证页（Cloudflare/Turnstile/验证码/滑块）与非机构登录页交给本机用户处理（macOS 通知，可选本地 `bark_cmd`），agent 从不自己过验证；每篇、每出版商每批最多 1 次，超时则冷却该出版商 |
| `fulltext.browser.sso` / `.sso_login` | 域名为空；登录关闭；每批 ≤2 次提交 | `sso.domains`：机构/WAYF 域，加入每篇的可访问域。`sso_login`：只在 `sso_login.domains` 内的 HTTPS 机构登录页上，**只用浏览器自动填充**并点登录；代码与 agent 都不读取、输入或记录凭据（只看输入框值长度是否 >0）；无自动填充、验证码/短信/MFA、登录失败一律交给用户，失败不重试；账本只记时间、域名和结果 |
| `fulltext.remote` | 关闭 | HTTP 通道（oa/elsevier/wiley/inst）经 ssh 在 `host` 上运行；远端须开启 egress_check |
| `package.dockerfile_mode` | selfcontained | Dockerfile 里用 pip 安装固定版本；`local_base` 只用于本机检查 |

## 5. S9 校准规则

GLM-5.3 + Claude Code 2.1.280 先跑第 1 次：

- 失败 → Opus 教师。教师通过记 **hard**；教师失败再跑 1 次，仍失败则暂存（`parked:teacher_failed`）。
- 通过 → GLM 第 2 次。失败记 **unstable**，通过记 **easy**。

hard 和 unstable 保留。reward=1 一律算通过，即使之后报了 API 错误。没有 reward 的 API/沙箱异常算 `infra`，不计入次数，连续超过 2 次则暂存。`allow_submit=false` 时只写出作业配置和启动命令，由人执行；`true` 时先在账本预留、再后台启动。结果用 `calib-ingest` 回填。

远端沙箱不执行 Dockerfile：提交前要为每题建好模板（可用"共享基础镜像 + 数据层"的薄镜像方式），并把准备好的任务副本目录配到 `calib.prepared_tasks_dir`。

## 6. 已验证 / 未验证

**已验证（实际跑过）**

- 测试：gen 共 255 个（226 个离线单测，含 S3 全文通道与浏览器批次的 149 个；16 个在 Docker 里执行任务代码的集成测试；13 个用本机真实 ego-browser 打本地测试页的动作脚本 live 测试，CI 上无 ego-browser 时跳过），每次推送由 GitHub Actions CI 运行。
- 集成测试覆盖的情况：断点续跑（中途停止，换新进程继续，已完成阶段不重复调用模型）；预算上限触发暂停，`resume` 后继续；quota 错误触发熔断并暂停；参考解崩溃 → S8 修复 → 恢复；注入泄漏 → L2 阻塞 → 修复后通过；修不好 → 2 轮后丢弃；S6A 截断续写；mid 不可区分时 S6M 只重写 mid；真值隔离探针；筛选丢弃；取不到全文 → 暂存。
- fake 模型端到端干跑：人造条目走完 S1→S9，再用模拟的校准结果走到 S10。
- 打包出的 Harbor 任务：用本机 Docker 实际跑 Harbor，oracle reward=1，nop reward=0（`tbsgen harbor-local`，Dockerfile 换成本地基础镜像）。
- 泄漏门回测：见 `eval/LEAK_BACKTEST.md`（召回 7/7；样本内误报 0/231；留出误报 1/81）。
- 真实论文端到端（6 篇，真实 Opus）：3 篇走完 S1→PKG 并通过 oracle=1 / nop=0；2 篇因 mid 与参考解拉不开被丢弃；1 篇仍在修复。S9 已真实提交学生模型（GLM）校准，每题 1 次，不足以给出难度标签。

**未验证**

- 校准的教师（Opus agent）分支。
- L4b 反解攻击、廉价模型各角色、S9b 变体、DOMF/CODE/MULTI 形态。
- S3 出版商通道（TDM、校园 IP、浏览器）只有用伪造响应的单测；真实下载依赖使用者自己的订阅与出口网络。浏览器 agent 的探索动作、handoff 与机构 SSO 自动填充登录只在本地测试页上验证过，尚未在真实出版商/机构站点端到端跑通。
- 专家时长是教师模型的自报值；价格是占位值，美元数字不是账单。

## 6b. 真实论文端到端测试后新增的门与检查

- `truth_isolation`（PROG，阻断）：判分时把 `solution_cli.py` 换成探针，探针尝试读 `/tests` 下除隐藏输入外的所有文件。要求探针经 `_metrics.run_solution` 执行，且什么都读不到。本地判分步骤与 Harbor 一致，以 root 运行在容器内副本上（含 `/app/data`）。
- `cli_contract`（PROG，阻断）：指令必须原样包含 `solution_cli.py <input_dir> <output_dir>`。
- S6B：
  - mid "只差 1–2 项"按判据计数，同名的 public/hidden 指标算一个；
  - 参考在某指标上明显差于 naive 时记 `ref_worse_than_naive`，送去修复。
