# tbsgen：TB-Science 出题管线 v2

出题管线 MVP：从检索结果开始，经过全文获取、可行性检查、规格、生成、机械门、修复、校准调度，最后打包成 Harbor 任务。**已在少量真实论文上端到端跑通（见 §6），还没有量产。**

## 0. 代码驱动出题（`code.grounded`，2026-11-12 起）

出题依赖**论文原文 + 作者配套代码**：没有作者代码的论文直接淘汰，不留例外通道。许可证只记录、不过滤
（任务包与训练数据都不进公开仓库，作者代码只在内部运行目录）。`code.grounded = false`（默认）时管线与以前完全相同。

```
S1 → S2a 代码信号预筛（不花全文额度） → S2 摘要筛 → S3 全文 → S3b 作者代码硬确认 → S3c 获取代码（构建机）
   → S4 可行性 → S4c 可运行性硬门（构建机 Claude Code agent） → S5G 出题 agent（构建机 Claude Code）
   → S6A_run → S6B → S6C → S7 → S8 → PKG → S9 → S9R → S10（后段不变）
```

| 阶段 | 产物 | 淘汰原因 |
|---|---|---|
| S2a | `S2a_code_signal.json`：元数据代码链接、检索层 `code_signals`（OpenAlex 全文 / PwC / GitHub README）、PwC 存档（`code.pwc_path`）；都没有时查一次 OpenAlex 全文标记 | `no_code_signal` |
| S3b | `S3b_code.json`：全文与元数据里的每个代码链接（上下文、Code Availability 段、参考文献区、标题词、README 是否引用本文、PwC official）→ 规则打分；全文和摘要都没写出的链接（只来自元数据、引用本文的 README 或 PwC mentioned）直接判为非作者代码（`rules:not_in_paper`，PwC official 除外）；模糊链接先查 README，仍模糊时 `roles.code_owner="jev"` 用 Jev 判定（不用 Opus）；作者链接在 GitHub / GitLab / Bitbucket / Zenodo / CRAN / PyPI / Bioconductor / SourceForge 上实际解析，记录 license、语言、大小 | `code_on_request`、`third_party_code_only`、`code_repo_unavailable`（404 / 私有 / 空仓库）、`data_only_no_code`、`code_host_unsupported`（Code Ocean、Figshare 等）、`no_code_link` |

S3b 规则在 2026-11-11 摸底的 224 篇（人工读上下文标注，93 篇有作者代码）上：规则直接判 own 的 38 篇全对；
“own + 模糊”合起来作为候选时论文级召回 0.95、精度 0.98，另外 50 篇作者代码只能落在“模糊”里，所以生产配置应设
`roles.code_owner = "jev"`（规则单独使用时召回只有约 0.41）；
模糊链接的最终归属：配了 Jev 时由 Jev 判定，Jev 不可用时该条目暂停（`code:code_owner_undecided`）；没配 Jev 时模糊链接不算作者代码。
规则直接判 own 需要论文自身的归属证据（链接旁的“our code is available…”类表述、Code Availability 段，Code Availability
段截止到下一个章节标题）或 PwC `is_official`；README 引用本文、PwC mentioned、元数据、标题词只让链接成为候选。仓库主机暂时无响应（网络、限流、5xx）时也只暂停（`code:code_host_unavailable`），
S2a 查不到 OpenAlex 时暂停（`code:openalex_unavailable`），都用 `tbsgen resume --reason code:` 恢复。私有仓库不算公开代码。
`tbsgen fetch` 投递全文时按条目自己 run 当时写进 `S3_queue.json` 的 `next_stage` 推进（旧条目默认 S4）。

| S3c | `S3c_code.json`：在构建机（`code.host`，ssh）上 `git clone --depth 1`（不拉 LFS、不拉子模块）或下载 Zenodo / CRAN / PyPI 源码包；固定 commit SHA（压缩包为 sha256）；记录大小（不含 .git）、语言、依赖文件、LICENSE 首行。代码只放在构建机的 `code.remote_root/<run_tag>/<item>/repo`，本机不执行、不保存第三方代码 | `code_too_large`（> `code.max_repo_mb`，默认 500）、`code_fetch_failed`、`code_host_unsupported`；网络失败暂停重试 |
| S4c | 构建机上的 Harbor 作业（`code.agent_launcher`，默认沿用教师路由：Opus 经 Substrate 中继）：agent 在容器里装环境（安装期只放行包索引和代码托管站 `code.install_hosts`，外加模型端点）、通过回执包装 `/work/tbsgen-run.sh` 跑一次作者的示例 / 测试 / 主入口（记录命令、退出码、日志；“可运行”必须有一条退出码 0、命令等于 `entry` 的回执），写 `runnability.json`、`HOWTO.md`（给下一步的“如何运行作者代码”）、`setup.sh`。MATLAB 先试 Octave | `code_not_runnable`、`code_needs_gpu`、`code_needs_commercial_license`（含 Mathematica）、`code_matlab_octave_incompatible`、`code_runnability_agent_failed` |
| S5G | 构建机上的 Claude Code 出题 agent：输入论文全文、作者仓库、S4c 的 HOWTO、harness 契约原文（S5_SPEC、S6A_GEN、METRICS_API、`_metrics.py` 模板）；输出与 S5 + S6A **完全相同**的产物（`_spec/spec.json`、`task/{instruction.md, generator, solution, tests/measure.py, baselines/mid.py, baselines/naive.py}`）加 `_spec/author_code.json`（包名、主要函数、参考解用了作者哪些代码、数据来源）。agent 可在容器里反复运行生成器、参考解、基线直到拉开差距。之后 S6A_run 起全部沿用旧阶段；S6B 要求重新出规格 / 重写参考解时回到 S5G（每篇最多 `code.authoring_max_jobs` 次） | `s5g_missing_files`、`spec_invalid`、`s5g_jobs_exhausted:*` |

防泄漏：任务镜像（PKG 生成的 Dockerfile）只有固定科学计算栈，agent 写的 `task/environment/` 一律丢弃；作者代码只能
被移植进参考解，或在构建期生成真值（存进 `generator/assets/`，附 PROVENANCE）。S7 的 `readymade` 门加入作者仓库名、
包名、主要函数名（`_spec/author_code.json`）；新 S7 门 `author_leak`（阻断）检查题面和公开数据文件是否点名这些名字；
PKG 打包时同样检查。S9R 审核清单新增 `i_author_code`：任务是否只是“重跑作者仓库”、是否泄漏作者实现。

预算：每个 agent 作业在账本里按 `code.runnability_est_usd` / `code.authoring_est_usd` 预留（阶段 S4c / S5G，教师模型），
结束后按 Claude Code 报告的 `cost_usd`（没有时按 token × `models.prices`）结算；单篇时长上限
`code.authoring_timeout_s`（默认 7200 s），估算费用上限 `code.authoring_max_usd`（默认 $20，传给 Claude Code
`--max-budget-usd`）。`code.authoring = "single_call"` 回退到旧的单次调用 S5 + S6A。

作者代码注入：S3c 把代码放在构建机 `<remote_root>/<run_tag 或 run 唯一 id>/<item>/pins/<pin>/repo`（每个 pin 不可变、原子发布、
符号链接全部删除）。agent 作业的 `environment/work/.tbsgen-remote-repo` 写明 `{repo_dir, pin}`；作业启动器须在构建机上调用
`gen/scripts/inject-author-repo.sh <task> <remote_root>`（拒绝链接路径、根目录外路径和 pin 不符），作业 Dockerfile 在
`/work/repo.PIN` 与 pin 不一致时拒绝构建，因此未注入时不会产生模型开销。agent 交付物按 trial 根目录逐级拒绝符号链接。
单篇预算按作业累计（`<stage>/spent.json` 以作业名为键，崩溃后可幂等重算；费用未知时按该作业上限计）。

沙箱（代码驱动运行必需）：`sandbox.backend = "ssh"`（`sandbox.host`、`sandbox.remote_dir`、`sandbox.docker_config`）让 S6A_run / S7
的生成器、参考解、基线在构建机的 docker（`--network none`、资源上限同本机）里跑：移植了作者代码的参考解也只在构建机执行。

配置（`[code]`）：`grounded`、`prescreen`（S2a 开关，默认开）、`openalex_flag`、`fulltext_terms`、`pwc_path`、
`github_client`（`gh` 用 gh CLI 自带认证 / `rest` 匿名）、`proxy`、`mailto`、`jev_own_threshold`（没有 maybe 放行开关）；
`[models.roles] code_owner = "rules" | "jev"`；S3c / S4c / S5G：`fetch`、`host`、`remote_root`、`run_tag`、
`max_repo_mb`、`fetch_timeout_s`、`runnability`、`runnability_timeout_s`、`runnability_max_usd`、`runnability_est_usd`、
`authoring`（`agent` | `single_call`）、`authoring_timeout_s`、`authoring_max_usd`、`authoring_est_usd`、
`authoring_max_jobs`、`agent_harbor_root`、`agent_launcher`、`agent`、`agent_environment`、`agent_model`、
`agent_effort`、`agent_allowed_hosts`、`install_hosts`、`agent_base_image`、`agent_cpus`、`agent_memory_mb`、
`agent_jobs_dir`、`agent_hints`。检索侧见 `retrieval/README.md` 的“代码通道”。

### 0b. 针对校准学生薄弱点出题与加难回路（`[difficulty]`，2026-11-12 起，仅代码驱动条目）

**弱点画像**：`difficulty.profile_path` 指向一份**内部** markdown 画像（不进仓库）。每条以 `## <ID> <标题>` 开头，
写清模型在什么情形下容易错、科学计算任务里合法考察它的出题杠杆、如何用 naive/mid/参考解证明杠杆起作用、禁止的假难度
做法。画像 id 取 front matter 的 `profile_id`（或 `difficulty.profile_id`，否则文件名），连同 sha256 前 16 位记入每题
`_spec/provenance.json`（`difficulty_profile`、`difficulty_levers`）和发布清单的 `difficulty` 字段。仓库只提供通用示例
`gen/examples/weakness_profile.example.md`。画像只能写抽象机制，**不得**写入任何评测题的题名、数据、方法细节、阈值或例子。

**S5G 针对性出题**：配了画像时，S5G 作业多一个输入 `/work/weakness_profile.md`，指令要求 agent 先读论文和作者代码、
找出本文真实存在的关键难点，再挑 1–3 个与画像匹配的杠杆，并把 mid/naive 设计成“漏掉该杠杆”的做法、实测分离；
另写 `_spec/difficulty_design.json`（`profile`、`levers[]`：`weakness_id`、`lever`、`paper_evidence`、`code_evidence`、
`why_naive_fails`、`why_mid_fails`、`reference_handles`、`metrics`；`artificial_difficulty_check`）。该文件只在内部，
不进任务包。结构不合格（id 不在画像里、指标不在 spec 里、1–3 条之外、字段为空）→ 带反馈退回 S5G，用完
`code.authoring_max_jobs` 后淘汰（`s5g_jobs_exhausted:difficulty_design_invalid`）。

**S7 完整性检查**（`difficulty.design_gate = "block" | "warn" | "off"`，默认 block；产物 `gates/difficulty.json`）：
每个杠杆至少有一个所列的非资源类指标，在所有种子上参考解过阈值、且 mid 或 naive 在所有种子上不过；题面不得出现画像
id 或“weakness profile / difficulty design”等字样；加难轮次中题目（生成器、参考解、度量、基线）必须与上一版不同。
不通过 → 退回 S5G 一次（`difficulty_reauthored`），再不通过淘汰（`difficulty_design_unproven`）。锁定阈值的审核修订 /
regate 不复查（它们不改科学内容）。

**S9R**：清单新增 `j_genuine_difficulty`：难度是否来自科学本身；审核包里附 `records/difficulty_design.json`、
`records/difficulty_check.json`、`records/hardening/round-k-record.json`（不含学生轨迹反馈）。含糊题面、隐藏规格、
不可推断的约定、格式陷阱、压在参考解边缘或靠运气的阈值、单纯时间压力都判 **blocking**。

**加难回路**（`difficulty.harden`，默认开；`harden_max_rounds = 2`）：S9 中学生（DeepSeek，内部角色仍叫 `glm`）**第 1 次
有效尝试就通过**时，不再跑第 2 次，而是：
1. 先持久化 `state.harden_pending`（轮次、触发作业、epoch），再写 `harden/round-k/feedback.json`：学生自己的最终总结
   （claude-code.txt 的 result 文本，截断 4000 字）、用时、费用、解题程序运行秒数、各指标的值/阈值/相对余量。**不给**
   verifier 输出、学生交付文件、逐对象结果、隐藏输入或真值；
2. 快照当前版本到 `harden/round-k/prev/`（task、_spec、pkg、gates、thresholds、测量、S5G 报告；`DONE` 标记完成），记录
   `harden/round-k/record.json`（上一版 task/pkg 哈希、标签、各次运行），撤下已发布条目，归档本轮校准
   （`calib/epoch-<e>/`，新版本作业名带 `-e<e+1>`，不同版本的结果不合并）；
3. 清掉上一版的构建计数与过期产物，回到 S5G。S5G 额外得到 `/work/previous_version/`、`/work/student_feedback.json`
   （以及更早轮次的用时与余量 `earlier_rounds.json`），指令要求按画像从科学上加难，不允许只收紧阈值、把题面写含糊、
   加格式陷阱或缩短时限；`difficulty_design.json` 必须写 `hardening.changes` 和 `why_previous_solution_fails_now`。
   每轮另给一份 S5G 额度（`harden_authoring_timeout_s` / `harden_authoring_max_usd`，默认等于 `code.authoring_*`）和一份
   `authoring_max_jobs`；
4. 新版本照常 S6A_run → S6B（阈值按新测量和分离规则重推）→ S6C → S7（含上面的完整性检查）→ PKG → S9（从学生重新校准）；
5. 新版本在 S5G..PKG 被淘汰 / 搁置（且尚未进入 S9R）时自动**回退**到上一版（`harden_revert`：文件从 `prev/` 还原，
   失败版本移到 `round-k/failed/`，上一版的校准记录从 epoch 归档移回），按原规则继续校准。回退原因若属于基础设施
   （`difficulty.revert_infra`：S5G 作业 launcher_failed / no_result，或 agent 以 NonZeroAgentExitCodeError /
   AgentSetupTimeoutError / EnvironmentStartTimeoutError 结束且缺交付文件；`code:*:launch_failed`；
   `model_error:*:network|stream|empty`），记 `harden_reverted[].infra`，**不消耗** `harden_max_rounds`，也不设
   `harden_exhausted`，上一版的学生通过记录会再触发一次加难；其他原因（费用上限 `s5g_budget_exhausted` /
   `item_harden_usd_cap`、agent 超时、交付不全、分离不成立等）计为一轮并设 `harden_exhausted=true`；
6. 轮次用尽（不计 infra 回退的轮数 ≥ `harden_max_rounds`，或总加难启动次数 ≥ `harden_max_attempts`，默认 4，防止
   infra 死循环；或 `harden_exhausted`）后按原规则：第 2 次学生运行 → easy / unstable。

可中断、可恢复、幂等：每一步先写状态再动文件；快照有 `DONE` 标记；epoch 只在仍为当前 epoch 时归档；重跑 S9 时用
`harden_pending` 继续同一轮，不会再次付费。账本里加难作业仍记为阶段 `S5G`；事件 `harden_start`、`harden_authored`、
`harden_reverted`、`difficulty_design_invalid`、`difficulty_design_failed`；每轮费用在 `harden/round-k/record.json`
（`author_jobs`、`author_cost_usd`）。旧的非代码驱动条目不走回路。

手动送入：`tbsgen harden --run R --config C --item ID --reason TEXT`，用于回路上线前已判 easy 的条目（要求当前校准有
学生通过记录、没有进行中的作业、尚未做过 S9R 审核）。

配置（`[difficulty]`）：`profile_path`、`profile_id`、`design_gate`、`harden`、`harden_max_rounds`、
`harden_max_attempts`、`harden_authoring_timeout_s`、`harden_authoring_max_usd`、`harden_exclude`。

**首选杠杆**：画像 front matter 可写 `preferred: [ID, ...]`（必须是画像里的条目）。S5G 指令要求优先使用（仍须来自论文
真实难点）；加难轮的 `difficulty_design.json` 必须至少含一个首选杠杆（S5G 设计校验），S7 还要求至少一个首选杠杆在
reference / mid / naive 实测中成立（`difficulty.preferred_problem`），否则按设计不成立处理（退回 S5G 一次，再不过淘汰 →
回退上一版）。首版出题不强制。

### 0c. 生产限流（2026-11-12 起，默认关）
- `budget.item_usd_cap`：单篇出题阶段（`budget.item_usd_cap_stages`，默认 S4c、S5、S5G、S6A、S6M、S6C、S8；
  不含 S9 校准和 S9R 审核）的已结算 + 预留费用上限。账本预留时检查，超出即 `dropped:item_usd_cap`（只影响该篇，
  模型不停）；S4c/S5G 作业的 `--max-budget-usd` 夹到剩余额度，剩余不足 $1 时 `s5g_budget_exhausted`。
- `budget.item_harden_usd_cap`（默认 None = 加难轮次与首版共用 `item_usd_cap`，即旧规则）：设了以后，单篇**所有加难轮
  合计**另有这份额度，首版不再与加难共用 `item_usd_cap`。`Pipeline.step` 在 `harden_active` 时把预留打上轮次标记
  （账本 `calls.harden` 列；旧账本自动加列，旧行视为首版），加难轮里 `item_usd_cap_stages` 加 S7 的花费都计入加难额度；
  超出即 `dropped:item_harden_usd_cap` → 回退上一版（计为一轮）。加难轮 S5G 的 `--max-budget-usd` 取
  `harden_authoring_max_usd` 与加难剩余额度的较小值。
- `code.daily_new_runnability` / `code.daily_new_authoring`：每天（北京时间）首次进入 S4c / S5G 的论文数上限，
  记在 run 目录 `s4c-admissions.jsonl` / `s5g-admissions.jsonl`（同一篇的重出题、加难轮次不重复计数）；
  超出时条目暂停 `budget:daily_s5g_papers`，次日 `tbsgen resume --reason budget:daily` 恢复（驱动跨日时执行）。只有作业真正启动才计入当天名额；
  设了单篇上限时，作业在账本里按整份额度预留（成本未知时不会少算），恢复时沿用同一预留额。准入检查、预留和启动在 run 级锁
  `.agent-slots.lock` 下完成。
- `code.max_concurrent_agent_jobs`：同时运行的 S4c + S5G 作业上限；满时条目 waiting（`queued: agent slots full`），
  下一轮自动重试。

## 1. 架构

```
src/tbsgen/
  config.py        全部配置与默认值（待定项都在这里）；config.toml 可覆盖
  budget.py        sqlite 预算账本：按 模型×阶段×条目 记 token/美元/次数；日预算、调用上限、单条目上限；熔断
  models.py        模型调用层：TeacherClient（Anthropic 兼容端点）/ FakeClient（脚本化）/ RecordingClient（录制）
  state.py         每条目一个状态目录 + state.json 状态机；events.jsonl
  pipeline.py      编排：逐条推进、断点续跑、预算/熔断暂停、resume
  codegate.py      代码驱动出题：S2a 代码信号预筛、S3b 作者代码确认与仓库解析（§0）
  codeagent.py     代码驱动出题：S3c 构建机获取代码、S4c / S5G 构建机 Claude Code agent 作业（§0）
  prompts_code.py  S4c 可运行性、S5G 出题 agent 的指令（含针对性出题、加难轮次附加段）
  difficulty.py    弱点画像加载、difficulty_design 校验、S7 杠杆完整性检查、加难回路的反馈 / 快照 / 回退（§0b）
  intake.py        S1 读检索层输出；S2 规则筛选（rules-v2）；S3 全文（调 fulltext.py）；S4 可行性（规则部分 + 官方题去重）
  blocklist.py     S2 前置黑名单：按 DOI 前缀（可加发表年份）直接丢弃拿不到全文的论文，配置 screen.blocklist；
                   自动复查（screen.blocklist_recheck）：每周少量试取，成功率够高自动解除规则
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
  package.py       PKG：打包为 Harbor 任务 + 结构校验（含网络策略）
  prerelease.py    上线前检查：现成实现检查（S7 `readymade` 门）+ 打包任务网络策略说明
  calib.py         S9：GLM 顺序校准 + Opus 教师分支的调度规则、Harbor 作业配置、结果收集
  review.py        S9R：校准后的审核 agent（审核任务构建、审核清单提示词、裁决契约、修订保工审查）
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
  S1_candidate.json S2a_code_signal.json S2_screen.json S2_jev.json S3_fulltext.json fulltext.txt S3b_code.json
  S3c_code.json S4_probe.json S4c/{run.json,task,HOWTO.md,run.log} S4c_runnability.json S5G/{run.json,task,report.md}
  S5G_result.json _spec/author_code.json
  _spec/{spec,depgraph,coverage,provenance}.json      出题侧，不进 Harbor
  task/                      工作树：generator/ solution/ baselines/ tests/ instruction.md
  runs/                      沙箱运行产物
  S6A_measure.json thresholds.json gates/attempt-<n>.json repair/round-<n>/
  pkg/                       Harbor 任务（只有这个目录交给 Harbor）
  calib/runs.json + 作业配置  S9（修订后旧校准归档到 calib/epoch-<k>/）
  review/runs.json r<k>/{task,verdict.json} verdict.json      S9R
  repair/review-<n>/{patch,before,review.json,thresholds.locked.json}   S9R 退回的修订
```

条目状态：`active`（还有下一阶段）、`waiting`（S9 等校准结果 / S9R 等审核结果）、`paused`（预算或熔断暂停，`resume` 后从原阶段继续）、`done`、`dropped`（带原因码）、`parked`（需要人工处理）。每个阶段先写完产物，再原子地更新 `state.json`；进程在阶段中途被杀，重跑时从该阶段重新开始。

## 2. 各阶段输入输出

| 阶段 | 输入 | 输出 | 模型 | 失败去向 |
|---|---|---|---|---|
| S1 | 检索层输出（`tbsretr` 的 `papers` 或 `eval/raw/*.json` 的 `selected`）或候选 JSONL | `S1_candidate.json` | 无 | — |
| S2 | 题录 + 摘要 | `S2_screen.json`（`score`、`keep`、`form_hint`、`method`、`reason`、`deferred`；Jev 模式另有各问题概率 `probabilities`、`decision`（keep/drop/uncertain）、`uncertain`、`jev_model`、`input_tokens`）；`S2_jev.json` 为按请求键缓存的 Jev 结果 | `roles.screen`：`rules`（默认）或 `jev`（TypeSafe Jev，失败回退规则） | `dropped: low_screen_score`（规则）/ `jev_*`（Jev）；无摘要时 `deferred`，S4 用全文开头重判 |
| S3 | DOI、OA 链接 | `fulltext.txt` + `S3_fulltext.json` | 无 | `parked: fulltext_unavailable`，写 `S3_queue.json` 并加入全文队列；`tbsgen fetch` 取到后写入 `fulltext.txt` 并推进到 S4 |
| S4 | 全文 + 筛选记录 | `S4_probe.json`（计算核心线索、与官方 70 题去重、形态禁令；Jev 模式另有 `jev`：可出题性各问题概率、`score`、`reasons`）；`S4_jev.json` 为按请求键缓存的 Jev 结果 | `roles.probe`：`rules`（默认）或 `jev`（全文开头 + 6 个可出题性问题，约 2k token/篇，≈$0.0001） | `dropped: no_compute_core / tbs_duplicate / interact_form_banned_until_isolation`；Jev：`jev_s4_no_self_contained_core / no_synthetic_truth / no_numeric_score / needs_restricted_resources` |
| S5 | 全文摘录 + 探针卡 | `_spec/spec.json`、`depgraph.json`、`coverage.json`、`provenance.json` | Opus | 校验不过重试 1 次，仍不过 `dropped: spec_invalid` |
| S6A | 规格 | `task/generator/`、`solution/`、`baselines/{mid,naive}.py`、`tests/measure.py` | Opus | 缺文件 `dropped` |
| S6A_run | 工作树 | `S6A_measure.json`：两个种子下 ref/mid/naive 的实测指标 | 无（Docker） | 生成器或参考解失败 → S8 |
| S6B | 实测指标 | `thresholds.json`（含 `decision`）：分离型指标取 ref 与 mid 的对数中点，非分离型取 ref 放宽 50%，取整一律向宽松方向，且保证 mid 仍不过；每次决策追加到 `S6B_decisions.json` | 无 | 见下方“S6B 决策与提前止损”；ref 不过 → S8 |
| S6C | 规格 + 阈值 + 实测 | `tests/test_outputs.py`、`instruction.md`、`baselines/exploit_*.py` | Opus | 缺文件 `dropped` |
| S7 | 完整工作树 | `gates/attempt-n.json` | 无（Docker） | 阻塞级 → S8；只有“阈值未写明”时由程序追加判据表 |
| S8 | 门日志 + 依赖图 + 相关文件 | `repair/round-n/`（补丁 + 审查结论） | Opus 修复；审查为机械 diff | 审查不过算用掉 1 轮；2 轮后 `dropped: repair_limit` |
| PKG | 通过 S7 的题 + 种子 0 产物 | `pkg/` + `PKG_check.json` | 无 | `parked: package_invalid` |
| S9 | `pkg/` | `calib/*.json` 作业配置 + 启动命令；`runs.json` | 不调用（只调度） | 见 §5 |
| S10 | 标签为 hard/unstable（channel main）或 frontier（channel frontier）的题 | `release/manifest.jsonl` | 无 | — |

**S7 机械门**：determinism（同种子两次哈希一致）；oracle（参考解在每个种子上都通过）；oracle_margin（参考解至少留出一半余量）；nop（空交付不过）；naive（每个种子至少 2 项不过）；mid（1–2 项不过，且全部判据都被评估）；exploit（作弊尝试全不过）；exploit_valid（阻断，v2.5：每个作弊基线必须在相同调用方式下真正跑完并产出可评分的输出——解题步骤退出码 0、写出了交付物、PROG 的每次 `solution_cli.py` 调用退出码 0、参考解被评估的判据都被评估、且不是所有指标都等于空交付值；作弊基线崩溃或没产出时它“没通过”不算拦住，送 S8 修复）；hidden_shift（只对 PROG）；thresholds_stated（每个阈值都在题面中出现）；leak；isolation（`environment/` 下没有 tests/truth/generator）；tdm_verbatim（全文来自 Elsevier/Wiley TDM 的论文：题面和 `environment/data` 下的文本文件与全文的最长逐字重合，空白规范化后不超过 200 字符，见 `tdmgate.py`）。

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
uv run tbsgen resume --run runs/R --only ID1,ID2 [--stage S9] [--reason budget:]  # 只恢复指定条目/阶段/原因前缀
uv run tbsgen breaker --run runs/R --model claude-opus-5-5 --reset  # 人工解除熔断
uv run tbsgen fetch --run runs/R --config config.toml [--max N] [--channels oa,browser] [--no-browser] [--dry-run]
                                                                    # --max 默认：远程模式 = fulltext.remote.batch_max（一次填满一批），否则 10
                                                                    # 清全文队列（含 ego-browser 通道），取到全文后交给等待的条目并推进到 S4
                                                                    # 只有真正发出请求并失败才退避；被限额/冷却/熔断跳过的不计次数，到最早恢复时间再到期
                                                                    # 部分轮次（任何显式 --channels，或本轮收回的旧远程作业范围窄于该条目适用的全部通道）
                                                                    # 只记录真正尝试过的通道的退避，不重新安排条目、不推迟未尝试的通道
uv run tbsgen fetch --run runs/R --only 10.1016/j.x.1,arxiv:2101.00001  # 只处理指定条目（不论是否到期）
uv run tbsgen remote deploy --run runs/R --config config.toml [--dry-run]  # 把本地 HEAD 部署到 fulltext.remote.code_dir 并核对两端指纹
uv run tbsgen remote status --run runs/R --config config.toml       # 两端 HEAD、指纹、远端作业
uv run tbsgen fulltext purge --run runs/R --channel elsevier,wiley [--before YYYY-MM-DD] [--dry-run]
                                                                    # 删除这些通道的全文与 raw PDF/XML（保留账本与 provenance）
uv run tbsgen fetch --run runs/R --status                          # 今日下载用量、冷却、熔断
uv run tbsgen fetch --run runs/R --finish-browser                  # 关闭本工具的 ego TaskSpace
uv run tbsgen fetch --run runs/R --browser --wait-for-egress         # 等出口进入 allow_cidrs 后跑一个浏览器批次
uv run tbsgen calib-ingest --run runs/R --item ID [--job-dir DIR]   # 回填校准结果（默认取该次提交的作业目录；作业未结束则拒绝）
uv run tbsgen calib watch --run runs/R --config config.toml [--only ID,..] [--once] [--interval 120] [--max-hours H]
                                                                    # 校准监控：回填已结束的作业 → 推进 S9（打标签或提交下一次：GLM → 教师）；
                                                                    # 前台循环，无未结束的 S9 条目时退出；allow_submit=false 且无运行中作业时也退出
uv run tbsgen calib prepare --run runs/R --config config.toml --item ID [--who teacher]  # 手动运行环境准备钩子（失败后重试）
uv run tbsgen review build --run runs/R --config config.toml --item ID [--round K] [--out DIR]  # 只生成 S9R 审核任务（回放/手动运行）
uv run tbsgen harbor-local --run runs/R [--item ID]                  # 本机 Harbor oracle/nop（换本地基础镜像）
uv run tbsgen run    --run runs/R --until S4                      # 只跑到 S4（不启动模型阶段，不留 model:config 暂停）
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

凭据：出题阶段的 teacher 模型走 Anthropic 兼容端点（`models.teacher.url` 或环境变量 `TBS_TEACHER_URL`，没有默认值；未设置时模型阶段以 `model:config` 暂停）。不设 `key_env` 时本进程不发送 key（适用于持有凭据的本机网关）。如果改走需要 key 的端点，设置 `models.teacher.key_env`，再用 `uv run tbsgen ...` 注入；key 只放在请求头里，不写日志或文件，错误信息会先脱敏。

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
| `screen.min_score` | 1 | 规则筛选（rules-v2）：每个词干只计一次、“survey”只在文献综述语境扣分。无 Jev key 时的默认、Jev 不可用时的回退 |
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
| `schedule.order` / `schedule.similar_title_jaccard` | round_robin / 0.5 | 条目推进顺序（每日调用上限下决定谁先进 S5）：`round_robin` 按 target 轮转、组内先按 S4 Jev 可出题性分数（`roles.probe = jev` 时；有 S4 分数的排在只有 S2 分数的前面），没有时回退到 S2 分数，组内标题实词 Jaccard ≥ 0.5 的近似重复后排；`score` 全局按分数；`id` 旧行为 |
| `fulltext.oa.europepmc_last / last_hosts / server_error_streak / server_error_cooldown_s` | True / ebi.ac.uk、europepmc.org / 5 / 1800 s | Europe PMC render（2026-11-08/09 有 38/49 次 HTTP 500）放到 OA 候选最后；同一站点连续 5 次 5xx 冷却 30 分钟，论文保持待取 |
| `roles.probe` / `probe.jev` | rules / max_chars 6000、thresholds 空、on_unavailable rules | `jev`：S4 在规则探针通过后再问 Jev 可出题性（`screen_jev.S4_QUESTIONS`），阈值见 `screen_jev.S4_THRESHOLDS`（core≤0.20、synthetic≤0.40、numeric≤0.30、restricted≥0.70 丢弃）。离线评估（e2e-prod-20261109 的 111 篇，真实 Jev，$0.0097）：拦下 10 篇（深度学习 QSM/场图网络 4、纯理论 2、药物警戒统计 1、DES-Y3 数据分析 1、QC 报告工具 1、CMB 系统误差分析 1），3 篇入库题和 9 篇出题失败题均未被拦下。失败题无法用这些概率区分（S4 score 对入库/失败 AUC 0.26，样本仅 3 对 9），因此 S4 只去掉明显不合适的论文，其余交给 S6B 止损 |
| `gen.mid_rework_max` | 1 | mid 不可区分时先由 S6M 只重写 `baselines/mid.py` |
| `gen.back_to_spec_max` | 0 | S6M 后仍拉不开时回 S5 的次数（e2e-prod-20261109：回 S5 的 5 篇无一救回）；>0 时只对“差一点”生效 |
| `gen.mid_far_above` | 2 | mid 失败判据数 ≥ max_mid_fail+此值（或全部判据且 ≥ max_mid_fail+1）直接丢弃 |
| `gen.ref_gross_ratio / ref_rewrite_max` | 10 / 1 | 参考比 naive 差这么多倍 → 重写 S6A 一次，再错则丢弃 |
| `gates.max_repair_rounds` | 2 | S8 修复轮数上限 |
| `gates.blocking` 中的 `readymade` | 阻塞 | 现成实现检查：任务镜像声明的依赖（requirements、Dockerfile 里的 pip install）、参考解/基线的 import、`environment/` 下随题分发的代码，与论文 `code_links`/`data_links` 的仓库名/包名、S5 规格的 `method_names` 匹配即阻塞，进 S8（去掉依赖、自己实现或改题）。已知配套仓库写入 `provenance.json` 的 `companion_code`。纯规则，不调模型。基于名称：覆盖 requirement 写法（`==`、`@ url`、`-e`、`#egg=`）、VCS/直接归档 URL 与文件（去版本号和 `#sha256=` 片段）、GitHub 等 `owner/repo`、GitLab 子组（取最后一段）、PyPI `/project/`、CRAN `package=` 与 `/web/packages/`、随题分发的源码/归档/编译产物（`.so`/`.pyd`/`.pyc` 等）；不保证发现改名、改名后 vendored 或混淆的实现，由校准和人工复核兜底 |
| `package.network.mode` | agent-allowlist | 打包任务的网络策略：`[agent]` 阶段 allowlist（`agent_hosts` + 启动时 `--allow-agent-host` 给的模型端点），`[verifier]` 断网，`[environment]` 公网只用于 agent 安装；`no-network` 全程断网（只适合 oracle）；`public` 旧行为，仅作对照 |
| `package.network.disable_index` | False | True 时镜像构建完成后去掉 pip/uv/conda 索引并设 `PIP_NO_INDEX=1` 等（减速带，不是屏障；默认关，避免 agent 安装自身运行时失败） |
| `calib.agent_allowed_hosts` | 空 | 校准作业里给 agent 阶段放行的模型端点主机（写成 `agents[].extra_allowed_hosts`）。allowlist 模式下为空且 `allow_submit=true` 时 S9 拒绝提交 |
| `gates.tdm_verbatim_max_chars / tdm_verbatim_channels` | 200 / elsevier, wiley | TDM 全文与公开文件的最长逐字重合上限（阻塞级） |
| `fulltext.not_entitled` | 开；elsevier, wiley；不过期 | TDM API 只给元数据/摘要（`short text N`）时记账本，该通道永不再请求该 DOI |
| `fulltext.remote.require_clean_tree / preflight` | True / True | 远程模式：本地包有未提交改动或远端指纹不一致时，在租约和提交之前拒绝（退出码 3） |
| `fulltext.remote.code_dir / deploy_sync` | None / `uv sync -q --no-dev` | `tbsgen remote deploy` 的远端仓库目录与依赖同步命令 |
| `calib.quick_pass_minutes` | None | GLM 首次通过且用时短时直接记 easy（**待定**，默认关） |
| `calib.keep_labels` | hard, unstable | 保留的难度标签 |
| `calib.variant_depth` | 0 | S9b 更难变体（**待定**，MVP 关闭） |
| `calib.infra_retries` | 2 | API/沙箱异常不计入校准次数 |
| `calib.allow_submit` | **False** | True 时：账本预留 → tmux 后台运行启动脚本；预留超总额则拒绝；没配 `harbor_root` 和启动脚本时拒绝提交 |
| `calib.harbor_root / student_launcher / teacher_launcher` | None | 启动脚本目录与脚本名（`./<launcher> run <作业配置>`）；未设置时只打印 `harbor run --config ...` |
| `calib.student_agent / teacher_agent / environment` | None | Harbor 自定义 agent/环境的 `module:Class`；未设置用内置 `claude-code` agent 和 `docker` 环境 |
| `calib.est_usd_per_run` | 20 | 每次校准作业的预留额，`calib-ingest` 冲销：Harbor 报了 `cost_usd` 用实报（`reported`）；没报（如超时）用 trial 的 token 用量 × `models.prices[模型]` 估算（`tokens`，来源依次为 agent_result、trajectory.json、Claude Code stream-json）；都没有则保留预留额并标 `reservation`——它仍占用预留额度，但不算进 `run_usd_total` 的实际花费门。`tbsgen ledger` 的 `usd_by_cost_kind` 分开列出 |
| `calib.teacher_available` | True | False：GLM 首次失败直接记 `parked:teacher_unavailable` |
| `calib.prepared_tasks_dir` | None | 远端沙箱已注册模板的任务副本目录；设置后缺模板的条目不提交 |
| `calib.prepare_command / prepare_cwd / prepare_timeout_s` | None / harbor_root / 3600 | 环境准备钩子：S9 写作业配置前对每题运行一次（bash），把 `pkg/` 变成 `<prepared_tasks_dir>/<item_id>`（如建镜像、注册沙箱模板、改写 task.toml）。占位符 `{item_id} {pkg} {out} {who}`（已 shell 转义），环境变量 `TBSGEN_ITEM_ID / TBSGEN_PKG / TBSGEN_PREPARED_DIR / TBSGEN_WHO`。结果连同 pkg 哈希记入 `calib/prepare.json`：同一 pkg 成功则复用、失败不自动重试（`tbsgen calib prepare` 重试），失败时条目 waiting 且不提交 |
| `calib.teacher_environment / teacher_agent_allowed_hosts / teacher_harbor_root / teacher_prepared_tasks_dir` | None（沿用共享项） | 教师独立路由：教师作业用自己的 Harbor 环境、agent 阶段放行主机、启动脚本目录和准备副本目录；未设置时与学生共用 `environment / agent_allowed_hosts / harbor_root / prepared_tasks_dir` |
| `calib.timeout_no_output / timeout_no_output_retries` | fail / 1 | 学生运行“超时且未交付”（`timeout_no_output`）的处理：`fail` 按普通失败（默认，标签规则不变）；`retry` 不计入决策、重跑学生（最多 N 次，用完后按失败计）；`park` 暂存 `parked:timeout_no_output` |
| `calib.watch_interval_s` | 120 | `tbsgen calib watch` 轮询间隔 |
| `fulltext.cascade` | cache → oa → elsevier → wiley → inst → browser | 下载级联顺序；每个通道有自己的 `enabled`，默认只开 cache 和 oa（见根 README「Full text (S3)」） |
| `fulltext.write_cache_dir / ledger_path / queue_path` | `<run>/fulltext_cache`、`<run>/fulltext_ledger.sqlite`、`<run>/fulltext_queue.jsonl` | 写入缓存（含来源 json）、下载账本、浏览器队列 |
| `fulltext.limits / global_limits / host_caps` | `dlledger.py` 中的默认值 | 每通道每天/每小时成功与尝试上限、同站点最小间隔、全局上限；429 冷却 6 小时，验证页冷却 24 小时 |
| `fulltext.egress_check` | 关闭 | 开启后，出口 IP 不在 `allow_cidrs` 内则拒绝所有网络通道 |
| `fulltext.use_env_proxy` | False | 下载不读 `HTTP(S)_PROXY` |
| `fulltext.browser.*` | 关闭；并发 2；验证页等 40 秒 | ego-browser 通道；S3 只入队，`tbsgen fetch` 执行；同一出版商间隔 60–120 秒，不点击任何验证 |
| `fulltext.browser.batch` / `.agent` | 窗口 600 s、余量 60 s；agent 关闭；每篇 ≤25 次工具调用、≤12 次页面加载，同出版商两次加载间隔 ≥2.5 s | `tbsgen fetch --browser [--wait-for-egress]`：出口门控的限时批次；agent 为 Hermes CLI，每篇独立 TaskSpace，唯一工具 `tbsgen-browser`（look [full|@REF]/open/click/click_xy/hover/scroll/select/type/screenshot/tabs/switch/back/wait_for/save_pdf/sso_login/handoff_status/finish；另有 vision_analyze 只能看本批次自己的截图；硬底线：凭据值不进入模型、日志或外部（不读字段值，snapshot 先去掉字段文本，截图遮盖字段、遮不住就拒绝，type/select 拒绝凭据字段），不自动解验证码（交接），只在允许域名内导航（含 SSO 域），文件由代码校验）；页面有任何登录或机构选择（WAYF）迹象时拒绝 click 并交给用户）；agent 的 stderr 尾部（凭据样式字符串打码）写入批次目录 `agent-stderr.log` 和 `agent_end` 日志 |
| `fulltext.browser.handoff` | 关闭；等 120 s | 验证页（Cloudflare/Turnstile/验证码/滑块）与非机构登录页交给本机用户处理（macOS 通知，可选本地 `bark_cmd`），agent 从不自己过验证；每篇、每出版商每批最多 1 次，超时则冷却该出版商 |
| `fulltext.browser.sso` | carsi.edu.cn、openathens.net、shibboleth.net | `sso.domains`：机构/WAYF 域，加入每篇的可访问域。`sso_login`（默认关闭）：`domains` 为机构自己的登录域（如 `authserver.nju.edu.cn`，HTTPS，同样加入可访问域），`tbsgen-browser N sso_login` 只在浏览器自动填充后点登录（不读取、不输出、不记录凭据值；表单须 POST 到登录域），每批最多 `max_submissions`（2）次、点击前扣减、失败不重试；验证码/短信/无自动填充时交给用户 |
| `fulltext.remote` | 关闭 | HTTP 通道（oa/elsevier/wiley/inst）经 ssh 在 `host` 上运行；远端须开启 egress_check |
| `package.dockerfile_mode` | selfcontained | Dockerfile 里用 pip 安装固定版本；`local_base` 只用于本机检查 |

**黑名单自动复查**（`screen.blocklist_recheck`）：某条黑名单规则在某个 target 的候选里占比超过 `min_share`（默认 10%）时，每个 ISO 周给它 `weekly_quota`（默认 5）篇试取额度：这些论文照常走 S2 筛选 → S3 下载（含正常的下载队列），不再在 S2 前丢弃。S3 取到全文记成功、因取不到全文暂存记失败（之后被 `tbsgen fetch` 补上会改记成功），S2 筛掉的不计。已判定的试取 ≥ `min_trials`（默认 5）且成功率 ≥ `lift_rate`（默认 60%）→ 自动解除该规则（`blocklist_lifted` 事件），否则维持（`blocklist_kept` 事件）并继续每周试取。状态在 `<run>/blocklist-trials.json`，`tbsgen review summary` 里的 `blocklist` 一节可以看到。

**下载器熔断与冷却**：`oa` 通道的熔断按**被拦截的不同站点数**计（`breaker_count_hosts`，默认 10 个站点），同一站点反复被拦只靠它自己的 24 h 冷却；一次被拦截的响应只计一次。跳转后被拦截时，冷却落在**最终站点**（计入熔断），起始站点只冷却不计数；`doi.org` 这类解析/跳转中介（`fulltext.RESOLVER_HOSTS`）不冷却。`fulltext.wiley.prefixes` 默认 `10.1002/ 10.1111/ 10.1029/ 10.1113/`，`fulltext.elsevier.prefixes` 默认 `10.1016/ 10.1006/ 10.1053/ 10.1067/`，与出版商别名表 `fulltext.PUBLISHERS` 一致（有测试防止不一致）。

## 5. S9 校准规则

GLM-5.3 + Claude Code 2.1.280 先跑第 1 次：

- 失败 → Opus 教师。教师通过记 **hard**；教师失败再跑 1 次。两次有效尝试都是真失败（不算 infra，也不能是超时）时记 **frontier**（`calib.frontier`，默认开启）；否则（含超时）暂存为 `parked:teacher_failed`。
- 通过 → GLM 第 2 次。失败记 **unstable**，通过记 **easy**。

hard 和 unstable 保留。reward=1 一律算通过，即使之后报了 API 错误。没有 reward 的 API/沙箱异常算 `infra`，不计入次数，连续超过 2 次则暂存。`allow_submit=false` 时只写出作业配置和启动命令，由人执行；`true` 时先在账本预留、再后台启动。结果用 `calib-ingest` 回填。

每次运行另记结果类型 `result_type`：`pass`、`fail_delivered`（交付了但没达标）、`fail_no_output`、`timeout_with_output`、`timeout_no_output`（超时且没写出交付文件，依据 verifier 的 `_solution_cli_run` 记录；无记录时按“已交付”处理）、`infra`。默认标签规则只看通过/失败；`calib.timeout_no_output` 可让“超时且未交付”的学生运行不单独决定结果（见 §4）。`release/manifest.jsonl` 每条带 `calib_runs`（每次运行的谁、结果类型、用时、费用及其类型）和 `result_types`。

远端沙箱不执行 Dockerfile：提交前要为每题建好模板（可用"共享基础镜像 + 数据层"的薄镜像方式），并把准备好的任务副本目录配到 `calib.prepared_tasks_dir`；配 `calib.prepare_command` 后由 S9 对每题自动调用你的准备脚本。在 `allow_submit=false` 时写出的作业意图（指向 `pkg/`）在准备副本就绪后提交时会自动改指向准备副本。

**只校准部分题**：把 `calib.allow_submit` 改为 true 后，`tbsgen run --only ID1,ID2`（或 `tbsgen calib watch --only ID1,ID2`）会推进这些 S9 waiting 且没有运行中作业的条目（未提交的意图、准备失败、提交被拒）；运行中的作业不会被重复提交，未列出的条目和其他暂停条目不受影响。预算/熔断暂停的条目用 `tbsgen resume --only ...` 单独恢复。之后用 `tbsgen calib watch --only ...` 等待作业结束、回填并推进（GLM → 教师 → 标签）。

## 5b. S9R 审核（校准之后、发布之前）

S9 打出标签后，**hard / unstable**（`calib.keep_labels`）必须先过审核才进 S10；**parked:teacher_failed** 也审核（`review.review_parked`，只作诊断，不会因此放行）；easy 不审。

**frontier**（没有任何模型做对的题）必须通过审核，且审核比一般题更严，才能进 S10：
- verdict 为 pass；
- 审核者复现参考解，`reference_check.reward` = 1；
- 清单 a（一致性）、b（条件完整）、e（无捷径/泄漏）、j（难度来自真实科学难度）都为 `ok`；
- `label_assessment.trustworthy` = true；
- S7 oracle 门禁通过。

任一项不满足，记为 `parked`（`frontier:unconfirmed`）。审核关闭时，frontier 一律不发布。发布记录中 `channel = "frontier"`，与 hard/unstable（`channel = "main"`）分开；附带 `frontier`（教师有效、infra、超时的次数）和 `reference_check`。frontier 题没有通过轨迹，下游不要把它放进 SFT 主体。

- **审核者**：能执行代码的 Claude Code agent（默认 Opus，`review.model` 未设则用 `calib.teacher_model`），全新上下文，在**题目自己的镜像**里运行（审核任务由 `review.build_task` 生成：原 Dockerfile 加 `COPY review /review`，`/tests`、`/solution` 照真实运行装好，可直接 `bash /solution/solve.sh && bash /tests/test.sh` 复现参考解）。网络与教师相同：agent 阶段只能连模型中继（`review.agent_allowed_hosts`，未设沿用教师的），verifier 阶段断网。
- **能看到**（`/review`）：题包全部文件（含 hidden/truth、参考解）、`extra/`（baselines、generator）、`records/`（thresholds.json、S6A 实测、S6B 决策、S7 门报告、S8 修复记录、spec、阶段历史）、`calibration/`（当前校准轮次每次运行的 result 摘要、verifier 输出与逐项指标、trace 摘要：工具调用序列、助手文字、最后的工具输出和最终消息）。
- **清单**（写在提示词里，逐项给结论与证据）：a 题面/格式说明/测试/评分一致性；b 缺失条件与测试检查题面未要求的东西；c 阈值是否以参考解为依据、是否合理；d 计时/随机性不稳定；e 捷径与泄漏；f 依赖网络或现成实现；g 结合校准 trace 判断失败/通过是否由题目缺陷或基础设施导致、标签是否可信；h 计时项是否应当参与判定（`resource_metric_gating`，只有速度本身是论文贡献时才为 true）。
- **输出** `/app/review_out/verdict.json`：`verdict ∈ {pass, revise, drop}`、`checklist`、`issues[]`（类别、`blocking|minor`、问题类别 `problem`、证据、修法）、`label_assessment`。每个 blocking 必须标 `problem`：
  - 局部问题（可修 → revise）：`statement`（题面表述/条件缺失或矛盾）、`format_notes`（格式说明）、`grader_io`（评分脚本的输入输出：路径、列、读哪个文件）、`exploit_baseline`（作弊基线缺失或错误）、`test_gap`（测试遗漏或检查了题面没要求的东西）、`timing_gate`（计时判定策略/余量，阈值仍锁定）；
  - 根本问题（不可修 → drop）：`reference_wrong`（参考解本身错误）、`truth_regen`（真值/数据需要重新生成）、`core_invalid`（题目核心不成立）、`calibration`（标签不可信且局部修改改变不了）。
- **裁决契约**（`review.normalize`）：**没有证据的 blocking 降为 minor**；**泄露真值/答案或可篡改评分的问题一律 blocking**（`"leak": true` 且有证据即升级）。`fix_kind` 由 `problem` 决定（审核自己写的 fix_kind 与类别矛盾时以类别为准并记录），裁决由 blocking 推出：有根本问题 → drop；只有局部问题 → revise；没有 blocking → pass。审核写的 verdict 与此不一致时以推出的为准（记在 `problems` 里），例如判了 revise 但 blocking 全是不可修类别 → drop。结构问题（未知 verdict 或类别、blocking 没有类别、verifier reward≠1、revise/drop 却没有带证据的 blocking）→ 裁决无效，本轮重新审核，超过 `review.infra_retries` 次 → dropped（`review:invalid`）。
- **没有人工出口**：每题去留由审核 agent 决定。`human` 只在显式打开 `review.allow_human = true`（默认 false）时才是合法裁决，并在修订用尽、修订被拒、无效裁决等出口改为 parked（`review:human`）。
- **路由**：pass → S10（minor 写进 manifest 的 `review.minor`）；parked 标签的 pass 仍保持 parked。revise → S8 审核修订（下条），修订轮数用完（`review.max_revisions`，默认 3）→ dropped。drop → dropped。已在 manifest 里的题若审核非 pass，追加一行 `withdrawn: true`（manifest 只追加，同一题以最后一行为准）。
- **审核修订**：S8 把审核 issues 交给模型，只允许改 `instruction.md`、`tests/test_outputs.py`、`tests/measure.py`、`generator/generate.py`（格式说明由生成器写出）和已有的 `baselines/exploit_*.py`；保工审查（`review.revision_review`）拒绝其他路径、拒绝 `check()` 的指标/阈值/方向有任何变化。**阈值始终锁定，不允许放宽**。修订被拒、或修订后参考解达不到锁定阈值 / S7 门 / PKG 检查失败：把失败原因作为反馈再修一轮（占用一轮修订），轮数用完 → dropped。通过 →
  - **只改了评分脚本**（`tests/test_outputs.py`、`tests/measure.py`；题面、数据、阈值都没变）且 `review.rescore_scoring_only = true`（默认）：保留当前校准，S9 先用新评分器在本地沙箱重评已有 GLM 与教师交付物（`items/<id>/calib/deliverables/<job>/`，用 `tbsgen review deliverable` 登记，例如从 trace 离线重放出的 `/app/output`）。所有运行的结果都不变 → 保留标签，直接进下一轮审核；结果变了、缺交付物或沙箱没写出 reward → 归档并重新校准。没有任何交付物的失败（`timeout_no_output`/`no_output`）不需要重评（S7 的 nop 门保证新评分器判空输出为失败）。结果写在 `calib/rescore.json`。
  - **其他修订**：旧校准记录归档到 `calib/epoch-<k>/`、`calib_epoch+1`、标签清空 → S6A_run 重新实测 → **S6B 锁定阈值**（新推导值只写进 `thresholds.relock.json` 供审核对照）→ S6C → S7 → PKG → S9 从 GLM 重新校准（作业名带 `-e<k>`）→ S9R 下一轮。
- **作业生命周期**与 S9 相同（intent → reserved → launched → `calib-ingest` / `calib watch` 回填），受 `calib.allow_submit` 控制；账本阶段记为 `S9R`。没有 verdict 的作业算 `infra`（基础设施，不算审核结论），超过 `review.infra_retries` 次 → parked（`review:infra`）。
- **远程启动器**：作业配置里的任务路径是本机路径；在另一台主机上跑时，启动器把任务同步过去并在本机作业目录写 `tbsgen-remote.json`（`local_task` / `remote_task`），回填时据此接受远端路径。

| 配置 | 默认 | 说明 |
|---|---|---|
| `review.enabled` | true | 关闭后 S9 标签直接按原规则走 |
| `review.model / agent / environment / agent_allowed_hosts / harbor_root / launcher / jobs_dir` | None（沿用教师路由） | 审核作业的模型、agent、Harbor 环境、agent 阶段放行主机、启动脚本目录与脚本、作业目录 |
| `review.est_usd_per_run` | None（= `calib.est_usd_per_run`） | 账本预留 |
| `review.timeout_sec` | 3600 | 审核任务的 agent 超时 |
| `review.max_revisions` | 3 | 每题最多修订轮数，用完 → dropped |
| `review.allow_human` | false | 显式打开后才允许 `human` 裁决与人工出口 |
| `review.rescore_scoring_only` | true | 只改评分脚本的修订先重评已有交付物，结果不变则保留校准 |
| `review.review_parked` | `["parked:teacher_failed"]` | 也要审核的暂存标签 |
| `review.infra_retries` | 2 | 无裁决 / 无效裁决的审核作业重试次数 |
| `review.trace_digest_kb` | 150 | 每次校准运行 trace 摘要的大小上限 |
| `review.summary_alerts.drop_rate` | 0.5 | `review summary` 的 drop 率提醒阈值 |

`tbsgen review build --run DIR --item ID [--round K] [--out DIR]` 只生成审核任务（不提交），用于回放或手动运行。

`tbsgen review summary --run DIR [--sample N] [--seed S]` 输出本批审核汇总：pass/revise/drop 数量、blocking 的问题类别 / 清单类别分布、drop 原因、每题修订轮数（及直方图）、随机抽 N 道 pass 题并列出它们的审核记录路径（`review/verdict.json`、`runs.json`、各轮 `r<k>/*.verdict.json`）供抽查、黑名单复查的解除/维持记录，以及异常提醒（drop 率超过 `review.summary_alerts.drop_rate`、同一包两次审核结论矛盾、无效裁决、`review:infra`）。

`tbsgen review deliverable --run DIR --item ID --job JOB --src DIR [--note TEXT]` 登记某次校准运行的交付物（其 `/app/output`），供只改评分时重评。

### 5c. 计时项：按机器速度换算，默认只记录

- **只记录、不判定（默认）**：资源类指标（运行时间、内存；`spec.is_resource_metric`）只在 `resource_metric_gating = true` 的题里参与判定（速度本身是论文贡献）。默认 false：`_metrics.check` 记下数值、锁定阈值、换算后的阈值与速度比（`gated: false`, `passed_raw`），但不让这一项失败；S6B/S7 也不用它区分 mid/naive。规格提示词与审核清单 h 都会问这一项。
- **速度换算**：`tests/_metrics.py` 里有一段固定的标准计算（确定性、纯 CPU、几百毫秒，单线程环境，`speed_bench()`，取 5 次中位数）。定阈值时（S6A 在题目自己的沙箱镜像里）测出的耗时写进 `thresholds.json` 的 `speed_bench_sec`；评分时在本机再测一次，速度比 = 本机 / 基准，限制在 [0.25, 8]，在 ±`thresholds.speed_deadband`（默认 0.1）内视为 1。时间类阈值按比例放大（`<=`）或缩小（`>=`），内存类不换算。`metrics.jsonl` 里 `threshold` 仍是锁定的原值（锁定检查照旧比对），换算后的值另记为 `threshold_effective`。旧阈值没有 `speed_bench_sec` 时不换算；锁定状态下重新实测（修订或 regate）会补记。

## 6. 已验证 / 未验证

**已验证（实际跑过）**

- 测试：gen 共 269 个（238 个离线单测，含 S3 全文通道与浏览器批次的 161 个；16 个在 Docker 里执行任务代码的集成测试；15 个用本机真实 ego-browser 打本地测试页的动作脚本 live 测试，CI 上无 ego-browser 时跳过），每次推送由 GitHub Actions CI 运行。
- 集成测试覆盖的情况：断点续跑（中途停止，换新进程继续，已完成阶段不重复调用模型）；预算上限触发暂停，`resume` 后继续；quota 错误触发熔断并暂停；参考解崩溃 → S8 修复 → 恢复；注入泄漏 → L2 阻塞 → 修复后通过；修不好 → 2 轮后丢弃；S6A 截断续写；mid 不可区分时 S6M 只重写 mid；真值隔离探针；筛选丢弃；取不到全文 → 暂存。
- fake 模型端到端干跑：人造条目走完 S1→S9，再用模拟的校准结果走到 S10。
- 打包出的 Harbor 任务：用本机 Docker 实际跑 Harbor，oracle reward=1，nop reward=0（`tbsgen harbor-local`，Dockerfile 换成本地基础镜像）。
- 泄漏门回测：见 `eval/LEAK_BACKTEST.md`（召回 7/7；样本内误报 0/231；留出误报 1/81）。
- 真实论文端到端（6 篇，真实 Opus）：3 篇走完 S1→PKG 并通过 oracle=1 / nop=0；2 篇因 mid 与参考解拉不开被丢弃；1 篇仍在修复。S9 已真实提交学生模型（GLM）校准，每题 1 次，不足以给出难度标签。

**未验证**

- 校准的教师（Opus agent）分支。
- L4b 反解攻击、廉价模型各角色、S9b 变体、DOMF/CODE/MULTI 形态。
- S3 出版商通道（TDM、校园 IP、浏览器）只有用伪造响应的单测；真实下载依赖使用者自己的订阅与出口网络。浏览器 agent 的探索动作与 handoff 只在本地测试页上验证过，尚未在真实出版商/机构站点端到端跑通。
- 专家时长是教师模型的自报值；价格是占位值，美元数字不是账单。

## 6b. 真实论文端到端测试后新增的门与检查

- **评分防篡改**（v2.5，`tests/_metrics.py` 与 `tests/test.sh` 模板）：`run_solution` 让解题程序在独立会话/进程组里以 `nobody` 运行，输出走临时文件；每次调用结束（含超时）后 SIGKILL 整个进程组，再以 `nobody` 身份 `kill(-1)` 并扫描 `/proc` 确认该 uid 无残留进程（仍有残留则该次 rc=-10）。首次运行前把 `/logs/verifier` 改成 `nobody` 不可写（Harbor 挂载为 777）。`test.sh` 先跑 pytest，再由 `python _metrics.py --finalize-reward RC` 最后写 reward：再次清理 `nobody` 进程、锁目录、删除先前的 reward 文件、以 O_EXCL|O_NOFOLLOW 新建 `reward.txt`，有残留则记 0。本地 S7 的 `runner.test` 直接运行同一个 `test.sh`，并在容器里多停 `gen.verifier_settle_sec`（默认 1 s）后读 `reward.txt` 判定通过与否。回归测试：`test_integration.py::test_detached_daemon_cannot_rewrite_reward`（S9R 审核 PoC：解题程序留下脱离的守护进程把 0 改成 1；c4752b1 上失败，reward=1）。
- **计时取中位数**（v2.5）：题目含资源类指标（`spec.is_resource_metric`）时，`run_solution` 对每个隐藏集计时 `thresholds.timing_repeats` 次（默认 3）取中位数；输出与返回码只取第一次（计分那次）。重复运行有单独的时间预算 `thresholds.timing_budget_sec`（默认 300 s，打包时加到题目的 verifier 超时上，计分运行仍有完整的原超时；每次重复的超时为剩余预算）。输出目录和 HOME 都放在 root 所有、0711 的父目录里（程序无法把它们换成符号链接）；重复前，计分那次的输出目录和 HOME 改为 root 所有、0700，重复运行既不能复制也不能改它们；若其中含符号链接、硬链接或特殊文件，则不做重复。每次运行前后对整个文件系统（/proc、/sys、/dev 除外，含 /dev/shm；按 ctime/xattr 比较）、SysV IPC 和内核 key 做快照，快照、封存、哈希的耗时都计入预算，超出即停止。报告的时间不低于第一次的 1/1.5（`MAX_SPEEDUP`），即使还有未覆盖的缓存通道，收益也有上限。以下任一情况只用第一次的时间，原因记在 `_solution_cli_run.timing_repeats_skipped`：任何一次运行改动了计分目录以外的状态、重复的输出与计分那次不同、预算不足。失败的重复记为 +inf。S6A 测量（定阈值）、S7 与打包后的评分用同一个模板，非资源类指标的判定不变。
- **verifier 时间上限**（v2.6）：一次 verifier 会话里，解题程序所有计分运行（每个隐藏集的第一次）共享一个总上限 `_metrics.SCORED_BUDGET_SEC` = min(`package.verifier_timeout`, `gen.test_timeout`) − `package.verifier_overhead_sec`（默认 900 − 120 = 780 s）。measure.py 传入更大的 timeout（例如 1800）时按剩余额度截断；额度用完的调用直接按超时处理（rc −9、耗时 inf），不会再跑到 Harbor 把整个 verifier 杀掉。计分上限 + 计时重复预算 + 开销 ≤ 题目的 verifier 超时，PKG 校验这一点（旧模板没有该上限的包会被判 invalid，需 `tbsgen regate`）。
- **题面写明 verifier 时间上限**（v2.7）：PROG 题打包时在 instruction.md 末尾追加“Verifier time limit”一节（`SCORED_BUDGET_SEC` 秒、沙箱 CPU 数，超时按失败计），幂等。起因：第二批 nqs 题审核 I2 指出 780 s 上限未写明。
- **输出格式一致门 `output_contract`**（v2.7，阻断）：题面里用反引号给出的 CSV 表头（在文件名所在行或其后 3 行内、不越过另一个输出文件名）必须与参考解在 seed 0 写出的同名文件表头列集合一致；不一致 → S8。起因：第二批 nano 题题面写 `material,rate,depth_nm,H_fit_GPa`，评分脚本却按 obs_id/H_fit 合并，照题面做的智能体（含 Opus 教师）4/5 指标为 inf。
- **校准并发上限**（v2.6）：`calib.max_concurrent_jobs`（默认 None = 不限）。全 run 范围内已启动、尚无结果的 S9 校准与 S9R 审核 Harbor 作业数达到上限时，新作业不准备、不预留账本、不写记录，条目置 waiting（reason 含 `queued: slots full`），下一次 `tbsgen run` / `calib watch` 自动重试。
- **`tbsgen regate`**（v2.5）：共享评分模板（`tests/_metrics.py`、`test.sh`）或 S7 门变更后，已打包/已发布的题按这条路径重新过门：在 manifest 里撤下，锁定阈值与数据哈希，从 S6A_run 重新测量（新模板，计时取中位数）→ S6B（锁定，参考解必须过锁定阈值，否则 parked）→ S6C（文件保留）→ S7（含 exploit_valid；只有作弊基线坏掉时走只改 `baselines/exploit_*.py` 的 S8，锁定阈值下也允许）→ PKG（检查数据与运行时阈值没有变化）→ S9。校准记录默认保留（题面、数据、阈值都没变；操作者需用新评分器重评教师先前的交付物作为证据），`--recalibrate` 则归档当前 epoch，从 GLM 重新开始。S9R 审核绑定包哈希，所以新包在 S10 前必须重新审核。
- `truth_isolation`（PROG，阻断）：判分时把 `solution_cli.py` 换成探针，探针尝试读 `/tests` 下除隐藏输入外的所有文件。要求探针经 `_metrics.run_solution` 执行，且什么都读不到。本地判分步骤与 Harbor 一致，以 root 运行在容器内副本上（含 `/app/data`）。
- `cli_contract`（PROG，阻断）：指令必须原样包含 `solution_cli.py <input_dir> <output_dir>`。
- S6B：
  - mid "只差 1–2 项"按判据计数，同名的 public/hidden 指标算一个；
  - 参考在某指标上明显差于 naive 时记 `ref_worse_than_naive`，送去修复。
- S6B 决策与提前止损（`spec.s6b_decision`；规则触发时写 `early_stop` 事件，条目 history 带 `early_stop`）：
  - mid 运行崩溃（非零退出，例如 import 了未挂载的 `solution/solve.py`）→ S6M（规则 `mid_crashed`；失败项计数无意义）；
  - mid 在每个种子上失败判据数 ≥ `max_mid_fail + gen.mid_far_above`（默认 2+2=4），或 ≥ `max_mid_fail+1` 且等于全部判据 → 直接 `dropped: mid_far_from_ref`（规则 `mid_far`），不走 S6M/S5；
  - 其余拉不开 → S6M 一次；之后默认直接丢弃（`gen.back_to_spec_max` 默认 0）；若打开，只有“差一点”（失败 1 到 max_mid_fail+1 项）才回 S5；
  - `ref_worse_than_naive` 且某指标参考比 naive 差 ≥ `gen.ref_gross_ratio` 倍（默认 10）→ 判为参考实现错误，带倍数回 S6A 重写全部代码一次（规则 `ref_gross`，`gen.ref_rewrite_max`=1），仍然如此 → `dropped: ref_grossly_wrong`；倍数较小的仍走 S8。
  - 依据：e2e-prod-20261109 的离线回放（9 篇丢弃题的真实 S6A/S6M/S8 产物重新在沙箱实测）：3 篇入库题首轮 S6B 均为 mid 失败 3/4 判据、S6M 后变成 1 项，规则不误伤；回 S5 的 5 篇无一救回。
