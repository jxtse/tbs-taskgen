# tbstrain 科学任务训练数据管线

从 taskgen 发布清单和 Harbor 校准作业生成 MinT SFT JSONL。每道入选题都保留难度计数、题目审核和轨迹审核结果，每条导出行都能追溯到 trial、片段和中间记录。

## 运行

在 `trainpipe/` 使用 Python 3.12 和项目虚拟环境。复制示例配置，在本地配置中填写运行目录、分词器、私有 wire profile 和评审模型。TOML 中的相对路径以配置文件目录为基准，命令行传入的路径以当前目录为基准。

```sh
cp examples/pipeline.toml examples/config.local.toml
.venv/bin/tbstrain run --config examples/config.local.toml
.venv/bin/python -m pytest -q
```

`run` 执行至本地验证。评审需要配置的模型端点。测试使用手写合成会话、进程内评审函数和 shell 替身，全程离线。运行目录、真实题面和轨迹、压缩指令、端点、账号、凭证及提交回执都应保存在私有存储。默认产物位于 Git 忽略的 `state/trainpipe/`。

| 命令 | 产物与判定 |
|---|---|
| `ingest --run DIR ... --job DIR ...` | `tasks.jsonl` 记录题包内容哈希和发布记录，`trials.jsonl` 记录会话路径、模型、异常、时长、费用与验证器路径。额外作业目录可省略。 |
| `difficulty` | `difficulty.jsonl` 给出学生及教师的有效尝试数、通过数、学生通过率和无效尝试数。同一校准作业优先使用原生 trial，避免重复计数。 |
| `filter-tasks` | `filtered.jsonl` 逐项检查 S10 已完成、审核通过、标签可信、题面卫生及留出集。相同 DOI 或标题词集 Jaccard ≥ 0.8 时优先保留通过次数较多的题。可选模块 `tbsgen.statement` 缺失时记录跳过。 |
| `screen` | `screen.jsonl` 给出 keep、strip 或 drop。检查得分、结束状态、模型身份、隐藏材料、离线任务联网及调用配对。只剥离可确认无实质结果的探查及其返回，保留同轮独立的并行调用。清理后的块记录在 `segments/`。 |
| `judge --kind task` | 对题面、评分程序、指标规格分配文本窗口，判断科学性、公平性、完整性和估计难度，记录 keep、flag 或 drop 及原文证据。 |
| `judge --kind trajectory` | 对清理后的轨迹分窗检查作弊、推理质量和浪费回合比例。任一窗口有 blocking 发现就淘汰，轨迹质量取各窗口最低分。 |
| `select` | `selection.jsonl` 和 `selection_report.md` 列出每层、领域、harness 的数量、监督 token 和每条排除原因。 |
| `export --target mint-sft` | 写出 `train.jsonl` 和逐行对应的 `train.sources.jsonl`。超长行记入账本，报告列出导出长度的 p50、p90、最大值。 |
| `validate FILE` | 检查 UTF-8、数值、消息及工具 schema、调用顺序、loss、长度与源块。与 FILE 同目录的 `<stem>.sources.jsonl` 用于逐行核验。失败返回非零，行号和原因在 `validate.ledger.jsonl`。 |
| `submit` | 本地校验通过后上传、读取平台报告，再执行训练 dry-run。加 `--confirm` 才会启动训练并读取 show 和 events。 |
| `rollout` | 根据已选题写出 `rollout/job.json` 和 `rollout/launch.sh`。作业由操作员启动，完成后用 `ingest --job` 纳入下一轮。 |

各子命令均支持 `--config FILE`。每阶段以输入内容、相关配置、上游产物键和阶段版本计算 SHA-256 键。完成结果位于 `<out>/.stages/`，同一输出目录用文件锁串行发布。重复执行会复用已有结果。当前阶段的摘要在 `reports/<stage>.json`，逐项原因在 `<stage>.ledger.jsonl`。中途失败的阶段可以重跑，已验证的评审请求缓存在 `.judge-cache/`。

## 难度与配比

学生通过率 ≤ 1/3 且教师至少通过一次为 hard，(1/3, 2/3] 为 medium，> 2/3 为 easy。所有模型均未通过为 unsolved。证据不足以定层的题标为 unclassified。环境、安装故障和无提交的 API 故障剔除，有提交且已评分的超时按实际得分统计。训练轨迹另需通过结束状态检查。

题目和轨迹评审都须 keep。easy 还需质量 ≥ 4。默认目标为 hard 45%、medium 40%、easy 15%。每题最多选两条，优先不同 harness，再取监督 token 较短者。按最大余数分配整数名额，并检查每题监督 token 占比上限。数据不足时会缩小入选集。默认 2% 上限意味着少于 50 道题不能构成非空训练集；小规模合成测试在本地配置中显式放宽上限。

## 私有 wire profile 和请求比对

从同版本的首请求生成一个新目录。所有输入的 system、tools 字段必须逐字节一致。提取器从 User-Agent 读取 Claude Code 版本；只有 body 的文件可使用 `--version VERSION`。系统提示、工具定义、原始会话和抓包均放在仓库外。

```sh
tbstrain wire-profile extract --from "$CAPTURE_A" "$CAPTURE_B" --out "$PROFILE"
tbstrain wire-check --captures "$CAPTURE_ROOT" --config examples/config.local.toml
tbstrain validate "$DATASET" --reference "$ACCEPTED_ROW" --config examples/config.local.toml
```

`[export.wire.claude-code]` 的 `dir` 指向该目录。里面的 `system.json` 保存有序 text 块，`tools.json` 保存完整 Anthropic 工具列表，`profile.json` 记录 harness、version 及以文件名为键的 sha256。可添加 `compaction.txt` 以提供训练摘要用的指令。每次加载都核验两个 JSON 文件的摘要。配置、版本或内容变化后重新执行筛选和选择。

Claude Code 以 parentUuid 重建每个活动窗口，合并 assistant 分片、补回并行结果。附件内部用单换行，连续附件用空行连接。归属提醒保留标签并折入用户回合。压缩窗口从摘要、信封附件和保留片段恢复状态。导出将 profile 内的系统块以空行连接为首条 system 消息，保留工具顺序、参数以及未调用的工具。

`wire-check` 递归查找两类抓包和相应 trial 的原生 session。首请求使用参考协议的请求控制参数，比较除 metadata 外的完整 body。压缩后的抓包依照代理规则移除注入的提示和 cache_control、折回单文本块，逐条比较消息。未指定配置时使用第一份首请求提供公共 system 和 tools。空目录、缺失会话或任何差异都会返回非零状态。

Codex 从 session_meta 和 turn_context 读取 instructions、base_instructions 和工具定义，也支持 additional_tools 记录。Pi 从 request 的系统消息和工具列表或 session/context 的 systemPrompt 和 tools 读取。显式空工具列表表示该次没有提供工具。缺少指令或工具时，使用对应的 `[export.wire.codex]` 或 `[export.wire.pi]` 目录补齐缺失字段。这两个目录沿用相同的文件和摘要格式。平台工具需要非空的 JSON 参数 schema，原生 custom 工具可在 profile 中按名称提供对应函数定义。

`validate --reference` 接受一个 JSON 行对象，检查顶层字段及类型，并按角色比较允许的消息字段与类型。参考行应覆盖待检查的角色和可选字段。普通验证和来源核验仍会执行。

## 平台格式

- 一行包含 `messages`、OpenAI function 形式的 `tools`、`loss_mask` 和 `meta`。并行工具结果按前一条 assistant 的调用 ID 顺序紧邻排列。行首为系统提示。末尾截到最后一个有内容、推理或工具调用的 assistant，裁去的平台消息条数写入 `meta.render_notes.dropped_tail`。
- Claude Code 在 `compact_boundary` 切段，合并同一消息 ID 的块。前段保留真实工具返回及系统提醒，再追加操作员提供的压缩指令和从下一段还原的摘要。续段按摘要、信封与保留片段恢复上下文。使用掩码训练时，保留片段和续跑前缀中的 assistant 均标记 loss 0。
- Codex 使用 Roller 的 ACTF 转换与 SFT 校验。发生压缩时需要捕获的根参与者调用、manifest 及原生边界。逐次匹配模型输出和上下文，仅在前缀一致时拼接。只有代理请求元数据的压缩会话记入跳过原因。Pi 支持原生 message 和 journal 中的 message_end，截至最后完整 assistant 答复，记录裁去的事件数、消息数和最后完成位置。
- 托管训练使用 `all_assistant_messages`，会忽略 `loss_mask`。选用该目标时，筛选阶段实际移除应屏蔽的旧 assistant 轮及其工具结果。需要保留完整重放上下文并使用掩码的 SDK 导出可设置 `train_on_what = "loss_mask"`，然后重新运行筛选和选择。验证默认拒绝 loss 为 0 的 assistant，`--allow-masked-assistant` 可用于检查其他训练方式生成的文件。
- 用配置的 tokenizer JSON 对角色、推理、工具 schema、调用、结果和正文估算模板 token。平台端最终以数据集报告为准。`submit` 要求 available、0 fatal、0 warn，`--allow-warn` 仅放行 warning。上传幂等键使用文件 SHA-256，训练幂等键绑定数据集版本、基座模型与超参文件哈希。CLI 原始 JSON 及失败响应位于 `submit-responses/`，报告记录是否观察到平台忽略 loss_mask。

## 配置

完整示例见 [pipeline.toml](examples/pipeline.toml)。

| 配置节 | 字段 |
|---|---|
| 顶层、`ingest` | `out` 指向专用私有产物目录。`runs`、`jobs` 为路径列表。 |
| `difficulty` | `student_model`、`teacher_model` 可显式指定；默认从发布校准角色推断。`hard_max`、`medium_max` 指定闭区间上界。 |
| `filter` | `review_allow` 默认仅 pass。`exclude` 接受官方基准 ID、DOI 或标题。 |
| `screen` | `models` 按 harness 指定期望模型，默认取 trial 配置。`task_dirs` 为容器内任务目录。`allow_timeout` 默认关闭。`hint_mode` 默认 exclude；strip 需在私有 `hint_texts` 中提供可逐字移除的提示，残留原文或无法定位提示即淘汰。`resume_marker` 可指定续跑标记。 |
| `judge` | `base_url`、`model`、`api_key_env` 配置 Messages API。`concurrency` 默认 64，可设 128。`timeout` 以秒计，`retries`、`backoff` 控制 429 和 5xx 重试。`window_chars`、`max_tokens` 限制输入窗口和输出。`input_tokens_per_million`、`output_tokens_per_million` 为每百万 token 美元单价。 |
| `select` | `shares` 指定三层权重，`per_task`、`max_token_share`、`easy_quality` 和 `seed` 指定题内数量、监督占比、easy 质量及同分顺序。 |
| `export` | `tokenizer` 必填。`compaction_instruction` 可兼容指定私有指令文件，profile 中的 `compaction.txt` 优先。`max_length` 默认 194560。`train_on_what` 默认 all_assistant_messages。 |
| `submit` | `cli` 指定程序，`cli_env` 指定可覆盖程序路径的环境变量。`config_file` 为训练超参 TOML/JSON，`base_model` 选择基座，`note` 最多 80 字符，`timeout` 默认为 30m。 |
| `rollout` | `harness`、`model`、`attempts`、`concurrency`、`job_name`、`jobs_dir` 选择作业参数。`agents.<harness>` 和 `environment` 透传 Harbor 配置，`harbor_cli` 指定启动脚本使用的命令。 |

## 增加 harness

在 `ingest.session_info` 中添加文件模式及 agent 名称识别，在 `native.CONVERTERS` 注册转换器。返回逐段的文本、thinking、工具调用和结果块、观测到的模型名，以及裁剪记录。为新格式补齐原生工具行为审核，并在 rollout 的 harness 选择及本地配置中登记 agent。用手写小会话覆盖模型不匹配、孤立工具、末轮不完整和一次完整导出。Roller 派生代码的版本与许可见 [NOTICE](src/tbstrain/vendor/roller/NOTICE)。

