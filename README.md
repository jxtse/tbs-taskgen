# tbs-taskgen：TB-Science 训练题出题管线 v2

把科学论文转成 Terminal-Bench-Science 风格的 Harbor 任务（题面 + 数据生成器 + 参考解 + 隐藏测试），用于 GLM-5.3 的 SFT 数据构造。

```
retrieval/   检索层 tbsretr：OpenAlex / Semantic Scholar / Exa / Crossref / arXiv / Europe PMC → RRF 融合 → 候选论文
gen/         出题层 tbsgen：S1 入库 → S2 筛选 → S3 全文 → S4 可行性 → S5 规格 → S6 生成+按实测定阈值
             → S7 机械门（含 4 层泄漏检查）→ S8 局部修复 → S9 难度校准（GLM 顺序校准 + Opus 教师）→ PKG 打包
docs/        DESIGN.md（设计与决策记录）、evidence/（旧管线数字复算、成本模型）、E2E_LOG.md（端到端测试记录）
```

## 当前状态

见 `docs/STATUS.md`。一句话：离线测试和假模型全流程已通过，真实论文端到端测试正在进行；还没有量产。

## 快速开始

需要 `uv`、Docker（沙箱跑生成器/参考解/基线）、Python 3.12。

```bash
cd retrieval && uv sync && uv run pytest -q          # 26 个离线单测
cd ../gen   && uv sync && uv run pytest -q           # 36 个单测+集成测试（集成测试需要 Docker）
docker build -t tbsgen-base:py311-sci-v2 -f gen/docker/Dockerfile.base gen/docker   # 沙箱镜像
uv run --directory gen python -m tbsgen.dryrun --out runs/dryrun   # 假模型端到端干跑
```

真实运行：

```bash
cd gen
uv run tbsgen ingest --run runs/R --candidates candidates.jsonl --config my.toml
uv run tbsgen run    --run runs/R --config my.toml
uv run tbsgen status --run runs/R ; uv run tbsgen report --run runs/R ; uv run tbsgen ledger --run runs/R
```

`my.toml` 必须显式写每日预算（`[budget] daily_usd_total`、`daily_calls_per_model`），否则真实模型调用一律拒绝。

## 机器相关配置

| 环境变量 | 默认 | 用途 |
|---|---|---|
| `TBS_WORKSPACE` | `/Volumes/Storage/hermes/workspace` | 工作区根 |
| `TBS_OLD_PIPE` | `$TBS_WORKSPACE/tbs-datapipe-20261015` | 旧管线：全文缓存 `literature/txt`、官方题文本 `tasks_text`（去重用） |
| `TBS_HARBOR_ROOT` | `$TBS_WORKSPACE/tbs-aliyun` | Harbor 作业脚本（S9 校准） |
| `TBS_LEAK_REFS` | `$TBS_OLD_PIPE/literature/tbs_task_refs.json` | 官方题来源论文列表，检索时排除 |

模型网关：`gen/src/tbsgen/config.py` 中 `models.opus.url`（Anthropic Messages 兼容端点），可在 config.toml 覆盖。检索 API key 只从环境变量读取（`OPENALEX_API_KEY`、`SEMANTIC_SCHOLAR_API_KEY`、`EXA_API_KEY_SECONDARY` 等，见 `retrieval/README.md`），仓库内不存任何 key。

## 不在仓库里的东西

- 运行目录 `runs/`、`state/`（账本、缓存、沙箱产物）
- `retrieval/eval/raw/`（10 个目标的原始检索响应，16 MB）
- 旧管线语料、官方题文本、论文全文（受版权/许可约束）

## 协作约定

- `main` 保持测试全绿；改动走分支 + PR。
- 改提示词时同步提高 `gen/src/tbsgen/prompts.py` 里的版本号（写入每题 provenance）。
- 新的端到端测试结论追加到 `docs/E2E_LOG.md`，不要覆盖旧记录。
