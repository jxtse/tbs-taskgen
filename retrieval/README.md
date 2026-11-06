# TB-Science 检索层 v2（`tbsretr`）

给定一个检索目标（方法主题或题型描述 + 若干关键词查询 + 可选种子 DOI），本层并行调用 7 个检索源，把结果归一化、去重，用 RRF 按名次融合，再做规则预筛，输出前 N 篇候选论文和一份代码/数据/文档资源清单。**本层不调用 LLM**，结果交给下游的 LLM 打分（默认送 60 篇）。

实测结论与数字见 `eval/REPORT.md`。

## 目录

```
src/tbsretr/
  config.py    默认配置（保守预算）；可用 config.toml 覆盖
  ledger.py    sqlite 账本：限速（跨进程）、重试/退避、429 冷却、每日/每月预算、响应缓存（不含 key）
  sources.py   7 个源的适配器：OpenAlex / Semantic Scholar / Exa / Tavily / Crossref / arXiv / Europe PMC
  model.py     Target / Paper / Resource；arXiv 学科路由、生物医学路由、Crossref 精确词抽取
  fusion.py    跨源去重（DOI/arXiv/OpenAlex/S2/标题）、RRF、规则预筛、资源融合
  pipeline.py  Retriever.retrieve(target) 编排：一阶段检索 → 种子 → 引用扩展/相似推荐 → 回查补全 → 融合
  cli.py       命令行入口 tbsretr
tests/         离线单测（tests/fixtures 是 2026-11-05 录制的真实响应，已去除 key）
scripts/record_fixtures.py  重新录制 fixtures
eval/          10 个目标的实测：run_eval.py / analyze.py / coverage.py / REPORT.md / 人工标签
state/         运行时生成：ledger.sqlite（账本）、cache.sqlite（响应缓存）；已加入 .gitignore
```

## Exa 后端

`sources.exa.backend` 有两种取值：

- `pool`（默认）：请求发到本地搜索池服务 `POST {pool_url}/v1/search`，`pool_url` 没有默认值，用配置或环境变量 `TBS_SEARCH_POOL_URL` 设置。服务持有 Exa key 并在多个 key 之间轮换，本进程不读 key。请求固定 `engine=exa`，不回退到 Tavily；`type`、`category`、`includeDomains`、highlights 长度通过 `exa_options` 原样传给 Exa。服务把 Exa 原始结果行放在 `raw` 里返回，解析与直连完全相同；费用按服务返回的 `usage.cost_usd`（即 Exa `costDollars.total`）记账。池服务地址未设置时 Exa 自动跳过。
- `direct`：直连 `https://api.exa.ai/search`，key 从 `key_env` 读取。离线测试的录制数据用这个后端。

搜索池服务需要支持 `exa_options` 字段，接口约定：请求 `{query, limit (1–20), engine, include_contents, exa_options}`，响应 `{provider, results: [{url, title, raw}], usage: {cost_usd}}`。

2026-11-06 对照：10 个目标的 40 条 Exa 查询经池服务重跑，与 11-05 直连录制结果相比，URL 重合率为 code 88%、data 84%、docs 84%、paper 93%，论文 DOI 数 79→80，费用 $0.28（40 次）。差异来自两次检索相隔一天的 Exa 索引变化，没有出现解析或字段缺失问题。

## 用法

```bash
cd retrieval
uv sync                                   # 独立 venv，只用标准库 + pytest
uv run tbsretr status                 # 各源是否可用（只显示是否有 key，不显示 key 值）+ 今日用量
uv run tbsretr run --targets T.json --out out/ [--only id1,id2] [--sources openalex,crossref] \
                                         [--seed-mode auto|pseudo|given|none] [--offline]
uv run tbsretr usage [--run RUN]          # 账本汇总（真实调用数、缓存命中数、美元、credits、429 次数、平均延迟）
uv run pytest -q                          # 离线单测
```

`T.json` 是目标列表，每个目标的字段如下：

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | ✓ | 例如 `method:column-generation-...` / `task:qsm-reconstruction` |
| `queries` | ✓ | 2–6 条短关键词查询（由上游的查询词生成器给出）。OpenAlex 用前 4 条，S2 用前 3 条，arXiv 和 Europe PMC 用前 2 条 |
| `description` | | 一段英文的方法或任务描述，供 Exa 的论文通道做语义检索 |
| `topic` / `name` | | 一个英文短语，供 Exa/Tavily 的代码、数据、文档通道使用 |
| `field` / `libs` | | 用于 arXiv 和 Europe PMC 的路由；`libs` 也用来抽取 Crossref 精确查询词 |
| `seed_dois` | | 种子论文，用于 OpenAlex 引用扩展和 S2 相似推荐。不给时，自动取 OpenAlex 关键词结果经预筛后的前 3 篇作为“伪种子” |
| `exclude_dois` | | 本目标额外要排除的 DOI |
| `openalex_filter` | | 额外的 OpenAlex 过滤条件，例如 `primary_topic.subfield.id:3104` |
| `arxiv_cats` / `biomed` | | 手动覆盖自动路由 |

Python 调用：

```python
from tbsretr import load_config, Retriever, Target, to_json
r = Retriever(load_config())
out = to_json(r.retrieve(Target(id="x", queries=["..."], description="...", seed_dois=["10.xxx/..."])))
out["papers"]      # 前 60 篇：doi/arxiv/openalex/s2 id、标题、摘要、年份、venue、oa_url、引用数、
                   # sources（命中的源）、ranks（各个列表里的名次）、rrf、code_links/data_links/code_words
out["resources"]   # 代码/数据/文档资源：url、kind、found_by（哪个源哪个通道排第几）、from_papers
out["sources"]     # 每个源的调用列表数、返回条数、状态（ok / skipped / stopped: 预算或冷却原因）、错误
out["dropped"]     # 预筛剔除的计数：综述、非英文、无摘要、TBS 出处、类型、年代
```

## 各源分工（默认配置，依据 eval/REPORT.md 的实测）

| 源 | 每个目标的调用 | 作用 |
|---|---|---|
| OpenAlex | 4 次 search + 1 次种子回查 + 1–2 次引用扩展 + 1–3 次 DOI 批量回查（list） | 主关键词检索。引用扩展取种子论文共同引用的参考文献。回查给其他源找到的 DOI 补摘要、OA 链接、类型和语言 |
| Semantic Scholar | 3 次 search + 1 次 paper/batch（另有约 40% 的 429 重试） | 论文检索；batch 回查补 `openAccessPdf` 和 S2 id。recommendations 实测几乎没有增益，默认关闭（`recommend=True` 可开启）。没有 key 时跳过，并在输出里标为 `skipped` |
| Exa | 4 次（code / data / docs / paper，paper 通道用 `category=publication`） | 代码仓库、数据集、软件文档；补充论文召回。默认经本地搜索池服务调用（见下文“Exa 后端”） |
| Tavily | 0 次（默认关闭） | 2026-11-05 起按项目决定从默认管线移除：实测质量明显低于 Exa（可用资源 35 vs 137，噪声 41 vs 0）。适配器保留，`sources.tavily.enabled=True` 可重开做对照 |
| Crossref | 3 次 search + 0–1 次 DOI 回查 | 兜底源。用工具名或作者名等精确词查询（例如 `V-SHARP ...`、`VerifyBamID ...`），同时提供 DOI 和出版商元数据。走 polite pool（带 mailto），间隔 ≥0.34 s |
| arXiv | 2 次（只在路由命中 astro-ph / physics / cond-mat / quant-ph / stat / math / cs 时调用） | 预印本。单连接，每 3.1 s 一次 |
| Europe PMC | 2 次（只用于生物医学目标） | 生物医学论文；OA/PMC 全文链接 |

融合规则：每个源的每类列表（例如 `openalex.search`）内，一篇论文按最佳名次计 `w/(60+rank)`，其余命中按 25% 计入；然后跨列表求和。权重在 `config.py: fusion.weights`。预筛规则沿用早期方法文献检索的排除口径：剔除综述/教程/观点类，剔除非英文（语言字段或拉丁字符占比 < 85%），摘要不足 200 字符的论文不剔除，只把分数 ×0.5 并标记 `abstract_missing`（`drop_no_abstract=True` 时改为硬剔除），剔除 `tbs_task_refs.json` 里的 TBS 题目出处 DOI，剔除勘误/社论/数据集等类型，剔除 1985 年以前的论文。资源清单另外剔除 URL 含 `terminal-bench-science` 的条目：TBS 仓库本身会泄题，实测 Exa 和 Tavily 都返回过它。

## 凭据

- key 只从环境变量读取，建议由密钥管理工具在运行时注入。**不要**把 key 写进仓库或任何文件。
- 环境变量名可在配置里改（`sources.<name>.key_env`），默认值：
  - OpenAlex：`OPENALEX_API_KEY`
  - Semantic Scholar：`SEMANTIC_SCHOLAR_API_KEY`
  - Exa：默认 `backend = "pool"`，本进程不需要 Exa key；`backend = "direct"` 时读 `EXA_API_KEY_SECONDARY`
  - Tavily：`TAVILY_API_KEY_SECONDARY`（同上）
- Crossref、arXiv、Europe PMC 不需要 key。OpenAlex 没有 key 也能跑，但同一 IP 共享每天 $0.10 的额度。S2、Exa、Tavily 没有 key 时会自动跳过。
- 防泄露措施：
  - S2、Exa、Tavily 的 key 只放在请求头里。
  - OpenAlex 的 `api_key` 在 URL 里，所以写缓存键、写账本、写日志之前都会先删掉这个参数（`util.strip_secret_params`）。
  - 错误响应体会先经过 `util.redact` 再记录。
  - 有单测检查账本和缓存里不出现 key。
  - 2026-11-05 用 6 把真实 key 扫描了全部产物、缓存和账本，0 命中。

## 预算与限速（默认值，`config.py` 可改）

| 源 | 请求间隔 | 每日上限 | 金额 / 额度上限 | 429 处理 |
|---|---|---|---|---|
| OpenAlex | 0.2 s | 2000 次 | $0.50/天（UTC 日）；响应头 `X-RateLimit-Remaining-USD` < $0.10 时停止 | 退避后冷却 1 h |
| S2 | 3.0 s（全部接口共享） | 1500 次 | – | 5/10/20/40/80 s 退避；仍失败则冷却 15 min |
| Exa | 0.3 s | 80 次 | $0.60/天、$9/月；按响应里的 `costDollars` 记账 | 冷却 1 h |
| Tavily | 1.0 s | 30 次 | 30 credits/天、900 credits/月（免费档每月 1000） | 冷却 1 h |
| Crossref | 0.34 s | 3000 次 | – | 冷却 30 min |
| arXiv | 3.1 s | 1000 次 | – | 冷却 1 h |
| Europe PMC | 0.3 s | 2000 次 | – | 冷却 30 min |

- 超出预算时抛 `BudgetExceeded`，该源在本次运行内停止，输出里标为 `stopped: ...`，其他源继续。
- 5xx 或网络错误按指数退避重试（3–5 次）。
- 计数和冷却都存在 sqlite 里，跨进程共享。缓存只命中、不计入预算。缓存默认保留 30 天。

## 局限

见 `eval/REPORT.md` 的“局限”一节。主要几点：
- 实测的相关性标签由单人按标题和摘要判断，属于主观判断。
- S2 的 key 在 1 次/3 s 的速率下仍有约 40% 的请求先返回 429。
- Europe PMC 在实测时有约 1 小时的 503 故障。
- arXiv 的查询形式在实测中途修过一次，已用修正版重跑了全部 arXiv 列表。
