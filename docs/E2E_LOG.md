# 端到端测试记录

按时间追加，不覆盖旧记录。

## 2026-11-06 e2e-iter1：6 篇真实论文，S1→PKG 真实 Opus，3 题真实 GLM 校准

术语：**参考解**（reference，`solution/solve.py`）、**mid 基线**（参考解只去掉一个关键推导节点）、**naive 基线**（去掉多个节点）、**nop**（什么都不交）、**隐藏集**（PROG 题的 `tests/hidden/`，只有判分时由 `solution_cli.py` 读取）。美元数字一律按管线占位价（输入 5 / 输出 25 美元每百万 token）计算，不是账单；GLM 校准的美元数取 Harbor 记录的 `cost_usd`，未结束的按预留额计。

### 论文与结果

| 条目 | 论文（DOI） | 领域 | 最远阶段 | 结局 | Opus 次数 | Opus $ | 修复轮 | 本机 oracle/nop | 沙箱 oracle/nop |
|---|---|---|---|---|---|---|---|---|---|
| 196db | 10.1007/s10182-025-00550-5 双变量 Fay–Herriot + VAR(1) 时间效应 | 小域估计 | S9 | GLM 校准中 | 9 | 3.62 | 2（含 1 次因管线修复重置） | 1 / 0 | 1 / 0 |
| d7c20 | 10.1186/s13059-020-1950-6 DecontX | 单细胞 | S9 | GLM 校准中 | 7 | 2.01 | 0 | 1 / 0 | 1 / 0 |
| e9e2a | 10.1002/net.20212 RCESPP 精确定价（VRP 列生成） | 运筹 | S9 | GLM 校准中 | 10 | 5.05 | 1 | 1 / 0 | 1 / 0 |
| d1df8 | 10.1002/qj.2236 迭代集合卡尔曼平滑 IEnKS | 资料同化 | S6A_run（修复中） | 见下 | 11 | 6.31 | 1 | – | – |
| 659ec | 10.1002/mrm.24998 iSMV 背景场去除 | MRI/QSM | S6B | 丢弃 mid_not_separable | 6 | 3.86 | 0 | – | – |
| 6642e | 10.1557/s43578-021-00386-0 纳米压痕热漂移校正 | 材料 | S6B | 丢弃 mid_not_separable | 7 | 2.80 | 2 | – | – |

（Opus 次数含 3 次被我手动中断、按预留额记账的调用。以上 6 篇旧管线都出过题，便于对照。）

合计（写本节时）：Opus 51 次 ≈ $24（S5 14 次 $4.0、S6A 13 次 $12.2、S6C $3.1、S6M $0.5、S8 $3.4）。S6A 是大头（单次输出 1.7–4.5 万 token）。

### 发现的失败模式与修复

1. **S6A 输出被截断**（修复）。第一篇论文 S6A 在 `max_tokens=32000` 处被截断（out_tok 恰为 32000），只解析出 `generator/generate.py`，条目被丢弃，原始回复也没留存。修复：保存每次原始回复；截断时丢弃最后一个不完整文件块；缺文件时追加 1 次续写调用，只要缺的文件；S6A 上限改为 64k。之后 S6A 单次 1.7–4.5 万 token，未再截断。
2. **生成的判分脚本自己以 root 执行 agent 程序**（修复，严重）。第一版 `measure.py` 用自己的 `subprocess.run` 运行 `solution_cli.py`，没走 `_metrics.run_solution` 的降权 + 锁 truth。另外 `_metrics` 的锁只锁 `dirname(input_dir)`：隐藏集是 `/tests/hidden/set1` 时，锁的是 `/tests/hidden`，`/tests/truth` 仍然可读。实测：一个直接读 `/tests/truth/*.npz` 输出 oracle 答案的 `solution_cli.py` 在本机 Harbor 拿到 reward 1。修复：
   - 锁覆盖整个 `/tests`（隐藏输入除外），并新增 `hidden_sets()`。
   - 新增阻断门 `truth_isolation`：用一个探针 CLI 尝试读 `/tests` 下所有文件，要求它必须经 `run_solution` 执行，且什么都读不到。
   - 本地判分步骤改为与 Harbor 一致：以 root 运行在容器内副本上。
   - S6A/S8 提示词写明 `_metrics` 的确切签名。

   修复后同一作弊解 reward 0，参考解 reward 1。S8 第一次修复时 Opus 猜不到函数签名，写出了"多种调用方式都试"的防御代码，结果仍然失败；提示词补签名后一次修好。
3. **指令与判分的 CLI 约定不一致**（修复）。指令写 `solution_cli.py --data <csv> --out <dir>`，判分按位置参数调用，照指令做的 agent 会全挂。新增阻断门 `cli_contract`；S6C 提示词要求原样给出调用行。
4. **阈值过松，naive 能过**（修复）。规则是"非区分指标 = 参考解变差 50%"，于是：
   - Fay–Herriot：RMSE 比参考 1.00、naive 1.48，阈值定在 1.6，naive 基线过了所有精度指标；
   - DecontX：Pearson 参考 0.996、naive 0.89，阈值定在 0.49。

   修复：若 naive 明显更差，阈值取参考与 naive 的对数中点，但不比参考变差 `min_ref_margin` 更紧。结果：Fay–Herriot 收到 1.3，DecontX 收到 0.89。
5. **mid 判定把公开集与隐藏集的同名指标重复计数**（修复）。RCESPP 的 mid 基线没过 `public_optimal_frac`、`hidden_optimal_frac`、`hidden_mean_rel_gap` 三项，按规则"mid 只差 1–2 项"被判不可区分，回到 S5 重新生成，花掉一整轮 S5+S6A。实际上这只是 2 个判据。修复：同名的 public/hidden 指标算 1 个判据（S6B 与 mid 门都改）。之后 RCESPP 通过。
6. **mid 不可区分时代价过高**（修复）。原先直接回 S5，重写规格和全部代码（≈$1.4/篇）。新增 S6M：只让 Opus 换一个 `mid_drop` 节点重写 `baselines/mid.py`（≈$0.12），仍不行再回 S5。仍然丢弃的两篇（iSMV、纳米压痕）在新规则下重算依旧不可区分：mid 挂 3–4 个不同判据，说明所选节点过于根本。丢弃是对的，不是门的问题。
7. **参考解在某指标上比 naive 差**（新增检查）。IEnKS 的隐藏集上参考解发散，filter RMSE 4.19，naive 只有 3.4–3.5，而阈值规则照样给出两者都能过的阈值。新增 S6B 状态 `ref_worse_than_naive`，走 S8 修复。
8. **参考指标为哨兵值**（修复）。判分脚本对缺失输出返回 1e6，参考解在该指标上就是 1e6，阈值被定为 1.5e6。现在 ≥1e5 的"≤"指标视为参考缺输出，走修复。附注：这次 1e6 本身是我在修 #2 时引入的回归——本地判分容器漏挂了 `/app/data`，而 Harbor 判分时有。已修复并补回归测试。
9. **初筛误杀方法论文**（修复）。iSMV（MRI 背景场）规则分 0，纳米压痕漂移校正因检索记录没有摘要得 -1，两篇都在 S2 被丢弃。修复：补方法类词表；无摘要时 S2 放行，到 S4 用全文开头复筛。
10. **S9 真实提交链路**（新增）。原 MVP 只写配置。现在：
    - 每次提交先在账本按 `est_usd_per_run` 预留，超总额即拒绝；
    - 用 tmux 后台启动；
    - `calib-ingest` 以 Harbor 的 `finished_at` 判断完成（`result.json` 在作业开始时就写了），并用 `cost_usd` 冲销预留额；
    - 支持 `teacher_available=false`：教师线路额度用尽时，GLM 首次失败即记 `parked:teacher_unavailable`；
    - 新增全运行级预算上限 `run_usd_total`/`run_calls_per_model`；
    - 每题调用上限只计出题调用。

    远端沙箱不能执行 Dockerfile，所以配置了"已准备模板目录"却没有该题模板时拒绝提交。本次有一题在模板就绪前被自动提交，环境启动即失败，没有用到模型。
11. 其他：
    - 新增 `tbsgen harbor-local`（本机 oracle/nop）和 `tbsgen reset`；
    - 检索层 `selected` 格式可直接作为候选输入；
    - harbor-local 会清掉外部的 `PYTHONPATH`/`DOCKER_CONFIG`（前者让 Harbor 导入失败，后者让 docker compose 插件消失）。

### 产出题目的质量（读过文件后的判断）

- **196db 双变量 Fay–Herriot（`items/doi-196db5a3b2c31394/pkg`）**
  - 优点：要求自己推导 8 个方差参数的 REML 得分与 Fisher 信息，写 Fisher scoring，给 EBLUP 和 Prasad–Rao 型 MSE（G1+G2+2G3）。3 个隐藏集的 (D,T)、φ、方差量级、相关符号、缺失比例都不同；规格估 5 h。
  - 对比：旧管线同一论文的题是"给区域面板数据、交 CRPS 概率预测"，专家时长标 0.5 h，指令里还写明了"oracle 用真实 DGP"。新题更接近论文本身的方法。
  - 问题：
    - 指令把 TBFH-VAR1 模型完整写出（需要推导的是估计和 MSE，不是模型本身）；
    - `hidden_mse_calib_logratio ≤ 0.079` 只比参考的 0.067 宽约 18%，对正确但数值细节不同的实现可能偏紧；
    - 精度类指标靠 naive 上限收紧到 1.3，mid 只在 φ 误差与 MSE 校准两项上被拦住。
- **d7c20 DecontX（`items/doi-d7c20d5934a9c912/pkg`）**
  - 优点：需要实现变分推断（Beta 先验超参数的 Newton 更新、每细胞污染比例、去污计数矩阵）。隐藏集换簇数、基因数、不均衡簇和重尾污染。
  - 问题：
    - 指令给出了完整生成模型，并点名 DecontX 方法，接近"按论文实现"，不需要自己选方法；
    - `contam_rmse` 阈值 0.023 而参考 0.015，mid（不估 η）在全局 RMSE 上与参考相同，只在去污矩阵和标记基因残留上被拦；
    - 专家时长估 4.5 h，我认为熟悉 DecontX 的人 2–3 h 可完成。
  - 对比：旧题偏启发式（按 lane 估泄漏比例）。
- **e9e2a RCESPP 精确定价（`items/doi-e9e2a454b2bc57f7/pkg`）**
  - 优点：三种变体（CAP、带途中载荷约束的 DC、TW）都要求精确最优，判分独立重算可行性与成本。隐藏集更大、约束更紧，所以必须实现带支配规则的标号算法，暴力法过不了。
  - 问题：
    - `hidden_feasible_frac ≥ 0.57` 与 `hidden_dc_optimal_frac ≥ 0.5` 很松（参考为 1.0，mid 也是 1.0），实际起作用的是 `hidden_optimal_frac ≥ 0.83` 和 `hidden_mean_rel_gap ≤ 2.3e-9`；
    - 后者近乎"必须全对"，接近二元判分。

### 未完成 / 未验证

- 3 题的 GLM 校准结果待回收，结论在下一节补充。
- 教师（Opus agent）分支未跑：校准线路上的 Opus key 额度已用尽（探测返回 429），只能跑 GLM 分支。
- L4b 反解、廉价模型角色仍未接。"专家时长"仍是 Opus 自报，未经人工核实。


## 2026-11-06（续）代码审查（PR #2）后的修复与新发现

### 审查意见处理（逐条见 PR #2 评论）

1. **（阻断）符号链接让 truth 重新可读；探针遍历失败被当作通过。** 修复：
   - 对 agent 目录放宽权限时不跟随符号链接、不处理目录外文件和硬链接；
   - 先放宽 `/app`，再锁 `/tests`，锁完逐项核验，核验失败就不运行 agent 程序；
   - 设置 `no_new_privs`；
   - 探针改为由可信端写出具体文件路径逐个 `open`：`/tests/...`、`/src/tests/...`，以及经预置软链 `/app/output/leak` 访问的 truth。

   本机 Harbor 实测：作弊解在 agent 阶段建好 `/app/output/leak -> /tests/truth`，再由 `solution_cli.py` 读隐藏真值。旧判分 reward 1.0，新判分 reward 0.0；参考解 oracle 1.0 / nop 0.0。
2. **本地判分残留未加锁的 `/src/tests` 副本。** 复制完成后把 `/src` 设为 0700；测试中的攻击脚本读不到 `/src/tests/truth/*` 和 `/src/tests/measure.py`（修复前能读到，已用测试反证）。
3. **S9 提交与回填存在崩溃窗口。** 修复：
   - 作业名对 (条目, 角色, 第几次) 固定；
   - 运行记录按 intent → reserved → launched 顺序逐步持久化，续跑时对账；
   - 若作业目录或启动日志已存在，则只补记录、不再启动；
   - 回填后若 item 状态未写入，可由 `calib-ingest` 恢复。
4. **回填未校验作业身份。** `calib-ingest` 现在校验 job_name、agent 模型、task 路径三者都与待回填的那次运行一致，每个作业目录只能消费一次。
5. **启动异常会泄漏预留额。** tmux 不存在、超时、返回非零时，预留按 0 结算，记录为 `launch_failed/infra`。启动脚本把退出码写到 `<log>.exit`，若脚本退出时没有生成作业目录，对账时同样处理。
6. **`run_usd_total` 管不住真实支出。** 现在每次提交前检查已结算的实际支出，达到上限即停止后续提交（`budget:run_total_actual`）。**仍有的限制**：单个 Harbor 作业没有硬费用上限，一次运行的实际费用可能远超预留额（测试用例中 $10 预留结算为 $150）。
7. **哨兵判断按全局绝对值，会误判。** 改为与 nop（空输出）的实测值比较：参考值等于空输出值即判为缺输出。两个方向、负值、零值、大量纲都有测试。
8. **S6M 换 mid 节点后 naive ⊇ mid 不再成立。** 现在会把新节点并入 `naive_drop`，必要时由同一次调用重写 `naive.py`，并重新做 `validate_spec`，不合法则丢弃。提示词优先引导从 `naive_drop` 中选节点。
9. **naive 上限被宽松取整破坏。** 取整后会重查 naive 是否仍被拒绝，不满足就增加有效位数。
10. **截断落在下一个文件头时误丢上一个完整文件。** 改为按块的实际边界判断；截断落在文件头、正文、END 标记上的情况都有测试。

测试：retrieval 29 passed；gen 65 passed（含 Docker 集成测试）。

### 新发现（已知问题，未修）：网络与现成实现捷径

- 打包出的 3 道已校准题 `task.toml` 均为 `network_mode = "public"`，agent 做题时可以联网。
- DecontX 题的指令点名了 DecontX。PyPI 上有现成实现（如 `decontx-python`），agent 理论上可以直接 `pip install`；Fay–Herriot、RCESPP 也可能有公开代码可抄。
- 在 GLM 校准中途做了只读核查（检索 `pip install`、`git clone`、`github.com`、`pypi.org`、`curl`、`wget`、WebFetch/WebSearch、`decontx` 包名），结果按题记录在下方"校准结果"一节。
- 下一轮会默认 no-network，并加"现成库/仓库捷径"检查。在此之前不再提交新的校准作业。

### 校准结果（GLM-5.3 + Claude Code 2.1.280，阿里云沙箱，每题 1 次）

| 题 | 结果 | 用时 | 费用（Harbor `cost_usd`） | 轨迹核查：pip install / git clone / github / pypi / curl / wget / WebFetch / WebSearch | 标签 |
|---|---|---|---|---|---|
| 196db Fay–Herriot | reward 1（6 项全过） | 76 min | $16.92 | 全部 0 次 | 待 GLM#2（未提交） |
| e9e2a RCESPP | reward 1（5 项全过） | 86 min | $10.00 | 全部 0 次 | 待 GLM#2（未提交） |
| d7c20 DecontX | reward 0，`AgentTimeoutError`（2 h 上限） | 120 min | 无记录（按 $30 预留额记账；按另两次的 token 单价折算约 $18–19） | 全部 0 次；"celda" 出现 7 次，都是 agent 在思考中回忆 DecontX 原始实现，不是安装或调用 | `parked:teacher_unavailable` |

- DecontX 超时时，隐藏集 4 项均已通过（`solution_cli.py` 已写好）。唯一失败的是公开集的 `contam_rmse`：`/app/output/contamination.csv` 没有生成。这次失败是"没在时限内交齐"，不是方法错。
- 实测的 3 次运行都没有用到网络捷径。但 `network_mode="public"` 仍是漏洞：没用不代表不能用。
- 与旧管线对比（旧题 GLM 3 次）：同一论文的旧题 Fay–Herriot 3/3、RCESPP 3/3、DecontX 1/3。新题首轮 GLM 2/3 一次通过，说明这两道新题对 GLM 仍可能偏容易。需要 GLM#2 才能判断属于 easy 还是 unstable，本次按要求未提交。
- 校准总花费：已结算 $26.91（两次通过），加超时一次按预留 $30 记，合计 $56.91。另有两次启动失败，未用到模型，均按 $0 结算。沙箱已清理（本次作业残留 0 个）。
