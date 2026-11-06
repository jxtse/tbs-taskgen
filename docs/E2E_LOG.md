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
