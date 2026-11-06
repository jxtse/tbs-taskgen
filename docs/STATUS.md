# 状态（2026-11-06）

## 已验证
- 检索层 29 个离线单测；10 个目标实测（见 `retrieval/eval/REPORT.md`，相关性为单人主观标注）。
- 出题层 49 个单测/集成测试（集成测试在本机 Docker 跑任务代码）。
- 假模型端到端干跑 S1→S10；本机 Harbor 0.23：oracle reward=1，nop reward=0。
- 泄漏门旧题回测：召回 7/7；样本内误报 0/231；留出误报 1/81（`gen/eval/LEAK_BACKTEST.md`）。
- 真实 Opus 端到端（e2e-iter1，6 篇论文，见 `E2E_LOG.md`）：
  - 3 篇走完 S1→PKG；
  - 本机 Harbor 与阿里云沙箱的 oracle=1 / nop=0 均通过；
  - 已真实提交 GLM-5.3 + Claude Code 2.1.280 校准。
- 真值隔离：读 `/tests/truth` 的作弊 `solution_cli.py` 在本机 Harbor 上由 reward 1 变为 0（`truth_isolation` 门）。

## 未验证
- 教师（Opus agent）校准分支：本次校准线路的 Opus 额度不可用，只跑了 GLM 分支。
- 校准结果是否落在"unstable / hard"区间（作业进行中，见 `E2E_LOG.md`）。
- L4b 反解攻击、廉价模型角色、S9b 变体、DOMF/CODE/MULTI 形态。
- 专家时长仍是 Opus 自报值。
- 美元数字按占位价计算，不是账单。

## 已定决策（2026-11-05）
1. 教师模型 Opus。
2. 难度校准：GLM-5.3 + Claude Code 2.1.280，顺序校准。
3. 难度合适 = GLM 2 次过 1 次（unstable），或 GLM 0 次过但 Opus 做出（hard）。
4. 检索不用 Tavily（适配器保留）。
5. 每日预算未定；量产等失败分析结果。

## 待定
- 每日预算与校准并发；
- 长程题比例与专家时长档位；
- 廉价模型选型；
- 证明题/交互题；
- 人工抽检比例。

详见 `DESIGN.md` §8。
