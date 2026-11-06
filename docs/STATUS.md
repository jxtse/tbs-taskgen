# 状态（2026-11-06）

## 已验证
- 检索层 26 个离线单测；10 个目标实测（见 `retrieval/eval/REPORT.md`，相关性为单人主观标注）。
- 出题层 36 个单测/集成测试（集成测试在本机 Docker 跑任务代码）。
- 假模型端到端干跑 S1→S10；本机 Harbor 0.23：oracle reward=1，nop reward=0。
- 泄漏门旧题回测：召回 7/7；样本内误报 0/231；留出误报 1/81（`gen/eval/LEAK_BACKTEST.md`）。

## 未验证（端到端测试进行中，见 `E2E_LOG.md`）
- 真实 Opus 的 S5/S6A/S6C/S8 产出能否过解析和各道门。
- 阿里云沙箱 / 自建 GPU 服务器 上的 Dockerfile 构建；S9 真实提交与 `calib-ingest`。
- L4b 反解攻击、廉价模型角色、S9b 变体、DOMF/CODE/MULTI 形态。
- 美元数字按占位价计算，不是账单。

## 已定决策（2026-11-05）
1. 教师模型 Opus。
2. 难度校准：GLM-5.3 + Claude Code 2.1.280，顺序校准。
3. 难度合适 = GLM 2 次过 1 次（unstable），或 GLM 0 次过但 Opus 做出（hard）。
4. 检索不用 Tavily（适配器保留）。
5. 每日预算未定；量产等失败分析结果。

## 待定
每日预算与校准并发；长程题比例与专家时长档位；廉价模型选型；证明题/交互题；人工抽检比例。详见 `DESIGN.md` §8。
