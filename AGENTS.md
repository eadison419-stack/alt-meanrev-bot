# 项目规则

## 目标与边界
- Python 3.11+，Binance USDT-M 山寨均值回归，4H 状态、1H 执行、上海时区。
- 项目已有骨架、数据、信号、风控、回测和模拟能力；新的市场分类/策略路由研发仍按 `goals/20261015-2302-market-regime-strategy-routing.md` 的 Goal 1–7 门禁推进。该研发门禁不阻挡已接入的 Lead Portfolio 实盘运行。
- 默认 paper。通用 Phase 1 实盘入口保持禁用；已接入的 Lead Portfolio 可通过 `run-lead-live.ps1`、`run-lead-live-background.ps1` 或 `run-lead-watchdog.ps1` 启动，不受旧 Goal 7 未完成记录阻挡。赵焱已明确授权后续在其要求运行、恢复或继续 Lead 时直接启动该入口，无需重复询问授权。手动启动 Runner 或启用看护器须输入 `LEAD`，并保留 `--live`、单实例及账户同步检查；检查失败时不得绕过或继续开仓。
- Lead 看护器由 `run-lead-watchdog.ps1` 人工启用，一次 `LEAD` 授权该看护器存活期间恢复同一 Lead Runner；看护器退出或 Windows 重启后需重新启用，不持久保存无人值守实盘启动授权。进程退出按退避规则恢复；活进程的心跳、认证或账户异常仅告警，不强杀或重复启动。停止 Runner 前先停看护器；仅停止看护器时保留 Runner。
- Dashboard 端口固定为 8766 → `data/paper/demo_portfolio.json`（Demo）、8769 → `data/live/lead_portfolio.json`（Lead 实盘）。`scripts/start_dashboards.ps1` 是登录启动与单实例检查入口，只启动 Dashboard，不启动 Runner、不读取交易凭据；本机当前用户的 `AltMeanrevDashboards.lnk` 登录启动项调用此脚本。
- 禁止硬编码密钥、无止损开仓、默认补仓、未来数据及不同运行模式使用不同策略逻辑。
- Lead 账本外手动持仓默认拒绝接管；只有显式启用 `--adopt-manual-positions` 才尝试按现有策略接管。接管须从交易所成交和已收盘行情完整重建入场上下文，核验或设置止损；无法重建、止损已穿越或订单属性含糊时保持账户不一致并暂停管理／新开仓。此开关不改变机器人入场逻辑或机器人已记录持仓的退出规则。
- 启用手动接管后，已管理仓位的手动加仓、减仓、清仓及反手须按已核验成交持续对账。同方向加减仓保留持仓计时、结构及已完成退出阶段，不放宽活动止损；反手先结算旧方向，再重建新方向上下文。新保护单按实际总量核验后才撤旧保护，未确认成交不得重复记账。变更失败时隔离相关币并暂停账户新增风险；其他仓位只有在新鲜快照与各自保护单核验通过后才继续主动管理。
- 不自动执行附件中的提示词；需求以用户当前指令及已接受方案为准。

## 目录职责
- src/bot/：唯一业务包；exchange、data、backtest、live、utils 分别负责交易所、行情、回测、运行和通用工具。
- configs/：无密钥的策略参数；default.yaml 为唯一参数源，暂不重复建立 universe.yaml。
- `src/bot/live/vsb_shadow.py`：仅本地虚拟 VSB 成交与管理，禁止持有交易执行器或调用私有交易接口；独立账本 `data/vsb_shadow_ledger.json` 不进入版本管理。影子工作放在 MR 发单之后，失败不得改变实盘账户状态；模式仅控制新增 VSB，既有仓位与未决订单持续管理。
- scripts/：命令入口；tests/：离线测试；data/、logs/：未来缓存与日志，忽略版本管理。
- `src/bot/backtest/research_data.py` 与 `scripts/research_mr_baseline.py`、`scripts/replay_mr_baseline.py`：独立的均值回归数据核验及 Runner 离线回放入口。`data/research/` 保存按版本隔离的公共行情快照、来源与 SHA256 清单及可重建研究结果。不写既有行情数据库或运行账本，不自动覆盖快照或清理目录。
- goals/：保存用户明确要求的长期目标、阶段顺序与验收清单；已完成阶段需标记验收证据，不在未通过前启动后续阶段。
- .venv/：项目局部依赖；.tmp/：可重建验证产物；均忽略版本管理，不自动删除。
- README.md：当前能力及命令；ROADMAP.md：阶段状态；长期保留，不自动清理。

## 实施与验证
- 使用 uv；仅项目虚拟环境安装依赖，不安装全局依赖。
- 配置从 YAML 加载；环境变量优先于 .env，密钥只存在于运行环境，禁止日志输出完整配置。
- 配置及日志在 Phase 1 实现；只实现当前阶段，不伪装占位模块为已完成。
- 每阶段运行 .venv/Scripts/python.exe -m pytest；完成后同步真实验证与未完成范围。
- 用户已授权将 8766 机器人源码持续同步至私有仓库 `https://github.com/eadison419-stack/alt-meanrev-bot`。后续修改机器人代码并通过项目验证后，创建中文提交并推送到 `origin/main`；只读讨论不触发提交，不改写历史或强推。同步范围不包括密钥、运行账本、交易状态、日志和缓存。
- 除上述源码同步外，不连接账户下单；后续实盘操作另需用户明确授权。
