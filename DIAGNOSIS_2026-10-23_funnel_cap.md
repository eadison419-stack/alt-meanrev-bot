# TREND 2x 做多偏差诊断小结（2026-10-23）

纯诊断，未改动任何策略/风控配置。回测埋点 `record_trend_funnel` 为 opt-in（默认 False），零回归（373 passed / 3 failed）。

## 问题

W2（2026-09-23..10-23，BTC −3.07% 阴跌）里，TREND 分类层多空平衡（~1.2:1），但成交层 2:1 做多（40 多 / 21 空）。要定位偏差出在哪一层。

## 一、漏斗定位（`scripts/diagnose_trend_funnel.py`）

| 层 | up/多 | down/空 | 比值 |
|---|---|---|---|
| L0 宇宙 4H 分类 | 297 | 244 | 1.22 |
| L1 top-20 分类 TREND | 427 | 354 | 1.21 |
| L2 1H 入场 OPEN | 101 | 86 | 1.17 |
| L3 路由后 queued | 98 | 86 | 1.14 |
| L4 实际成交 | 39 | 21 | **1.86** |

**结论**：偏差不在分类、不在 top-20 热度、不在 1H 入场、不在路由（这些全平衡），而在**成交/风控层**。砍短边的是两个对称 notional cap：

- `per_symbol_notional_cap`（$3000 = margin 1000 × 杠杆 3）拒空 23 vs 拒多 0
- `directional_exposure_cap`（0.9×equity）拒空 30 vs 拒多 5

**机制**：TREND 按风险 sizing（`notional ≈ $50 / 止损距离%`）。空头信号期望 notional 平均 $4377、最高 $12270（止损仅 ~1%），多头从不超 $3000（止损 >1.6%）→ 固定 cap 系统性砍掉紧止损空头。

澄清：用户怀疑的 adaptive pool `direction_bias` 只在 live（multi_runner/trend_match），回测里没有，不是 40/21 的来源；top-20 热度评分用 abs 收益、方向中性，也不是来源。

## 二、cap 反事实四宫格（`scripts/diagnose_cap_ab.py`）

变体 baseline(1000, 0.9) / per_symbol_loose(20000, 0.9) / dir_loose(1000, 1.8) / both_loose(20000, 1.8)。表内为（多/空, net, PF, maxDD, 空头净）。

| 变体 | W2 阴跌 | | W1 牛市 | |
|---|---|---|---|---|
| baseline | 37/21, −487, 0.661, 8.89%, +36.6 | | 76/3, +851, 1.52, 5.80%, −67 | |
| per_symbol_loose | 37/21, −420, 0.708, 8.73%, +104 | | 80/5, +769, 1.44, 5.10%, −115 | |
| dir_loose | 38/28, −488, 0.694, 9.74%, +22 | | 84/3, +692, 1.39, 6.11%, −67 | |
| both_loose | 38/38, −521, 0.706, 11.50%, −50 | | 89/6, +539, 1.27, 6.63%, −106 | |

**结论**：

- `per_symbol_notional_cap` 放开方向**跨窗口相反**（W2 +67 改善 / W1 −81 恶化）。W2 的改善是单笔 XRP +66.65 撑起的窗口噪音，不是稳健 edge。
- `directional_exposure_cap` 放开在两窗口都恶化；`both_loose` 两窗口都是最差（W1 +851→+539）。
- → **两个 notional cap 都是有效保护层，维持不变，不改。**

机制：cap 按"方向敞口 + 单币 notional"起作用——牛市拒多（per_symbol L22 / directional L74），熊市拒空（per_symbol S23 / directional S30）。放开后的边际成交都是追高/追跌的紧止损单，负期望。

## 三、元结论

- TREND 本质是**做多山寨 beta**，PnL 由 BTC 方向主导（W1 +851 / W2 −487），非市场中性。
- "翻空不足"（2:1 做多）的根因是策略本身，不是风控把它憋死——空头信号有产出（L2 86 个），但成交的空头单大部分亏，放更多进来只会更亏。
- 风控 cap 不是问题。

## 四、本轮产物

- `scripts/diagnose_trend_funnel.py`：漏斗定位（含向量化 L0 宇宙分类）。
- `scripts/diagnose_cap_ab.py`：cap 反事实四宫格。
- `src/bot/backtest/portfolio.py`：新增 opt-in `record_trend_funnel` 埋点，`result.trend_funnel` / `trend_funnel_fill_rejections` 字段（默认空）。

## 五、下一步方向

- ② 多头加 BTC 方向过滤（本轮已证实多头在 BTC 阴跌时持续放血）。
- ① 短边信号质量（已部分回答：空头信号有、但负 edge）。
- 退出方式：`trend_structure_stop_hit` 是最大输家引擎（W1 −962 / W2 −1065），可单独诊断结构止损参数。

## 六、BTC 方向过滤前置诊断（`scripts/diagnose_btc_direction.py`）

多头按入场时 BTC 4H 状态分组：

| 入场时 BTC | W2 阴跌 | | W1 牛市 | |
|---|---|---|---|---|
| below EMA50 | 6笔 −216 (avg −36.0) | | 12笔 +167 (avg +13.9) | |
| above EMA50 | 31笔 −308 (avg −9.9) | | 64笔 +751 (avg +11.7) | |
| 24h 下跌 | 19笔 −346 (avg −18.2) | | 23笔 +256 (avg +11.1) | |
| 24h 上涨 | 18笔 −178 (avg −9.9) | | 53笔 +663 (avg +12.5) | |

**结论**：BTC 方向在熊市能区分多头盈亏，但在牛市方向性消失（below EMA50 的多头照样赚）。简单过滤器（BTC 下跌禁多头）两窗口合计仅 +49~+90 微弱正收益，牛市误伤，regime 依赖强 → **不推荐**。

## 七、结构止损诊断（`scripts/diagnose_structure_stop.py` + `result.trend_stops` 埋点）

`trend_structure_stop_hit` 是最大放血源：两窗口 100% 亏损（W1 26笔 −1015、W2 24笔 −975），持有中位数 5.5–7h，多头占绝对多数。但**止损距离并不紧**：W2 中位 2.81%、W1 中位 3.09%，比赢家 trailing_stop（1.97% / 3.40%）还宽。

**结论**：放血根因不是止损参数，而是**入场方向错**——趋势末端追入，入场后几小时就反向 ~3% 打到止损、从未触发 breakeven。赢家 trailing_stop（100% 胜、持 15–20h）是少数趋势延续单。→ 唯一能动的方向是**入场时机**。

## 八、本轮收敛

TREND 亏损根因链条：做多山寨 beta → PnL 由 BTC 方向主导 → 2:1 做多偏差是 cap 挡的（但 cap 是保护层）→ BTC 方向过滤依据弱 → 最大放血源是入场信号在趋势末端追高。**工程/风控路径已穷尽（cap/止损参数都不该动），剩下只能动入场时机。**

## 九、入场时机 A/B（`scripts/diagnose_adx_filter_ab.py` + `diagnose_entry_timing.py`）

入场特征诊断：structure_stop 单 vs trailing_stop 单，`|dev_ema|` 无区分（都 <0.2%，入场时价格紧贴 EMA50），但 **ADX 有区分**（W1 45 vs 34、W2 36.6 vs 31.5）、streak 略长。→ 过滤维度选 ADX 上限。

ADX 上限过滤跨窗口一致改善（本轮首个有 edge 的发现）：

| 变体 | W1 net | W1 PF | W1 maxDD | W2 net | W2 PF | W2 maxDD |
|---|---|---|---|---|---|---|
| baseline | +851 | 1.52 | 5.80% | −487 | 0.66 | 8.89% |
| adx45 | +717 | 1.69 | 6.69% | −260 | 0.80 | 8.16% |
| adx40 | +1002 | 2.52 | 4.50% | −55 | 0.95 | 7.15% |
| **adx35** | **+951** | **3.11** | **2.95%** | **+27** | **1.04** | **4.71%** |

- **adx35 最优**：两窗口合计 +978，structure_stop 放血单 W1 26→8、W2 24→13，赢家 trailing_stop 基本保留。
- adx45 太松（W1 +717 反而劣于 baseline，40–45 区间的单是正 edge 被误砍）。
- **机制**：ADX > 35~40 入场 = 趋势过热末端追高 = structure_stop 放血主源；过滤后只留趋势早期单。

**性质**：这是策略改动（新增入场过滤），已两窗口一致验证、方向稳健；正式采纳仍需用户本机全样本 + 更多 OOS（adx35 交易数减半、样本小）。

## 十、更早 OOS 窗口（W3）推翻了 adx35 结论

补了第三个 30 天窗口 W3（~2026-07-25..08-24，`diagnose_adx_filter_ab.py --window 3`）。这是**干净牛市**：baseline 29 笔 +616 PF2.24，structure_stop 仅 7 笔（对比 W1 26 / W2 24 笔放血）。

| 变体 | W3 net | W3 PF | W3 maxDD |
|---|---|---|---|
| baseline | +616 | 2.24 | 3.64% |
| adx45 | +425 | 2.11 | 3.64% |
| adx40 | +417 | 2.37 | 2.85% |
| **adx35** | **+133** | **1.77** | 1.54% |

**结论**：

- 所有 ADX 上限在干净牛市里**单调砍净利**（+616→+425→+417→+133），因为高 ADX 追高单在强趋势里继续走，不是放血源。
- **adx35 被否决**：W3 净利被砍 78%、PF 反而跌破 baseline（1.77<2.24），此前"两窗口最优"是过拟合近期震荡。
- adx40 三窗口 PF 全 ≥baseline、maxDD 全降、三窗口总净 +1364 vs baseline +980，但改善集中在震荡/熊（W1/W2），干净牛（W3）净利 −32%。本质是 **chop 过滤器，不是稳健 alpha**。

**落地建议**：不把 ADX 上限过滤设为默认开启。adx40 至多算一个"震荡市降回撤"的风控 overlay（且降回撤主要来自减少敞口的机械效应），需用户单独拍板，不作为 alpha 改动落地。

## 十一、market_risk 门是方向中性缩仓，且系统性砍收益（已落地修复）

补了 `scripts/diagnose_market_risk_ab.py`。market_risk 门（`classify_market_risk`）是方向中性的波动检测器（|24h move|≥1×ATR 且 ADX≥20，或单 bar |body|≥1.5×ATR），三窗口都触发 RISK_ELEVATED ~31–39%，牛熊不分。关掉缩仓（mr_off = `market_risk.enabled=false`）后三窗口收益全部改善：

| 窗口 | baseline（缩仓开） | mr_off | 差 |
|---|---|---|---|
| W3 干净牛 | +616 / PF2.24 | +1472 / PF3.28 | +856 |
| W1 震荡牛 | +851 / PF1.52 | +853 / PF1.40 | +2 |
| W2 阴跌 | −487 / PF0.66 | −371 / PF0.77 | +116 |

机制（trade 级，`result.market_risk_execution_summary`）：RISK_ELEVATED 里成交的单是趋势赢家，却被 0.5× 半仓（W3 31 笔里 14 笔半仓 notional $14k vs 正常 $29k）。趋势策略里"高波动=强趋势=钱所在"，方向中性缩仓砍的就是 trailing_stop 赢家。

**落地（2026-10-23，已执行）**：`elevated_size_multiplier 0.5 → 1.0`，保留 PAUSE_NEW 极端下跌停机。改动 config.py 约束 `lt=1→le=1`、default.yaml、4 处测试断言；pytest 373 passed / 3 failed 零回归。剩余可做：downside-only 方向化缩仓（只在下行缩，保留熊市保护），需新信号 + 回测。

## 十二、工程清理（2026-10-23）

§九/§十 的 ADX 上限入场过滤已被 W3 OOS 窗口否决（本质是 chop 过滤器，非稳健 alpha），相关 opt-in 代码属死代码。本会话做纯工程回滚，零策略变更、零回归（373 passed / 3 failed，3 个仍为既有失败）：

- 移除 `trend_adx_max` opt-in：`src/bot/trend_strategy.py`（`evaluate_trend_entry` 签名 + `trend_adx_overheated` 过滤块）、`src/bot/backtest/portfolio.py`（`run_portfolio_backtest` 签名 + 透传）。全库 grep 无残留。
- 删除 `scripts/diagnose_adx_filter_ab.py`（唯一传入 `trend_adx_max` 的脚本，回滚后必然报错；实验结论已留存于 §九/§十）。
- 其余诊断脚本**保留**（均为可复跑证据链）：btc_direction / cap_ab / confirm_bars / entry_timing / market_risk_ab（仍用于 §十一 60 天验证）/ range_sensitivity / regime_conflict / structure_stop / trend_funnel / trend_regime；`run_range_ab.py` / `run_ab_experiment.py` 仍用于 MR B+ 候选验证。
- 未动的实验脚手架（默认 no-op，属未决 MR 重设计候选，非本次清理范围）：`range_structure_mode`（默认 neutral）、`mr_entry_mode`/`mr_deviation_pct`（回测 only，默认 legacy）、`symbol_regime_assessment` 透传、`record_trend_funnel`/`trend_stops` 埋点。

## 十三、downside-only 方向化缩仓（B）——已否决（2026-10-23）

B 提案：market_risk 只在"下行"方向缩仓（RISK_ELEVATED 且 BTC 24h 净动为负 → 缩 0.5×；上行保持 1.0×），PAUSE_NEW 不动。实现为 `run_portfolio_backtest` opt-in 参数 `market_risk_downside_only` / `market_risk_downside_multiplier`（默认 False/0.5，零 live/config 改动），脚本 `scripts/diagnose_market_risk_downside_ab.py`。30 天×3 窗口 A/B（真实资金费）：

| 窗口 | baseline（mult 1.0） | downside_only | Δ |
|---|---|---|---|
| W1 震荡牛 | +842 / PF1.39 / DD6.53% / 81笔 | +926 / PF1.44 / DD5.81% / 81笔 | +84 |
| W2 阴跌 | −253 / PF0.83 / DD7.63% / 55笔(L34/S21) | −330 / PF0.77 / DD9.08% / 64笔(L34/S30) | −77 |
| W3 干净牛 | +1472 / PF3.28 / DD3.57% / 30笔 | +1452 / PF3.18 / DD4.07% / 34笔 | −20 |

**结论：否决。** 三窗口合计净 −13（wash），maxDD 在 2/3 窗口反而更差。机制：

- W2 阴跌：downside 缩仓把熊市里少数盈利的空单（baseline 空 +36）也半仓，且 0.5× notional 压到 per_symbol_notional_cap 之下 → 9 笔原本被 cap 拒掉的"追跌紧止损"空单漏进来（空 21→30），正是漏斗诊断确认的负 edge 单。净 −77。
- W1 震荡牛：无空单 cap 交互（仅 3 空），缩仓减少下行段多单亏损 → +84，但这是唯一改善窗口、regime 依赖强。
- W3 干净牛：下行罕见，缩仓误砍少数赢家 + 4 笔 cap 漏单 → −20。

加上此前已否决的（§11 方向中性缩仓、`diagnose_btc_direction` BTC 方向入场过滤），"方向化市场门降敞口"这条线已关闭：对 long-beta TREND，没有简单的市场方向门能产生稳定 edge。

**清理（本会话）**：回滚 opt-in 参数 + 删除脚本（与 §12 trend_adx_max 同原则：证伪即回滚）。零回归（373 passed / 3 failed）。

**口径说明**：W2 baseline 本会话 −253 与 §11 的 −371 有出入，是 utcnow 窗口漂移（缓存末 bar 2026-10-22 20:00 UTC、约 18h stale）；W3 baseline +1472.01 与 §11 的 +1472 完全吻合，证明 baseline=multiplier 1.0 状态正确、A/B 方法论无误。

## 十四、live/回测一致性：热点宇宙豁免路径（2026-10-23）

`paper.py` 的 TREND 入场有一条"热点宇宙豁免"路径（live/paper 专用，回测不模拟）：

- 正常：只有 top-N 热点宇宙内的币（`symbol_in_hot_universe=True`）能开 TREND 单；榜外币 `allow_entries=False` 被拦。
- 豁免（paper.py:2242-2256）：榜外币（`symbol_in_hot_universe is False`）若 `trend_confirmation_pass=True`（TREND 连续 `confirm_bars` 根确认）+ `confirm_bars==3` + 有效 pullback（pullback_price / stop_price / 未过期），照样开 TREND 单（`hot_universe_exempted=True`）。两处硬编码 `confirm_bars==3` 在 paper.py:2249 与 :2318（后者是 expired_candidate 上报路径）。
- 回测（portfolio.py:734-735）：`if symbol not in eligible and symbol not in active: continue` —— 榜外且未持仓的币直接跳过，**无豁免**。

**结论**：live/paper 的 TREND 入场宇宙 = top-20 ∪（榜外确认趋势的豁免单），回测只覆盖 top-20。回测系统性**低估** live 的 TREND 单数与敞口。confirm_bars 4→3 后趋势确认 +31.2%，豁免触发面随之扩大，但此路径在 confirm_bars=3 下从未做过 A/B（豁免单的 edge 未验证）。

**影响**：paper 已在该 live 路径上跑豁免（paper.py 即 live 路径），所以 paper PnL 与回测预期会系统性不一致；正式上线前须量化或收紧。

**建议（待用户拍板）**：① 文档化（本节即记录，已做）；② 补回测 opt-in 模拟豁免（允许榜外 `trend_confirmation_pass` 的币进场），量化豁免单的 edge 与量级，让回测 ≈ paper；③ 暂不建议直接收紧（豁免单正负未知）。

## 十五、热点宇宙豁免 A/B 结果（2026-10-23，已量化）

补回测 opt-in `hot_universe_exemption`（默认 `False` 零行为变更），模拟 live 的豁免路径：榜外且未持仓的币若 `classify_symbol_regime` 得 TREND 且 `trend_confirmation_pass=True`，则不再 skip、走 TREND 入场路径；`result.hot_universe_exemption_records` 记录被豁免币的成交（symbol/side/net_pnl/exit_reason）。脚本 `scripts/diagnose_hot_universe_exemption_ab.py`。30 天×3 窗口（真实资金费，`--cache data/market.sqlite --funding-cache data/ab_funding.pkl`）：

| 窗口 | baseline（豁免关） | exempt（豁免开） | Δ | 豁免单 |
|---|---|---|---|---|
| W3 干净牛 07-25..08-24 | +1472.01 / PF3.28 / DD3.57% / 30笔 | +458.66 / PF1.22 / DD11.90% / 82笔 | −1013 | 59笔 −253.53 |
| W1 震荡牛 08-24..09-23 | +711.98 / PF1.31 / DD6.53% / 81笔 | −595.79 / PF0.83 / DD12.93% / 131笔 | −1308 | 62笔 −804.24 |
| W2 阴跌 09-23..10-23 | −252.99 / PF0.83 / DD7.63% / 55笔 | −472.98 / PF0.80 / DD9.81% / 78笔 | −220 | 32笔 −138.13 |
| **三窗口合计** | **+1931.00** | **−610.11** | **−2541** | **153笔 −1195.90** |

**结论：豁免单是强负 edge，方向三窗口全一致，建议收紧/关闭 live 豁免（策略改动，需用户拍板）。**

- 三窗口合计：豁免把净利从 +$1931 打到 −$610（净 −$2541）；153 笔豁免单合计 −$1195.90，自身即净亏。
- 每窗口 maxDD 全面恶化（3.57→11.90、6.53→12.93、7.63→9.81）、PF 全面下降（3.28→1.22、1.31→0.83、0.83→0.80）。
- 豁免单以 `trend_structure_stop_hit` / `trend_structure_break` 为主，正是 §7 确认的"趋势末端追高/追跌"放血模式；少数 `trend_trailing_stop_hit` 赢家（如 W3 LINK +233、+201）撑不住整体。
- 次生效应：豁免单不止自己亏，还挤占 cap/敞口——exempted 笔数 > TREND 净增笔数（W3 59 vs 52、W1 62 vs 50、W2 32 vs 23），说明部分 baseline 合格单被挤掉/缩小。

**性质**：纯 backtest-only opt-in，不改 live/default。这是 live 豁免路径的首次 A/B 量化，结论稳健（三窗口方向一致、且与 §7/§9 的"追高追跌放血"机制自洽）。

**建议（待用户拍板，本次未动 live）**：
1. 收紧或关闭 live 豁免：paper.py:2242-2256 的 `exemption_candidate` 门、:2318 的 `expired_candidate` 上报路径、两处硬编码 `confirm_bars==3`。
2. 回测 opt-in `hot_universe_exemption` 保留但默认 `False`（生产回测基线仍是 top-20；需要"paper≈回测"口径时再显式打开）。

**附：本会话顺手做的纯性能优化**（零行为变更，零回归 373/3）：`evaluate_time_filter` 原在决策循环里对每个 symbol 每根 bar 无条件计算，但其结果只被 MR 入场分支消费、TREND/UNCERTAIN 路径从不读取。已把它的计算延迟到 MR 分支内（`backtest/portfolio.py`），30 天豁免回测从 >178s 降到 ~143–151s。

## 十六、宇宙口径核查：formal 池门与 regime-TREND 反相关，§15 结论作废（2026-10-24）

用户追问"可交易币种到底是不是热点前20"，核对代码后确认：live 的 `eligible`（`symbol_in_hot_universe`）**不是热度前20，而是自适应池正式候选**（`enable_adaptive_pools=true` 默认，match score 过阈值 `trend_match>=75` 或 MR>=65），热度分只排序不做 top-N。而回测一直用 `_historical_rank(top_n=20)`=热度前20。据此实现 `run_portfolio_backtest(adaptive_universe=False 默认零变更)`（`scripts/diagnose_universe_ab.py`），并核实 match 打分与 regime 分类的关系：

**核心反相关（已验，W3 干净牛）**：`trend_match>=75` 与 4H regime 的 TREND 基本反相关——regime-TREND bar 的 trend_match 中位数 **59.5**（<75 门槛），仅 **10.5%** 落在 formal 池。MR-formal 恒为 0（RANGE 不可达，`mean_reversion_match>=65` 从不单独触发）。→ formal 池门挡掉 ~90% 的 regime-TREND 信号，`adaptive`（纯 formal 池、无豁免）几乎不出单：W3 仅 2 笔。

**§15 结论作废**：live 的豁免路径不是"小后门"，而是 TREND 单的**主通道**（W3 84 笔里 83 笔豁免、W1 121 笔里 119 笔豁免），且用**正确边界**（formal 池之外）测，豁免单是**正 edge**（W3 +432.04、W1 +475.55）。§15 用"热度前20之外"作边界是错的，测出的是另一拨（低量能）单。真正坏的是 **formal 池门本身**（trend_match 这把"区间位置/动量"尺子挡掉了 regime 确认的趋势），豁免正是把它救回来的机制。

**净效果（30 天×窗口，真实资金费，2026-10-24 utcnow）**：

| 窗口 | hot（回测现状） | adaptive+豁免（≈live） |
|---|---|---|
| W3 干净牛 | 33笔 +1437.76 / PF3.11 / DD3.57% | 84笔 +383.42 / PF1.19 / DD11.07% |
| W1 震荡牛 | 78笔 +772.55 / PF1.35 / DD6.53% | 121笔 +443.28 / PF1.13 / DD11.35% |
| W2 阴跌 | 55笔 −252.99 / PF0.83 / DD7.63% | 79笔 −360.02 / PF0.68 / DD~11%* |

\* W2 是拆两个 15 天半段并行跑再合并的近似（单跑 30 天超 178s 沙箱上限）：09-24..10-09 46笔 +221.13（豁免41笔 +383.53）、10-09..10-24 33笔 −581.15（豁免25笔 −531.54）；边界有少量强制平仓失真。

豁免单分窗口合计（正确边界 = formal 池之外）：W3 83笔 **+432.04**、W1 119笔 **+475.55**、W2 66笔 **−148.01**（早期 +383.53 / 后期 −531.54）。→ 豁免不是均匀正 edge，而是**顺 BTC 方向**（牛/震荡正、阴跌负），与"TREND=做多 beta"一致；净合计三窗口 +759.58。

→ 回测（热度前20）系统性**高估** live：净利 ~4x、maxDD ~3x。live 实际在交易"formal 池 ∪ 豁免"这个更杂、更差的集合。

**数据 caveat**：缓存是现货镜像，volume 对 20M 期货阈值 → 候选偏少（2–15 vs live 期货 ~30–40），影响绝对量级；但 `trend_match` vs regime-TREND 的反相关是代码属性、非数据 artifact。

**结论（待用户拍板）**：① 回测默认 universe 是否要切到 live 口径（`adaptive_universe`）；② 更根本的问题——live 的 formal 池门（`trend_match>=75`）本身是个坏门（挡掉 regime 确认的趋势），要不要改 live 的宇宙定义；③ §15 的"收紧豁免"建议撤回。

## 十七、路 B + 选项一 + 选项二：入场时机与 BTC regime 门全部证伪（2026-10-24）

用户选择"先做选项一（BTC regime 多头总闸）+ 选项二（下跌段空头）"，并先试了路 B（入场时机 alpha）。三条线全部收敛到同一个负结果，把 TREND 策略的优化空间彻底关死。

**路 B（入场时机 alpha，`scripts/diagnose_entry_timing2.py`）**：赢家（trailing_stop）vs 输家（structure_stop）在入场那一刻的四个特征——ADX、确认 streak、顺趋势累计 move、|dev_ema|——**全部跨窗口反转或无效**。ADX：W1/W2 输家高、W3 输家低；streak：W1/W2 输家长、W3 输家短；dir_move：三窗口三个方向；|dev| 恒 <0.15% 无区分。streak 分桶里赢家输家同桶共存、SS 恒亏 TR 恒赚。→ 赢家/输家的差别是"入场后继续走 vs 反转"，入场时不可观测。路 B 无解。

**选项一（BTC 多头总闸 `btc_bull_long_only`，`scripts/diagnose_btc_gate_ab.py`）**：

| 窗口 | baseline | gate（close>EMA50） | Δ |
|---|---|---|---|
| W3 干净牛 | +1482 / DD3.57% | +1590 / DD3.49% | +108 |
| W1 震荡牛 | +727 / DD6.53% | +185 / DD4.46% | **−542** |
| W2 阴跌 | −204 / DD7.63% | −84 / DD7.04% | +120 |
| 合计 | +2005 | +1691 | **−314** |

EMA50 6bar 斜率版 W1 仍误伤（+727→+222）。根因：震荡牛里 BTC 反复穿越 EMA50，放血单（structure_stop）和赢家单（trailing_stop）**同时开出来**，宏观门分不开；gate 砍掉 11 笔里混着 3 笔放血 + 5 笔赢家（+860）。"简单方向门"家族第三次失败（btc_direction、downside_only、本总闸）。唯一价值 = 降回撤，非增收益。

**选项二（下跌段空头，`scripts/diagnose_short_edge.py`，hot+豁免口径三窗口）**：空头整体 W1 18笔 −585（胜率 6%）/ W2 31笔 +125 / W3 40笔 −938（胜率 15%），合计 −1398。关键：**BTC 阴跌 regime（below EMA50 + slope down）下空单照样亏**——W1 17笔 −626（0% 胜率）、W3 30笔 −503（20% 胜率）、W2 25笔 +156（靠早期 3 笔 trailing 撑）。放血源 = structure_stop_hit（W1 10笔 −448 / W3 15笔 −692），追空进场后反弹打止损，和追高一模一样；空头赢家 trailing_stop 样本极小（1/6/4 笔）。→ "阴跌才做空"不成立，空头在阴跌 regime 下本来就亏。

**元结论（收敛）**：TREND 唯一的正 edge 是趋势延续单（trailing_stop 赢家），但它在入场时**不可观测**（路 B）；宏观 regime 门（无论"牛市才开多"还是"阴跌才做空"）都无法把它和放血单分开（选项一/二）。根子在入场信号层：`evaluate_trend_entry` 不区分趋势延续 vs 趋势末端。策略盈亏最终由 BTC 方向主导、**无可挖掘的稳定 alpha**。现有框架已穷尽。

**剩下的现实选项（用户决策）**：① 接受"牛市 beta 放大器"定位——只在明显牛市跑（`btc_bull_long_only` 或手动开关），忍受震荡误伤或干脆接受回撤；② 换策略内核（不是调这个 TREND 策略）。落地：`adaptive_universe`、`btc_bull_long_only` 两个 opt-in 均默认零变更，零回归 373/3。

## 十八、横截面相对强弱腿（独立做多 top-K 动量）——已证伪（2026-10-24）

CTA 换内核后的最后一条未探索方向：给 CTA 补一条"横截面相对强弱腿"——按 N 日动量排名、做多 top-K 最强币，退出复用 CTA 的 2×ATR 止损 + reverse + be2 + 1% 风险 + max_pos=5。脚本 `scripts/diagnose_cta_rs.py`（自包含；warmup 修正：全窗口帧算指标+动量、只在窗口内交易，使动量在窗口首日即可算）。

**口径校准（重要，避免被数字误导）**：window-frame 基线今天（Oct24 统一窗口）重跑 = baseline **+11.23%** / be2 **+13.79%**（旧记忆 +12.39%/+14.96% 是 Oct23/Oct24 混合窗口的旧数）。warmup 修正让 donchian 反而 −1.59%（被 warmup 放行的"窗口首日突破"是净负）——但 warmup 对 rs 和 donchian 是同一套，同框架内对比才公平：donchian **+9.64%** / donchian_be2 **+11.45%**。

**lookback 扫描（4H 根数 → total%）**：

| lookback | 6 (1d) | 12 (2d) | 18 (3d) | 24 (4d) | 30 (5d) | 42 (7d) | 84 (14d) |
|---|---|---|---|---|---|---|---|
| total% | +10.77 | +11.16 | **+12.79** | +5.45 | +5.21 | +8.85 | +6.42 |

非单调、rs18 是孤立尖峰（邻域 rs24 崩到 +5.45%）→ 过拟合，非稳健 edge。唯一跑赢 donchian 的是 rs18（+12.79 vs +9.64），但它是单点尖峰、不是平滑趋势。

**结构发现**：① be2 对 RS 是**负**的（rs18_be2 +8.85 vs rs18 +12.79；rs84_be2 +6.09 vs +6.42）——因 RS 出场以 reverse（趋势破位）为主、stop 极少（W1 36 reverse vs 6 stop），与 Donchian 相反（Donchian 靠 stop 出场、be2 把止损单转保本才有用）；② top-3 集中化两处都更差（rs18_k3 +10.06 vs rs18 +12.79）；③ min_mom>0 正动量过滤是 wash（rs84_pos +6.69 vs rs84 +6.42）；④ 所有 RS 变体 W2 阴跌全亏（仍做多 beta）。

**结论**：独立相对强弱腿是死胡同，与本项目元结论一致——选币类 long-beta 无稳定 alpha。脚本保留作证据链、未接生产；RS 作为 Donchian 的排序 overlay/过滤门**未测**（用户决定不测，期望值低）。策略研究到此正式收敛：现有框架已穷尽，剩余只有 OOS 前瞻验证（be2 已在模拟盘跑）。
