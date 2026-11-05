# 热点筛选、策略路由与策略模块边界

本文记录当前实现及经确认的趋势策略模块边界。热点分数负责排序；逐币状态负责选择策略路由；大盘状态只控制是否允许新开仓。均值回归条件与执行路径不由本文的趋势模块改动。

## 数据流与职责边界

```text
Binance 公共市场元数据 + 24h ticker 快照
  └─ src/bot/data/klines.py::filter_tradable_crypto_universe
       └─ 基本可交易性硬筛（不按热点排名或 Top-N 截断）
            └─ src/bot/live/multi_runner.py::_refresh_universe
                 ├─ 对全部基本合格候选拉取 1H OHLCV
                 ├─ hot_activity_metrics：只保留已收盘 1H K 线并计算特征
                 └─ rank_hot_candidates：全体候选横截面评分，再保留配置的 Top-N

多币 Runner 对每个持仓、待处理和合格候选分别评估
  └─ 过滤已收盘 1H / 4H OHLCV
       ├─ src/bot/market_risk.py：BTC 已收盘 4H 大盘风险状态
       ├─ src/bot/symbol_regime.py：候选币已收盘 4H 趋势／区间／不确定状态
       └─ src/bot/live/paper.py：路由后调用信号生成、模拟成交与退出管理
            ├─ RANGE → src/bot/signals.py::generate_signal（现有均值回归）
            ├─ TREND → Paper／Demo 通过统一风控后进入 45 分钟回踩待单与趋势退出；Live 不启用
            ├─ UNCERTAIN → 不生成新开仓
            └─ src/bot/strategy_router.py + src/bot/risk.py：来源、持仓冲突、大盘暂停与统一开仓风控
```

`filter_tradable_crypto_universe` 只负责基本准入，不计算热点、不做 Top-N 截断，也不生成开仓信号；`rank_hot_candidates` 在全部通过基本门槛的候选上计算热点分，最终按分数截取配置的 Top-N。只有最终候选池进入新信号状态分类；现有持仓、待单仍按各自路径继续管理。热点分没有额外的最低门槛或独立准入判断。`classify_regimes` 和 `generate_signal` 继续负责均值回归状态及信号语义。`classify_symbol_regime` 的结果只选新信号路由；大盘 `PAUSE_NEW` 禁止新开仓。已开仓的币继续由持仓来源管理，其他策略不能新增或接管。执行前的资金费、止损、仓位与组合风险由 Paper／Runner／Risk 模块处理。

## 行情时效口径

- 任何从 OHLCV 计算的放量、涨跌、ATR、ADX、4H 状态和 1H 信号，只使用对应周期已收盘 K 线。实时 Runner 在 `src/bot/live/multi_runner.py::cycle` 按 `open_time + timeframe <= now` 过滤；`classify_regimes` 的输入契约也要求每行均为已完成 K 线。分类函数本身不会根据墙钟时间剔除未收盘行，因此调用方必须遵守该契约。
- 基本准入中的 24h `quoteVolume` 与热点评分中的成交笔数 `count` 来自交易所 ticker 的滚动 24h 快照，不是 K 线指标。它们不是未收盘 K 线值；此口径与基于收盘 K 线计算的 OHLCV 特征分开记录。
- 4H 状态在策略决策的 1H 收盘时，只能使用当时已经收盘的 4H 行。Paper Runner 和多币 Runner 均按 K 线收盘时间选择可用状态；回测按历史决策时间对齐。

## 热点初筛与复筛

### 输入、硬筛与输出

主要实现位于 `src/bot/data/klines.py`：

- `filter_tradable_crypto_universe(markets, tickers, settings, ...)` 输入 CCXT 市场元数据、ticker 和 `default.yaml` 的 `universe` 配置，返回**全部**基本合格币，不接收排名上限。保留活跃 USDT 结算永续、Binance 元数据中 `contractType=PERPETUAL`、`underlyingType=COIN` 且 `underlyingSubType` 含 `Crypto` 的合约；排除 BTC、ETH 和配置排除项；要求有有效 `onboardDate` 且上市至少 `min_list_days`，并要求 ticker `quoteVolume >= min_quote_volume_24h`。默认门槛分别为 14 天和 20,000,000 USDT。
- 24h 成交笔数来自 `ticker.info.count`，不存在时回退到 `ticker.count`。它只参与评分，不单独设置成交笔数硬门槛。
- 当前项目没有单独的盘口点差或深度准入过滤；基本流动性门槛是 24h 报价成交额。多币实时流程由 `src/bot/live/multi_runner.py::_refresh_universe` 编排：先对全部基本合格币取 1H OHLCV（请求 26 根以容纳边界／未收盘数据），然后 `rank_hot_candidates` 在全部候选上计算横截面分数，最后只把配置的 `top_n` 放入新信号分类候选池；Runner 构造器要求 `1 <= top_n <= 20`。因此不再存在“先按 24h 成交额预取 Top-N 再算热点分”的步骤。
- 候选池门槛和参数位置：`configs/default.yaml::universe.min_quote_volume_24h`、`min_list_days`、`exclude_symbols`；运行候选数上限来自 `scripts/run_multi_paper.py --top`，受 Runner 的 20 个上限约束。
- 每次 Runner 更新候选池会输出结构化 `hot_universe_refreshed` 日志，含 `basic_tradable_count`、`top_n`、评分权重、排序依据和最终候选逐币分数／排名。最终池外的币不会生成新信号；热点分只负责在基本合格币中排序，不存在额外的热点分阈值。

### 收盘 K 线特征与热点分

`hot_activity_metrics(rows, now_ms)` 先按 `row[0] + 1h <= now_ms` 仅保留已收盘 1H K 线，并按开盘时间去重排序。少于 25 根、价格或成交额非法时，相关 OHLCV 指标返回缺失。

- 每根 1H K 线估算报价成交额：`base_volume × (open + high + low + close) / 4`。
- 1H 相对放量：最新已收盘 1H 估算报价成交额 ÷ 其前 24 根已收盘 1H 估算报价成交额的中位数。
- 4H 涨跌：最近 5 个已收盘 1H 收盘价的首尾变化，即 4 个小时段；24H 涨跌：最近 25 个已收盘 1H 收盘价的首尾变化，即 24 个小时段。排名使用两者的绝对值。
- `rank_hot_candidates` 对当前合格候选各指标做横截面中秩百分位，平分取平均秩。热点总分为：
  - 相对放量百分位 × 40%；
  - （4H 绝对涨跌百分位 + 24H 绝对涨跌百分位）÷ 2 × 30%；
  - （24h 报价成交额百分位 + 24h 成交笔数百分位）÷ 2 × 30%。
- 缺失的放量、涨跌或成交笔数不被伪造成原始数值；实现中其百分位分量为 0，不单独剔除该候选。报价成交额因硬筛门槛而必须有效。
- 排序依次为热点分降序、24h 报价成交额降序、symbol 升序，确保平分时顺序确定。

## 均值回归 4H 状态

主要实现位于 `src/bot/regime.py::classify_regimes`，指标实现位于 `src/bot/indicators.py::indicator_frame`。输入为已收盘 4H OHLCV；调用方传入未收盘行将违反接口契约。

### 指标与状态优先级

- ATR(14)、ADX(14) 使用 Wilder 递归平滑；`ATR% = 100 × ATR / close`，并在最近 30 根上取当前值的百分位中秩。
- 每行的区间高低点取此前 48 根 4H K 线的最高价和最低价，通过 `shift(1)` 排除当前行。位置为 `(当前 close - 前 48 根 low) / (前 48 根 high - 前 48 根 low)`；当前行若创新低／新高，`new_low`／`new_high` 也可将其判为 LOW／HIGH。
- 六根 4H 回看净变化为 `close[t] - close[t-6]`。
- 状态按代码顺序判定，首个命中即返回：
  1. `SIDEWAYS`：ADX < 20 且 ATR% 最近 30 根百分位 ≤ 30。
  2. `IMPULSE_UP`／`IMPULSE_DOWN`：当前已收盘 K 线实体 `close - open` 分别 ≥ `+1.5 × ATR` 或 ≤ `-1.5 × ATR`。
  3. `LOW`：区间位置 ≤ 30%，或当前已收盘 K 线向下突破此前 48 根低点。
  4. `HIGH`：区间位置 ≥ 70%，或当前已收盘 K 线向上突破此前 48 根高点。
  5. `SLOW_DOWN`／`SLOW_UP`：ADX ≥ 20 且六根净变化分别 ≤ `-1 × ATR` 或 ≥ `+1 × ATR`。
  6. 以上均未命中时为 `MID`。
- 缺少 ATR、ADX、ATR 百分位、前 48 根高低点、区间位置或六根净变化时，该行状态为 `None`；实时 Runner 还要求至少 `lookback.min_regime_bars_4h`（当前 80）根已收盘 4H 历史。

配置位置为 `configs/default.yaml`：`timeframes.regime`、`lookback.range_bars_4h`、`atr_len`、`adx_len`、`min_regime_bars_4h`、`regime.low_pct`、`high_pct`、`adx_sideways`、`atrpct_percentile`、`atrpct_lookback`、`slow_lookback_4h`、`slow_move_atr_mult`、`signals.impulse_atr_mult`。

## 1H 入场组合与后续过滤

`src/bot/signals.py::generate_signal` 将最近已收盘 1H K 线与可用 4H 状态组合：

- 多单候选：允许做多、4H 为 `LOW`、已收盘 1H 收阳（`close > open`），reason 为 `low_zone_1h_bullish_close`。
- 空单候选：允许做空、4H 为 `HIGH`、已收盘 1H 收阴（`close < open`），reason 为 `high_zone_1h_bearish_close`。十字线不满足任一方向的阴阳条件。
- `SIDEWAYS` 或 `flatten_only` 阻止新信号；其他 4H 状态不匹配时不生成对应的均值回归入场。
- 多单 `long_entry.block_if_4h_lower_lows` 默认检查最近 4 根已收盘 4H 低点；严格连续下移或历史不足时拒绝。`require_reversal_sign` 默认要求最近 3 根 1H 低点不下降，或当前阳线的正下影线长度不短于实体。它是 4H LOW + 1H 收阳入场条件之外的独立确认；阳线本身不替代止跌证据。
- 空单 `short_entry.block_if_4h_higher_highs` 默认在最近 4 根已收盘 4H 高点严格连续上升时拒绝。
- `entry_rejection_reason` 还检查参考价相对此前 48 根 4H 区间边界的位置（越界至少 0.5 × 1H ATR 时拒绝）和开仓结构位有效性；具体结构值由调用方提供。
- `src/bot/time_filter.py::evaluate_time_filter` 以已收盘 1H K 线收盘时间和 `Asia/Shanghai` 时区计算时段及极端波动限制。资金费、冷却、止损宽度、持仓数、名义敞口、费用／滑点和组合风险预算不属于热点排名或 `generate_signal`，后续在 `src/bot/live/paper.py` 与 `src/bot/risk.py` 检查。
- 已持仓标的继续由其原 Paper 管理路径处理；`generate_signal` 在持仓状态下不追加同币仓位，反向信号遵循现有 close-first 配置，不由热点榜接管持仓。

关键参数均以 `configs/default.yaml` 为准：`long_entry.*`、`short_entry.*`、`session.*`、`signals.allow_long`、`signals.allow_short`、`signals.reverse_same_bar`。本说明不复制或另设参数源。

## 大盘门与逐币策略路由

- `src/bot/market_risk.py::fetch_market_risk_status` 从 BTCUSDT 永续已收盘 4H K 线计算 `NORMAL`、`RISK_ELEVATED` 或 `PAUSE_NEW`。只有 `PAUSE_NEW` 禁止新开仓；同时由 Runner 将开仓预算设为 0，使普通待执行开仓与回撤待执行开仓也被拒绝。监控、止损和策略平仓不受该开仓门影响。
- `src/bot/symbol_regime.py::classify_symbol_regime` 使用候选币已收盘 4H K 线输出 `TREND`、`RANGE` 或 `UNCERTAIN`；行情分类不直接使用 BTC 标签。`src/bot/strategy_router.py::resolve_strategy_route` 选择来源并检查大盘状态和现有持仓来源。
- `RANGE` 调用原 `generate_signal`，不改变现有 4H 均值回归状态、1H 阴阳线、结构、时段及出场条件。`symbol_regime.trend_adx_min` 默认 25，TREND 连续确认数 `confirm_bars` 默认 4；RANGE 使用独立的 `range_confirm_bars`，默认 2，以免调整趋势确认影响均值回归分类。本地 Paper 与锁定 Demo 主机的 Demo Runner 对通过路由的 TREND 调用 `evaluate_trend_entry`，经归属、资金费、切换冷却、仓位和组合风控预检后写入 `pending`；新鲜公开 ticker 触及目标价后，本地 Paper 模拟成交，Demo 通过 `DemoExecutionService` 发送测试委托并挂保护止损。均值回归继续使用原有 Paper／Demo 成交路径。Live Runner 尚不可用，Live 趋势执行不获准。`UNCERTAIN` 只记录不确定原因，不生成新开仓信号。
- 每个 Paper／Demo 路由决策保存事件：执行模式、币种、分类状态、大盘状态、候选动作／方向／来源／原因、最终动作、信号时间、持仓来源、放行或拒绝原因。可从 Runner 日志及 Dashboard 最近事件核对。策略生成的信号来源 `MEAN_REVERSION`／`TREND` 与交易对、方向、已收盘信号时间一起传递；成交另记录 `execution_source`，Demo 为 `binance_usdm_demo`。未选策略或被路由拦截的结果标为 `ROUTER`，候选来源单独保留。
- 持仓结构退出、保护止损、持仓时间和其他管理先于新信号路由继续执行。只有 `OPEN` 动作经过策略路由和来源一致性检查；`CLOSE` 不因状态不匹配或 `PAUSE_NEW` 被拦截。

## Goal 5 趋势策略模块边界

趋势决策规则位于 `src/bot/trend_strategy.py`，使用 `configs/default.yaml::trend_strategy`。该模块仍是无执行副作用的决策模块；读取输入并返回计划／退出信号，不读取账户或下单。`src/bot/live/paper.py::PaperTrader.update` 与 `_execute_pending` 负责本地 Paper／Demo 的趋势路由及待单；本地 Paper 生成模拟成交，Demo 只在 `execution_mode=demo_usdm` 且执行适配器 host 精确匹配 Demo REST 主机时发送委托。`src/bot/live/multi_runner.py` 为 TREND 待单轮询新鲜 ticker。其他执行模式不能启用趋势，Live 入口仍由 `resolve_execution_mode` 明确拒绝。

### 入场与仓位

- 输入是同币已收盘 1H／4H OHLCV、带时区的评估时间、逐币分类所需配置、市场风险状态、持仓／待单来源、权益及交易所数量步长。独立入口 `evaluate_trend_entry` 再次过滤未收盘 K 线，并拒绝 BTC／ETH、配置排除项、历史不足、间隔、过期、分类不确定或过滤冲突。
- 4H `TREND` 状态和方向来自既有 `classify_symbol_regime`。做多要求最新已收盘 1H 收阳且收盘价高于 EMA20；做空镜像要求收阴且收盘价低于 EMA20。EMA 使用 `adjust=False`。
- 入场信号创建仅作为待回踩候选：多头目标价在信号收盘价下方 `0.4 × 1H ATR` 范围内取较浅回撤（并参考信号 K 中点），空头镜像。计划有效 45 分钟，且不晚于信号 K 收盘后下一小时边界。本地 Paper 与 Demo Runner 均使用单币新鲜 ticker 检查目标触价。Paper 按观察价模拟；Demo 按现有 `DemoExecutionService.open` 发送委托、核验实际成交及交易所保护止损。两者共享最大名义、方向敞口、单笔风险、资金费门、冷却和数量精度检查。手续费／资金费按现有 Paper 账本或 Demo 交易所成交处理，不新增周期资金费率结算。
- 初始止损默认使用截至信号 K（包含该根）最近 6 根已收盘 1H 摆动低／高点外侧 `0.4 × ATR`。可选 `stop_mode: atr` 使用 `2 × ATR`。止损距离小于 `0.3 × ATR` 或大于 `2.5 × ATR` 时拒绝候选，不自动扩大或压缩。
- 单笔风险预算默认权益的 0.5%。数量按 `风险预算 ÷（止损距离 + 双边 taker 费用及滑点估算）` 计算，再按剩余总名义敞口、现有杠杆配置且最多 3 倍、最小数量和步长取整约束。资金费及共享组合预算由 Goal 6 统一风控处理，不在该生成器中重复实现。
- `PAUSE_NEW`、同币已有仓位、任何来源的同币待单以及路由拒绝均阻止新候选。RANGE／UNCERTAIN 不能生成 TREND 候选。

### 趋势退出与止损管理

- `evaluate_trend_exit` 只管理 `source=TREND` 的持仓。Paper 在完整 1H 评估时调用它；结构失效使用入场时保存的摆动结构位，依据完整 1H 收盘价跌破多头结构位／升破空头结构位退出；EMA50 也按已收盘 1H 收盘价反向突破退出。Paper 每轮 `monitor_tick` 按当前趋势 `protective_stop` 检查止损。Demo 中每当该已收盘 K 线评估提高止损，Paper 会调用 `DemoExecutionService.replace_stop` 更新交易所 Algo `STOP_MARKET`：先提交新保护单，再撤销旧保护单，并把交易所精度后的价格写回趋势退出状态。若更新失败，保留最后确认的本地止损、记录 `protective_stop_update_failed` 并把执行状态标为 `inconsistent`，不伪称新止损已生效。
- 初始止损和已有保护止损使用已完成 1H K 线的 high／low 检查；可选当前成交价只用于检查当前保护止损是否已触发，不参与指标或结构信号计算。每根完整 K 线仅处理一次，entry 所在未完整 K 线排除。
- 最大有利波动达到 `1.5R` 后将止损移至近似覆盖入场 taker 费及退出 taker 费／滑点的保本位；达到 `2.5R` 后启用 `2 × 1H ATR` 移动止损。止损只向保护方向移动；先用 K 线开始前生效的止损检查本根，再在收盘后更新，避免用本根极值倒推同根内新止损触发。Demo 的新交易所保护单确认后才更新本地止损价；本地 Paper 只更新模拟状态。
- 没有新增时间退出、分批止盈或反转开仓条件。上述参数是经用户确认的趋势策略边界，全部位于 `trend_strategy` 配置，均值回归配置和行为保持原样。

## 模块索引

| 职责 | 实现入口 | 输入 | 输出／职责 |
|---|---|---|---|
| 市场硬筛与热点指标 | `src/bot/data/klines.py`：`filter_tradable_crypto_universe`、`hot_activity_metrics`、`rank_hot_candidates` | 市场元数据、ticker、已收盘 1H OHLCV、`universe` 配置 | 全部基本合格候选、热点分及分项指标 |
| 实时扫描编排 | `src/bot/live/multi_runner.py`：`_refresh_universe`、`cycle` | 公共 ticker、OHLCV、候选参数 | 全体候选评分后选出的最多 20 个观察标的及每币更新调用 |
| 指标 | `src/bot/indicators.py`：`indicator_frame` | 已收盘 OHLCV、ATR／ADX 周期 | ATR、方向指标、ADX |
| 4H 状态 | `src/bot/regime.py`：`classify_regimes`、`classify_regime` | 已收盘 4H OHLCV、区间及状态参数 | 每根 K 线的因果状态与快照 |
| 1H 时段过滤 | `src/bot/time_filter.py`：`evaluate_time_filter` | 已收盘 1H 历史、决策收盘时刻 | 多空时段许可与极端行情标志 |
| 均值回归信号 | `src/bot/signals.py`：`generate_signal`、`entry_rejection_reason` | 4H 状态、已收盘 1H、过滤标记、结构与区间 | `OPEN`／`CLOSE`／`NONE` 信号及原因 |
| 大盘风险状态 | `src/bot/market_risk.py`：`fetch_market_risk_status` | BTC 已收盘 4H OHLCV | `NORMAL`／`RISK_ELEVATED`／`PAUSE_NEW` |
| 逐币分类与路由 | `src/bot/symbol_regime.py`、`src/bot/strategy_router.py` | 候选币已收盘 4H 分类、市场状态及持仓来源 | 策略来源、是否允许新开仓及拒绝原因 |
| 趋势候选与退出判断 | `src/bot/trend_strategy.py`：`evaluate_trend_entry`、`evaluate_trend_exit`；Paper 调用点：`src/bot/live/paper.py::PaperTrader.update` | 已收盘 1H／4H、权益与数量规则、已有持仓／待单上下文 | TREND 候选与已收盘 bar 退出信号 |
| Paper／Demo 开仓风险、回踩待单与执行 | `src/bot/live/paper.py::_execute_pending`、`_advance_pullback_pending`；`src/bot/live/multi_runner.py::cycle` | 趋势计划、公开 ticker、持仓来源、资金费与共享预算 | 本地 Paper 模拟成交，锁定 USDⓈ-M Demo 主机后发送 Demo TREND 委托；Live TREND 不启用 |
| 通用开仓风险与账本 | `src/bot/risk.py::guard_signal`、`src/bot/live/paper.py` | 信号、仓位、资金费、价格、风控参数 | 接受／拒绝、统一风控记录、Paper／Demo 执行路径 |

## 独立旧版均值回归 Paper Runner

`scripts/run_legacy_mr_paper.py` 以 `MultiPaperRunner(strategy_mode="mean_reversion_only")` 执行旧版 LOW/HIGH 均值回归信号，不启用逐币趋势路由。它单独使用 `data/paper/legacy_low_high_portfolio.json` 和 `logs/legacy_low_high_paper.log`，本地虚拟权益默认 5,000 USDT。BTC 大盘风险状态为 `PAUSE_NEW` 时禁止新开仓，风险上限及仓位规则读取 `configs/default.yaml`。

入场使用已收盘 4H 状态和已收盘 1H 方向，沿用 `generate_signal` 的时段、反转、结构过滤及现有止损／持仓管理。旧回测另启用 EMA20 中轨回撤退出；独立 Runner 在每根新收盘 1H K 线上只处理一次该保护退出。该运行入口仅创建 Binance 公共行情客户端，不连接 Demo 或正式账户。

本次 Goal 1 只增加说明和源代码注释；没有改变函数逻辑、配置值、筛选阈值、开平仓条件或账本。
