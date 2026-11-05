# 策略机器人完整复盘与 Windows 重建说明

- 代码／配置／日志复核时间：2026-10-23 00:17（北京时间）；进程现场复核时间：00:17:09。
- 代码基线：当前 E:\alt-meanrev-bot 工作区；项目不是 Git 仓库，本包不宣称对应某个 Git commit。
- 参数基线：configs/default.yaml。
- 当前运行态是只读快照，不代表策略长期表现，也不随源码安装包迁移。
- 不包含真实密钥、.env、市场数据库、账本、订单状态、运行日志、研究缓存或虚拟环境。

## 结论先行

这不是“一个打分器触发买卖”的单策略，而是“全市场硬筛与热点评分 → 两个互斥策略候选池 → 逐币 4H 三态分类 → 策略路由 → 1H 入场条件 → 统一风险 → Paper／Demo 执行与持仓退出”的串联系统。每一段只放行上游剩下的候选，因此候选池分数高不等于 4H 可路由，更不等于有交易信号。

最近一组完整的逐币路由证据有 39 个样本，39/39 为 UNCERTAIN，且没有信号拒绝、风控拒绝、待单或成交事件。因此在该评估批次里，交易没有走到统一风控或执行层；直接瓶颈是逐币 4H 分类没有得到 TREND／RANGE。后续最新候选刷新虽然有 37 个 TREND 与 2 个 MEAN_REVERSION 正式候选，但刷新只产生候选清单，不是路由评估。新 Runner 未完成该批候选的首轮路由验证。

当前还有独立的运行可用性问题：截至 2026-10-23 00:17:09（北京时间），watchdog PID 10320 在运行，父进程 PID 6900；没有 Python Runner 进程。最新尝试于 00:14:31 启动 Demo Runner PID 6004，00:14:49 API 请求失败，watchdog 于 00:15:01 记录 1800 秒退避，预计下一次尝试不早于 00:45 左右。日志未提供足以归因具体 API 根因的安全错误详情。它不会解释之前 39 个 UNCERTAIN，却会阻止新的路由周期发生。

最近账本文件修改时间为 2026-10-22 23:35:17；其中状态标签为 synchronized，Demo 交易所快照时间为 23:34:05；在该快照里，positions、pending_by_symbol、exchange_snapshot.positions、exchange_orders_by_symbol、foreign_order_ids、inconsistencies 均为空。最后一条账本策略路由事件时间为 23:05:14。快照比当前时刻旧，不能据此断言交易所现在仍为空仓。安装包不复制该状态。

## 一、策略从输入到出场的实际链路

### 1. 市场硬筛与热点候选

入口主要是 src/bot/data/klines.py 与 src/bot/live/multi_runner.py::_refresh_universe。

- 只考虑活跃 Binance USDT 结算、Crypto 类永续合约；BTC、ETH 默认排除；要求 24 小时报价成交额至少 20,000,000 USDT、上市至少 14 天。没有独立的盘口点差或深度硬门。
- 行情热度指标仅使用已收盘 1H K 线：最近完整 1H 的估算报价成交额 ÷ 前 24 根中位数，4H／24H 绝对涨跌，24h 报价成交额及 ticker 成交笔数。横截面中秩百分位权重分别为：相对放量 40%；4H 与 24H 绝对涨跌均分 30%；报价成交额与成交笔数均分 30%。缺失分项不伪造为市场观测值，排名分项按零分处理。
- 活跃候选先得到最多 20 个 hot Top-N 对照榜；但 adaptive_pools.enabled 当前为 true，实际新信号候选由匹配分的正式池并集决定，不保证不超过 20 个。
- 趋势匹配分由区间位置、EMA50 距离、近期收益、ADX 上升、EMA50 斜率和结构打分；多空取配置定义的最高方向分。均值回归匹配分由路由、ADX、EMA50 斜率／距离、结构及旧均值回归状态打分，并附趋势倾向、冲击、结构冲突和极值惩罚。所有这些是候选筛选分，不是方向信号。
- 正式池阈值：TREND 75、MEAN_REVERSION 65。观察池分别从门槛的 85% 到门槛以下；观察币只记录，不进入路由。
- 正式和观察资格互斥分配；正式优先；多个正式池同时达标时选“超过该池门槛的分差较大者”，平分优先 TREND。Runner 后续只允许该币被分配的来源策略开仓。若币被分给 TREND，但 4H 路由选出 MEAN_REVERSION，开仓会因 source_not_in_adaptive_candidate_pool 被拒；不会自动补回另一个策略池。

### 2. BTC 大盘风险预算

入口 src/bot/market_risk.py::classify_market_risk，Runner 在新开仓前获取 BTC/USDT 永续 4H 状态。

- 用已收盘 BTC 4H K 线，至少 80 根，连续且最近数据未超过 4 小时 5 分钟；缺失、缺口、过期或指标无效会 fail closed 为 PAUSE_NEW。
- NORMAL 按常态预算；RISK_ELEVATED 将新单与组合预算乘以 0.5；PAUSE_NEW 将新开仓预算置零。大盘状态不选择币种策略；持仓退出仍继续。
- 本次最新可读运行日志的 BTC 状态在 2026-10-22 23:34 为 RISK_ELEVATED / btc_volatility_or_trend_elevated，而非 PAUSE_NEW。这只会降低开仓额度，不能单独解释没有信号。当前 Runner 已退出，不能把该旧状态说成现在的 BTC 状态。

### 3. 逐币 4H 路由分类

入口 src/bot/symbol_regime.py::classify_symbol_regime。只对候选池并集及现有持仓／待单执行对应评估；只用连续已收盘 4H K 线。当前至少 80 根、历史无缺口、最新收盘未超过 4 小时 5 分钟。

- TREND_UP：ADX14 ≥25；收盘价 > EMA20 > EMA50；EMA50 三根 4H 间隔斜率 ≥0.10%；最近两组各 4 根 K 线结构为更高高点且更高低点。TREND_DOWN 镜像要求。
- TREND 候选必须连续 4 根已收盘 4H K 线属于同一方向。该持续确认约束有意降低单根翻转，但状态切换慢。
- RANGE：ADX14 ≤18；EMA50 三根斜率绝对值 ≤0.05%；收盘价距 EMA50 ≤1.5%；结构为 NEUTRAL，表示高点与低点均没有形成同向扩大或冲突；并连续 2 根 4H K 线符合。
- 因而 ADX 大于 18 且小于 25 时，没有 TREND 或 RANGE 原始候选。即使 ADX 通过，斜率、EMA 顺序、距离、结构和连续确认仍须分别满足。数据缺失、历史不足、K 线缺口、过期也为 UNCERTAIN。UNCERTAIN 明确不生成新开仓。
- RANGE 路由选 MEAN_REVERSION，TREND 路由选 TREND；BTC 状态不覆盖逐币状态。执行中的仓位仍按其开仓来源管理。

### 4. 均值回归 4H 状态和 1H 入场

逐币分类器先确认 RANGE，之后旧均值回归分类器 src/bot/regime.py::classify_regimes 才被调用；这两套 4H 分类职责不同，形成串联过滤。

旧分类器按首个命中顺序判断：SIDEWAYS（ADX<20 且 ATR% 的近 30 根百分位≤30）→ IMPULSE_UP／DOWN（实体幅度≥1.5 ATR）→ LOW／HIGH（此前 48 根区间位置≤30%／≥70%，或本根新低／新高）→ SLOW_UP／DOWN（ADX≥20 且六根净变动≥1 ATR）→ MID。RANGE 路由并不会强迫旧分类结果变成 LOW／HIGH。

- 做多候选要求旧 4H 状态 LOW、最新已收盘 1H 收阳、允许做多。默认还要求最近 3 根 1H 低点不下降，或当前阳线正下影线长度不短于实体；最近 4 根 4H 低点严格下移会阻断。低于此前 48 根 4H 区间低点至少 0.5×1H ATR 时拒绝。
- 做空候选要求旧 4H 状态 HIGH、最新已收盘 1H 收阴、允许做空；最近 4 根 4H 高点严格上升会阻断。高于此前区间高点至少 0.5×1H ATR 时拒绝。
- SIDEWAYS 或 flatten_only 阻止新开仓。1H 时段过滤使用北京时间会话及快速波动条件；白天某些大幅波动会暂时禁对应方向或当日多头。当前 night_new_entries=true，夜间不因时段本身禁开。
- 新 RANGE 路由和旧 SIDEWAYS／LOW／HIGH 状态的组合是第二层环境过滤；影子里的 range_structure_sideways_priority_v1 只改变诊断反事实，不改变正式状态或订单。

### 5. 趋势入场与退出

入口 src/bot/trend_strategy.py::evaluate_trend_entry，由 PaperRunner 在 TREND 路由时调用。

- 4H 趋势确认之外，1H 多头要求已收盘 1H 收阳且收盘在 EMA20 上方；空头要求收阴且收盘在 EMA20 下方。
- 决策生成回踩计划，不等于立即成交。当前配置虽然 general execution.wait_pullback=false，TREND 自身仍设置 0.4×1H ATR 回踩目标，最长 45 分钟，且不晚于信号收盘后的下一小时边界；Runner 以新鲜 ticker 触及计划价格作为触发条件。不能把启动日志的 “immediate” 理解成趋势单马上成交。
- 初始止损默认基于信号时最近 6 根 1H 摆动高／低点外加 0.4 ATR；止损距离须在 0.3–2.5 ATR，否则拒绝。趋势策略目标风险为权益 0.5%；止盈／保护管理含 1.5R 保本、2.5R 后 2 ATR 跟踪，以及结构／EMA50 退出。
- Demo 在条件到达后通过固定 USDⓈ-M Demo 主机发测试开仓并维护交易所保护止损；本地 Paper 模拟成交。Live 当前拒绝执行。

### 6. 统一风控、待单和成交

主要位于 src/bot/risk.py::guard_signal 与 src/bot/live/paper.py。

- 只有策略先形成 OPEN，才会进入通用风控。主要拒绝条件有：当日权益回撤达到 4%；PAUSE_NEW 或风控状态无效；资金费缺失或绝对值≥0.001；出场／跨策略切换冷却；同币持仓或来源冲突；最多 5 个持仓；单币名义上限（目标保证金 1,000 USDT × 最高 3 倍，再按大盘状态缩放）；止损风险加估算进出手续费和滑点后超权益 1%；总敞口超权益 1.8 倍；方向敞口超权益 0.9 倍；数量低于交易所最小值。
- TREND 仓位另先按权益 0.5% 风险预算、止损距离、手续费／滑点反推数量，再经过共享统一风控。常规单笔最大止损风险默认权益 1%；taker 费率 5 bps，常规模拟滑点 3 bps，风险估算滑点上限 0.15%。RISK_ELEVATED 同时缩小单币及组合预算；这改变“能否批准／能开多大”，不把 UNCERTAIN 变成 OPEN。
- MEAN_REVERSION 在 execution.wait_pullback=false 下走当前直接执行路径；确认发生延迟超过 60 秒会拒绝，Demo 的本地／交易所校验可能再拒绝。TREND 必须先通过 preflight 风控再创建 45 分钟待单；触价后会重新检查时效、路由、来源、费用和执行条件。待单过期／未触价不等于策略未形成信号。
- 均值回归默认启用 exit_proposal_v1：价格有利触及入场 ATR 的 1.0 倍／1.8 倍时分别减仓初始数量的 40%，剩余仓位使用 0.8／0.5 ATR 跟踪；达到 1 ATR 有利波动可把止损移至费用调整后的保本附近。结构止损缓冲 0.25 ATR、旧状态衰减阈值 0.5 ATR、连续 6 根 1H 无进展且 MFE 小于 0.4 ATR 时平仓、最长 24 根 1H 持有；TP1 后满足保本条件可豁免普通最长持仓。配置中旧 time_exit_enabled=false，因此不能把 12／24 小时检查当作开启的时间退出。另有 exit_management 的 1.5R 保本、2.5R 后 2 ATR 跟踪设置；它属于另一退出管理路径，不应与均值回归 v1 参数混为一套规则。TREND 使用独立的 1.5R 保本、2.5R 后 2 ATR 跟踪与结构／EMA50 退出。退出管理独立于新开仓；PAUSE_NEW 与不再入池不会阻止持仓管理。

## 二、错配分析

### A. 候选分数与路由状态是两套模型

热点分负责“近期活动度”；趋势／均值回归匹配分负责“特征接近策略画像”；逐币 4H 分类负责“当前是否满足硬定义的 TREND 或 RANGE”。这些目标不同，不能用高分替代状态。当前 Runner 只对正式候选并集运行 4H 分类，因此正式池扩大仍可能零路由。

### B. TREND／MR 互斥分池会切掉已满足另一策略门槛的许可

2026-10-22 23:35:16 的刷新记录（事件 at 为 23:34:03）里，94 个基本合格标的对应 hot Top-20；实际 adaptive pool 正式并集为 TREND 37、MEAN_REVERSION 2。匹配两个正式门槛的币被分配冲突共 3 个，另有 1 个 TREND 正式／MR 观察冲突。NEAR 的两个正式分分别为 82.41／76.11，按余量分给 MR；BCH 为 88.00／65.74、ASTER 为 97.92／76.75，均双正式达标却分给 TREND。若 BCH／ASTER 此后路由到 RANGE，MEAN_REVERSION 来源不在允许集合，会被 source_not_in_adaptive_candidate_pool 拒绝。该错误出现在候选资格与路由状态的衔接处；不应仅凭分数决定永远只允许一种来源，也不代表应当降分数门槛。

### C. 两个 4H 分类器串联，增加 RANGE 实际可达难度

新分类器要求 RANGE 条件连续 2 根 4H，旧分类器却继续优先输出 SIDEWAYS 或 MID；只有之后落到 LOW／HIGH 且 1H 反转成立，才可能生成旧均值回归信号。历史 TRX 影子显示 SIDEWAYS 会阻止提案；结构单项放宽的既有离线实验没有产生 MR 提案，因此不能推断跳过 SIDEWAYS 或改一项 RANGE 结构条件就足够。

### D. 4H 硬状态之间有阈值空档，且结构证据支配分类

当前 ADX 18–25 属于状态定义空档；同时 EMA／斜率／结构条件要求合取。2026-10-22 23:05 完整路由批次中 39/39 为 UNCERTAIN，原因分布为 no_clear_regime 20、trend_components_conflict 9、confirmation_not_met 8、trend_structure_conflict 2。失败条件重叠，不能把数字相加成独立失败币数。RANGE 结构非 NEUTRAL 为 39/39、EMA50 斜率不平 33/39、ADX 超 RANGE 上限 30/39、离 EMA50 过远 27/39；TREND_UP 结构不符 37/39。该批次只有 2 个 near_range、26 个 near_trend。分类零路由是该批次无信号的直接策略原因。

### E. 重复路由计算不是“两道新的独立趋势门”

MultiRunner 先分类，再把状态传给 Paper；Paper 根据该状态选路由；TREND 分支中的 evaluate_trend_entry 又用相同的已收盘 4H 输入调用同一 classify_symbol_regime，随后还会做 route 应用／来源一致性检查。若时间戳、配置及 K 线输入一致，这几次调用大多是重复计算同一判断，不构成额外的市场过滤条件。真正新增的趋势过滤是 1H 方向与 EMA20、计划价／止损构建、风控和回踩成交。重复调用值得做逐步一致性审计；在确认存在输入偏差或分类不一致前，不应为“去重”改决策逻辑。

### F. 信号时钟与执行时钟不同

候选池每 5 分钟左右刷新，但逐币开仓判定只在闭合 1H 决策点执行；4H 分类仅能使用当时最新闭合 4H bar。TREND 出信号后还要在 45 分钟内等 quote 触价。当前在线 Runner 在 23:20–23:40 期间刷新过候选池，但没有新路由记录；下一个小时边界前进程退出。随后多个 Runner 重试在 Demo 初始化 API 检查处失败。因此候选刷新不能证明路由已完成，也不能据此称“信号成立但没成交”。

## 三、现场证据分层（北京时间）

### 运行态核对

- 检查时间：2026-10-23 00:17:09。当前 watch_demo_paper.ps1 PID 10320，父 PID 6900；未发现 Python Runner。上一窗口 PID 18644／23096、watchdog PID 9288 不是当前 PID，不据此判断运行状态。
- 最新尝试于 2026-10-23 00:14:31 启动 Runner PID 6004；00:14:49 API 请求失败；00:15:01 watchdog 记录 1800 秒退避。具体远端错误原因未确认。
- 最新完整路由日志仍是 23:05:14。该批 39 条逐币状态观察均为 UNCERTAIN。23:20 的新 Runner 在 23:21、23:28、23:35 有候选刷新，23:35 正式池 37／2，最后一次刷新不等于路由事件。之后没有新路由、信号、风控拒绝、待单或成交事件。
- 同一候选刷新里有 3 个“双正式达标”分配冲突（NEAR、BCH、ASTER）和 1 个趋势正式／均值回归观察冲突（DOT）。因此互斥分池问题确实可观测，但是否造成后续可交易机会损失仍待固定历史数据回放。
- 最近 Demo 账本状态文件的本地修改时刻 23:35:17，快照时间 23:34:05；当时状态标签 synchronized 且本地／交易所仓位、待单、挂单、不一致均为空；最后路由事件 23:05:14。停止运行后没有新快照；当前账户状态不能从该文件推断。
- 最近日志中的 BTC 大盘风险在 23:34 为 RISK_ELEVATED。这不是 23 日当前状态；它缩小新仓预算但不是 39 条 UNCERTAIN 的来源。

### “没有信号”与“信号后被拦截”的区分

- 没有形成信号：状态为 UNCERTAIN；或 RANGE 下旧 4H 状态不是 LOW／HIGH；或 1H 方向／反转／时段／结构条件失败。观察名单和打分不会绕过这些条件。
- 信号形成后路由／风险拒绝：要有 OPEN 候选，之后能看到 strategy_route_evaluated 的 candidate_action 与最终 action、signal_rejected，或 order_intent.allowed=false 及具体风险原因。
- 信号成立但待单未成交：重点看 pending_entry_started、pending_entry_cancelled、pending quote touch、paper_fill 或 Demo fill／保护止损事件。
- 目前最新 39 条状态样本全为 UNCERTAIN，且没有对应的 OPEN、风控拒绝或待单事件；本次证据支持第一类（4H 未路由成功）。对于本轮之后的时间，Runner 停止导致没有观测，属于运行数据缺口，不可声称“新 Runner 也验证为 UNCERTAIN”。

## 四、既有研究与证据局限

- 研究缓存只到 2026-10-16；使用固定候选快照、14 天训练／8 天样本外切分。基准样本为训练 54 笔，净收益 +446.55 USDT、最大回撤 8.37%、PF 1.342；样本外 25 笔，-218.85 USDT、回撤 3.21%、PF 0.564。
- 允许 UNCERTAIN 走 MR：训练 107 笔、-225.02、回撤 10.77%、PF 0.930；样本外 57 笔、-398.33、回撤 5.66%、PF 0.710。直接让 UNCERTAIN 参与 MR 不受现有证据支持。
- 只对 UNCERTAIN 原始方向候选绕过到 TREND：训练 90 笔、-426.97、回撤 10.29%、PF 0.807；样本外 48 笔、-225.02、回撤 5.12%、PF 0.750。也不应直接放行 UNCERTAIN。
- 只放宽 RANGE 结构条件的回放没有形成 MEAN_REVERSION 提案，净收益与状态统计和基线相同；重复这条实验方向优先级低。
- 将趋势结构 CONFLICT 放入趋势原始候选但保留其他约束，训练从 54 笔／+446.55 变为 73 笔／-455.64；样本外从 25 笔／-218.85 变为 38 笔／+500.44，同时状态翻转增加。训练／样本外方向相反，结论是不稳，不建议正式调整。
- 这些数据不覆盖 2026-10-17 之后市场，也不是 Demo 实际成交；回测的 1H 撮合近似不能精确复现 45 分钟趋势回踩。8 天样本外和少量交易不足以支持收益结论。Goal 7 在 ROADMAP 与目标文件中仍标为进行中。
- ROADMAP 最后一次全量测试记录为 370 passed、3 failed；已记载的失败为一个 1H 回踩未收盘 K 线许可测试及两项风险限额断言。本次只读打包没有运行测试，也没有独立确认这些旧失败目前是否仍存在。

## 五、优先离线对照方案

所有实验固定缓存版本、币池快照、BTC 4H 状态、资金费、费率／滑点、1H／4H 已收盘边界、信号和退出规则。一次只改变下面指定的单项；先输出逐币逐小时的逐门审计，再看成交指标。不得拿观察名单、match score 或扩大状态数作为策略有效证据。

### 优先级 1：只对双正式达标币取消来源互斥

- 基准：当前 TREND 75／MR 65 门槛、exclusive largest-margin 分配。
- 实验：唯一变化是正式候选同时满足两个门槛时，让该币两个来源都具备候选池许可；实际只由原 4H 路由决定来源。若趋势池币被路由 RANGE，MR 不再因 source_not_in_adaptive_candidate_pool 被拒；所有硬门、4H 分类、信号、风控、执行和退出不变。
- 原因：最新 23:35 刷新有 NEAR、BCH、ASTER 三个双正式达标币；BCH 与 ASTER 被分给 TREND，分池后可能错过 4H RANGE 所需 MR 资格。此实验直接隔离“分配资格错配”，不放宽市场状态。
- 输出指标：每个窗口双达标数量及归属；该类币的 RANGE／TREND／UNCERTAIN 状态占比；按来源的原始 OPEN、候选池拒绝、风控批准、pending、成交数；净收益、PF、胜率、最大回撤、换手和状态翻转。候选数或提案增多不算通过。
- 限制与判断：至少 2 个不重叠样本外窗口，并逐窗报告；有效 OOS 成交少于 30 笔时只判“样本不足”。只有增加的交易来自被恢复的另一策略资格，成本后表现不劣于基准、回撤不恶化且两窗口方向一致，才进入下一阶段审查。若双池样本过少，结论是“不确定”，不是采用。

### 优先级 2：只把 TREND 连续确认从 4 根降为 3 根

- 基准：symbol_regime.confirm_bars=4；其他 RANGE、ADX、EMA、斜率、结构、1H、回踩和风险规则完全不变。
- 实验：唯一变化为 TREND 同向连续已收盘 4H 候选数量从 4 改为 3。RANGE 仍为 2；不可同时调 ADX、结构或 EMA 条件。
- 原因：最近完整路由批次里 8/39 最终原因为 confirmation_not_met，近 TREND 信号也存在；缩短一根 4H 确认是清晰、可追踪的单条件候选。它不会解决所有 UNCERTAIN 原始冲突，所以先做反事实命中统计，再决定是否进入收益回放。
- 输出指标：被新规则改变的 state transition 数；新增 TREND 状态中原始条件和方向；每个状态保持时长／翻转；1H 过滤后 OPEN、风控通过、45 分钟触价成交；净收益、PF、回撤和趋势／均值回归来源归因。
- 判断：若新进入 TREND 主要靠缩短确认后仍频繁翻回、触价率没有改善，拒绝；如果 8 天 OOS 仍不足 30 笔或结果跨窗口方向不一致，保持基线并延长留出观察。不得仅因趋势候选变多而修改正式配置。

在以上两项之前不再重复“只放宽 RANGE 结构”或“UNCERTAIN 直接路由策略”的实验；现有回放未支持它们。当前没有依据把匹配分门槛、结构、确认根数、路由和执行时间一起调整。

## 六、代码与安装入口

| 职责 | 入口 |
|---|---|
| YAML／运行环境配置 | configs/default.yaml；src/bot/config.py::load_settings |
| 市场硬筛、活动度分数 | src/bot/data/klines.py |
| 候选池分配及周期 | src/bot/live/multi_runner.py::_refresh_universe、cycle |
| BTC 风险 | src/bot/market_risk.py::classify_market_risk |
| 新 4H 三态 | src/bot/symbol_regime.py::classify_symbol_regime |
| MR 旧状态 | src/bot/regime.py::classify_regimes |
| MR 信号及结构过滤 | src/bot/signals.py::generate_signal |
| 策略来源路由 | src/bot/strategy_router.py |
| TREND 信号与退出计划 | src/bot/trend_strategy.py |
| 统一风控 | src/bot/risk.py::guard_signal |
| Paper／Demo 主循环、信号事件和成交 | src/bot/live/paper.py；src/bot/live/multi_runner.py |
| Windows 安装 | install.ps1；只创建局部 .venv 并装锁定依赖 |
| 本地 Paper 手动入口 | run-local-paper.ps1；强制本地模拟模式 |
| USDⓈ-M Demo 凭据输入 | scripts/setup_demo_credentials.ps1；需要手动选择执行 |

项目当前 Windows 主机 Python 为 3.12。安装 ZIP 需要网络下载 uv 和锁文件依赖，不是离线安装程序。源码包不带 Demo 账户数据、已建仓状态或运行日志；另一台电脑先按本地 Paper 手动启动，Demo 账户核对与状态恢复是单独工作，不包含在安装步骤内。



