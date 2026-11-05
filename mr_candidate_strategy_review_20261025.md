# mr_candidate 策略完整复盘（2026-10-25）

> 这是 `run_portfolio_backtest(..., mr_candidate=True)` 所代表的**均值回归候选策略**的完整规格。
> 用途：迁移到另一台电脑做回测复现 / OOS 验证。
> 重要边界：**这是回测+模拟盘候选，OOS 验证已证伪其 edge（见文末）**。可作为研究/前瞻验证工具，不当作可稳定盈利的策略。

---

## 一、一句话定性

无策略路由、无 4H RANGE 门的独立均值回归：**热度 top-20 选币 → EMA50 偏离触发入场 → 统一风控 → 现有退出 + EMA20 中轨保护**。

## 二、策略规则（完整）

### 1. 选币（Universe）

1. 基本可交易过滤 `filter_tradable_crypto_universe`（USDT-M 永续、剔除不可交易/低流动性合约）。
2. 热度打分 `score_hot_candidates`，横截面分位排名后加权：
   - **40%** 量能爆发比 `volume_burst_ratio`
   - **30%** 价格活跃度 =（|4H 收益| 与 |24H 收益| 的分位均值）
   - **30%** 流动性 =（24H 成交额 与 24H 成交笔数 的分位均值）
3. 排序：`hot_score` 降序 → `quote_volume_24h` 降序 → symbol 升序。
4. 取前 **top_n = 20**。

### 2. 入场（Entry）

`mr_entry_mode="ema_deviation"`，`mr_deviation_pct=0.01`。信号函数 `_range_deviation_signal`：

- 计算 **4H close 的 EMA50**（`symbol_regime.ema_slow_bars = 50`）。
- `deviation = close / EMA50 − 1`（百分比偏离）。
- **开多**：`deviation ≤ −1%` 且 1H 阳线（close>open）且 `allow_long_entry`。
- **开空**：`deviation ≥ +1%` 且 1H 阴线（close<open）且 `allow_short_entry`。
- 时间过滤 `evaluate_time_filter`（session 早盘 08:00、午盘 12:00、晚盘 20:00，`night_new_entries=true`）。
- 成交：**下一根 1H 开盘价** + 滑点（`max_slippage_pct=0.15%`）。

### 3. 仓位与风控（Sizing / Risk）

- **固定名义**：`margin_per_symbol_usdt(1000) × leverage(3) = $3000/仓`（`calculate_order_quantity`）。
- 上限：`max_symbols=5`、总名义 ≤ `1.8×equity`、单方向名义 ≤ `0.9×equity`、单币名义 ≤ `$3000`。
- 单笔风险 ≤ `1% equity`（`max_single_trade_risk_pct=0.01`）。
- 止损 `calculate_adaptive_stop`：**结构止损** = swing 6 根 1H + `0.4 ATR` buffer；min `0.3 ATR`（太窄则拒）、max `2.5 ATR`（太宽则拒）、hard cap `2.0 ATR`。
- 冷却期 `cooldowns`：止损 6h / 止盈 2h / 策略退出 0h / 策略切换 1h / 强平 18h。
- 费用：taker `5 bps`（0.05%）、滑点 `3 bps`（0.03%）。

### 4. 退出（Exit，candidate 配置）

共享退出链（按 bar 内检查顺序大致为）：

1. **保护止损**：结构止损（如上），击穿按止损价/跳空开盘价成交（`stop_hit`/`stop_gap`）。
2. **中轨保护**（candidate 独有，`bollinger_mid_exit=True`）：
   - 中轨 = **1H close 的 EMA20**（`bollinger_mid_period=20, ema=True`）。
   - 多头：入场价 < 中轨 → 收盘 ≥ 中轨时武装（`reached_mid`）→ 之后收盘 < 中轨则**全平**（`bollinger_mid_rejection`）。
   - 空头对称（入场价 > 中轨 → 收盘 ≤ 中轨武装 → 收盘 > 中轨全平）。
   - 无 buffer（`buffer_sigma=0`）、无"站稳 N 根"（`hold_bars=0`）、全平（`partial=1.0`）。
3. **结构失效**：收盘跌破结构低 − `0.25 ATR`（多头）/ 反之（空头）。
4. **regime 消退**：进度 < `0.5 ATR`。
5. **无进展**：连续 `6 bars` 且浮盈 < `0.4 ATR` → 平。
6. **最大持仓**：`24 bars`。
7. **分段保护**：浮盈 ≥ `1.0 ATR`（`stops.move_to_breakeven_atr`）后，止损锁定扣除费用和滑点后的 `+0.2R`；进度 ≥ `1.5R` 后提高至 `+0.5R`。
8. **移动止损**：进度 ≥ `2.0R`（`mr_candidate_exit.trailing_trigger_r`）触发，沿用原跟踪距离 = `trailing_atr_mult(2.0) × initial_risk / stop_atr_mult(2.0)` ≈ **1R**。
9. **信号层退出**（`generate_signal` 的退出侧）：4H 冲动（IMPULSE_UP/DOWN）、早盘冲动（morning_move UP/DOWN）、新高/新低（high_zone_new_high / low_zone_new_low）、反向信号（reverse_signal_close_first）。

**显式关闭**：`exit_proposal_v1=False`（无 TP1/TP2 分批止盈）；`breakeven_r_mult=None`（分段保护用 ATR 1.0、随后按 R 口径递进）。

## 三、`mr_candidate` 预设开关

`run_portfolio_backtest(..., mr_candidate=True)` 等价于：

```python
enable_trend_routing  = False   # 无路由、无 RANGE 门
mr_entry_mode         = "ema_deviation"
mr_deviation_pct      = 0.01
exit_proposal_v1      = False
breakeven_r_mult      = None    # 改用 1.0 ATR / +0.2R、1.5R / +0.5R 的分段保护
bollinger_mid_exit    = True
bollinger_mid_ema     = True    # EMA20
bollinger_mid_period  = 20
bollinger_mid_hold_bars = 0
bollinger_mid_buffer_sigma = 0.0
bollinger_mid_partial = 1.0    # 全平
```

其余（选币 top-20、风控、共享退出）继承 `configs/default.yaml`。

## 四、关键代码位置

| 模块 | 文件 | 关键内容 |
|---|---|---|
| 回测入口 | `src/bot/backtest/portfolio.py` | `run_portfolio_backtest` + `mr_candidate` 预设（函数体开头）；`_range_deviation_signal`（L38）；中轨保护退出块（~L796）；退出链（~L764–940） |
| 选币打分 | `src/bot/data/klines.py` | `score_hot_candidates`（L141） |
| 旧 8 态分类 | `src/bot/regime.py` | `classify_regimes`（用于退出/regime 判定） |
| 信号/退出 | `src/bot/signals.py` | `generate_signal`（退出侧 + legacy 入场） |
| 风控 | `src/bot/risk.py` | `assess_entry` / `calculate_order_quantity` / `calculate_adaptive_stop` |
| 参数 | `configs/default.yaml` | 所有数值参数 |
| OOS 脚本 | `scripts/diagnose_mr_candidate_oos.py` | base vs candidate 验证 |

## 五、迁移到另一台电脑

### 需要的东西

1. **整个项目目录**（源码 `src/`、`configs/`、`scripts/`、`tests/`）。
2. **数据缓存**（可重新拉，或直接拷贝）：
   - `data/market.sqlite`：30 币 × ~400 天现货镜像 K 线（1H + 4H）。
   - `data/ab_funding.pkl`：30 币资金费率缓存（8h/4h 结算）。
3. **Python 环境**：依赖 `ccxt`、`pandas`、`pydantic`、`loguru`（项目已有 `install.ps1`）。

### 运行

```bash
python scripts/diagnose_mr_candidate_oos.py --window all --variant candidate
python scripts/diagnose_mr_candidate_oos.py --window all --window-days 60
```

若数据缓存路径不同，用 `--cache` / `--funding-cache` 指定。

### 模拟盘交易（新增 live runner）

回测之外，现已提供一个独立的模拟盘入口 `scripts/run_mr_candidate_paper.py`（复用 CTA 的 `MultiPaperState` + `DemoExecutionService`），把 `mr_candidate` 原样接到 live 执行：

```bash
python scripts/run_mr_candidate_paper.py --once   # local_paper（本地虚拟成交，无 API key），跑一轮
python scripts/run_mr_candidate_paper.py          # local_paper，连续跑
python scripts/run_mr_candidate_paper.py --demo   # demo_usdm（币安 USDT-M 测试网，需测试网 API key）
```

要点：

- 选币 = **动态热度 top-20**（`filter_tradable_crypto_universe` → `hot_activity_metrics` → `score_hot_candidates`），用实时 tickers。
- 入场 = ema_deviation（4H EMA50 偏离 ≥1% + 1H 方向），多空双向，固定 $3000 名义/仓、最多 `--max-positions`（默认 5）。
- 退出 = 结构止损（demo 挂在交易所 / local 本地跟踪）、1.0 ATR 锁 +0.2R、1.5R 锁 +0.5R、移动止损 2.0R、结构失效、regime 消退、无进展、最大持仓、**EMA20 中轨保护**。
- 状态落 `data/paper/mr_candidate_portfolio.json`，日志 `logs/mr_candidate_paper.log`，权益历史自动记录（可复用 `scripts/analyze_cta_oos.py` 思路做 OOS 快报）。
- 沙箱 local_paper 一轮已验证：live 数据拉到、ema_deviation 正确开出 2 个多仓（STRK/SUI，$3000 名义、结构止损、regime=LOW）。

## 六、诚实边界（务必先读）

1. **模拟盘 runner 已就位，但策略本身 OOS 已证伪**：`run_mr_candidate_paper.py` 已把 mr_candidate 接到 live 执行（动态热度 top-20 + ema_deviation + 中轨保护），可做真实前瞻 OOS 验证；但历史回测上中轨保护的 edge 不成立（见下），所以只用于观察/验证，别当可稳定盈利的策略实盘。
2. **OOS 已证伪中轨保护的 edge**：中轨保护在调参的 5 窗口（W1–W5）上 Δ=+1844，但换到更早 7 个窗口（W6–W12）Δ=−426。它在震荡/阴跌窗口锁利避回撤，在强上涨窗口砍爆炸性赢家（W7 几乎全由 PEPEUSDT 一只币驱动，+15336 被砍 −683）。
3. **MR 本身无稳定 alpha**：收益高度集中于少数爆炸币、受市场 regime 主导（部分窗口 +150%、部分 −20%）。这条线多次证伪，建议只作研究/回测，不当可稳定盈利的策略落地。
