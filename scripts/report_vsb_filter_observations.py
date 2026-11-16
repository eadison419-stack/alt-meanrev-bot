"""Read-only report for VSB signals rejected by the shared time filter."""

import argparse
import json
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 3, 6, 12, 24)


def summarize(events):
    observations = [event for event in events
                    if event.get("event") == "vsb_time_filter_observation"]
    grouped = {}
    for observation in observations:
        reason = ",".join(observation.get("filter_reasons", [])) or "unknown"
        rows = grouped.setdefault(reason, {horizon: [] for horizon in HORIZONS})
        for horizon in HORIZONS:
            result = observation.get("horizons", {}).get(f"{horizon}h", {})
            if result.get("status") != "complete":
                continue
            if result.get("observation_gaps", 0):
                continue
            rows[horizon].append(result)
    return observations, grouped


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def render_report(events):
    observations, grouped = summarize(events)
    active = sum(event.get("status") == "tracking" for event in observations)
    completed = sum(event.get("status") == "complete" for event in observations)
    lines = ["VSB 时段过滤反事实观察（信号收盘价基准，未计手续费／滑点）",
             f"已记录：{len(observations)}；跟踪中：{active}；24 小时观察完成：{completed}"]
    if not observations:
        lines.append("暂无样本；新版本加载后，符合 VSB 条件但被时段过滤拦截的信号会开始记录。")
        return "\n".join(lines)
    for reason, horizons in sorted(grouped.items()):
        lines.extend(["", f"过滤原因：{reason}"])
        for horizon in HORIZONS:
            rows = horizons[horizon]
            if not rows:
                lines.append(f"  {horizon:>2}h：完整样本 0")
                continue
            close_r = _mean(rows, "forward_close_r")
            mfe_r = _mean(rows, "mfe_r")
            mae_r = _mean(rows, "mae_r")
            win_rate = 100 * sum(float(row["forward_close_r"]) > 0 for row in rows) / len(rows)
            stop_rate = 100 * sum(bool(row.get("initial_stop_touched")) for row in rows) / len(rows)
            lines.append(
                f"  {horizon:>2}h：n={len(rows)}，均值 R={close_r:+.3f}，"
                f"正收益率={win_rate:.1f}% ，平均 MFE={mfe_r:.3f}R，"
                f"平均 MAE={mae_r:.3f}R，初始止损触及率={stop_rate:.1f}%"
            )
    lines.append("说明：这是被过滤信号的价格路径统计，不等同于可成交收益；限价订单、资金费及费用未模拟。")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path,
                        default=PROJECT_ROOT / "data" / "vsb_shadow_ledger.json",
                        help="VSB shadow ledger path (default: data/vsb_shadow_ledger.json)")
    args = parser.parse_args(argv)
    try:
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"无法读取影子账本：{exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"影子账本 JSON 无效：{exc}", file=sys.stderr)
        return 2
    print(render_report(ledger.get("events", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
