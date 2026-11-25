import os
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify

app = Flask(__name__)

VERSION = "2.0.0"
ENGINE_NAME = "Gold Analysis Engine"
SYMBOL = "XAU/USD"

TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]
MIN_COMPLETE_CANDLES = 10

# Pivot settings. These are rule parameters, not universal market truths.
SWING_LEFT = 2
SWING_RIGHT = 2
INTERNAL_LEFT = 1
INTERNAL_RIGHT = 1

# Signal settings. Tune only through out-of-sample testing.
MIN_SETUP_SCORE = 70
MIN_RISK_REWARD = 1.5
MAX_ENTRY_DISTANCE_ATR = 1.5
EQUAL_LEVEL_ATR_FRACTION = 0.10
ATR_PERIOD = 14
PRICE_EPSILON = 1e-9


# ============================================================
# COMMON / VALIDATION
# ============================================================

def parse_time(value):
    if value is None or value == "":
        return None
    try:
        value = str(value).strip()
        if value.endswith(("Z", "z")):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (ValueError, TypeError, OverflowError):
        return None


def same_price(a: float, b: float) -> bool:
    return math.isclose(
        float(a), float(b),
        rel_tol=PRICE_EPSILON,
        abs_tol=PRICE_EPSILON
    )


def validate_candle(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    try:
        values = {k: float(raw[k]) for k in ("open", "high", "low", "close")}
        if not all(math.isfinite(v) for v in values.values()):
            return None
        if values["low"] > values["high"]:
            return None
        if not (
            values["low"] <= values["open"] <= values["high"]
            and values["low"] <= values["close"] <= values["high"]
        ):
            return None
        timestamp = raw.get("datetime")
        if parse_time(timestamp) is None:
            return None
        return {
            "datetime": str(timestamp),
            **values,
            "is_incomplete": raw.get("is_incomplete", False) is True
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def normalize_candles(section: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = section.get("candles")
    if not isinstance(raw, list):
        raise ValueError("Missing candles array")

    valid, invalid_count = [], 0
    for item in raw:
        candle = validate_candle(item)
        if candle is None:
            invalid_count += 1
        else:
            valid.append(candle)

    complete = [c for c in valid if not c["is_incomplete"]]
    parsed = [(c, parse_time(c["datetime"])) for c in complete]
    parsed.sort(key=lambda item: item[1])

    for i in range(1, len(parsed)):
        if parsed[i][1] == parsed[i - 1][1]:
            raise ValueError("Duplicate timestamps found")

    ordered = [item[0] for item in parsed]
    aware = [item[1].utcoffset() is not None for item in parsed]
    if aware and any(aware) and not all(aware):
        raise ValueError("Mixed timezone-aware and naive timestamps")

    meta = {
        "received": len(raw),
        "valid": len(valid),
        "invalid": invalid_count,
        "incomplete_excluded": len(valid) - len(complete),
        "complete": len(ordered),
        "timestamp_timezone_aware": bool(aware and all(aware)),
        "timestamp_basis": section.get("timestamp_basis", "unknown"),
    }
    return ordered, meta


def calculate_atr(candles: List[Dict[str, Any]], period: int = ATR_PERIOD) -> float:
    if not candles:
        return 0.0
    trs = []
    for i, c in enumerate(candles):
        if i == 0:
            tr = c["high"] - c["low"]
        else:
            prev_close = candles[i - 1]["close"]
            tr = max(
                c["high"] - c["low"],
                abs(c["high"] - prev_close),
                abs(c["low"] - prev_close)
            )
        trs.append(max(0.0, tr))
    sample = trs[-period:]
    return sum(sample) / len(sample) if sample else 0.0


# ============================================================
# MARKET STRUCTURE
# ============================================================

def find_pivots(candles, left=2, right=2):
    highs, lows = [], []
    if len(candles) < left + right + 1:
        return highs, lows

    for i in range(left, len(candles) - right):
        current = candles[i]
        neighbors = candles[i-left:i] + candles[i+1:i+right+1]

        if all(current["high"] > c["high"] for c in neighbors):
            highs.append({
                "index": i,
                "price": current["high"],
                "datetime": current["datetime"],
                "confirmed_at_index": i + right
            })

        if all(current["low"] < c["low"] for c in neighbors):
            lows.append({
                "index": i,
                "price": current["low"],
                "datetime": current["datetime"],
                "confirmed_at_index": i + right
            })
    return highs, lows


def classify_structure(highs, lows):
    if len(highs) < 2 or len(lows) < 2:
        return {
            "trend": "UNCONFIRMED",
            "structure": [],
            "high_type": None,
            "low_type": None,
            "reason": "At least two confirmed swing highs and lows are required"
        }

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]

    if same_price(h1["price"], h2["price"]):
        high_type = "EQH"
    elif h2["price"] > h1["price"]:
        high_type = "HH"
    else:
        high_type = "LH"

    if same_price(l1["price"], l2["price"]):
        low_type = "EQL"
    elif l2["price"] > l1["price"]:
        low_type = "HL"
    else:
        low_type = "LL"

    if high_type == "HH" and low_type == "HL":
        trend = "BULLISH"
    elif high_type == "LH" and low_type == "LL":
        trend = "BEARISH"
    else:
        trend = "RANGING"

    structure = [
        {"type": high_type, "price": h2["price"], "datetime": h2["datetime"], "index": h2["index"]},
        {"type": low_type, "price": l2["price"], "datetime": l2["datetime"], "index": l2["index"]}
    ]
    return {
        "trend": trend,
        "structure": structure,
        "high_type": high_type,
        "low_type": low_type,
        "reason": "Based on the two latest confirmed swing highs and lows"
    }


def trend_at_index(highs, lows, index):
    known_highs = [p for p in highs if p["confirmed_at_index"] < index]
    known_lows = [p for p in lows if p["confirmed_at_index"] < index]
    return classify_structure(known_highs, known_lows)["trend"]


def detect_structure_events(candles, highs, lows):
    """
    Historical, close-confirmed break detection.
    A pivot is eligible only after its right-side confirmation candles have closed.
    Each pivot level can trigger at most one event.
    """
    events = []
    broken_highs, broken_lows = set(), set()

    for i in range(1, len(candles)):
        prev_close = candles[i - 1]["close"]
        close = candles[i]["close"]
        prior_trend = trend_at_index(highs, lows, i)

        eligible_highs = [
            (n, p) for n, p in enumerate(highs)
            if p["confirmed_at_index"] < i and n not in broken_highs
        ]
        eligible_lows = [
            (n, p) for n, p in enumerate(lows)
            if p["confirmed_at_index"] < i and n not in broken_lows
        ]

        crossed_up = [(n, p) for n, p in eligible_highs
                      if prev_close <= p["price"] < close]
        crossed_down = [(n, p) for n, p in eligible_lows
                        if prev_close >= p["price"] > close]

        # Record all levels crossed on this candle, but never re-emit the same pivot.
        for n, pivot in crossed_up:
            event_type = (
                "BOS" if prior_trend == "BULLISH"
                else "CHoCH" if prior_trend == "BEARISH"
                else "BULLISH_STRUCTURE_BREAK"
            )
            events.append({
                "event": event_type,
                "direction": "BULLISH",
                "level": pivot["price"],
                "pivot_index": pivot["index"],
                "pivot_datetime": pivot["datetime"],
                "confirmed_at_index": pivot["confirmed_at_index"],
                "break_candle_index": i,
                "break_datetime": candles[i]["datetime"],
                "previous_close": prev_close,
                "latest_close": close,
                "trend_before_break": prior_trend,
                "confirmed_by_close": True
            })
            broken_highs.add(n)

        for n, pivot in crossed_down:
            event_type = (
                "BOS" if prior_trend == "BEARISH"
                else "CHoCH" if prior_trend == "BULLISH"
                else "BEARISH_STRUCTURE_BREAK"
            )
            events.append({
                "event": event_type,
                "direction": "BEARISH",
                "level": pivot["price"],
                "pivot_index": pivot["index"],
                "pivot_datetime": pivot["datetime"],
                "confirmed_at_index": pivot["confirmed_at_index"],
                "break_candle_index": i,
                "break_datetime": candles[i]["datetime"],
                "previous_close": prev_close,
                "latest_close": close,
                "trend_before_break": prior_trend,
                "confirmed_by_close": True
            })
            broken_lows.add(n)

    return events


def analyze_structure_layer(candles, left, right, layer_name):
    highs, lows = find_pivots(candles, left, right)
    state = classify_structure(highs, lows)
    events = detect_structure_events(candles, highs, lows)
    return {
        "layer": layer_name,
        "trend": state["trend"],
        "structure": state["structure"],
        "swing_highs": highs[-10:],
        "swing_lows": lows[-10:],
        "structure_events": events[-20:],
        "total_confirmed_highs": len(highs),
        "total_confirmed_lows": len(lows),
        "analysis_note": state["reason"]
    }


# ============================================================
# LIQUIDITY / FVG / ORDER BLOCK / PREMIUM-DISCOUNT
# ============================================================

def detect_equal_levels(pivots, atr):
    tolerance = max(0.01, atr * EQUAL_LEVEL_ATR_FRACTION)
    pools, used = [], set()
    for i, first in enumerate(pivots):
        if i in used:
            continue
        matches = []
        for j in range(i + 1, len(pivots)):
            if abs(first["price"] - pivots[j]["price"]) <= tolerance:
                matches.append((j, pivots[j]))
        if matches:
            group = [first] + [item[1] for item in matches]
            for j, _ in matches:
                used.add(j)
            pools.append({
                "type": "EQUAL_HIGH" if "high" in first else "EQUAL_LEVEL",
                "level": sum(p["price"] for p in group) / len(group),
                "touches": len(group),
                "pivots": [{"price": p["price"], "datetime": p["datetime"], "index": p["index"]} for p in group]
            })
    return pools


def detect_liquidity(candles, highs, lows, atr):
    tolerance = max(0.01, atr * EQUAL_LEVEL_ATR_FRACTION)
    equal_highs, equal_lows = [], []
    for pivots, output, label in ((highs, equal_highs, "EQUAL_HIGHS"),
                                  (lows, equal_lows, "EQUAL_LOWS")):
        consumed = set()
        for i, p in enumerate(pivots):
            if i in consumed:
                continue
            group = [p]
            for j in range(i + 1, len(pivots)):
                if abs(p["price"] - pivots[j]["price"]) <= tolerance:
                    group.append(pivots[j])
                    consumed.add(j)
            if len(group) >= 2:
                output.append({
                    "type": label,
                    "level": sum(x["price"] for x in group) / len(group),
                    "touches": len(group),
                    "last_touch": group[-1]["datetime"]
                })

    sweeps = []
    for i, candle in enumerate(candles):
        known_highs = [p for p in highs if p["confirmed_at_index"] < i]
        known_lows = [p for p in lows if p["confirmed_at_index"] < i]
        for p in known_highs:
            if candle["high"] > p["price"] and candle["close"] < p["price"]:
                sweeps.append({
                    "type": "BUY_SIDE_LIQUIDITY_SWEEP",
                    "level": p["price"],
                    "datetime": candle["datetime"],
                    "index": i,
                    "pivot_datetime": p["datetime"]
                })
        for p in known_lows:
            if candle["low"] < p["price"] and candle["close"] > p["price"]:
                sweeps.append({
                    "type": "SELL_SIDE_LIQUIDITY_SWEEP",
                    "level": p["price"],
                    "datetime": candle["datetime"],
                    "index": i,
                    "pivot_datetime": p["datetime"]
                })

    return {
        "equal_highs": equal_highs[-10:],
        "equal_lows": equal_lows[-10:],
        "sweeps": sweeps[-20:]
    }


def detect_fvgs(candles):
    zones = []
    for i in range(2, len(candles)):
        a, c = candles[i - 2], candles[i]
        if c["low"] > a["high"]:
            zones.append({
                "type": "BULLISH_FVG",
                "low": a["high"],
                "high": c["low"],
                "midpoint": (a["high"] + c["low"]) / 2,
                "formed_at": c["datetime"],
                "formed_index": i
            })
        elif c["high"] < a["low"]:
            zones.append({
                "type": "BEARISH_FVG",
                "low": c["high"],
                "high": a["low"],
                "midpoint": (c["high"] + a["low"]) / 2,
                "formed_at": c["datetime"],
                "formed_index": i
            })

    # Mark whether price has subsequently traded back into each zone.
    for zone in zones:
        later = candles[zone["formed_index"] + 1:]
        zone["mitigated"] = any(
            c["low"] <= zone["high"] and c["high"] >= zone["low"]
            for c in later
        )
    return zones[-20:]


def detect_order_blocks(candles, atr):
    blocks = []
    if len(candles) < 4:
        return blocks
    bodies = [abs(c["close"] - c["open"]) for c in candles]
    for i in range(2, len(candles)):
        current = candles[i]
        avg_body = sum(bodies[max(0, i - 10):i]) / max(1, len(bodies[max(0, i - 10):i]))
        displacement = abs(current["close"] - current["open"]) >= max(avg_body * 1.5, atr * 0.5)
        if not displacement:
            continue
        direction = "BULLISH" if current["close"] > current["open"] else "BEARISH"
        # Find the nearest prior candle opposing the displacement.
        for j in range(i - 1, max(-1, i - 6), -1):
            prior = candles[j]
            opposing = (
                prior["close"] < prior["open"] if direction == "BULLISH"
                else prior["close"] > prior["open"]
            )
            if opposing:
                blocks.append({
                    "type": direction + "_ORDER_BLOCK",
                    "low": prior["low"],
                    "high": prior["high"],
                    "datetime": prior["datetime"],
                    "formed_by": current["datetime"],
                    "formed_index": i,
                    "mitigated": any(
                        later["low"] <= prior["high"] and later["high"] >= prior["low"]
                        for later in candles[i + 1:]
                    )
                })
                break
    return blocks[-15:]


def premium_discount(candles, highs, lows):
    if not highs or not lows:
        return {"status": "unavailable", "range_high": None, "range_low": None, "equilibrium": None}
    # Use the latest confirmed external high/low, not future/unconfirmed pivots.
    high = highs[-1]["price"]
    low = lows[-1]["price"]
    if high <= low:
        return {"status": "unavailable", "range_high": high, "range_low": low, "equilibrium": None}
    eq = (high + low) / 2
    price = candles[-1]["close"]
    return {
        "status": "available",
        "range_high": high,
        "range_low": low,
        "equilibrium": eq,
        "current_price": price,
        "zone": "DISCOUNT" if price < eq else "PREMIUM" if price > eq else "EQUILIBRIUM"
    }


# ============================================================
# TIMEFRAME + MULTI-TIMEFRAME ANALYSIS
# ============================================================

def analyze_timeframe(section):
    if not isinstance(section, dict):
        return {"status": "error", "message": "Invalid timeframe object"}

    try:
        candles, data_quality = normalize_candles(section)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if len(candles) < MIN_COMPLETE_CANDLES:
        return {
            "status": "insufficient_data",
            **data_quality,
            "reason": "Insufficient complete candles"
        }

    atr = calculate_atr(candles)
    swing = analyze_structure_layer(candles, SWING_LEFT, SWING_RIGHT, "SWING")
    internal = analyze_structure_layer(candles, INTERNAL_LEFT, INTERNAL_RIGHT, "INTERNAL")
    highs, lows = find_pivots(candles, SWING_LEFT, SWING_RIGHT)

    liquidity = detect_liquidity(candles, highs, lows, atr)
    fvgs = detect_fvgs(candles)
    order_blocks = detect_order_blocks(candles, atr)
    pd = premium_discount(candles, highs, lows)

    return {
        "status": "analyzed",
        **data_quality,
        "timestamp_reliable": True,
        "latest_datetime": candles[-1]["datetime"],
        "latest_close": candles[-1]["close"],
        "atr": atr,
        "trend": swing["trend"],
        "structure": swing["structure"],
        "swing_highs": swing["swing_highs"],
        "swing_lows": swing["swing_lows"],
        "structure_events": swing["structure_events"],
        "internal_structure": {
            "trend": internal["trend"],
            "structure": internal["structure"],
            "swing_highs": internal["swing_highs"],
            "swing_lows": internal["swing_lows"],
            "structure_events": internal["structure_events"]
        },
        "liquidity": liquidity,
        "fair_value_gaps": fvgs,
        "order_blocks": order_blocks,
        "premium_discount": pd,
        "structure_quality": {
            "confirmed_swing_highs": swing["total_confirmed_highs"],
            "confirmed_swing_lows": swing["total_confirmed_lows"],
            "confirmed_internal_highs": internal["total_confirmed_highs"],
            "confirmed_internal_lows": internal["total_confirmed_lows"]
        },
        "analysis_note": swing["analysis_note"]
    }


def calculate_overall_structure(results):
    usable = [
        results[tf]["trend"] for tf in TIMEFRAMES
        if results.get(tf, {}).get("status") == "analyzed"
        and results[tf]["trend"] in ("BULLISH", "BEARISH")
    ]
    if len(usable) < 3:
        return "UNCONFIRMED"
    bullish, bearish = usable.count("BULLISH"), usable.count("BEARISH")
    if bullish >= 3 and bullish > bearish:
        return "BULLISH"
    if bearish >= 3 and bearish > bullish:
        return "BEARISH"
    return "CONFLICTING"


# ============================================================
# ENTRY / RISK ENGINE
# ============================================================

def latest_signal_event(tf_result):
    events = tf_result.get("structure_events", [])
    if not events:
        return None
    return events[-1]


def latest_active_zone(tf_result, direction):
    expected_fvg = "BULLISH_FVG" if direction == "BUY" else "BEARISH_FVG"
    expected_ob = "BULLISH_ORDER_BLOCK" if direction == "BUY" else "BEARISH_ORDER_BLOCK"

    zones = []
    for z in tf_result.get("fair_value_gaps", []):
        if z["type"] == expected_fvg and not z["mitigated"]:
            zones.append({
                "source": "FVG", "low": z["low"], "high": z["high"],
                "datetime": z["formed_at"], "formed_index": z["formed_index"]
            })
    for z in tf_result.get("order_blocks", []):
        if z["type"] == expected_ob and not z["mitigated"]:
            zones.append({
                "source": "ORDER_BLOCK", "low": z["low"], "high": z["high"],
                "datetime": z["datetime"], "formed_index": z["formed_index"]
            })
    return zones[-1] if zones else None


def build_entry_setup(results, overall):
    # Entry timing is assessed on 15m; 1h is a secondary confirmation.
    fast = results.get("15min", {})
    hourly = results.get("1h", {})
    four_hour = results.get("4h", {})
    daily = results.get("1day", {})

    if any(x.get("status") != "analyzed" for x in (fast, hourly, four_hour, daily)):
        return {
            "status": "WAIT",
            "score": 0,
            "reason": "Required 15min, 1h, 4h and 1day data is not all available"
        }

    candidates = []
    for direction in ("BUY", "SELL"):
        wanted = "BULLISH" if direction == "BUY" else "BEARISH"
        score, reasons = 0, []

        if overall == wanted:
            score += 15
            reasons.append("Overall multi-timeframe structure agrees")
        if four_hour["trend"] == wanted:
            score += 15
            reasons.append("4h structure agrees")
        if daily["trend"] == wanted:
            score += 15
            reasons.append("1day structure agrees")
        if hourly["trend"] == wanted:
            score += 10
            reasons.append("1h structure agrees")
        if fast["trend"] == wanted:
            score += 10
            reasons.append("15min structure agrees")

        event = latest_signal_event(fast)
        if event and event["direction"] == wanted and event["event"] in ("BOS", "CHoCH"):
            score += 15
            reasons.append("Recent 15min close-confirmed BOS/CHoCH agrees")

        sweep_type = "SELL_SIDE_LIQUIDITY_SWEEP" if direction == "BUY" else "BUY_SIDE_LIQUIDITY_SWEEP"
        if any(s["type"] == sweep_type for s in fast["liquidity"]["sweeps"][-3:]):
            score += 10
            reasons.append("Recent liquidity sweep supports the setup")

        zone = latest_active_zone(fast, direction)
        if zone:
            score += 10
            reasons.append("Unmitigated 15min FVG/order-block zone exists")

        candidates.append((score, direction, reasons, zone))

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, direction, reasons, zone = candidates[0]

    price = fast["latest_close"]
    atr = fast["atr"]
    if score < MIN_SETUP_SCORE or zone is None or atr <= 0:
        return {
            "status": "WAIT",
            "score": score,
            "score_is_probability": False,
            "reason": "No setup currently meets the configured confluence and zone requirements",
            "best_candidate": direction,
            "reasons": reasons
        }

    distance = 0.0
    if price < zone["low"]:
        distance = zone["low"] - price
    elif price > zone["high"]:
        distance = price - zone["high"]

    if distance > atr * MAX_ENTRY_DISTANCE_ATR:
        return {
            "status": "WAIT",
            "score": score,
            "score_is_probability": False,
            "reason": "Price is too far from the identified entry zone",
            "best_candidate": direction,
            "entry_zone": zone,
            "reasons": reasons
        }

    entry_low, entry_high = zone["low"], zone["high"]
    entry = (entry_low + entry_high) / 2
    buffer = max(atr * 0.20, 0.01)

    if direction == "BUY":
        stop = zone["low"] - buffer
        risk = entry - stop
        targets = [entry + risk * rr for rr in (1.5, 2.0, 3.0)]
    else:
        stop = zone["high"] + buffer
        risk = stop - entry
        targets = [entry - risk * rr for rr in (1.5, 2.0, 3.0)]

    if risk <= 0:
        return {"status": "WAIT", "score": score, "reason": "Invalid risk calculation"}

    return {
        "status": "POTENTIAL_SETUP",
        "direction": direction,
        "score": score,
        "score_is_probability": False,
        "entry_zone": {"low": entry_low, "high": entry_high, "midpoint": entry},
        "entry_reference": entry,
        "stop_loss": stop,
        "targets": {"tp1": targets[0], "tp2": targets[1], "tp3": targets[2]},
        "risk_reward": {"tp1": 1.5, "tp2": 2.0, "tp3": 3.0},
        "zone_source": zone["source"],
        "reasons": reasons,
        "warning": "A rule-based research signal, not a guarantee or an automatic trade instruction."
    }


# ============================================================
# API
# ============================================================

def build_analysis(data):
    if not isinstance(data, dict):
        return None, "Invalid report object"

    timeframes = data.get("timeframes")
    if not isinstance(timeframes, dict):
        return None, "Missing timeframes object"

    results, total_valid = {}, 0
    for tf in TIMEFRAMES:
        section = timeframes.get(tf)
        if not isinstance(section, dict):
            return None, f"Missing timeframe: {tf}"
        result = analyze_timeframe(section)
        if result["status"] == "error":
            return None, f"{tf}: {result['message']}"
        total_valid += result.get("valid", 0)
        results[tf] = result

    overall = calculate_overall_structure(results)
    entry = build_entry_setup(results, overall)

    return {
        "status": "success",
        "symbol": data.get("summary", {}).get("symbol", SYMBOL),
        "engine": ENGINE_NAME,
        "version": VERSION,
        "received_timeframes": len(results),
        "total_valid_candles": total_valid,
        "overall_structure": overall,
        "market_structure": results,
        "entry_analysis": entry,
        "decision": "WAIT" if entry["status"] == "WAIT" else "POTENTIAL_SETUP",
        "message": (
            "Analysis completed. Entry score is a rule-confluence score, not a win probability. "
            "Validate all parameters with historical out-of-sample testing before live use."
        )
    }, None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ready",
        "engine": ENGINE_NAME,
        "version": VERSION,
        "symbol": SYMBOL,
        "timeframes": TIMEFRAMES,
        "modules": [
            "Data Validation", "Swing/Internal Market Structure",
            "Historical BOS/CHoCH", "Liquidity Pools/Sweeps",
            "Fair Value Gaps", "Basic Order Blocks",
            "Premium/Discount", "Multi-Timeframe Analysis",
            "Rule-Based Entry Scoring", "Risk/Reward Levels"
        ]
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    if isinstance(data, list):
        if len(data) != 1 or not isinstance(data[0], dict):
            return jsonify({"status": "error", "message": "Expected one report object"}), 400
        data = data[0]

    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Invalid JSON structure"}), 400

    try:
        result, error = build_analysis(data)
        if error:
            return jsonify({"status": "error", "message": error}), 400
        return jsonify(result)
    except Exception:
        app.logger.exception("Analysis engine error")
        return jsonify({"status": "error", "message": "Internal analysis error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
