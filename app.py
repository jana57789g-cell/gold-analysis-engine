import os
import math
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]
PIVOT_LEFT = 2
PIVOT_RIGHT = 2
MIN_COMPLETE_CANDLES = 10
PRICE_EPSILON = 1e-9


def parse_time(value):
    """Parse ISO-like timestamps without inventing a timezone."""
    if value is None or value == "":
        return None
    try:
        text = str(value).strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except (ValueError, TypeError, OverflowError):
        return None


def validate_candle(candle):
    if not isinstance(candle, dict):
        return None
    try:
        values = {key: float(candle[key]) for key in ("open", "high", "low", "close")}
        if not all(math.isfinite(value) for value in values.values()):
            return None
        if values["low"] > values["high"]:
            return None
        if not (values["low"] <= values["open"] <= values["high"] and
                values["low"] <= values["close"] <= values["high"]):
            return None
        return {
            "datetime": candle.get("datetime"),
            **values,
            "is_incomplete": candle.get("is_incomplete", False) is True
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _same_price(a, b):
    return math.isclose(a, b, rel_tol=PRICE_EPSILON, abs_tol=PRICE_EPSILON)


def find_pivots(candles):
    highs, lows = [], []
    for i in range(PIVOT_LEFT, len(candles) - PIVOT_RIGHT):
        current = candles[i]
        left = candles[i-PIVOT_LEFT:i]
        right = candles[i+1:i+PIVOT_RIGHT+1]

        # Strict pivots avoid arbitrarily choosing one candle from a flat plateau.
        if all(current["high"] > c["high"] for c in left + right):
            highs.append({
                "index": i, "price": current["high"],
                "datetime": current["datetime"],
                "confirmed_at_index": i + PIVOT_RIGHT
            })
        if all(current["low"] < c["low"] for c in left + right):
            lows.append({
                "index": i, "price": current["low"],
                "datetime": current["datetime"],
                "confirmed_at_index": i + PIVOT_RIGHT
            })
    return highs, lows


def classify_structure(highs, lows):
    if len(highs) < 2 or len(lows) < 2:
        return {
            "trend": "UNCONFIRMED", "structure": [],
            "reason": "Not enough confirmed swing points"
        }

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]

    if _same_price(h1["price"], h2["price"]):
        high_type = "EQH"
    else:
        high_type = "HH" if h2["price"] > h1["price"] else "LH"

    if _same_price(l1["price"], l2["price"]):
        low_type = "EQL"
    else:
        low_type = "HL" if l2["price"] > l1["price"] else "LL"

    if high_type == "HH" and low_type == "HL":
        trend = "BULLISH"
    elif high_type == "LH" and low_type == "LL":
        trend = "BEARISH"
    else:
        trend = "RANGING"

    return {
        "trend": trend,
        "structure": [
            {"type": high_type, "price": h2["price"]},
            {"type": low_type, "price": l2["price"]}
        ],
        "reason": "Based on the two latest confirmed swing highs and lows"
    }


def detect_breaks(candles, highs, lows, previous_trend):
    """Report only a fresh close-break on the latest complete candle."""
    if len(candles) < 2:
        return []

    latest_index = len(candles) - 1
    latest_close = candles[-1]["close"]
    events = []

    def fresh_break(pivot, direction):
        confirm_index = pivot["confirmed_at_index"]
        if confirm_index >= latest_index:
            return False
        level = pivot["price"]
        earlier = candles[confirm_index + 1:latest_index]
        if direction == "BULLISH":
            return (latest_close > level and
                    not any(c["close"] > level for c in earlier))
        return (latest_close < level and
                not any(c["close"] < level for c in earlier))

    if highs and fresh_break(highs[-1], "BULLISH"):
        event = ("BOS" if previous_trend == "BULLISH" else
                 "CHoCH" if previous_trend == "BEARISH" else
                 "BULLISH_STRUCTURE_BREAK")
        events.append({
            "event": event, "direction": "BULLISH",
            "level": highs[-1]["price"], "confirmed_by_close": True
        })

    if lows and fresh_break(lows[-1], "BEARISH"):
        event = ("BOS" if previous_trend == "BEARISH" else
                 "CHoCH" if previous_trend == "BULLISH" else
                 "BEARISH_STRUCTURE_BREAK")
        events.append({
            "event": event, "direction": "BEARISH",
            "level": lows[-1]["price"], "confirmed_by_close": True
        })

    return events


def analyze_timeframe(section):
    if not isinstance(section, dict):
        return {"status": "error", "message": "Invalid timeframe object"}

    candles_input = section.get("candles")
    if not isinstance(candles_input, list):
        return {"status": "error", "message": "Missing candles array"}

    valid = [c for item in candles_input if (c := validate_candle(item)) is not None]
    complete = [c for c in valid if not c["is_incomplete"]]

    if len(complete) < MIN_COMPLETE_CANDLES:
        return {
            "status": "insufficient_data",
            "received": len(candles_input), "valid": len(valid),
            "complete": len(complete),
            "reason": f"At least {MIN_COMPLETE_CANDLES} complete candles required"
        }

    parsed = [parse_time(c["datetime"]) for c in complete]
    if any(t is None for t in parsed):
        return {
            "status": "insufficient_data",
            "received": len(candles_input), "valid": len(valid),
            "complete": len(complete), "timestamp_reliable": False,
            "reason": "Cannot establish chronological order: one or more timestamps are invalid"
        }

    aware_flags = [t.utcoffset() is not None for t in parsed]
    if any(aware_flags) and not all(aware_flags):
        return {
            "status": "insufficient_data",
            "received": len(candles_input), "valid": len(valid),
            "complete": len(complete), "timestamp_reliable": False,
            "reason": "Mixed timezone-aware and timezone-naive timestamps"
        }

    indexed = list(zip(complete, parsed))
    indexed.sort(key=lambda pair: pair[1])
    ordered = [pair[0] for pair in indexed]
    ordered_times = [pair[1] for pair in indexed]

    if any(ordered_times[i] == ordered_times[i-1] for i in range(1, len(ordered_times))):
        return {
            "status": "insufficient_data",
            "received": len(candles_input), "valid": len(valid),
            "complete": len(complete), "timestamp_reliable": False,
            "reason": "Duplicate timestamps found; candle sequence is ambiguous"
        }

    highs, lows = find_pivots(ordered)
    structure = classify_structure(highs, lows)
    breaks = detect_breaks(ordered, highs, lows, structure["trend"])

    timezone_aware = all(aware_flags)
    return {
        "status": "analyzed",
        "received": len(candles_input),
        "valid": len(valid),
        "complete": len(ordered),
        "timestamp_reliable": True,
        "timestamp_timezone_aware": timezone_aware,
        "timestamp_basis": section.get("timestamp_basis", "unknown"),
        "latest_close": ordered[-1]["close"],
        "trend": structure["trend"],
        "structure": structure["structure"],
        "swing_highs": highs[-5:],
        "swing_lows": lows[-5:],
        "structure_events": breaks,
        "analysis_note": structure["reason"]
    }


def build_analysis(data):
    timeframes = data.get("timeframes")
    if not isinstance(timeframes, dict):
        return None, "Missing timeframes object"

    results, total_valid, trends = {}, 0, []
    for tf in TIMEFRAMES:
        section = timeframes.get(tf)
        if not isinstance(section, dict):
            return None, f"Missing timeframe: {tf}"
        result = analyze_timeframe(section)
        if result["status"] == "error":
            return None, f"{tf}: {result['message']}"
        total_valid += result.get("valid", 0)
        if result["status"] == "analyzed":
            trends.append(result["trend"])
        results[tf] = result

    usable = [trend for trend in trends if trend in ("BULLISH", "BEARISH")]
    if len(usable) >= 3 and len(set(usable)) == 1:
        overall = usable[0]
    elif len(usable) >= 3:
        overall = "CONFLICTING"
    else:
        overall = "UNCONFIRMED"

    return {
        "status": "success",
        "symbol": data.get("summary", {}).get("symbol", "XAU/USD"),
        "engine": "Gold Analysis Engine",
        "version": "1.2",
        "received_timeframes": len(results),
        "total_valid_candles": total_valid,
        "market_structure": results,
        "overall_structure": overall,
        "decision": "WAIT",
        "message": (
            "Market structure analysis completed. "
            "Liquidity and trade-entry modules are not implemented yet."
        )
    }, None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ready", "engine": "Gold Analysis Engine",
        "version": "1.2", "analysis": "Market Structure Engine"
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
