import os
import math
from datetime import datetime, timezone

from flask import Flask, request, jsonify

app = Flask(__name__)

TIMEFRAMES = [
    "15min", "1h", "4h",
    "1day", "1week", "1month"
]

PIVOT_LEFT = 2
PIVOT_RIGHT = 2


def parse_time(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        return None


def validate_candle(candle):
    if not isinstance(candle, dict):
        return None

    try:
        values = {
            key: float(candle[key])
            for key in ("open", "high", "low", "close")
        }

        if not all(math.isfinite(v) for v in values.values()):
            return None

        if values["low"] > values["high"]:
            return None

        if not (
            values["low"] <= values["open"] <= values["high"]
            and values["low"] <= values["close"] <= values["high"]
        ):
            return None

        return {
            "datetime": candle.get("datetime"),
            **values,
            "is_incomplete": bool(
                candle.get("is_incomplete", False)
            )
        }

    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def find_pivots(candles):
    highs = []
    lows = []

    left = PIVOT_LEFT
    right = PIVOT_RIGHT

    for i in range(left, len(candles) - right):
        current = candles[i]

        previous = candles[i-left:i]
        following = candles[i+1:i+right+1]

        is_high = all(
            current["high"] > c["high"]
            for c in previous + following
        )

        is_low = all(
            current["low"] < c["low"]
            for c in previous + following
        )

        if is_high:
            highs.append({
                "index": i,
                "price": current["high"],
                "datetime": current["datetime"]
            })

        if is_low:
            lows.append({
                "index": i,
                "price": current["low"],
                "datetime": current["datetime"]
            })

    return highs, lows


def classify_structure(highs, lows):
    if len(highs) < 2 or len(lows) < 2:
        return {
            "trend": "UNCONFIRMED",
            "structure": [],
            "reason": "Not enough confirmed swing points"
        }

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]

    higher_high = h2["price"] > h1["price"]
    higher_low = l2["price"] > l1["price"]

    lower_high = h2["price"] < h1["price"]
    lower_low = l2["price"] < l1["price"]

    structure = []

    structure.append({
        "type": "HH" if higher_high else (
            "LH" if lower_high else "EQH"
        ),
        "price": h2["price"]
    })

    structure.append({
        "type": "HL" if higher_low else (
            "LL" if lower_low else "EQL"
        ),
        "price": l2["price"]
    })

    if higher_high and higher_low:
        trend = "BULLISH"

    elif lower_high and lower_low:
        trend = "BEARISH"

    else:
        trend = "RANGING"

    return {
        "trend": trend,
        "structure": structure,
        "reason": "Based on last two confirmed swing highs and lows"
    }


def detect_breaks(candles, highs, lows, previous_trend):
    if not candles or not highs or not lows:
        return []

    events = []

    latest = candles[-1]
    close = latest["close"]

    resistance = highs[-1]
    support = lows[-1]

    if close > resistance["price"]:
        event = (
            "BOS" if previous_trend == "BULLISH"
            else "CHoCH" if previous_trend == "BEARISH"
            else "BULLISH_STRUCTURE_BREAK"
        )

        events.append({
            "event": event,
            "direction": "BULLISH",
            "level": resistance["price"],
            "confirmed_by_close": True
        })

    elif close < support["price"]:
        event = (
            "BOS" if previous_trend == "BEARISH"
            else "CHoCH" if previous_trend == "BULLISH"
            else "BEARISH_STRUCTURE_BREAK"
        )

        events.append({
            "event": event,
            "direction": "BEARISH",
            "level": support["price"],
            "confirmed_by_close": True
        })

    return events


def analyze_timeframe(section):
    candles_input = section.get("candles")

    if not isinstance(candles_input, list):
        return {
            "status": "error",
            "message": "Missing candles array"
        }

    valid = []

    for item in candles_input:
        candle = validate_candle(item)

        if candle:
            valid.append(candle)

    complete = [
        c for c in valid
        if not c["is_incomplete"]
    ]

    if len(complete) < 10:
        return {
            "status": "insufficient_data",
            "received": len(candles_input),
            "valid": len(valid),
            "complete": len(complete),
            "reason": "At least 10 complete candles required"
        }

    parsed_times = [
        parse_time(c["datetime"])
        for c in complete
    ]

    time_reliable = all(t is not None for t in parsed_times)

    if time_reliable:
        complete.sort(
            key=lambda c: parse_time(c["datetime"])
        )

    else:
        # Original data order is retained if timestamps
        # cannot be parsed reliably.
        # Input order must be verified separately.
        pass

    highs, lows = find_pivots(complete)

    structure = classify_structure(highs, lows)

    breaks = detect_breaks(
        complete,
        highs,
        lows,
        structure["trend"]
    )

    return {
        "status": "analyzed",
        "received": len(candles_input),
        "valid": len(valid),
        "complete": len(complete),
        "timestamp_reliable": time_reliable,
        "latest_close": complete[-1]["close"],
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

    results = {}
    total_valid = 0
    trends = []

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

    usable = [
        value for value in trends
        if value in ("BULLISH", "BEARISH")
    ]

    if len(usable) >= 3 and len(set(usable)) == 1:
        overall = usable[0]
    elif len(usable) >= 3:
        overall = "CONFLICTING"
    else:
        overall = "UNCONFIRMED"

    return {
        "status": "success",
        "symbol": data.get("summary", {}).get(
            "symbol", "XAU/USD"
        ),
        "engine": "Gold Analysis Engine",
        "version": "1.1",
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
        "status": "ready",
        "engine": "Gold Analysis Engine",
        "version": "1.1",
        "analysis": "Market Structure Engine"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "status": "error",
            "message": "No JSON data received"
        }), 400

    if isinstance(data, list):
        if len(data) != 1 or not isinstance(data[0], dict):
            return jsonify({
                "status": "error",
                "message": "Expected one report object"
            }), 400

        data = data[0]

    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Invalid JSON structure"
        }), 400

    try:
        result, error = build_analysis(data)

        if error:
            return jsonify({
                "status": "error",
                "message": error
            }), 400

        return jsonify(result)

    except Exception:
        app.logger.exception("Analysis engine error")

        return jsonify({
            "status": "error",
            "message": "Internal analysis error"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
        
