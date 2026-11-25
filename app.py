
import os
import math
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

VERSION = "1.3"

TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]

MIN_COMPLETE_CANDLES = 10
PRICE_EPSILON = 1e-9

SWING_LEFT = 2
SWING_RIGHT = 2

INTERNAL_LEFT = 1
INTERNAL_RIGHT = 1


# ==============================
# DATA VALIDATION
# ==============================

def parse_time(value):
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
            "is_incomplete": candle.get("is_incomplete", False) is True
        }

    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def same_price(a, b):
    return math.isclose(
        a, b,
        rel_tol=PRICE_EPSILON,
        abs_tol=PRICE_EPSILON
    )


# ==============================
# PIVOT DETECTION
# ==============================

def find_pivots(candles, left=2, right=2):

    highs = []
    lows = []

    for i in range(left, len(candles) - right):

        current = candles[i]

        left_candles = candles[i-left:i]
        right_candles = candles[i+1:i+right+1]

        neighbors = left_candles + right_candles

        if all(
            current["high"] > c["high"]
            for c in neighbors
        ):

            highs.append({
                "index": i,
                "price": current["high"],
                "datetime": current["datetime"],
                "confirmed_at_index": i + right
            })

        if all(
            current["low"] < c["low"]
            for c in neighbors
        ):

            lows.append({
                "index": i,
                "price": current["low"],
                "datetime": current["datetime"],
                "confirmed_at_index": i + right
            })

    return highs, lows


# ==============================
# MARKET STRUCTURE
# ==============================

def classify_structure(highs, lows):

    if len(highs) < 2 or len(lows) < 2:

        return {
            "trend": "UNCONFIRMED",
            "structure": [],
            "reason": "Not enough confirmed swing points"
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

    return {
        "trend": trend,

        "structure": [
            {
                "type": high_type,
                "price": h2["price"],
                "datetime": h2["datetime"],
                "index": h2["index"]
            },
            {
                "type": low_type,
                "price": l2["price"],
                "datetime": l2["datetime"],
                "index": l2["index"]
            }
        ],

        "reason": "Based on the two latest confirmed swing highs and lows"
    }


# ==============================
# BOS / CHoCH DETECTION
# ==============================

def detect_breaks(candles, highs, lows, previous_trend):

    if len(candles) < 2:
        return []

    latest_index = len(candles) - 1
    previous_index = latest_index - 1

    latest_close = candles[-1]["close"]
    previous_close = candles[-2]["close"]

    events = []

    def check_breaks(pivots, direction):

        detected = []

        for pivot in pivots:

            confirm_index = pivot["confirmed_at_index"]

            if confirm_index > previous_index:
                continue

            level = pivot["price"]

            if direction == "BULLISH":

                crossed = (
                    previous_close <= level
                    and latest_close > level
                )

            else:

                crossed = (
                    previous_close >= level
                    and latest_close < level
                )

            if not crossed:
                continue

            if direction == "BULLISH":

                event_type = (
                    "BOS"
                    if previous_trend == "BULLISH"
                    else "CHoCH"
                    if previous_trend == "BEARISH"
                    else "BULLISH_STRUCTURE_BREAK"
                )

            else:

                event_type = (
                    "BOS"
                    if previous_trend == "BEARISH"
                    else "CHoCH"
                    if previous_trend == "BULLISH"
                    else "BEARISH_STRUCTURE_BREAK"
                )

            detected.append({

                "event": event_type,
                "direction": direction,

                "level": level,

                "pivot_index": pivot["index"],

                "pivot_datetime": pivot["datetime"],

                "confirmed_at_index": confirm_index,

                "break_candle_index": latest_index,

                "break_datetime": candles[-1]["datetime"],

                "previous_close": previous_close,

                "latest_close": latest_close,

                "confirmed_by_close": True

            })

        return detected

    events.extend(check_breaks(highs, "BULLISH"))
    events.extend(check_breaks(lows, "BEARISH"))

    return events


# ==============================
# STRUCTURE ENGINE
# ==============================

def analyze_structure_layer(
    candles,
    left,
    right,
    layer_name
):

    highs, lows = find_pivots(
        candles,
        left=left,
        right=right
    )

    structure = classify_structure(highs, lows)

    events = detect_breaks(
        candles,
        highs,
        lows,
        structure["trend"]
    )

    return {

        "layer": layer_name,

        "trend": structure["trend"],

        "structure": structure["structure"],

        "swing_highs": highs[-5:],

        "swing_lows": lows[-5:],

        "structure_events": events,

        "total_confirmed_highs": len(highs),

        "total_confirmed_lows": len(lows),

        "analysis_note": structure["reason"]

    }


# ==============================
# TIMEFRAME ANALYSIS
# ==============================

def analyze_timeframe(section):

    if not isinstance(section, dict):

        return {
            "status": "error",
            "message": "Invalid timeframe object"
        }

    candles_input = section.get("candles")

    if not isinstance(candles_input, list):

        return {
            "status": "error",
            "message": "Missing candles array"
        }

    valid = []

    for item in candles_input:

        candle = validate_candle(item)

        if candle is not None:
            valid.append(candle)

    complete = [
        c for c in valid
        if not c["is_incomplete"]
    ]

    if len(complete) < MIN_COMPLETE_CANDLES:

        return {
            "status": "insufficient_data",
            "received": len(candles_input),
            "valid": len(valid),
            "complete": len(complete),
            "reason": "Insufficient complete candles"
        }

    parsed = [
        parse_time(c["datetime"])
        for c in complete
    ]

    if any(t is None for t in parsed):

        return {
            "status": "insufficient_data",
            "received": len(candles_input),
            "valid": len(valid),
            "complete": len(complete),
            "timestamp_reliable": False,
            "reason": "Invalid timestamps"
        }

    aware_flags = [
        t.utcoffset() is not None
        for t in parsed
    ]

    if any(aware_flags) and not all(aware_flags):

        return {
            "status": "insufficient_data",
            "received": len(candles_input),
            "valid": len(valid),
            "complete": len(complete),
            "timestamp_reliable": False,
            "reason": "Mixed timezone-aware and naive timestamps"
        }

    indexed = list(zip(complete, parsed))

    indexed.sort(key=lambda pair: pair[1])

    ordered = [pair[0] for pair in indexed]

    ordered_times = [pair[1] for pair in indexed]

    if any(
        ordered_times[i] == ordered_times[i-1]
        for i in range(1, len(ordered_times))
    ):

        return {
            "status": "insufficient_data",
            "received": len(candles_input),
            "valid": len(valid),
            "complete": len(complete),
            "timestamp_reliable": False,
            "reason": "Duplicate timestamps found"
        }

    # Main swing structure

    swing = analyze_structure_layer(
        ordered,
        SWING_LEFT,
        SWING_RIGHT,
        "SWING"
    )

    # Internal structure

    internal = analyze_structure_layer(
        ordered,
        INTERNAL_LEFT,
        INTERNAL_RIGHT,
        "INTERNAL"
    )

    return {

        "status": "analyzed",

        "received": len(candles_input),

        "valid": len(valid),

        "complete": len(ordered),

        "timestamp_reliable": True,

        "timestamp_timezone_aware": all(aware_flags),

        "timestamp_basis": section.get(
            "timestamp_basis",
            "unknown"
        ),

        "latest_close": ordered[-1]["close"],

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

        "structure_quality": {

            "confirmed_swing_highs":
                swing["total_confirmed_highs"],

            "confirmed_swing_lows":
                swing["total_confirmed_lows"],

            "confirmed_internal_highs":
                internal["total_confirmed_highs"],

            "confirmed_internal_lows":
                internal["total_confirmed_lows"]

        },

        "analysis_note": swing["analysis_note"]

    }


# ==============================
# MULTI-TIMEFRAME ENGINE
# ==============================

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
        trend for trend in trends
        if trend in ("BULLISH", "BEARISH")
    ]

    if len(usable) >= 3 and len(set(usable)) == 1:

        overall = usable[0]

    elif len(usable) >= 3:

        overall = "CONFLICTING"

    else:

        overall = "UNCONFIRMED"

    return {

        "status": "success",

        "symbol": data.get(
            "summary", {}
        ).get("symbol", "XAU/USD"),

        "engine": "Gold Analysis Engine",

        "version": VERSION,

        "received_timeframes": len(results),

        "total_valid_candles": total_valid,

        "market_structure": results,

        "overall_structure": overall,

        "decision": "WAIT",

        "message": (
            "Market Structure Engine v1.3 completed. "
            "Liquidity, SMC, ICT and trade-entry modules "
            "are not implemented yet."
        )

    }, None


# ==============================
# FLASK ROUTES
# ==============================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "status": "ready",

        "engine": "Gold Analysis Engine",

        "version": VERSION,

        "analysis": "Market Structure Engine",

        "modules": [
            "Swing Structure",
            "Internal Structure",
            "BOS",
            "CHoCH",
            "Multi-Timeframe Structure"
        ]

    })


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json(silent=True)

    if data is None:

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

    app.run(
        host="0.0.0.0",
        port=port
    )
