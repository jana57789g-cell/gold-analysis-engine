
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

TIMEFRAMES = [
    "15min", "1h", "4h",
    "1day", "1week", "1month"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ready",
        "engine": "Gold Analysis Engine",
        "version": "1.0"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "status": "error",
            "message": "No JSON data received"
        }), 400

    # Support n8n's single-item array output
    if isinstance(data, list):
        if len(data) != 1 or not isinstance(data[0], dict):
            return jsonify({
                "status": "error",
                "message": "Expected one report object"
            }), 400
        data = data[0]

    timeframes = data.get("timeframes")

    if not isinstance(timeframes, dict):
        return jsonify({
            "status": "error",
            "message": "Missing timeframes object"
        }), 400

    results = {}
    total_candles = 0

    for tf in TIMEFRAMES:
        section = timeframes.get(tf)

        if not isinstance(section, dict):
            return jsonify({
                "status": "error",
                "message": f"Missing timeframe: {tf}"
            }), 400

        candles = section.get("candles")

        if not isinstance(candles, list) or not candles:
            return jsonify({
                "status": "error",
                "message": f"Missing candles: {tf}"
            }), 400

        valid = []

        for candle in candles:
            try:
                values = {
                    key: float(candle[key])
                    for key in ("open", "high", "low", "close")
                }

                if not all(
                    values["low"] <= values[key] <= values["high"]
                    for key in ("open", "close")
                ):
                    continue

                if values["low"] > values["high"]:
                    continue

                valid.append({
                    "datetime": candle.get("datetime"),
                    **values,
                    "is_incomplete": bool(
                        candle.get("is_incomplete", False)
                    )
                })

            except (KeyError, TypeError, ValueError):
                continue

        if not valid:
            return jsonify({
                "status": "error",
                "message": f"No valid candles: {tf}"
            }), 400

        valid.sort(
            key=lambda x: x["datetime"] or "",
            reverse=True
        )

        complete = [
            c for c in valid
            if not c["is_incomplete"]
        ]

        results[tf] = {
            "received": len(candles),
            "valid": len(valid),
            "complete": len(complete),
            "latest_close": (
                complete[0]["close"] if complete else None
            )
        }

        total_candles += len(valid)

    return jsonify({
        "status": "success",
        "symbol": data.get("summary", {}).get(
            "symbol", "XAU/USD"
        ),
        "engine": "Gold Analysis Engine",
        "received_timeframes": len(results),
        "total_valid_candles": total_candles,
        "validation": results,
        "decision": "WAIT",
        "message": "Data validated. Technical analysis is not implemented yet."
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
