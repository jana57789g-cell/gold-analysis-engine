import os
import math
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify

VERSION = "4.0.1"
ENGINE_NAME = "Gold Analysis Engine"
SYMBOL = "XAU/USD"
TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]
MAX_CANDLES = 300
MIN_COMPLETE_CANDLES = 30

app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(ENGINE_NAME.replace(" ", "_"))

# ------------------------- Data validation -------------------------
def _parse_time(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        # Accept seconds or milliseconds since epoch.
        stamp = float(value)
        if abs(stamp) > 1e12:
            stamp /= 1000.0
        return datetime.fromtimestamp(stamp, tz=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_or_invalid_timestamp")
    raw = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(value.strip(), fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            raise ValueError("unparseable_timestamp")
    if dt.tzinfo is None:
        # Naive timestamps are interpreted as UTC, consistently and explicitly.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid_{field}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid_{field}")
    if not math.isfinite(number):
        raise ValueError(f"non_finite_{field}")
    return number


def _extract_candle_array(value: Any) -> Optional[List[dict]]:
    """Accept a timeframe as a candle array or a common n8n wrapper object."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        # Common shapes: {"candles": [...]}, {"data": [...]}, {"items": [...]}
        for key in ("candles", "data", "items", "records", "values"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
        # Some nodes wrap the timeframe one level deeper.
        for key in ("result", "output", "body"):
            nested = value.get(key)
            if isinstance(nested, dict):
                extracted = _extract_candle_array(nested)
                if extracted is not None:
                    return extracted
    return None


def normalize_payload(payload: Any) -> Tuple[Dict[str, List[dict]], List[str]]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return {}, ["payload_must_be_object"]

    # Accept either {"timeframes": {...}} or the timeframe keys at the root.
    raw_frames = payload.get("timeframes", payload)
    if not isinstance(raw_frames, dict):
        return {}, ["timeframes_must_be_object"]

    # Also accept a common n8n wrapper: {"data": {"timeframes": {...}}}.
    if not any(tf in raw_frames for tf in TIMEFRAMES):
        for wrapper_key in ("data", "body", "result", "output"):
            nested = raw_frames.get(wrapper_key)
            if isinstance(nested, dict):
                candidate = nested.get("timeframes", nested)
                if isinstance(candidate, dict) and any(tf in candidate for tf in TIMEFRAMES):
                    raw_frames = candidate
                    break

    normalized: Dict[str, List[dict]] = {}
    for tf in TIMEFRAMES:
        raw_value = raw_frames.get(tf)
        if raw_value is None:
            errors.append(f"{tf}:missing_timeframe")
            continue
        rows = _extract_candle_array(raw_value)
        if rows is None:
            errors.append(f"{tf}:candles_must_be_array_or_wrapped_array")
            continue
        if len(rows) > MAX_CANDLES:
            rows = rows[-MAX_CANDLES:]

        parsed: List[dict] = []
        row_errors: List[str] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                row_errors.append(f"{tf}[{idx}]:candle_must_be_object")
                continue
            try:
                ts_value = row.get("datetime", row.get("timestamp", row.get("time", row.get("date"))))
                dt = _parse_time(ts_value)
                o = _finite_number(row.get("open"), "open")
                h = _finite_number(row.get("high"), "high")
                l = _finite_number(row.get("low"), "low")
                c = _finite_number(row.get("close"), "close")
                if min(o, h, l, c) <= 0:
                    raise ValueError("non_positive_ohlc")
                if h < max(o, c, l) or l > min(o, c, h) or h < l:
                    raise ValueError("inconsistent_ohlc")
                vol = row.get("volume")
                if vol is not None:
                    vol = _finite_number(vol, "volume")
                    if vol < 0:
                        raise ValueError("negative_volume")
                incomplete = row.get("is_incomplete", row.get("incomplete", False))
                if not isinstance(incomplete, bool):
                    raise ValueError("invalid_is_incomplete")
                parsed.append({"datetime": dt.isoformat(), "_ts": dt.timestamp(), "open": o, "high": h,
                               "low": l, "close": c, "volume": vol, "is_incomplete": incomplete})
            except Exception as exc:
                row_errors.append(f"{tf}[{idx}]:{str(exc)}")

        # Sort by UTC instant; duplicate instants are rejected rather than silently merged.
        parsed.sort(key=lambda x: x["_ts"])
        unique: List[dict] = []
        seen = set()
        for candle in parsed:
            key = candle["_ts"]
            if key in seen:
                row_errors.append(f"{tf}:duplicate_timestamp:{candle['datetime']}")
                continue
            seen.add(key)
            unique.append(candle)
        complete = [x for x in unique if not x["is_incomplete"]]
        if len(complete) < MIN_COMPLETE_CANDLES:
            row_errors.append(f"{tf}:insufficient_complete_candles:{len(complete)}<{MIN_COMPLETE_CANDLES}")
        errors.extend(row_errors)
        normalized[tf] = complete
    return normalized, errors


def data_quality_report(frames: Dict[str, List[dict]], errors: List[str]) -> dict:
    return {
        "status": "VALID" if not errors and all(tf in frames and len(frames[tf]) >= MIN_COMPLETE_CANDLES for tf in TIMEFRAMES) else "INVALID",
        "max_candles_per_timeframe": MAX_CANDLES,
        "timeframes": {tf: {"complete_count": len(frames.get(tf, [])),
                            "latest_datetime": frames[tf][-1]["datetime"] if frames.get(tf) else None,
                            "latest_close": frames[tf][-1]["close"] if frames.get(tf) else None}
                       for tf in TIMEFRAMES},
        "errors": errors,
        "self_healing": {"applied": ["UTC timestamp normalization", "chronological sorting", "safe numeric parsing"],
                         "not_applied": ["OHLC price alteration", "fabricating missing candles", "marking incomplete candles complete"]}
    }

# ------------------------- Indicators and structure -------------------------
def atr(candles: List[dict], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        cur, prev = candles[i], candles[i - 1]
        trs.append(max(cur["high"] - cur["low"], abs(cur["high"] - prev["close"]), abs(cur["low"] - prev["close"])))
    sample = trs[-period:]
    return sum(sample) / len(sample) if sample else 0.0


def pivots(candles: List[dict], left: int = 3, right: int = 3) -> Tuple[List[dict], List[dict]]:
    highs, lows = [], []
    for i in range(left, len(candles) - right):
        h, l = candles[i]["high"], candles[i]["low"]
        if all(h > candles[j]["high"] for j in range(i-left, i)) and all(h >= candles[j]["high"] for j in range(i+1, i+right+1)):
            highs.append({"index": i, "price": h, "datetime": candles[i]["datetime"]})
        if all(l < candles[j]["low"] for j in range(i-left, i)) and all(l <= candles[j]["low"] for j in range(i+1, i+right+1)):
            lows.append({"index": i, "price": l, "datetime": candles[i]["datetime"]})
    return highs, lows


def structure_summary(candles: List[dict]) -> dict:
    highs, lows = pivots(candles)
    close = candles[-1]["close"]
    recent_highs, recent_lows = highs[-4:], lows[-4:]
    hh = len(recent_highs) >= 2 and recent_highs[-1]["price"] > recent_highs[-2]["price"]
    hl = len(recent_lows) >= 2 and recent_lows[-1]["price"] > recent_lows[-2]["price"]
    lh = len(recent_highs) >= 2 and recent_highs[-1]["price"] < recent_highs[-2]["price"]
    ll = len(recent_lows) >= 2 and recent_lows[-1]["price"] < recent_lows[-2]["price"]
    direction = "BULLISH" if hh and hl else "BEARISH" if lh and ll else "RANGE"
    last_high = recent_highs[-1] if recent_highs else None
    last_low = recent_lows[-1] if recent_lows else None
    bos = "BULLISH_BOS" if last_high and close > last_high["price"] else "BEARISH_BOS" if last_low and close < last_low["price"] else None
    prev_dir = "BULLISH" if hh and hl else "BEARISH" if lh and ll else "RANGE"
    choch = "BULLISH_CHOCH" if prev_dir == "BEARISH" and bos == "BULLISH_BOS" else "BEARISH_CHOCH" if prev_dir == "BULLISH" and bos == "BEARISH_BOS" else None
    return {"direction": direction, "bos": bos, "choch": choch,
            "swing_high": last_high, "swing_low": last_low,
            "higher_high": hh, "higher_low": hl, "lower_high": lh, "lower_low": ll}


def candle_rejection(c: dict, side: str) -> float:
    rng = max(c["high"] - c["low"], 1e-12)
    body = abs(c["close"] - c["open"])
    lower_wick = min(c["open"], c["close"]) - c["low"]
    upper_wick = c["high"] - max(c["open"], c["close"])
    wick = lower_wick if side == "BUY" else upper_wick
    close_position = (c["close"] - c["low"]) / rng
    close_bonus = close_position if side == "BUY" else 1 - close_position
    return min(1.0, 0.55 * wick / rng + 0.25 * (1 - body / rng) + 0.20 * close_bonus)


def liquidity_map(candles: List[dict], current: float, volatility: float) -> dict:
    highs, lows = pivots(candles)
    tolerance = max(volatility * 0.18, current * 0.0005)
    def clusters(points, label):
        groups = []
        for p in points:
            placed = False
            for g in groups:
                if abs(p["price"] - sum(x["price"] for x in g) / len(g)) <= tolerance:
                    g.append(p); placed = True; break
            if not placed: groups.append([p])
        out = []
        for g in groups:
            if len(g) >= 2:
                price = sum(x["price"] for x in g) / len(g)
                out.append({"price": round(price, 5), "touches": len(g), "type": label})
        return out
    eqh = clusters(highs, "EQUAL_HIGHS")
    eql = clusters(lows, "EQUAL_LOWS")
    # Sweep: latest candle takes a prior pivot and closes back inside.
    last = candles[-1]
    prior_highs = [p["price"] for p in highs if p["index"] < len(candles)-1]
    prior_lows = [p["price"] for p in lows if p["index"] < len(candles)-1]
    swept_high = bool(prior_highs and last["high"] > max(prior_highs[-5:]) and last["close"] < max(prior_highs[-5:]))
    swept_low = bool(prior_lows and last["low"] < min(prior_lows[-5:]) and last["close"] > min(prior_lows[-5:]))
    above = sorted([x for x in eqh if x["price"] > current], key=lambda x: x["price"])
    below = sorted([x for x in eql if x["price"] < current], key=lambda x: x["price"], reverse=True)
    return {"buy_side_liquidity": above, "sell_side_liquidity": below,
            "equal_highs": eqh, "equal_lows": eql,
            "recent_sweep": "BUY_SIDE_SWEPT" if swept_high else "SELL_SIDE_SWEPT" if swept_low else None,
            "nearest_above": above[0] if above else None, "nearest_below": below[0] if below else None}


def build_zones(candles: List[dict], current: float, side: str, tf: str) -> List[dict]:
    highs, lows = pivots(candles)
    vol = atr(candles)
    if vol <= 0: return []
    points = lows if side == "BUY" else highs
    zones = []
    for p in points[-20:]:
        i = p["index"]
        candle = candles[i]
        if side == "BUY":
            lo, hi = candle["low"], max(candle["open"], candle["close"])
            reaction = max((x["high"] - candle["low"] for x in candles[i+1:min(i+6, len(candles))]), default=0)
        else:
            lo, hi = min(candle["open"], candle["close"]), candle["high"]
            reaction = max((candle["high"] - x["low"] for x in candles[i+1:min(i+6, len(candles))]), default=0)
        width = max(hi-lo, vol*0.08)
        if width > vol*1.5: continue
        age = len(candles)-1-i
        if age > 120: continue
        touches = sum(1 for x in candles[i+1:] if x["low"] <= hi and x["high"] >= lo)
        broken = any((x["close"] < lo-vol*0.05 if side == "BUY" else x["close"] > hi+vol*0.05) for x in candles[i+1:])
        if broken: continue
        distance = (current-lo if side == "BUY" else hi-current)
        if distance < -vol*0.1: continue
        reaction_score = min(1.0, reaction/(vol*2.0))
        freshness = max(0.0, 1-age/120)
        touch_score = 1.0 if touches <= 1 else max(0.0, 1-(touches-1)*0.18)
        score = 100*(0.35*reaction_score + 0.30*freshness + 0.25*touch_score + 0.10*(1-min(width/vol,1)))
        zones.append({"side": side, "timeframe": tf, "low": round(lo, 5), "high": round(hi, 5),
                      "mid": round((lo+hi)/2, 5), "score": round(score, 1), "age_candles": age,
                      "tests": touches, "reaction_strength": round(reaction_score, 2), "valid": True,
                      "distance": round(abs((lo+hi)/2-current), 5)})
    return zones

# ------------------------- Scenario, path and risk -------------------------
def analyze_frame(tf: str, candles: List[dict]) -> dict:
    return {"timeframe": tf, "structure": structure_summary(candles), "atr": round(atr(candles), 5),
            "latest_close": candles[-1]["close"], "latest_datetime": candles[-1]["datetime"]}


def deduplicate_zones(zones: List[dict], current: float, side: str) -> List[dict]:
    zones = sorted(zones, key=lambda z: (-z["score"], z["distance"]))
    kept = []
    for z in zones:
        if any(abs(z["mid"]-k["mid"]) <= max(z["high"]-z["low"], k["high"]-k["low"]) * 1.5 for k in kept):
            continue
        kept.append(z)
    return sorted(kept, key=lambda z: z["distance"])


def make_trade_plan(zone: Optional[dict], side: str, current: float, volatility: float, liquidity: dict) -> Optional[dict]:
    if not zone or zone["score"] < 55 or volatility <= 0:
        return None
    entry_low, entry_high = zone["low"], zone["high"]
    entry = zone["mid"]
    stop = entry_low - volatility*0.25 if side == "BUY" else entry_high + volatility*0.25
    risk = abs(entry-stop)
    if risk <= 0: return None
    candidates = []
    pool = liquidity["buy_side_liquidity"] if side == "BUY" else liquidity["sell_side_liquidity"]
    for item in pool:
        target = item["price"]
        if (target > entry if side == "BUY" else target < entry): candidates.append(target)
    # R-multiple fallback is explicitly labeled when no mapped liquidity target exists.
    targets = sorted(candidates, reverse=False) if side == "BUY" else sorted(candidates, reverse=True)
    tps = targets[:3]
    source = "LIQUIDITY"
    if not tps:
        tps = [entry + risk*r*(1 if side == "BUY" else -1) for r in (1.5, 2.0, 3.0)]
        source = "R_MULTIPLE_FALLBACK"
    rr = abs(tps[0]-entry)/risk if tps else 0
    if rr < 1.3: return None
    return {"side": side, "entry_zone": {"low": entry_low, "high": entry_high}, "entry_reference": round(entry, 5),
            "stop_loss": round(stop, 5), "targets": [{"name": f"TP{i+1}", "price": round(x, 5)} for i,x in enumerate(tps)],
            "first_target_rr": round(rr, 2), "target_method": source,
            "confirmation_required": "Wait for a rejection candle or lower-timeframe structure shift from the zone.",
            "invalidation": "A candle close beyond the stop/invalidation boundary cancels the setup."}


def analyze_market(frames: Dict[str, List[dict]]) -> dict:
    frame_reports = {tf: analyze_frame(tf, frames[tf]) for tf in TIMEFRAMES}
    current = frames["15min"][-1]["close"]
    # Weight higher timeframes more heavily, but retain intraday context.
    weights = {"15min": 1, "1h": 1.5, "4h": 2, "1day": 2.5, "1week": 3, "1month": 3}
    score = 0.0; total = 0.0
    for tf, report in frame_reports.items():
        direction = report["structure"]["direction"]
        score += weights[tf] * (1 if direction == "BULLISH" else -1 if direction == "BEARISH" else 0)
        total += weights[tf]
    normalized = score/total if total else 0
    bias = "BULLISH" if normalized >= 0.25 else "BEARISH" if normalized <= -0.25 else "NEUTRAL"
    vol = atr(frames["15min"])
    liquidity = liquidity_map(frames["15min"], current, vol)
    all_buy, all_sell = [], []
    for tf in TIMEFRAMES:
        all_buy += build_zones(frames[tf], current, "BUY", tf)
        all_sell += build_zones(frames[tf], current, "SELL", tf)
    buy_zones = deduplicate_zones(all_buy, current, "BUY")
    sell_zones = deduplicate_zones(all_sell, current, "SELL")
    # Scenario follows the directional bias while keeping the opposite case visible.
    if bias == "BULLISH": primary, secondary = "UPSIDE", "DOWNSIDE"
    elif bias == "BEARISH": primary, secondary = "DOWNSIDE", "UPSIDE"
    else: primary, secondary = "WAIT_FOR_BREAKOUT", "RANGE_CONTINUATION"
    buy_plan = make_trade_plan(buy_zones[0] if buy_zones else None, "BUY", current, vol, liquidity)
    sell_plan = make_trade_plan(sell_zones[0] if sell_zones else None, "SELL", current, vol, liquidity)
    nearest_resistance = liquidity.get("nearest_above")
    nearest_support = liquidity.get("nearest_below")
    path = {"from_price": current, "primary_scenario": primary, "secondary_scenario": secondary,
            "upside_obstacle": nearest_resistance, "downside_obstacle": nearest_support,
            "bullish_confirmation": "15min/1h close above the nearest confirmed swing high, followed by a successful retest.",
            "bearish_confirmation": "15min/1h close below the nearest confirmed swing low, followed by a failed retest.",
            "warning": "Scenario is probabilistic; it is not a guaranteed price prediction."}
    decision = "WAIT_FOR_CONFIRMATION"
    if bias == "BULLISH" and buy_plan: decision = "BUY_SETUP_WATCH"
    elif bias == "BEARISH" and sell_plan: decision = "SELL_SETUP_WATCH"
    return {"engine": ENGINE_NAME, "version": VERSION, "symbol": SYMBOL, "decision": decision,
            "market": {"current_price": current, "directional_bias": bias, "bias_strength": round(abs(normalized)*100, 1),
                       "timeframe_alignment_score": round(normalized, 3)},
            "scenario": path, "timeframes": frame_reports,
            "liquidity": liquidity,
            "reversal_zones": {"buy": buy_zones[:10], "sell": sell_zones[:10]},
            "trade_plans": {"buy": buy_plan, "sell": sell_plan},
            "risk_notice": "Analysis only, not financial advice. No zone or target is guaranteed to hold."}

# ------------------------- API -------------------------
@app.get("/")
def home():
    return jsonify({"engine": ENGINE_NAME, "version": VERSION, "status": "online", "symbol": SYMBOL,
                    "timeframes": TIMEFRAMES, "max_candles": MAX_CANDLES})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "version": VERSION})

@app.post("/analyze")
def analyze_endpoint():
    try:
        payload = request.get_json(silent=True)
        frames, errors = normalize_payload(payload)
        quality = data_quality_report(frames, errors)
        if quality["status"] != "VALID":
            return jsonify({"engine": ENGINE_NAME, "version": VERSION, "decision": "NO_TRADE_INVALID_DATA",
                            "data_validation": quality}), 422
        result = analyze_market(frames)
        result["data_validation"] = quality
        return jsonify(result), 200
    except Exception:
        logger.exception("Unhandled analysis error")
        return jsonify({"engine": ENGINE_NAME, "version": VERSION, "decision": "ANALYSIS_ERROR",
                        "error": {"code": "internal_analysis_error", "message": "Analysis failed safely. Check server logs."}}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
