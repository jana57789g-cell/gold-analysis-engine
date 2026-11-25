import os
import math
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify

VERSION = "4.1.0"
ENGINE_NAME = "Gold Analysis Engine"
SYMBOL = "XAU/USD"
TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]
MAX_CANDLES = 300
MIN_COMPLETE_CANDLES = 30

app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("gold_analysis_engine")

# ---------------------------- Input handling ----------------------------
def parse_time(value: Any) -> datetime:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
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
        dt = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(value.strip(), fmt)
                break
            except ValueError:
                pass
        if dt is None:
            raise ValueError("unparseable_timestamp")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid_{field}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid_{field}")
    if not math.isfinite(number):
        raise ValueError(f"non_finite_{field}")
    return number


def extract_candles(value: Any) -> Optional[List[dict]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("candles", "data", "items", "records", "values", "prices"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
        for key in ("result", "output", "body", "response"):
            nested = value.get(key)
            found = extract_candles(nested)
            if found is not None:
                return found
    return None


def unwrap_payload(payload: Any) -> Any:
    # n8n sometimes passes a single item as [{"json": {...}}].
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        item = payload[0]
        payload = item.get("json", item)
    if not isinstance(payload, dict):
        return payload
    candidate = payload
    for _ in range(4):
        if isinstance(candidate, dict) and "timeframes" in candidate:
            return candidate["timeframes"]
        if isinstance(candidate, dict) and any(tf in candidate for tf in TIMEFRAMES):
            return candidate
        if isinstance(candidate, dict):
            next_value = None
            for key in ("json", "data", "body", "result", "output", "response"):
                if key in candidate and isinstance(candidate[key], (dict, list)):
                    next_value = candidate[key]
                    break
            if next_value is None:
                break
            candidate = next_value
        else:
            break
    return candidate


def normalize_payload(payload: Any) -> Tuple[Dict[str, List[dict]], List[str], Dict[str, Any]]:
    errors: List[str] = []
    notes: List[str] = []
    raw_frames = unwrap_payload(payload)
    if not isinstance(raw_frames, dict):
        return {}, ["timeframes_must_be_object"], {"normalization_notes": notes}

    frames: Dict[str, List[dict]] = {}
    for tf in TIMEFRAMES:
        raw = raw_frames.get(tf)
        if raw is None:
            errors.append(f"{tf}:missing_timeframe")
            continue
        rows = extract_candles(raw)
        if rows is None:
            errors.append(f"{tf}:candles_must_be_array")
            continue
        if len(rows) > MAX_CANDLES:
            rows = rows[-MAX_CANDLES:]
            notes.append(f"{tf}:trimmed_to_last_{MAX_CANDLES}_candles")
        parsed, row_errors = [], []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                row_errors.append(f"{tf}[{idx}]:candle_must_be_object")
                continue
            try:
                dt = parse_time(row.get("datetime", row.get("timestamp", row.get("time", row.get("date")))))
                o = finite_number(row.get("open"), "open")
                h = finite_number(row.get("high"), "high")
                low = finite_number(row.get("low"), "low")
                c = finite_number(row.get("close"), "close")
                if min(o, h, low, c) <= 0:
                    raise ValueError("non_positive_ohlc")
                if h < max(o, c, low) or low > min(o, c, h):
                    raise ValueError("inconsistent_ohlc")
                vol = row.get("volume")
                if vol is not None:
                    vol = finite_number(vol, "volume")
                    if vol < 0:
                        raise ValueError("negative_volume")
                incomplete = row.get("is_incomplete", row.get("incomplete", False))
                if not isinstance(incomplete, bool):
                    raise ValueError("invalid_is_incomplete")
                parsed.append({"datetime": dt.isoformat(), "_ts": dt.timestamp(), "open": o,
                               "high": h, "low": low, "close": c, "volume": vol,
                               "is_incomplete": incomplete})
            except Exception as exc:
                row_errors.append(f"{tf}[{idx}]:{str(exc)}")
        parsed.sort(key=lambda x: x["_ts"])
        if len(parsed) > 1 and any(parsed[i]["_ts"] < parsed[i-1]["_ts"] for i in range(1, len(parsed))):
            notes.append(f"{tf}:chronological_sort_applied")
        unique, seen = [], set()
        for candle in parsed:
            stamp = candle["_ts"]
            if stamp in seen:
                row_errors.append(f"{tf}:duplicate_timestamp:{candle['datetime']}")
                continue
            seen.add(stamp)
            unique.append(candle)
        complete = [x for x in unique if not x["is_incomplete"]]
        if len(complete) < MIN_COMPLETE_CANDLES:
            row_errors.append(f"{tf}:insufficient_complete_candles:{len(complete)}<{MIN_COMPLETE_CANDLES}")
        errors.extend(row_errors)
        frames[tf] = complete
    return frames, errors, {"normalization_notes": notes}


def timeframe_seconds(tf: str) -> int:
    return {"15min": 900, "1h": 3600, "4h": 14400, "1day": 86400,
            "1week": 604800, "1month": 2678400}[tf]


def data_quality_report(frames: Dict[str, List[dict]], errors: List[str], notes: dict) -> dict:
    now = datetime.now(timezone.utc).timestamp()
    report = {}
    warnings = []
    for tf in TIMEFRAMES:
        candles = frames.get(tf, [])
        latest = candles[-1] if candles else None
        future_seconds = max(0.0, latest["_ts"] - now) if latest else 0.0
        stale_seconds = max(0.0, now - latest["_ts"]) if latest else None
        if latest and future_seconds > max(1800, timeframe_seconds(tf) * 0.25):
            warnings.append(f"{tf}:latest_candle_timestamp_is_in_the_future")
        report[tf] = {"input_complete_count": len(candles), "latest_datetime": latest["datetime"] if latest else None,
                      "latest_close": round(latest["close"], 5) if latest else None,
                      "future_seconds": round(future_seconds, 1),
                      "age_seconds": round(stale_seconds, 1) if stale_seconds is not None else None}
    valid = not errors and all(len(frames.get(tf, [])) >= MIN_COMPLETE_CANDLES for tf in TIMEFRAMES)
    return {"status": "VALID" if valid else "INVALID", "max_candles_per_timeframe": MAX_CANDLES,
            "timeframes": report, "errors": errors, "warnings": warnings,
            "self_healing": {"applied": ["UTC timestamp normalization", "chronological sorting", "safe numeric parsing", "duplicate instant detection"],
                             "not_applied": ["OHLC price alteration", "fabricating missing candles", "marking incomplete candles complete"]},
            **notes}

# ---------------------------- Market calculations ----------------------------
def atr(candles: List[dict], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        cur, prev = candles[i], candles[i-1]
        trs.append(max(cur["high"]-cur["low"], abs(cur["high"]-prev["close"]), abs(cur["low"]-prev["close"])))
    sample = trs[-period:]
    return sum(sample)/len(sample) if sample else 0.0


def pivots(candles: List[dict], left: int = 3, right: int = 3) -> Tuple[List[dict], List[dict]]:
    highs, lows = [], []
    for i in range(left, len(candles)-right):
        h, lo = candles[i]["high"], candles[i]["low"]
        if all(h > candles[j]["high"] for j in range(i-left, i)) and all(h >= candles[j]["high"] for j in range(i+1, i+right+1)):
            highs.append({"index": i, "price": h, "datetime": candles[i]["datetime"]})
        if all(lo < candles[j]["low"] for j in range(i-left, i)) and all(lo <= candles[j]["low"] for j in range(i+1, i+right+1)):
            lows.append({"index": i, "price": lo, "datetime": candles[i]["datetime"]})
    return highs, lows


def structure_summary(candles: List[dict]) -> dict:
    highs, lows = pivots(candles)
    recent_highs, recent_lows = highs[-3:], lows[-3:]
    hh = len(recent_highs) >= 2 and recent_highs[-1]["price"] > recent_highs[-2]["price"]
    hl = len(recent_lows) >= 2 and recent_lows[-1]["price"] > recent_lows[-2]["price"]
    lh = len(recent_highs) >= 2 and recent_highs[-1]["price"] < recent_highs[-2]["price"]
    ll = len(recent_lows) >= 2 and recent_lows[-1]["price"] < recent_lows[-2]["price"]
    trend = "BULLISH" if hh and hl else "BEARISH" if lh and ll else "RANGE"
    last_close = candles[-1]["close"]
    # BOS/CHoCH are events, not a second independent trend label.
    last_high = highs[-1] if highs else None
    last_low = lows[-1] if lows else None
    bos = "BULLISH_BOS" if last_high and last_close > last_high["price"] else "BEARISH_BOS" if last_low and last_close < last_low["price"] else None
    choch = "BULLISH_CHOCH" if trend == "BEARISH" and bos == "BULLISH_BOS" else "BEARISH_CHOCH" if trend == "BULLISH" and bos == "BEARISH_BOS" else None
    return {"direction": trend, "bos": bos, "choch": choch,
            "swing_high": last_high, "swing_low": last_low,
            "higher_high": bool(hh), "higher_low": bool(hl), "lower_high": bool(lh), "lower_low": bool(ll),
            "pivot_high_count": len(highs), "pivot_low_count": len(lows)}


def rejection_strength(candle: dict, side: str) -> float:
    span = max(candle["high"]-candle["low"], 1e-9)
    body = abs(candle["close"]-candle["open"])
    lower_wick = min(candle["open"], candle["close"])-candle["low"]
    upper_wick = candle["high"]-max(candle["open"], candle["close"])
    wick = lower_wick if side == "BUY" else upper_wick
    close_pos = (candle["close"]-candle["low"])/span
    close_quality = close_pos if side == "BUY" else 1-close_pos
    return round(max(0.0, min(1.0, 0.55*wick/span + 0.25*(1-body/span) + 0.20*close_quality)), 3)


def cluster_pivots(points: List[dict], tolerance: float, label: str) -> List[dict]:
    groups: List[List[dict]] = []
    for point in points:
        target = next((g for g in groups if abs(point["price"]-sum(x["price"] for x in g)/len(g)) <= tolerance), None)
        if target is None:
            groups.append([point])
        else:
            target.append(point)
    out = []
    for group in groups:
        if len(group) >= 2:
            out.append({"price": round(sum(x["price"] for x in group)/len(group), 5),
                        "touches": len(group), "type": label,
                        "first_seen": group[0]["datetime"], "last_seen": group[-1]["datetime"]})
    return out


def liquidity_map(candles: List[dict], current: float, volatility: float) -> dict:
    highs, lows = pivots(candles)
    tol = max(volatility*0.18, current*0.0005)
    eqh = cluster_pivots(highs, tol, "EQUAL_HIGHS")
    eql = cluster_pivots(lows, tol, "EQUAL_LOWS")
    above = sorted([x for x in eqh if x["price"] > current], key=lambda x: x["price"])
    below = sorted([x for x in eql if x["price"] < current], key=lambda x: x["price"], reverse=True)
    recent = candles[-3:]
    prior_highs = [p["price"] for p in highs if p["index"] < len(candles)-3]
    prior_lows = [p["price"] for p in lows if p["index"] < len(candles)-3]
    sweep = None
    if prior_highs and any(c["high"] > max(prior_highs[-5:]) and c["close"] < max(prior_highs[-5:]) for c in recent):
        sweep = "BUY_SIDE_SWEPT"
    elif prior_lows and any(c["low"] < min(prior_lows[-5:]) and c["close"] > min(prior_lows[-5:]) for c in recent):
        sweep = "SELL_SIDE_SWEPT"
    return {"buy_side_liquidity": above, "sell_side_liquidity": below,
            "equal_highs": eqh, "equal_lows": eql, "recent_sweep": sweep,
            "nearest_above": above[0] if above else None,
            "nearest_below": below[0] if below else None,
            "note": "Liquidity pools are inferred from historical swing clusters; actual resting orders are not observable."}

# ---------------------------- Zones and trade plans ----------------------------
def build_zones(candles: List[dict], current: float, side: str, tf: str) -> List[dict]:
    highs, lows = pivots(candles)
    volatility = atr(candles)
    if volatility <= 0:
        return []
    points = lows if side == "BUY" else highs
    zones = []
    for pivot in points[-35:]:
        i = pivot["index"]
        base = candles[i]
        if side == "BUY":
            lo, hi = base["low"], max(base["open"], base["close"])
            future = candles[i+1:min(i+7, len(candles))]
            impulse = max((x["high"]-base["low"] for x in future), default=0.0)
        else:
            lo, hi = min(base["open"], base["close"]), base["high"]
            future = candles[i+1:min(i+7, len(candles))]
            impulse = max((base["high"]-x["low"] for x in future), default=0.0)
        width = hi-lo
        if width <= 0 or width > volatility*1.6:
            continue
        age = len(candles)-1-i
        if age > 150:
            continue
        later = candles[i+1:]
        broken = any((x["close"] < lo-volatility*0.05 if side == "BUY" else x["close"] > hi+volatility*0.05) for x in later)
        if broken:
            continue
        # Count distinct returns into the zone, not every candle inside one visit.
        tests, inside = 0, False
        for x in later:
            overlaps = x["low"] <= hi and x["high"] >= lo
            if overlaps and not inside:
                tests += 1
            inside = overlaps
        if tests > 3:
            continue
        midpoint = (lo+hi)/2
        distance = abs(current-midpoint)
        if side == "BUY" and midpoint > current+volatility*0.25:
            continue
        if side == "SELL" and midpoint < current-volatility*0.25:
            continue
        impulse_score = min(1.0, impulse/(volatility*2.2))
        fresh_score = max(0.0, 1-age/150)
        test_score = {0: 1.0, 1: 0.92, 2: 0.72, 3: 0.45}.get(tests, 0.0)
        narrow_score = max(0.0, 1-width/(volatility*1.6))
        reaction = rejection_strength(candles[min(i+1, len(candles)-1)], side) if i+1 < len(candles) else 0.0
        score = 100*(0.38*impulse_score + 0.22*fresh_score + 0.22*test_score + 0.10*narrow_score + 0.08*reaction)
        zones.append({"side": side, "timeframe": tf, "low": round(lo, 5), "high": round(hi, 5),
                      "mid": round(midpoint, 5), "score": round(score, 1), "age_candles": age,
                      "tests": tests, "impulse_strength": round(impulse_score, 3),
                      "reaction_strength": reaction, "distance": round(distance, 5),
                      "valid": True, "status": "VALID_UNCONFIRMED"})
    return zones


def combine_zone_confluence(zones: List[dict]) -> List[dict]:
    # Keep one representative zone and count genuinely overlapping timeframe evidence.
    ranked = sorted(zones, key=lambda z: (z["score"], -z["distance"]), reverse=True)
    kept = []
    for zone in ranked:
        overlaps = [x for x in kept if max(zone["low"], x["low"]) <= min(zone["high"], x["high"]) + max(zone["high"]-zone["low"], x["high"]-x["low"])*0.25]
        if overlaps:
            best = overlaps[0]
            existing_tfs = best.get("confluence_timeframes", [best["timeframe"]])
            if zone["timeframe"] not in existing_tfs:
                best["confluence_timeframes"] = sorted(set(existing_tfs + [zone["timeframe"]]))
                best["score"] = round(min(98.0, best["score"] + min(5.0, zone["score"]*0.05)), 1)
        else:
            copy = dict(zone)
            copy["confluence_timeframes"] = [zone["timeframe"]]
            kept.append(copy)
    return sorted(kept, key=lambda z: (z["distance"], -z["score"]))


def make_trade_plan(zone: Optional[dict], side: str, current: float, volatility: float,
                    liquidity: dict, structure_ok: bool, data_warnings: List[str]) -> Optional[dict]:
    if not zone or volatility <= 0:
        return None
    # Stronger gate: weak reaction/impulse zones are not promoted to trade plans.
    if zone["score"] < 68 or zone["impulse_strength"] < 0.45 or zone["tests"] > 2:
        return None
    low, high = zone["low"], zone["high"]
    entry = zone["mid"]
    stop = low-volatility*0.22 if side == "BUY" else high+volatility*0.22
    risk = abs(entry-stop)
    if risk <= 0:
        return None
    pools = liquidity["buy_side_liquidity"] if side == "BUY" else liquidity["sell_side_liquidity"]
    targets = sorted([p["price"] for p in pools if (p["price"] > entry if side == "BUY" else p["price"] < entry)], reverse=(side == "SELL"))
    # Do not invent targets when no mapped liquidity target exists.
    selected = []
    for target in targets:
        if not selected or abs(target-selected[-1]) >= volatility*0.25:
            selected.append(target)
        if len(selected) == 3:
            break
    if not selected:
        return None
    rr = abs(selected[0]-entry)/risk
    if rr < 1.5:
        return None
    current_in_zone = low <= current <= high
    proximity = abs(current-entry)/volatility
    activation = ("Price is inside the zone; wait for a confirmed rejection/structure shift before entry."
                  if current_in_zone else "Wait for price to reach the zone, then require a rejection candle and lower-timeframe structure shift.")
    blocked = bool(data_warnings)
    return {"side": side, "entry_zone": {"low": low, "high": high}, "entry_reference": round(entry, 5),
            "current_price_distance_atr": round(proximity, 2), "current_price_inside_zone": current_in_zone,
            "stop_loss": round(stop, 5),
            "targets": [{"name": f"TP{i+1}", "price": round(p, 5)} for i, p in enumerate(selected)],
            "first_target_rr": round(rr, 2), "target_method": "OBSERVED_LIQUIDITY",
            "zone_score": zone["score"], "zone_timeframes": zone.get("confluence_timeframes", [zone["timeframe"]]),
            "confirmation_required": activation,
            "setup_status": "DATA_REVIEW_REQUIRED" if blocked else ("WAITING_FOR_ZONE" if not current_in_zone else "WAITING_FOR_CONFIRMATION"),
            "trade_allowed_now": False,
            "invalidation": "Cancel if a candle closes beyond the stop-loss boundary or the zone is decisively broken.",
            "structure_context": "Aligned" if structure_ok else "Mixed; require stronger confirmation."}

# ---------------------------- Scenario/path report ----------------------------
def analyze_market(frames: Dict[str, List[dict]], quality: dict) -> dict:
    reports = {tf: {"timeframe": tf, "structure": structure_summary(frames[tf]),
                    "atr": round(atr(frames[tf]), 5), "latest_close": round(frames[tf][-1]["close"], 5),
                    "latest_datetime": frames[tf][-1]["datetime"]} for tf in TIMEFRAMES}
    current = frames["15min"][-1]["close"]
    weights = {"15min": 1.0, "1h": 1.5, "4h": 2.0, "1day": 2.5, "1week": 3.0, "1month": 3.0}
    raw_score, total = 0.0, 0.0
    directions = {}
    for tf, report in reports.items():
        direction = report["structure"]["direction"]
        directions[tf] = direction
        raw_score += weights[tf] * (1 if direction == "BULLISH" else -1 if direction == "BEARISH" else 0)
        total += weights[tf]
    normalized = raw_score/total if total else 0.0
    bias = "BULLISH" if normalized >= 0.25 else "BEARISH" if normalized <= -0.25 else "NEUTRAL"
    bullish_n = sum(1 for x in directions.values() if x == "BULLISH")
    bearish_n = sum(1 for x in directions.values() if x == "BEARISH")
    alignment = max(bullish_n, bearish_n)/len(TIMEFRAMES)
    vol15 = atr(frames["15min"])
    liquidity = liquidity_map(frames["15min"], current, vol15)

    # Cross-timeframe close comparison is a diagnostic, not an assumption that all candles close together.
    close_checks = {}
    mismatch_warnings = []
    for tf in TIMEFRAMES:
        close = frames[tf][-1]["close"]
        delta = close-current
        pct = delta/current*100 if current else 0
        close_checks[tf] = {"latest_close": round(close, 5), "difference_from_15min": round(delta, 5),
                            "difference_percent": round(pct, 3), "latest_datetime": frames[tf][-1]["datetime"]}
        if tf not in ("15min", "1h") and abs(delta) > max(vol15*2.5, current*0.012):
            mismatch_warnings.append(f"{tf}:latest_close_materially_differs_from_15min_close")

    all_buy, all_sell = [], []
    for tf in TIMEFRAMES:
        all_buy.extend(build_zones(frames[tf], current, "BUY", tf))
        all_sell.extend(build_zones(frames[tf], current, "SELL", tf))
    buy_zones = combine_zone_confluence(all_buy)
    sell_zones = combine_zone_confluence(all_sell)
    strong_buy = [z for z in buy_zones if z["score"] >= 68 and z["impulse_strength"] >= 0.45 and z["tests"] <= 2]
    strong_sell = [z for z in sell_zones if z["score"] >= 68 and z["impulse_strength"] >= 0.45 and z["tests"] <= 2]
    buy_zone = strong_buy[0] if strong_buy else None
    sell_zone = strong_sell[0] if strong_sell else None

    data_warnings = list(quality.get("warnings", [])) + mismatch_warnings
    bullish_structure_ok = bullish_n >= 3 and bias == "BULLISH"
    bearish_structure_ok = bearish_n >= 3 and bias == "BEARISH"
    buy_plan = next((plan for z in strong_buy
                     if (plan := make_trade_plan(z, "BUY", current, vol15, liquidity, bullish_structure_ok, data_warnings)) is not None), None)
    sell_plan = next((plan for z in strong_sell
                      if (plan := make_trade_plan(z, "SELL", current, vol15, liquidity, bearish_structure_ok, data_warnings)) is not None), None)

    nearest_highs = [r["structure"]["swing_high"]["price"] for r in reports.values() if r["structure"]["swing_high"] and r["structure"]["swing_high"]["price"] > current]
    nearest_lows = [r["structure"]["swing_low"]["price"] for r in reports.values() if r["structure"]["swing_low"] and r["structure"]["swing_low"]["price"] < current]
    upside = min(nearest_highs) if nearest_highs else (liquidity["nearest_above"]["price"] if liquidity["nearest_above"] else None)
    downside = max(nearest_lows) if nearest_lows else (liquidity["nearest_below"]["price"] if liquidity["nearest_below"] else None)
    if bias == "BULLISH": primary, secondary = "UPSIDE_CONTINUATION_IF_CONFIRMED", "DOWNSIDE_PULLBACK"
    elif bias == "BEARISH": primary, secondary = "DOWNSIDE_CONTINUATION_IF_CONFIRMED", "UPSIDE_PULLBACK"
    else: primary, secondary = "RANGE_UNTIL_BREAKOUT", "WAIT_FOR_CONFIRMATION"

    path = {"from_price": round(current, 5), "primary_scenario": primary, "secondary_scenario": secondary,
            "upside_path_level": round(upside, 5) if upside is not None else None,
            "downside_path_level": round(downside, 5) if downside is not None else None,
            "upside_liquidity": liquidity["nearest_above"], "downside_liquidity": liquidity["nearest_below"],
            "bullish_confirmation": "15min close above the latest confirmed swing high, then successful retest; 1h should not contradict.",
            "bearish_confirmation": "15min close below the latest confirmed swing low, then failed retest; 1h should not contradict.",
            "warning": "These are conditional scenarios, not a guaranteed forecast or exact future price."}
    if bias == "BULLISH" and buy_plan:
        decision = "BUY_SETUP_WATCH"
    elif bias == "BEARISH" and sell_plan:
        decision = "SELL_SETUP_WATCH"
    else:
        decision = "WAIT_FOR_CONFIRMATION"

    return {"engine": ENGINE_NAME, "version": VERSION, "symbol": SYMBOL, "decision": decision,
            "market": {"current_price": round(current, 5), "directional_bias": bias,
                       "bias_strength": round(abs(normalized)*100, 1), "timeframe_alignment_score": round(alignment, 3),
                       "timeframe_direction_votes": {"bullish": bullish_n, "bearish": bearish_n, "range": 6-bullish_n-bearish_n},
                       "confidence_note": "Strength is a rule-based alignment score, not a statistically calibrated probability."},
            "scenario": path,
            "price_path": {"current_price": round(current, 5), "upside_obstacle": path["upside_path_level"],
                           "downside_obstacle": path["downside_path_level"],
                           "next_upside_liquidity": liquidity["nearest_above"], "next_downside_liquidity": liquidity["nearest_below"],
                           "path_status": "CONDITIONAL_LEVELS_ONLY"},
            "timeframes": reports,
            "cross_timeframe_data_check": {"closes": close_checks, "warnings": mismatch_warnings},
            "liquidity": liquidity,
            "reversal_zones": {"buy": buy_zones[:12], "sell": sell_zones[:12],
                               "strong_buy_candidates": strong_buy[:5], "strong_sell_candidates": strong_sell[:5]},
            "entry_analysis": {"buy_candidate": buy_zone, "sell_candidate": sell_zone,
                               "buy_setup_available": buy_plan is not None, "sell_setup_available": sell_plan is not None,
                               "note": "A zone is not an entry signal. Wait for price to reach it and confirm rejection/structure shift."},
            "trade_plans": {"buy": buy_plan, "sell": sell_plan},
            "data_warnings": data_warnings,
            "risk_notice": "Analysis only, not financial advice. No zone, entry, stop or target is guaranteed to hold."}

# -------------------------------- API --------------------------------
@app.get("/")
def home():
    return jsonify({"engine": ENGINE_NAME, "version": VERSION, "status": "online", "symbol": SYMBOL,
                    "timeframes": TIMEFRAMES, "max_candles_per_timeframe": MAX_CANDLES,
                    "min_complete_candles": MIN_COMPLETE_CANDLES})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "version": VERSION})

@app.post("/analyze")
def analyze_endpoint():
    try:
        payload = request.get_json(silent=True)
        frames, errors, notes = normalize_payload(payload)
        quality = data_quality_report(frames, errors, notes)
        if quality["status"] != "VALID":
            return jsonify({"engine": ENGINE_NAME, "version": VERSION, "decision": "NO_TRADE_INVALID_DATA",
                            "data_validation": quality}), 422
        result = analyze_market(frames, quality)
        result["data_validation"] = quality
        return jsonify(result), 200
    except Exception:
        logger.exception("Unhandled analysis error")
        return jsonify({"engine": ENGINE_NAME, "version": VERSION, "decision": "ANALYSIS_ERROR",
                        "error": {"code": "internal_analysis_error", "message": "Analysis failed safely. Check server logs."}}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
