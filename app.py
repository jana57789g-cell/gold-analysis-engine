import os
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify

app = Flask(__name__)

VERSION = "2.3.0"
ENGINE_NAME = "Gold Analysis Engine"
SYMBOL = "XAU/USD"
TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]
MIN_COMPLETE_CANDLES = 10
SWING_LEFT = 2
SWING_RIGHT = 2
INTERNAL_LEFT = 1
INTERNAL_RIGHT = 1
ATR_PERIOD = 14
PRICE_EPSILON = 1e-9
EQUAL_LEVEL_ATR_FRACTION = 0.10
MAX_ZONE_AGE_CANDLES = 80
MAX_RETURNED_ZONES = 5
MAX_DIRECTION_ZONES = 3
MIN_SETUP_SCORE = 70
MAX_ENTRY_DISTANCE_ATR = 1.5


# ============================================================
# VALIDATION AND DATA QUALITY
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
    return math.isclose(float(a), float(b), rel_tol=PRICE_EPSILON,
                        abs_tol=PRICE_EPSILON)


def validate_candle(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    try:
        values = {k: float(raw[k]) for k in ("open", "high", "low", "close")}
        if not all(math.isfinite(v) for v in values.values()):
            return None
        if values["low"] > values["high"]:
            return None
        if not (values["low"] <= values["open"] <= values["high"]
                and values["low"] <= values["close"] <= values["high"]):
            return None
        stamp = raw.get("datetime")
        if parse_time(stamp) is None:
            return None
        return {"datetime": str(stamp), **values,
                "is_incomplete": raw.get("is_incomplete", False) is True}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def normalize_candles(section: Dict[str, Any]):
    raw = section.get("candles")
    if not isinstance(raw, list):
        raise ValueError("Missing candles array")
    valid, invalid = [], 0
    for item in raw:
        candle = validate_candle(item)
        if candle is None:
            invalid += 1
        else:
            valid.append(candle)
    complete = [c for c in valid if not c["is_incomplete"]]
    parsed = [(c, parse_time(c["datetime"])) for c in complete]
    parsed.sort(key=lambda x: x[1])
    for i in range(1, len(parsed)):
        if parsed[i][1] == parsed[i - 1][1]:
            raise ValueError("Duplicate timestamps found")
    ordered = [x[0] for x in parsed]
    aware = [x[1].utcoffset() is not None for x in parsed]
    if aware and any(aware) and not all(aware):
        raise ValueError("Mixed timezone-aware and naive timestamps")
    basis = section.get("timestamp_basis", "unknown")
    meta = {
        "received": len(raw), "valid": len(valid), "invalid": invalid,
        "incomplete_excluded": len(valid) - len(complete),
        "complete": len(ordered),
        "timestamp_timezone_aware": bool(aware and all(aware)),
        "timestamp_basis": basis,
        "timestamp_reliable": bool(aware and all(aware) and basis not in (None, "", "unknown"))
    }
    return ordered, meta


def calculate_atr(candles, period=ATR_PERIOD):
    if not candles:
        return 0.0
    trs = []
    for i, c in enumerate(candles):
        if i == 0:
            tr = c["high"] - c["low"]
        else:
            pc = candles[i - 1]["close"]
            tr = max(c["high"] - c["low"], abs(c["high"] - pc),
                     abs(c["low"] - pc))
        trs.append(max(0.0, tr))
    sample = trs[-period:]
    return sum(sample) / len(sample) if sample else 0.0


# ============================================================
# STRUCTURE
# ============================================================

def find_pivots(candles, left=2, right=2):
    highs, lows = [], []
    for i in range(left, max(left, len(candles) - right)):
        current = candles[i]
        neighbors = candles[i-left:i] + candles[i+1:i+right+1]
        if neighbors and all(current["high"] > c["high"] for c in neighbors):
            highs.append({"index": i, "price": current["high"],
                          "datetime": current["datetime"],
                          "confirmed_at_index": i + right})
        if neighbors and all(current["low"] < c["low"] for c in neighbors):
            lows.append({"index": i, "price": current["low"],
                         "datetime": current["datetime"],
                         "confirmed_at_index": i + right})
    return highs, lows


def classify_structure(highs, lows):
    if len(highs) < 2 or len(lows) < 2:
        return {"trend": "UNCONFIRMED", "structure": [],
                "reason": "At least two confirmed swing highs and lows are required"}
    h1, h2, l1, l2 = highs[-2], highs[-1], lows[-2], lows[-1]
    ht = "EQH" if same_price(h1["price"], h2["price"]) else (
        "HH" if h2["price"] > h1["price"] else "LH")
    lt = "EQL" if same_price(l1["price"], l2["price"]) else (
        "HL" if l2["price"] > l1["price"] else "LL")
    trend = ("BULLISH" if ht == "HH" and lt == "HL" else
             "BEARISH" if ht == "LH" and lt == "LL" else "RANGING")
    return {
        "trend": trend,
        "structure": [
            {"type": ht, "price": h2["price"], "datetime": h2["datetime"], "index": h2["index"]},
            {"type": lt, "price": l2["price"], "datetime": l2["datetime"], "index": l2["index"]}
        ],
        "reason": "Based on the two latest confirmed swing highs and lows"
    }


def trend_at_index(highs, lows, index):
    return classify_structure(
        [p for p in highs if p["confirmed_at_index"] < index],
        [p for p in lows if p["confirmed_at_index"] < index]
    )["trend"]


def detect_structure_events(candles, highs, lows):
    events, broken_h, broken_l = [], set(), set()
    for i in range(1, len(candles)):
        prev_close, close = candles[i-1]["close"], candles[i]["close"]
        prior = trend_at_index(highs, lows, i)
        for n, p in enumerate(highs):
            if p["confirmed_at_index"] >= i or n in broken_h:
                continue
            if prev_close <= p["price"] < close:
                kind = ("BOS" if prior == "BULLISH" else
                        "CHoCH" if prior == "BEARISH" else "BULLISH_STRUCTURE_BREAK")
                events.append({
                    "event": kind, "direction": "BULLISH", "level": p["price"],
                    "pivot_index": p["index"], "pivot_datetime": p["datetime"],
                    "confirmed_at_index": p["confirmed_at_index"],
                    "break_candle_index": i, "break_datetime": candles[i]["datetime"],
                    "previous_close": prev_close, "latest_close": close,
                    "trend_before_break": prior, "confirmed_by_close": True
                })
                broken_h.add(n)
        for n, p in enumerate(lows):
            if p["confirmed_at_index"] >= i or n in broken_l:
                continue
            if prev_close >= p["price"] > close:
                kind = ("BOS" if prior == "BEARISH" else
                        "CHoCH" if prior == "BULLISH" else "BEARISH_STRUCTURE_BREAK")
                events.append({
                    "event": kind, "direction": "BEARISH", "level": p["price"],
                    "pivot_index": p["index"], "pivot_datetime": p["datetime"],
                    "confirmed_at_index": p["confirmed_at_index"],
                    "break_candle_index": i, "break_datetime": candles[i]["datetime"],
                    "previous_close": prev_close, "latest_close": close,
                    "trend_before_break": prior, "confirmed_by_close": True
                })
                broken_l.add(n)
    return events


def analyze_structure_layer(candles, left, right):
    highs, lows = find_pivots(candles, left, right)
    state = classify_structure(highs, lows)
    return {
        "trend": state["trend"], "structure": state["structure"],
        "swing_highs": highs[-10:], "swing_lows": lows[-10:],
        "structure_events": detect_structure_events(candles, highs, lows)[-20:],
        "total_confirmed_highs": len(highs), "total_confirmed_lows": len(lows),
        "analysis_note": state["reason"]
    }


# ============================================================
# LIQUIDITY, IMBALANCE, ORDER BLOCKS
# ============================================================

def detect_liquidity(candles, highs, lows, atr):
    tolerance = max(0.01, atr * EQUAL_LEVEL_ATR_FRACTION)
    equal_highs, equal_lows = [], []
    for pivots, output, label in (
        (highs, equal_highs, "EQUAL_HIGHS"),
        (lows, equal_lows, "EQUAL_LOWS")
    ):
        used = set()
        for i, pivot in enumerate(pivots):
            if i in used:
                continue
            group = [pivot]
            for j in range(i + 1, len(pivots)):
                if abs(pivot["price"] - pivot["price"]) <= tolerance and abs(
                    pivot["price"] - pivots[j]["price"]) <= tolerance:
                    group.append(pivots[j])
                    used.add(j)
            if len(group) >= 2:
                output.append({
                    "type": label,
                    "level": sum(x["price"] for x in group) / len(group),
                    "touches": len(group), "last_touch": group[-1]["datetime"]
                })
    sweeps = []
    for i, c in enumerate(candles):
        for p in highs:
            if p["confirmed_at_index"] < i and c["high"] > p["price"] and c["close"] < p["price"]:
                sweeps.append({"type": "BUY_SIDE_LIQUIDITY_SWEEP",
                               "level": p["price"], "datetime": c["datetime"],
                               "index": i, "pivot_datetime": p["datetime"]})
        for p in lows:
            if p["confirmed_at_index"] < i and c["low"] < p["price"] and c["close"] > p["price"]:
                sweeps.append({"type": "SELL_SIDE_LIQUIDITY_SWEEP",
                               "level": p["price"], "datetime": c["datetime"],
                               "index": i, "pivot_datetime": p["datetime"]})
    return {"equal_highs": equal_highs[-10:], "equal_lows": equal_lows[-10:],
            "sweeps": sweeps[-20:]}


def detect_fvgs(candles):
    zones = []
    for i in range(2, len(candles)):
        a, c = candles[i-2], candles[i]
        if c["low"] > a["high"]:
            low, high, typ = a["high"], c["low"], "BULLISH_FVG"
        elif c["high"] < a["low"]:
            low, high, typ = c["high"], a["low"], "BEARISH_FVG"
        else:
            continue
        later = candles[i+1:]
        mitigated = any(x["low"] <= high and x["high"] >= low for x in later)
        zones.append({"type": typ, "low": low, "high": high,
                      "midpoint": (low + high) / 2,
                      "formed_at": c["datetime"], "formed_index": i,
                      "mitigated": mitigated})
    return zones[-30:]


def detect_order_blocks(candles, atr):
    blocks, seen = [], set()
    bodies = [abs(c["close"] - c["open"]) for c in candles]
    for i in range(2, len(candles)):
        cur = candles[i]
        prior_bodies = bodies[max(0, i-10):i]
        avg = sum(prior_bodies) / max(1, len(prior_bodies))
        if abs(cur["close"] - cur["open"]) < max(avg * 1.5, atr * 0.5):
            continue
        if same_price(cur["close"], cur["open"]):
            continue
        direction = "BULLISH" if cur["close"] > cur["open"] else "BEARISH"
        for j in range(i-1, max(-1, i-6), -1):
            p = candles[j]
            opposing = p["close"] < p["open"] if direction == "BULLISH" else p["close"] > p["open"]
            if not opposing:
                continue
            key = (direction, j)
            if key in seen:
                break
            seen.add(key)
            mitigated = any(x["low"] <= p["high"] and x["high"] >= p["low"]
                            for x in candles[i+1:])
            blocks.append({
                "type": direction + "_ORDER_BLOCK", "low": p["low"], "high": p["high"],
                "datetime": p["datetime"], "formed_by": cur["datetime"],
                "formed_index": i, "origin_index": j, "mitigated": mitigated
            })
            break
    return blocks[-30:]


def premium_discount(candles, highs, lows):
    if not highs or not lows:
        return {"status": "unavailable"}
    # Use the most recent completed swing range, not arbitrary last high/low ordering.
    hp, lp = highs[-1], lows[-1]
    if hp["price"] <= lp["price"]:
        return {"status": "unavailable", "range_high": hp["price"], "range_low": lp["price"]}
    eq = (hp["price"] + lp["price"]) / 2
    price = candles[-1]["close"]
    return {"status": "available", "range_high": hp["price"], "range_low": lp["price"],
            "equilibrium": eq, "current_price": price,
            "zone": "DISCOUNT" if price < eq else "PREMIUM" if price > eq else "EQUILIBRIUM"}


def analyze_timeframe(section):
    if not isinstance(section, dict):
        return {"status": "error", "message": "Invalid timeframe object"}
    try:
        candles, quality = normalize_candles(section)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    if len(candles) < MIN_COMPLETE_CANDLES:
        return {"status": "insufficient_data", **quality, "reason": "Insufficient complete candles"}
    atr = calculate_atr(candles)
    swing = analyze_structure_layer(candles, SWING_LEFT, SWING_RIGHT)
    internal = analyze_structure_layer(candles, INTERNAL_LEFT, INTERNAL_RIGHT)
    highs, lows = find_pivots(candles, SWING_LEFT, SWING_RIGHT)
    return {
        "status": "analyzed", **quality, "latest_datetime": candles[-1]["datetime"],
        "latest_close": candles[-1]["close"], "atr": atr,
        "trend": swing["trend"], "structure": swing["structure"],
        "swing_highs": swing["swing_highs"], "swing_lows": swing["swing_lows"],
        "structure_events": swing["structure_events"],
        "internal_structure": internal,
        "liquidity": detect_liquidity(candles, highs, lows, atr),
        "fair_value_gaps": detect_fvgs(candles),
        "order_blocks": detect_order_blocks(candles, atr),
        "premium_discount": premium_discount(candles, highs, lows),
        "structure_quality": {
            "confirmed_swing_highs": swing["total_confirmed_highs"],
            "confirmed_swing_lows": swing["total_confirmed_lows"],
            "confirmed_internal_highs": internal["total_confirmed_highs"],
            "confirmed_internal_lows": internal["total_confirmed_lows"]
        },
        "analysis_note": swing["analysis_note"]
    }


# ============================================================
# MULTI-TIMEFRAME BIAS
# ============================================================

def calculate_overall_structure(results):
    usable = [results[tf]["trend"] for tf in TIMEFRAMES
              if results.get(tf, {}).get("status") == "analyzed"
              and results[tf]["trend"] in ("BULLISH", "BEARISH")]
    if len(usable) < 3:
        return "UNCONFIRMED"
    bull, bear = usable.count("BULLISH"), usable.count("BEARISH")
    if bull >= 3 and bull > bear:
        return "BULLISH"
    if bear >= 3 and bear > bull:
        return "BEARISH"
    return "CONFLICTING"


def latest_signal_event(result):
    events = result.get("structure_events", [])
    return max(events, key=lambda e: e.get("break_candle_index", -1)) if events else None


# ============================================================
# ZONE SCORING / DEFENSE ZONES / TRADE PLANS
# ============================================================

def zone_distance(price, zone):
    if price < zone["low"]:
        return zone["low"] - price
    if price > zone["high"]:
        return price - zone["high"]
    return 0.0


def collect_active_zones(result, direction):
    fvg_type = "BULLISH_FVG" if direction == "BUY" else "BEARISH_FVG"
    ob_type = "BULLISH_ORDER_BLOCK" if direction == "BUY" else "BEARISH_ORDER_BLOCK"
    items = []
    for z in result.get("fair_value_gaps", []):
        if z.get("type") == fvg_type and not z.get("mitigated", True):
            items.append({"source": "FVG", "sources": ["FVG"], "low": z["low"],
                          "high": z["high"], "datetime": z["formed_at"],
                          "formed_index": z["formed_index"]})
    for z in result.get("order_blocks", []):
        if z.get("type") == ob_type and not z.get("mitigated", True):
            items.append({"source": "ORDER_BLOCK", "sources": ["ORDER_BLOCK"],
                          "low": z["low"], "high": z["high"],
                          "datetime": z["datetime"], "formed_index": z["formed_index"]})
    items.sort(key=lambda x: x["formed_index"])
    merged = []
    for z in items:
        match = next((m for m in merged
                      if min(m["high"], z["high"]) >= max(m["low"], z["low"])), None)
        if match:
            old_index = match["formed_index"]
            match["low"] = min(match["low"], z["low"])
            match["high"] = max(match["high"], z["high"])
            match["sources"] = sorted(set(match["sources"] + z["sources"]))
            match["source"] = "+".join(match["sources"])
            if z["formed_index"] >= old_index:
                match["formed_index"] = z["formed_index"]
                match["datetime"] = z["datetime"]
        else:
            merged.append(dict(z))
    return merged


def score_zone(zone, direction, tf_result, price, atr):
    age = max(0, tf_result.get("complete", 0) - 1 - zone.get("formed_index", 0))
    score = 30.0
    sources = zone.get("sources", [])
    if "FVG" in sources:
        score += 7
    if "ORDER_BLOCK" in sources:
        score += 9
    if len(sources) > 1:
        score += 12
    score += 12 if age <= 8 else 7 if age <= 20 else 2 if age <= MAX_ZONE_AGE_CANDLES else -15
    distance = zone_distance(price, zone)
    if atr > 0:
        score += 10 if distance <= atr * 0.5 else 5 if distance <= atr * MAX_ENTRY_DISTANCE_ATR else -min(18, distance / atr * 3)
    wanted_sweep = "SELL_SIDE_LIQUIDITY_SWEEP" if direction == "BUY" else "BUY_SIDE_LIQUIDITY_SWEEP"
    if any(s.get("type") == wanted_sweep and
           abs(s.get("level", 0) - (zone["low"] + zone["high"]) / 2) <= max(atr, 0.01) * 1.5
           for s in tf_result.get("liquidity", {}).get("sweeps", [])[-8:]):
        score += 10
    pd = tf_result.get("premium_discount", {})
    if pd.get("status") == "available":
        if direction == "BUY" and pd.get("zone") == "DISCOUNT":
            score += 4
        if direction == "SELL" and pd.get("zone") == "PREMIUM":
            score += 4
    return round(max(0, min(100, score)), 2)


def get_zone_candidates(results, direction, price, fast_atr):
    candidates = []
    for tf in TIMEFRAMES:
        data = results.get(tf, {})
        if data.get("status") != "analyzed":
            continue
        atr = data.get("atr", 0) or fast_atr
        for zone in collect_active_zones(data, direction):
            age = max(0, data.get("complete", 0) - 1 - zone["formed_index"])
            if age > MAX_ZONE_AGE_CANDLES:
                continue
            score = score_zone(zone, direction, data, price, atr)
            # Cross-timeframe confluence: independent active zones near this zone.
            midpoint = (zone["low"] + zone["high"]) / 2
            confluence = []
            for other_tf in TIMEFRAMES:
                other = results.get(other_tf, {})
                if other_tf == tf or other.get("status") != "analyzed":
                    continue
                other_atr = other.get("atr", 0) or fast_atr
                for oz in collect_active_zones(other, direction):
                    omid = (oz["low"] + oz["high"]) / 2
                    if abs(omid - midpoint) <= max(fast_atr, other_atr, 0.01) * 0.35:
                        confluence.append(other_tf)
                        break
            score = min(100, score + min(15, len(confluence) * 5))
            candidates.append({
                **zone, "timeframe": tf, "zone_score": round(score, 2),
                "age_candles": age, "confluence_timeframes": confluence,
                "distance_from_current_price": zone_distance(price, zone),
                "within_entry_distance": zone_distance(price, zone) <= fast_atr * MAX_ENTRY_DISTANCE_ATR
            })
    # De-duplicate zones across timeframes but preserve their timeframe evidence.
    candidates.sort(key=lambda z: z["zone_score"], reverse=True)
    unique = []
    for z in candidates:
        mid = (z["low"] + z["high"]) / 2
        match = next((u for u in unique if abs(
            (u["low"] + u["high"]) / 2 - mid) <= max(fast_atr * 0.15, 0.01)), None)
        if match:
            tfs = set(match.get("timeframes", [match["timeframe"]]))
            tfs.add(z["timeframe"])
            tfs.update(z.get("confluence_timeframes", []))
            match["timeframes"] = sorted(tfs, key=lambda x: TIMEFRAMES.index(x))
            match["zone_score"] = min(100, max(match["zone_score"], z["zone_score"]) + 3)
            match["sources"] = sorted(set(match["sources"] + z["sources"]))
            match["source"] = "+".join(match["sources"])
            match["low"] = min(match["low"], z["low"])
            match["high"] = max(match["high"], z["high"])
        else:
            item = dict(z)
            item["timeframes"] = [z["timeframe"]]
            unique.append(item)
    return unique


def nearest_level(levels, entry, direction):
    valid = [x for x in levels if (x > entry if direction == "BUY" else x < entry)]
    if not valid:
        return None
    return min(valid) if direction == "BUY" else max(valid)


def build_trade_plan(zone, direction, price, fast_atr, results):
    low, high = float(zone["low"]), float(zone["high"])
    entry = (low + high) / 2
    buffer = max(fast_atr * 0.20, (high - low) * 0.15, 0.01)
    stop = low - buffer if direction == "BUY" else high + buffer
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    if direction == "BUY":
        rr = [entry + risk * x for x in (1.5, 2, 3, 4)]
        pool_key, pivot_key = "equal_highs", "swing_highs"
        liquidity_direction = "ABOVE_ENTRY"
    else:
        rr = [entry - risk * x for x in (1.5, 2, 3, 4)]
        pool_key, pivot_key = "equal_lows", "swing_lows"
        liquidity_direction = "BELOW_ENTRY"

    pools = []
    for tf in TIMEFRAMES:
        data = results.get(tf, {})
        if data.get("status") != "analyzed":
            continue
        pools.extend(x["level"] for x in data.get("liquidity", {}).get(pool_key, []))
        pools.extend(x["price"] for x in data.get(pivot_key, []))
    pools = sorted(set(x for x in pools if x > entry),) if direction == "BUY" else sorted(
        set(x for x in pools if x < entry), reverse=True)
    # Liquidity targets are informational; RR targets are retained where a pool is absent.
    targets = {}
    for name, target in zip(("tp1", "tp2", "tp3", "tp4"), rr):
        pool = nearest_level(pools, entry, direction)
        if pool is not None and abs(pool - entry) < abs(target - entry) * 1.25:
            target = pool
            pools.remove(pool)
        targets[name] = round(target, 5)
    return {
        "entry_zone": {"low": round(low, 5), "high": round(high, 5),
                       "midpoint": round(entry, 5)},
        "entry_reference": round(entry, 5),
        "stop_loss": round(stop, 5),
        "targets": targets,
        "risk_reward": {"tp1": 1.5, "tp2": 2.0, "tp3": 3.0, "tp4": 4.0},
        "risk_distance": round(risk, 5),
        "distance_from_current_price": round(zone_distance(price, zone), 5),
        "liquidity_target_type": "EQUAL_HIGHS" if direction == "BUY" else "EQUAL_LOWS",
        "liquidity_targets": [round(x, 5) for x in pools[:5]],
        "stop_method": "Beyond zone boundary with ATR/zone-width buffer",
        "target_method": "Risk/reward targets adjusted toward nearby opposing liquidity"
    }


def defense_label(score):
    if score >= 80:
        return "VERY_STRONG_CONFLUENCE"
    if score >= 65:
        return "STRONG_CONFLUENCE"
    if score >= 50:
        return "MODERATE_CONFLUENCE"
    return "WEAK_CONFLUENCE"


def public_zone(zone, direction, role, plan):
    return {
        "role": role, "direction": direction,
        "source": zone.get("source"), "sources": zone.get("sources", []),
        "timeframe": zone.get("timeframe"),
        "timeframes": zone.get("timeframes", [zone.get("timeframe")]),
        "datetime": zone.get("datetime"), "age_candles": zone.get("age_candles"),
        "zone_score": zone.get("zone_score"),
        "defense_strength": defense_label(zone.get("zone_score", 0)),
        "confluence_timeframes": zone.get("confluence_timeframes", []),
        "within_entry_distance": zone.get("within_entry_distance"),
        **plan
    }


# ============================================================
# SCENARIO MAP: BOTH DIRECTIONS, PATH, CONFIRMATION, INVALIDATION
# ============================================================

def collect_projection_levels(results, direction, current_price):
    downward = direction == "SELL"
    candidates = []
    for tf in TIMEFRAMES:
        data = results.get(tf, {})
        if data.get("status") != "analyzed":
            continue
        liq_key = "equal_lows" if downward else "equal_highs"
        piv_key = "swing_lows" if downward else "swing_highs"
        src = "EQUAL_LOWS" if downward else "EQUAL_HIGHS"
        for item in data.get("liquidity", {}).get(liq_key, []):
            level = item.get("level")
            if isinstance(level, (int, float)) and (level < current_price if downward else level > current_price):
                candidates.append((float(level), tf, src))
        for item in data.get(piv_key, []):
            level = item.get("price")
            if isinstance(level, (int, float)) and (level < current_price if downward else level > current_price):
                candidates.append((float(level), tf, "SWING_LOW" if downward else "SWING_HIGH"))
    candidates.sort(key=lambda x: abs(x[0] - current_price))
    merged = []
    for price, tf, source in candidates:
        match = next((x for x in merged if abs(x["price"] - price) <= max(0.01, current_price * 0.00005)), None)
        if match:
            if tf not in match["timeframes"]:
                match["timeframes"].append(tf)
            if source not in match["sources"]:
                match["sources"].append(source)
        else:
            merged.append({"price": round(price, 5), "distance": round(abs(price-current_price), 5),
                           "timeframes": [tf], "sources": [source]})
    return merged[:7]


def build_scenario_map(results, candidates, overall):
    fast = results.get("15min", {})
    if fast.get("status") != "analyzed":
        return {"status": "UNAVAILABLE", "reason": "15min data unavailable"}
    price = fast["latest_close"]
    atr = fast.get("atr", 0) or 0
    candidate_map = {x["direction"]: x for x in candidates}
    scenarios = {}
    for direction in ("BUY", "SELL"):
        is_buy = direction == "BUY"
        pivot_key = "swing_highs" if is_buy else "swing_lows"
        invalid_key = "swing_lows" if is_buy else "swing_highs"
        confirm_pivots = fast.get(pivot_key, [])
        invalid_pivots = fast.get(invalid_key, [])
        confirm = confirm_pivots[-1]["price"] if confirm_pivots else None
        invalid = invalid_pivots[-1]["price"] if invalid_pivots else None
        event = latest_signal_event(fast)
        expected = "BULLISH" if is_buy else "BEARISH"
        recent_event_agrees = bool(event and event.get("direction") == expected)
        levels = collect_projection_levels(results, direction, price)
        zone_direction = direction
        best_zones = candidate_map.get(direction, {}).get("zones", [])
        nearest_zone = None
        if best_zones:
            nearest_zone = min(best_zones, key=lambda z: z["distance_from_current_price"])
        plan = build_trade_plan(nearest_zone, direction, price, atr, results) if nearest_zone else None
        scenarios[direction.lower()] = {
            "direction": direction,
            "confluence_score": candidate_map.get(direction, {}).get("base_score", 0),
            "score_is_probability": False,
            "path_targets": levels,
            "nearest_reaction_zone": public_zone(nearest_zone, direction, "SCENARIO_ZONE", plan)
                if nearest_zone and plan else None,
            "confirmation_level": confirm,
            "confirmation_rule": (
                "15min candle close above latest confirmed swing high, then hold/retest"
                if is_buy else
                "15min candle close below latest confirmed swing low, then hold/retest"
            ),
            "invalidation_level": invalid,
            "invalidation_rule": (
                "15min close below latest confirmed swing low with bullish structure failure"
                if is_buy else
                "15min close above latest confirmed swing high with bearish structure failure"
            ),
            "recent_structure_event_agrees": recent_event_agrees,
            "scenario_status": "CONDITIONAL"
        }
    best = max(candidates, key=lambda x: x["base_score"]) if candidates else None
    reliable = bool(fast.get("timestamp_reliable"))
    return {
        "status": "SCENARIO_MAP" if reliable else "DATA_TIME_UNVERIFIED",
        "as_of": fast.get("latest_datetime"), "current_price": price,
        "timestamp_basis": fast.get("timestamp_basis", "unknown"),
        "timestamp_reliable": reliable,
        "leading_direction": best["direction"] if best else "UNCONFIRMED",
        "leading_scenario": scenarios.get(best["direction"].lower()) if best else None,
        "scenarios": scenarios,
        "reference_range": {
            "lower_1_atr": round(price-atr, 5), "upper_1_atr": round(price+atr, 5),
            "atr_15min": round(atr, 5),
            "note": "ATR volatility reference, not a forecast"
        },
        "warning": "Both paths are conditional scenarios, not guaranteed predictions. Scores are confluence scores, not probabilities."
    }


# ============================================================
# ENTRY ENGINE
# ============================================================

def build_entry_setup(results, overall):
    fast, hourly, four_hour, daily = (results.get(x, {}) for x in ("15min", "1h", "4h", "1day"))
    if any(x.get("status") != "analyzed" for x in (fast, hourly, four_hour, daily)):
        return {"status": "WAIT", "score": 0, "score_is_probability": False,
                "reason": "Required 15min, 1h, 4h and 1day data is not all available"}
    price, atr = fast["latest_close"], fast["atr"]
    if atr <= 0:
        return {"status": "WAIT", "score": 0, "score_is_probability": False,
                "reason": "15min ATR unavailable"}

    candidates = []
    for direction in ("BUY", "SELL"):
        wanted = "BULLISH" if direction == "BUY" else "BEARISH"
        score, reasons = 0, []
        weights = [(overall, 15, "Overall multi-timeframe structure"),
                   (four_hour["trend"], 15, "4h structure"),
                   (daily["trend"], 15, "1day structure"),
                   (hourly["trend"], 10, "1h structure"),
                   (fast["trend"], 10, "15min structure")]
        for actual, points, label in weights:
            if actual == wanted:
                score += points
                reasons.append(label + " agrees")
        event = latest_signal_event(fast)
        if event and event.get("direction") == wanted and event.get("event") in ("BOS", "CHoCH"):
            age = fast.get("complete", 0) - 1 - event.get("break_candle_index", 0)
            if 0 <= age <= 8:
                score += 15
                reasons.append("Recent close-confirmed 15min BOS/CHoCH agrees")
        sweep_type = "SELL_SIDE_LIQUIDITY_SWEEP" if direction == "BUY" else "BUY_SIDE_LIQUIDITY_SWEEP"
        if any(s.get("type") == sweep_type and
               0 <= fast.get("complete", 0)-1-s.get("index", 0) <= 5
               for s in fast.get("liquidity", {}).get("sweeps", [])):
            score += 10
            reasons.append("Recent liquidity sweep supports direction")
        zones = get_zone_candidates(results, direction, price, atr)
        if zones:
            score += 10
            reasons.append("Active multi-timeframe reaction zone detected")
        candidates.append({
            "direction": direction, "base_score": min(100, score),
            "reasons": reasons, "zones": zones
        })

    candidates.sort(key=lambda x: x["base_score"], reverse=True)
    projection = build_scenario_map(results, candidates, overall)
    ranked = {}
    for c in candidates:
        items = []
        for i, z in enumerate(c["zones"][:MAX_DIRECTION_ZONES]):
            plan = build_trade_plan(z, c["direction"], price, atr, results)
            if plan:
                items.append(public_zone(z, c["direction"], "RANK_" + str(i+1), plan))
        ranked[c["direction"].lower()] = items

    chosen = candidates[0]
    direction, score, zones = chosen["direction"], chosen["base_score"], chosen["zones"]
    main = zones[0] if zones else None
    plan = build_trade_plan(main, direction, price, atr, results) if main else None
    qualified = bool(main and plan and main["within_entry_distance"] and score >= MIN_SETUP_SCORE)
    alternatives = sorted(
        [z for z in zones[1:] if z["zone_score"] >= 50],
        key=lambda z: z["distance_from_current_price"]
    )
    backup = alternatives[0] if alternatives else None
    extra = []
    for z in zones[1:MAX_RETURNED_ZONES]:
        p = build_trade_plan(z, direction, price, atr, results)
        if p and z is not backup:
            extra.append(public_zone(z, direction, "ADDITIONAL", p))

    return {
        "status": "POTENTIAL_SETUP" if qualified else "WAIT",
        "direction": direction, "best_candidate": direction,
        "score": score, "score_is_probability": False,
        "alternative_direction_score": candidates[1]["base_score"],
        "current_price": price,
        "reason": (
            "Qualified potential setup: zone is nearby and direction score meets threshold"
            if qualified else
            "No confirmed entry: score, distance, or active reaction-zone requirements are not met"
        ),
        "reasons": chosen["reasons"],
        "primary_zone": public_zone(main, direction, "PRIMARY", plan) if main and plan else None,
        "nearest_alternative_zone": (
            public_zone(backup, direction, "NEAREST_ALTERNATIVE",
                        build_trade_plan(backup, direction, price, atr, results))
            if backup else None
        ),
        "additional_zones": extra,
        "ranked_zones_by_direction": ranked,
        "price_projection": projection,
        "scenario_summary": {
            "leading_direction": projection.get("leading_direction"),
            "buy_path": projection.get("scenarios", {}).get("buy", {}).get("path_targets", []),
            "sell_path": projection.get("scenarios", {}).get("sell", {}).get("path_targets", [])
        },
        "warning": (
            "Defense strength means confluence of historical technical evidence, not an unbreakable level. "
            "SL/TP values are planning references, not guaranteed outcomes."
        )
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
        "status": "success", "symbol": data.get("summary", {}).get("symbol", SYMBOL),
        "engine": ENGINE_NAME, "version": VERSION,
        "received_timeframes": len(results), "total_valid_candles": total_valid,
        "overall_structure": overall, "market_structure": results,
        "entry_analysis": entry,
        "decision": "WAIT" if entry.get("status") == "WAIT" else "POTENTIAL_SETUP",
        "message": "Research analysis only. Scores are not win probabilities; validate with historical out-of-sample and walk-forward tests."
    }, None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ready", "engine": ENGINE_NAME, "version": VERSION,
        "symbol": SYMBOL, "timeframes": TIMEFRAMES,
        "modules": [
            "Data validation and timestamp quality",
            "Swing and internal market structure",
            "BOS / CHoCH close-confirmed events",
            "Liquidity pools and sweeps",
            "Fair Value Gaps and displacement Order Blocks",
            "Multi-timeframe active reaction zones",
            "Cross-timeframe zone confluence and defense-strength ranking",
            "BUY and SELL conditional price-path scenarios",
            "Confirmation and invalidation levels",
            "Entry zone, Stop Loss, TP1, TP2, TP3 and TP4",
            "Risk/reward and opposing liquidity targets"
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
