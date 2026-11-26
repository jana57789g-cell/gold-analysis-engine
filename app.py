import os
import math
import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify

VERSION = "5.4.0"
ENGINE_NAME = "Gold Analysis Engine"
SYMBOL = "XAU/USD"
TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]
TF_SECONDS = {
    "15min": 900, "1h": 3600, "4h": 14400,
    "1day": 86400, "1week": 604800, "1month": 2678400
}
MAX_CANDLES = 300
# 25 completed candles are sufficient for the engine's ATR(14) and pivot logic.
# 30 remains the preferred minimum; 25-29 candles are accepted but marked DEGRADED.
MIN_COMPLETE_CANDLES = int(os.getenv("MIN_COMPLETE_CANDLES", "25"))
if not 1 <= MIN_COMPLETE_CANDLES <= MAX_CANDLES:
    raise ValueError("MIN_COMPLETE_CANDLES must be between 1 and MAX_CANDLES")
PREFERRED_COMPLETE_CANDLES = 30
FUTURE_TOLERANCE_SECONDS = 60
CANDLE_TIMESTAMP_TYPE = os.getenv("CANDLE_TIMESTAMP_TYPE", "open").strip().lower()
if CANDLE_TIMESTAMP_TYPE not in {"open", "close"}:
    raise ValueError("CANDLE_TIMESTAMP_TYPE must be 'open' or 'close'")

app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("gold_analysis_engine")


# ---------------------------------------------------------------------------
# Safe input normalization
# ---------------------------------------------------------------------------

def parse_time(value: Any, source_timezone: Optional[str] = None) -> datetime:
    if isinstance(value, bool):
        raise ValueError("invalid_timestamp")
    if isinstance(value, (int, float)):
        stamp = float(value)
        if not math.isfinite(stamp):
            raise ValueError("invalid_timestamp")
        if abs(stamp) > 1e12:
            stamp /= 1000.0
        return datetime.fromtimestamp(stamp, tz=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_or_invalid_timestamp")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        dt = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            raise ValueError("unparseable_timestamp")
    if dt.tzinfo is None:
        # Naive exchange-local timestamps must never be guessed as UTC.
        if not source_timezone:
            raise ValueError("timezone_required_for_naive_timestamp")
        try:
            source_tz = ZoneInfo(source_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown_source_timezone:{source_timezone}") from exc
        dt = dt.replace(tzinfo=source_tz)
    return dt.astimezone(timezone.utc)


def number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError("invalid_" + field)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("invalid_" + field)
    if not math.isfinite(result):
        raise ValueError("non_finite_" + field)
    return result


def extract_candles(value: Any) -> Optional[List[dict]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("candles", "data", "items", "records", "values", "prices"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
        for key in ("result", "output", "body", "response"):
            if key in value:
                found = extract_candles(value[key])
                if found is not None:
                    return found
    return None


def unwrap_payload(payload: Any) -> Any:
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0].get("json", payload[0])
    candidate = payload
    for _ in range(6):
        if isinstance(candidate, dict):
            if isinstance(candidate.get("timeframes"), dict):
                return candidate["timeframes"]
            if any(tf in candidate for tf in TIMEFRAMES):
                return candidate
            nested = None
            for key in ("json", "data", "body", "result", "output", "response"):
                if isinstance(candidate.get(key), (dict, list)):
                    nested = candidate[key]
                    break
            if nested is None:
                break
            candidate = nested
        elif isinstance(candidate, list) and len(candidate) == 1 and isinstance(candidate[0], dict):
            candidate = candidate[0].get("json", candidate[0])
        else:
            break
    return candidate


# Twelve Data source contract for this XAU/USD workflow:
# - Intraday nodes explicitly request timezone=UTC.
# - Daily/weekly/monthly bars ignore the timezone query parameter and use
#   the XAU/USD Commodity exchange timezone, Australia/Sydney.
# Every value can be overridden explicitly through environment variables.
DEFAULT_SOURCE_TIMEZONES = {
    "15min": "UTC", "1h": "UTC", "4h": "UTC",
    "1day": "Australia/Sydney", "1week": "Australia/Sydney",
    "1month": "Australia/Sydney",
}


def find_source_timezone(payload: Any, timeframe: str,
                         timeframe_payload: Any = None) -> Optional[str]:
    """Resolve the timestamp timezone for one timeframe, never by guessing.

    Precedence: per-timeframe environment override, JSON mapping override,
    explicit timeframe metadata, global override, then the documented
    Twelve Data XAU/USD source contract above.
    """
    tf_env = os.getenv("SOURCE_TIMEZONE_" + timeframe.upper(), "").strip()
    if tf_env:
        zone = tf_env
    else:
        mapping_raw = os.getenv("SOURCE_TIMEZONE_BY_TIMEFRAME", "").strip()
        mapping = {}
        if mapping_raw:
            try:
                mapping = json.loads(mapping_raw)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid_SOURCE_TIMEZONE_BY_TIMEFRAME_JSON") from exc
            if not isinstance(mapping, dict):
                raise ValueError("SOURCE_TIMEZONE_BY_TIMEFRAME_must_be_object")
        zone = str(mapping.get(timeframe, "")).strip()
        if not zone and isinstance(timeframe_payload, (dict, list)):
            keys = {"source_timezone", "exchange_timezone", "timezone", "tz"}
            found = []
            def walk(obj: Any, depth: int = 0) -> None:
                if depth > 5:
                    return
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if str(key).lower() in keys and isinstance(value, str) and value.strip():
                            found.append(value.strip())
                        elif isinstance(value, (dict, list)):
                            walk(value, depth + 1)
                elif isinstance(obj, list):
                    for item in obj[:5]:
                        walk(item, depth + 1)
            walk(timeframe_payload)
            unique = list(dict.fromkeys(found))
            if len(unique) > 1:
                raise ValueError(f"{timeframe}:conflicting_source_timezones:" + ",".join(unique))
            zone = unique[0] if unique else ""
        if not zone:
            zone = os.getenv("SOURCE_TIMEZONE", "").strip()
        if not zone:
            zone = DEFAULT_SOURCE_TIMEZONES.get(timeframe, "")
    if not zone:
        return None
    if zone.lower() in {"exchange_local", "local", "server_local"}:
        raise ValueError("source_timezone_must_be_an_iana_timezone_not_a_convention_label")
    try:
        ZoneInfo(zone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown_source_timezone:{zone}") from exc
    return zone


def _month_after(dt: datetime) -> datetime:
    year = dt.year + (1 if dt.month == 12 else 0)
    month = 1 if dt.month == 12 else dt.month + 1
    # Preserve the local calendar date where possible (e.g. Jan 31 -> Feb 28).
    import calendar
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def candle_end_time(candle_dt: datetime, timeframe: str) -> datetime:
    """Return interval end using calendar arithmetic for monthly candles."""
    if CANDLE_TIMESTAMP_TYPE == "close":
        return candle_dt
    if timeframe == "1month":
        return _month_after(candle_dt)
    from datetime import timedelta
    return candle_dt + timedelta(seconds=TF_SECONDS[timeframe])


def normalize_payload(payload: Any) -> Tuple[Dict[str, List[dict]], List[str], List[str], Dict[str, dict]]:
    errors: List[str] = []
    notes: List[str] = []
    diagnostics: Dict[str, dict] = {}
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    raw_frames = unwrap_payload(payload)
    if not isinstance(raw_frames, dict):
        return {}, ["timeframes_must_be_object"], notes, diagnostics

    frames: Dict[str, List[dict]] = {}
    for tf in TIMEFRAMES:
        raw = raw_frames.get(tf)
        if raw is None:
            errors.append(f"{tf}:missing_timeframe")
            diagnostics[tf] = {"input_count": 0, "parsed_count": 0, "complete_count": 0,
                               "incomplete_count": 0, "invalid_count": 0, "duplicate_count": 0}
            continue
        rows = extract_candles(raw)
        if rows is None:
            errors.append(f"{tf}:candles_must_be_array")
            diagnostics[tf] = {"input_count": 0, "parsed_count": 0, "complete_count": 0,
                               "incomplete_count": 0, "invalid_count": 0, "duplicate_count": 0}
            continue

        try:
            source_timezone = find_source_timezone(payload, tf, raw)
            if source_timezone:
                notes.append(f"{tf}:naive_timestamps_interpreted_in_timezone:{source_timezone}")
            else:
                notes.append(f"{tf}:timezone_not_assumed:naive_timestamps_will_be_rejected")
        except (ValueError, ZoneInfoNotFoundError) as exc:
            errors.append(f"{tf}:{str(exc)}")
            frames[tf] = []
            diagnostics[tf] = {"input_count": len(rows), "parsed_count": 0, "complete_count": 0,
                               "incomplete_count": 0, "invalid_count": len(rows), "duplicate_count": 0,
                               "source_timezone": None}
            continue

        input_count = len(rows)
        parsed: List[dict] = []
        invalid_count = 0
        future_count = 0
        for idx, row in enumerate(rows):
            try:
                if not isinstance(row, dict):
                    raise ValueError("candle_must_be_object")
                raw_dt = row.get("datetime", row.get("timestamp", row.get("time", row.get("date"))))
                dt = parse_time(raw_dt, source_timezone)
                o = number(row.get("open"), "open")
                h = number(row.get("high"), "high")
                lo = number(row.get("low"), "low")
                c = number(row.get("close"), "close")
                if min(o, h, lo, c) <= 0:
                    raise ValueError("non_positive_ohlc")
                if h < max(o, c, lo) or lo > min(o, c, h):
                    raise ValueError("inconsistent_ohlc")
                vol = row.get("volume")
                if vol is not None:
                    vol = number(vol, "volume")
                    if vol < 0:
                        raise ValueError("negative_volume")
                incomplete = row.get("is_incomplete", row.get("incomplete", False))
                if not isinstance(incomplete, bool):
                    raise ValueError("invalid_is_incomplete")

                # Both timestamp conventions are supported, but never guessed.
                if dt.timestamp() > now_ts + FUTURE_TOLERANCE_SECONDS:
                    future_count += 1
                    errors.append(f"{tf}[{idx}]:future_timestamp_by_{int(dt.timestamp() - now_ts)}_seconds")
                end_dt = candle_end_time(dt, tf)
                time_incomplete = end_dt.timestamp() > now_ts + FUTURE_TOLERANCE_SECONDS
                parsed.append({
                    "datetime": dt.isoformat(), "_ts": dt.timestamp(),
                    "open": o, "high": h, "low": lo, "close": c,
                    "volume": vol, "is_incomplete": bool(incomplete or time_incomplete)
                })
            except Exception as exc:
                invalid_count += 1
                errors.append(f"{tf}[{idx}]:{str(exc)}")

        parsed.sort(key=lambda x: x["_ts"])
        seen = set()
        unique = []
        duplicate_count = 0
        for candle in parsed:
            ts = candle["_ts"]
            if ts in seen:
                duplicate_count += 1
                errors.append(f"{tf}:duplicate_timestamp:{candle['datetime']}")
                continue
            seen.add(ts)
            unique.append(candle)

        # Sort first, then retain the most recent MAX_CANDLES; this also works
        # when the source returns candles in reverse or mixed order.
        if len(unique) > MAX_CANDLES:
            unique = unique[-MAX_CANDLES:]
            notes.append(f"{tf}:trimmed_to_latest_{MAX_CANDLES}_chronological_candles")

        complete = [x for x in unique if not x["is_incomplete"]]
        incomplete_count = len(unique) - len(complete)
        diagnostics[tf] = {
            "input_count": input_count,
            "parsed_count": len(parsed),
            "complete_count": len(complete),
            "incomplete_count": incomplete_count,
            "invalid_count": invalid_count,
            "duplicate_count": duplicate_count,
            "future_timestamp_count": future_count,
            "timestamp_type": CANDLE_TIMESTAMP_TYPE,
            "source_timezone": source_timezone
        }
        if len(complete) < MIN_COMPLETE_CANDLES:
            errors.append(f"{tf}:insufficient_complete_candles:{len(complete)}<{MIN_COMPLETE_CANDLES}")
        elif len(complete) < PREFERRED_COMPLETE_CANDLES:
            notes.append(f"{tf}:degraded_candle_count:{len(complete)}<{PREFERRED_COMPLETE_CANDLES}")
        frames[tf] = complete

    return frames, errors, notes, diagnostics

# ---------------------------------------------------------------------------
# Data quality: future timestamps are a hard block, not a soft warning.
# Datetimes are assumed to be candle OPEN times, unless the provider contract
# says otherwise. The engine never shifts or fabricates timestamps.
# ---------------------------------------------------------------------------

def data_quality_report(frames: Dict[str, List[dict]], errors: List[str],
                        notes: List[str], diagnostics: Optional[Dict[str, dict]] = None,
                        now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    now_ts = now.timestamp()
    diagnostics = diagnostics or {}
    report, warnings, blockers = {}, [], []
    degraded = False
    for tf in TIMEFRAMES:
        candles = frames.get(tf, [])
        latest = candles[-1] if candles else None
        counts = diagnostics.get(tf, {})
        if latest:
            latest_dt = datetime.fromtimestamp(latest["_ts"], tz=timezone.utc)
            end_dt = candle_end_time(latest_dt, tf)
            future = latest["_ts"] - now_ts
            age_after_close = max(0.0, now_ts - end_dt.timestamp())
            if future > FUTURE_TOLERANCE_SECONDS:
                blockers.append(f"{tf}:future_timestamp_by_{int(future)}_seconds")
            # A month is not stale merely because the prior completed month ended
            # more than one day ago. Use a 45-day monthly freshness allowance.
            stale_limit = max(TF_SECONDS[tf] * 2.5, 45 * 86400 if tf == "1month" else 0)
            if age_after_close > stale_limit:
                warnings.append(f"{tf}:latest_completed_candle_is_stale")
            if len(candles) < PREFERRED_COMPLETE_CANDLES:
                degraded = True
                warnings.append(f"{tf}:below_preferred_candle_count:{len(candles)}<{PREFERRED_COMPLETE_CANDLES}")
            report[tf] = {
                **counts,
                "complete_count": len(candles),
                "latest_datetime": latest["datetime"],
                "latest_close": round(latest["close"], 5),
                "timestamp_assumption": f"candle_{CANDLE_TIMESTAMP_TYPE}_time; explicit offset/epoch or configured IANA timezone required",
                "future_seconds": round(max(0.0, future), 1),
                "age_after_candle_close_seconds": round(age_after_close, 1)
            }
        else:
            report[tf] = {**counts, "complete_count": 0, "latest_datetime": None,
                          "latest_close": None, "future_seconds": None,
                          "age_after_candle_close_seconds": None}
            blockers.append(f"{tf}:no_usable_candles")

    if errors:
        blockers.extend(errors)
    valid_counts = all(len(frames.get(tf, [])) >= MIN_COMPLETE_CANDLES for tf in TIMEFRAMES)
    valid = not blockers and valid_counts
    status = "INVALID" if not valid else "DEGRADED" if degraded else "VALID"
    return {
        "status": status,
        "analysis_blocked": not valid,
        "max_candles_per_timeframe": MAX_CANDLES,
        "minimum_complete_candles": MIN_COMPLETE_CANDLES,
        "preferred_complete_candles": PREFERRED_COMPLETE_CANDLES,
        "source_timezones_by_timeframe": {tf: diagnostics.get(tf, {}).get("source_timezone") for tf in TIMEFRAMES},
        "candle_timestamp_type": CANDLE_TIMESTAMP_TYPE,
        "timeframes": report,
        "errors": errors,
        "warnings": warnings,
        "blockers": blockers,
        "normalization_notes": notes,
        "self_healing": {
            "applied": ["explicit-offset/epoch UTC conversion", "chronological sorting before latest-candle selection", "safe numeric parsing", "calendar-aware monthly candle boundaries"],
            "not_applied": [
                "OHLC alteration", "timestamp shifting", "fabricating candles",
                "marking incomplete candles complete", "silently ignoring future timestamps"
            ]
        }
    }

# ---------------------------------------------------------------------------
# Technical analysis primitives
# ---------------------------------------------------------------------------

def atr(candles: List[dict], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        cur, prev = candles[i], candles[i - 1]
        trs.append(max(cur["high"] - cur["low"],
                       abs(cur["high"] - prev["close"]),
                       abs(cur["low"] - prev["close"])))
    sample = trs[-period:]
    return sum(sample) / len(sample) if sample else 0.0


def pivots(candles: List[dict], left: int = 3, right: int = 3) -> Tuple[List[dict], List[dict]]:
    highs, lows = [], []
    for i in range(left, len(candles) - right):
        h, lo = candles[i]["high"], candles[i]["low"]
        if all(h > candles[j]["high"] for j in range(i-left, i)) and all(
            h >= candles[j]["high"] for j in range(i+1, i+right+1)
        ):
            highs.append({"index": i, "price": h, "datetime": candles[i]["datetime"]})
        if all(lo < candles[j]["low"] for j in range(i-left, i)) and all(
            lo <= candles[j]["low"] for j in range(i+1, i+right+1)
        ):
            lows.append({"index": i, "price": lo, "datetime": candles[i]["datetime"]})
    return highs, lows


def structure_summary(candles: List[dict]) -> dict:
    highs, lows = pivots(candles)
    rh, rl = highs[-3:], lows[-3:]
    hh = len(rh) >= 2 and rh[-1]["price"] > rh[-2]["price"]
    hl = len(rl) >= 2 and rl[-1]["price"] > rl[-2]["price"]
    lh = len(rh) >= 2 and rh[-1]["price"] < rh[-2]["price"]
    ll = len(rl) >= 2 and rl[-1]["price"] < rl[-2]["price"]
    direction = "BULLISH" if hh and hl else "BEARISH" if lh and ll else "RANGE"
    close = candles[-1]["close"]
    last_high = highs[-1] if highs else None
    last_low = lows[-1] if lows else None
    bos = ("BULLISH_BOS" if last_high and close > last_high["price"] else
           "BEARISH_BOS" if last_low and close < last_low["price"] else None)
    choch = ("BULLISH_CHOCH" if direction == "BEARISH" and bos == "BULLISH_BOS" else
             "BEARISH_CHOCH" if direction == "BULLISH" and bos == "BEARISH_BOS" else None)
    return {
        "direction": direction, "bos": bos, "choch": choch,
        "swing_high": last_high, "swing_low": last_low,
        "higher_high": bool(hh), "higher_low": bool(hl),
        "lower_high": bool(lh), "lower_low": bool(ll),
        "pivot_high_count": len(highs), "pivot_low_count": len(lows)
    }


def rejection_strength(candle: dict, side: str) -> float:
    span = max(candle["high"] - candle["low"], 1e-9)
    body = abs(candle["close"] - candle["open"])
    lower = min(candle["open"], candle["close"]) - candle["low"]
    upper = candle["high"] - max(candle["open"], candle["close"])
    wick = lower if side == "BUY" else upper
    pos = (candle["close"] - candle["low"]) / span
    quality = pos if side == "BUY" else 1 - pos
    return round(max(0.0, min(1.0, 0.55*wick/span + 0.25*(1-body/span) + 0.20*quality)), 3)


def cluster_pivots(points: List[dict], tolerance: float, label: str) -> List[dict]:
    groups: List[List[dict]] = []
    for point in points:
        group = next((g for g in groups if abs(point["price"] -
                     sum(x["price"] for x in g)/len(g)) <= tolerance), None)
        if group is None:
            groups.append([point])
        else:
            group.append(point)
    out = []
    for group in groups:
        if len(group) >= 2:
            out.append({
                "price": round(sum(x["price"] for x in group)/len(group), 5),
                "touches": len(group), "type": label,
                "first_seen": group[0]["datetime"], "last_seen": group[-1]["datetime"]
            })
    return out


def liquidity_map(candles: List[dict], current: float, volatility: float) -> dict:
    highs, lows = pivots(candles)
    tolerance = max(volatility * 0.18, current * 0.0005)
    eqh = cluster_pivots(highs, tolerance, "EQUAL_HIGHS")
    eql = cluster_pivots(lows, tolerance, "EQUAL_LOWS")
    above = sorted([x for x in eqh if x["price"] > current], key=lambda x: x["price"])
    below = sorted([x for x in eql if x["price"] < current],
                   key=lambda x: x["price"], reverse=True)
    recent = candles[-3:]
    old_highs = [p["price"] for p in highs if p["index"] < len(candles)-3]
    old_lows = [p["price"] for p in lows if p["index"] < len(candles)-3]
    sweep = None
    if old_highs:
        level = max(old_highs[-5:])
        if any(c["high"] > level and c["close"] < level for c in recent):
            sweep = {"side": "BUY_SIDE_SWEPT", "level": round(level, 5)}
    if sweep is None and old_lows:
        level = min(old_lows[-5:])
        if any(c["low"] < level and c["close"] > level for c in recent):
            sweep = {"side": "SELL_SIDE_SWEPT", "level": round(level, 5)}
    return {
        "buy_side_liquidity": above, "sell_side_liquidity": below,
        "equal_highs": eqh, "equal_lows": eql, "recent_sweep": sweep,
        "nearest_above": above[0] if above else None,
        "nearest_below": below[0] if below else None,
        "note": "Liquidity pools are inferred from historical swing clusters; resting orders are not directly visible."
    }


# ---------------------------------------------------------------------------
# Reversal zones, ranked by distance first and strength second.
# Nearby and distant zones are reported separately.
# ---------------------------------------------------------------------------

def build_zones(candles: List[dict], current: float, side: str, tf: str) -> List[dict]:
    highs, lows = pivots(candles)
    volatility = atr(candles)
    if volatility <= 0:
        return []
    points = lows if side == "BUY" else highs
    zones = []
    for pivot in points[-45:]:
        i = pivot["index"]
        base = candles[i]
        if side == "BUY":
            lo, hi = base["low"], max(base["open"], base["close"])
            future = candles[i+1:min(i+7, len(candles))]
            impulse = max((x["high"] - base["low"] for x in future), default=0.0)
        else:
            lo, hi = min(base["open"], base["close"]), base["high"]
            future = candles[i+1:min(i+7, len(candles))]
            impulse = max((base["high"] - x["low"] for x in future), default=0.0)
        width = hi - lo
        if width <= 0 or width > volatility * 1.8:
            continue
        age = len(candles) - 1 - i
        if age > 180:
            continue
        later = candles[i+1:]
        broken = any((x["close"] < lo - volatility*0.05 if side == "BUY"
                      else x["close"] > hi + volatility*0.05) for x in later)
        if broken:
            continue
        tests, inside = 0, False
        for candle in later:
            overlaps = candle["low"] <= hi and candle["high"] >= lo
            if overlaps and not inside:
                tests += 1
            inside = overlaps
        if tests > 3:
            continue
        mid = (lo + hi) / 2
        # A buy zone must be below price; a sell zone must be above price.
        if side == "BUY" and mid >= current:
            continue
        if side == "SELL" and mid <= current:
            continue
        impulse_score = min(1.0, impulse / max(volatility*2.2, 1e-9))
        freshness = max(0.0, 1 - age/180)
        test_score = {0: 1.0, 1: 0.92, 2: 0.72, 3: 0.45}.get(tests, 0)
        narrow = max(0.0, 1 - width/max(volatility*1.8, 1e-9))
        reaction = rejection_strength(candles[i+1], side) if i+1 < len(candles) else 0
        score = 100*(0.38*impulse_score + 0.22*freshness +
                     0.22*test_score + 0.10*narrow + 0.08*reaction)
        distance = abs(current-mid)
        zones.append({
            "side": side, "timeframe": tf, "low": round(lo, 5),
            "high": round(hi, 5), "mid": round(mid, 5),
            "score": round(score, 1), "age_candles": age, "tests": tests,
            "impulse_strength": round(impulse_score, 3),
            "reaction_strength": reaction, "distance": round(distance, 5),
            "distance_atr": round(distance/volatility, 2),
            "strength_label": "STRONG" if score >= 75 else "MODERATE" if score >= 60 else "WEAK",
            "status": "VALID_UNCONFIRMED"
        })
    return zones


def combine_zone_confluence(zones: List[dict]) -> List[dict]:
    ranked = sorted(zones, key=lambda z: (-z["score"], z["distance"]))
    kept = []
    for zone in ranked:
        match = next((x for x in kept if
                      max(zone["low"], x["low"]) <= min(zone["high"], x["high"]) +
                      max(zone["high"]-zone["low"], x["high"]-x["low"])*0.20), None)
        if match:
            tfs = set(match["timeframes"])
            is_new_timeframe = zone["timeframe"] not in tfs
            tfs.add(zone["timeframe"])
            match["timeframes"] = sorted(tfs)
            match["score"] = round(min(98, max(match["score"], zone["score"]) +
                                       (2 if is_new_timeframe else 0)), 1)
            match["low"] = round(min(match["low"], zone["low"]), 5)
            match["high"] = round(max(match["high"], zone["high"]), 5)
            match["mid"] = round((match["low"] + match["high"]) / 2, 5)
        else:
            kept.append({
                "side": zone["side"], "low": zone["low"], "high": zone["high"],
                "mid": zone["mid"], "score": zone["score"],
                "distance": zone["distance"], "distance_atr": zone["distance_atr"],
                "strength_label": zone["strength_label"],
                "timeframes": [zone["timeframe"]],
                "age_candles": zone["age_candles"], "tests": zone["tests"],
                "impulse_strength": zone["impulse_strength"],
                "reaction_strength": zone["reaction_strength"],
                "status": zone["status"]
            })
    return sorted(kept, key=lambda z: (z["distance"], -z["score"]))


def zone_collections(zones: List[dict], volatility: float) -> dict:
    near_limit = max(volatility * 2.0, 0.0)
    near = [z for z in zones if z["distance"] <= near_limit]
    far = [z for z in zones if z["distance"] > near_limit]
    return {
        "nearest": zones[:5],
        "nearby": near[:8],
        "distant": far[:8],
        "strong_nearby": [z for z in near if z["score"] >= 75][:5],
        "strong_distant": [z for z in far if z["score"] >= 75][:5]
    }


# ---------------------------------------------------------------------------
# Scenario, path and trade plan construction
# ---------------------------------------------------------------------------

def make_trade_plan(zone: Optional[dict], side: str, current: float,
                    volatility: float, liquidity: dict,
                    data_blocked: bool) -> Optional[dict]:
    if not zone or volatility <= 0:
        return None
    low, high, entry = zone["low"], zone["high"], zone["mid"]
    # No plan is issued from a weak zone.
    if zone["score"] < 60:
        return None
    stop = low - volatility*0.22 if side == "BUY" else high + volatility*0.22
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    # Long targets are overhead buy-side liquidity; short targets are
    # downside sell-side liquidity. Never reverse these pools.
    pools = liquidity["buy_side_liquidity"] if side == "BUY" else liquidity["sell_side_liquidity"]
    targets = sorted(
        [p["price"] for p in pools if (p["price"] > entry if side == "BUY" else p["price"] < entry)],
        reverse=(side == "SELL")
    )
    targets = [p for p in targets if abs(p-entry) >= risk*1.5]
    selected = []
    for target in targets:
        if not selected or abs(target-selected[-1]) >= volatility*0.25:
            selected.append(target)
        if len(selected) == 3:
            break
    # A plan without an observed target is incomplete and must not be emitted.
    if not selected:
        return None
    rr = abs(selected[0]-entry)/risk
    current_inside = low <= current <= high
    if current_inside:
        activation = "Price is inside the zone; wait for a rejection candle and a confirmed lower-timeframe structure shift."
    else:
        activation = "Wait for price to enter the zone, then require a rejection candle and a confirmed lower-timeframe structure shift."
    status = "DATA_REVIEW_REQUIRED" if data_blocked else (
        "WAITING_FOR_ZONE" if not current_inside else "WAITING_FOR_CONFIRMATION"
    )
    return {
        "side": side,
        "entry_zone": {"low": low, "high": high},
        "entry_reference": entry,
        "distance_from_current": round(abs(current-entry), 5),
        "distance_atr": zone["distance_atr"],
        "current_price_inside_zone": current_inside,
        "stop_loss": round(stop, 5),
        "targets": [{"name": f"TP{i+1}", "price": round(p, 5)} for i, p in enumerate(selected)],
        "first_target_rr": round(rr, 2) if rr is not None else None,
        "target_method": "OBSERVED_LIQUIDITY",
        "target_count": len(selected),
        "zone_score": zone["score"], "zone_strength": zone["strength_label"],
        "zone_timeframes": zone["timeframes"],
        "confirmation_required": activation,
        "setup_status": status,
        "trade_allowed_now": False,
        "invalidation": "Cancel if price closes beyond the stop boundary or the zone is decisively broken.",
        "structure_context": "Check lower-timeframe BOS/CHoCH before activation."
    }


def build_price_path(current: float, volatility: float, structure: dict,
                     liquidity: dict, zones_buy: List[dict],
                     zones_sell: List[dict]) -> dict:
    upside = [p["price"] for p in liquidity["buy_side_liquidity"] if p["price"] > current]
    downside = [p["price"] for p in liquidity["sell_side_liquidity"] if p["price"] < current]
    nearest_buy_zone = zones_buy[0] if zones_buy else None
    nearest_sell_zone = zones_sell[0] if zones_sell else None
    upside_target = min(upside) if upside else (
        structure["swing_high"]["price"] if structure.get("swing_high") and structure["swing_high"]["price"] > current else None
    )
    downside_target = max(downside) if downside else (
        structure["swing_low"]["price"] if structure.get("swing_low") and structure["swing_low"]["price"] < current else None
    )
    bullish_steps = []
    bearish_steps = []
    if upside_target is not None:
        bullish_steps.append({"level": round(upside_target, 5), "reason": "nearest_observed_buy_side_liquidity"})
    if nearest_sell_zone:
        bullish_steps.append({"zone": [nearest_sell_zone["low"], nearest_sell_zone["high"]],
                              "reason": "overhead_supply_zone_to_reclaim"})
    if downside_target is not None:
        bearish_steps.append({"level": round(downside_target, 5), "reason": "nearest_observed_sell_side_liquidity"})
    if nearest_buy_zone:
        bearish_steps.append({"zone": [nearest_buy_zone["low"], nearest_buy_zone["high"]],
                              "reason": "underlying_demand_zone_to_test"})
    return {
        "current_price": round(current, 5),
        "bullish_scenario": {
            "trigger": "A 15min/1h bullish structure shift, followed by a close above the nearest local swing high.",
            "possible_path": bullish_steps,
            "invalidation": "A decisive close below the latest confirmed 15min swing low."
        },
        "bearish_scenario": {
            "trigger": "A 15min/1h bearish structure shift, followed by a close below the nearest local swing low.",
            "possible_path": bearish_steps,
            "invalidation": "A decisive close above the latest confirmed 15min swing high."
        },
        "directional_note": "These are conditional paths, not guaranteed forecasts. Levels are based only on observed candle structure and inferred liquidity."
    }


def analyze_market(frames: Dict[str, List[dict]], quality: dict) -> dict:
    primary = frames["15min"]
    current = primary[-1]["close"]
    per_tf, votes = {}, {"BULLISH": 0, "BEARISH": 0, "RANGE": 0}
    all_zones = {"BUY": [], "SELL": []}
    for tf in TIMEFRAMES:
        candles = frames[tf]
        structure = structure_summary(candles)
        volatility = atr(candles)
        per_tf[tf] = {
            "direction": structure["direction"],
            "bos": structure["bos"], "choch": structure["choch"],
            "atr": round(volatility, 5),
            "latest_close": round(candles[-1]["close"], 5),
            "latest_datetime": candles[-1]["datetime"],
            "structure": structure
        }
        votes[structure["direction"]] += 1
        # Zones are built against each timeframe's own close; distance is
        # recalculated against the common live reference price below.
        for side in ("BUY", "SELL"):
            all_zones[side].extend(build_zones(candles, current, side, tf))

    current_vol = atr(primary)
    structures = [per_tf[tf]["structure"] for tf in TIMEFRAMES]
    aligned_score = round(max(votes.values()) / len(TIMEFRAMES), 3)
    bias = max(votes, key=votes.get)
    if votes[bias] < 3:
        bias = "NEUTRAL"

    liquidity = liquidity_map(primary, current, current_vol)
    buy_zones = combine_zone_confluence(all_zones["BUY"])
    sell_zones = combine_zone_confluence(all_zones["SELL"])
    buy_groups = zone_collections(buy_zones, current_vol)
    sell_groups = zone_collections(sell_zones, current_vol)

    # Pick the nearest qualifying zone for the plan; do not let a distant
    # high-scoring zone hide a nearer valid opportunity.
    buy_plan_zone = min((z for z in buy_zones if z["score"] >= 60),
                        key=lambda z: z["distance"], default=None)
    sell_plan_zone = min((z for z in sell_zones if z["score"] >= 60),
                         key=lambda z: z["distance"], default=None)
    blocked = quality["analysis_blocked"]
    buy_plan = make_trade_plan(buy_plan_zone, "BUY", current, current_vol, liquidity, blocked)
    sell_plan = make_trade_plan(sell_plan_zone, "SELL", current, current_vol, liquidity, blocked)

    path = build_price_path(current, current_vol, per_tf["15min"]["structure"],
                            liquidity, buy_zones, sell_zones)
    decision = "DATA_INVALID" if blocked else (
        "WAIT_FOR_CONFIRMATION" if bias == "NEUTRAL" else "WAIT_FOR_SETUP"
    )
    return {
        "engine": ENGINE_NAME, "version": VERSION, "symbol": SYMBOL,
        "decision": decision, "current_price": round(current, 5),
        "market": {
            "directional_bias": bias, "timeframe_votes": votes,
            "timeframe_alignment_score": aligned_score,
            "confidence_note": "Alignment score is a rule-based agreement measure, not a calibrated probability."
        },
        "timeframe_analysis": per_tf,
        "market_structure": {
            "15min": per_tf["15min"]["structure"],
            "1h": per_tf["1h"]["structure"],
            "4h": per_tf["4h"]["structure"]
        },
        "liquidity": liquidity,
        "reversal_zones": {
            "buy": buy_groups, "sell": sell_groups,
            "all_buy_zones": buy_zones[:20], "all_sell_zones": sell_zones[:20]
        },
        "entry_analysis": {
            "buy_plan": buy_plan, "sell_plan": sell_plan,
            "note": "A zone is not an entry signal. Wait for price interaction and confirmation."
        },
        "price_path": path,
        "data_validation": quality,
        "risk_notice": "Analytical scenarios only; no trade is authorized automatically."
    }


# ---------------------------------------------------------------------------
# Human-readable report for Telegram / AI formatting node
# ---------------------------------------------------------------------------

def fmt_price(value: Any) -> str:
    return "غير متاح" if value is None else f"{float(value):,.2f}"


def format_plan(name: str, plan: Optional[dict]) -> List[str]:
    if not plan:
        return [f"{name}: لا توجد خطة مستوفية لشروط المحرك حاليًا."]
    zone = plan["entry_zone"]
    lines = [
        f"{name}:",
        f"- منطقة الدخول: {fmt_price(zone['low'])} إلى {fmt_price(zone['high'])}",
        f"- السعر المرجعي: {fmt_price(plan['entry_reference'])}",
        f"- البعد عن السعر الحالي: {plan['distance_atr']} ATR",
        f"- وقف الخسارة: {fmt_price(plan['stop_loss'])}",
        f"- الأهداف: " + (" | ".join(f"{x['name']} {fmt_price(x['price'])}" for x in plan["targets"])
                            if plan["targets"] else "لا توجد 3 تجمعات سيولة مؤكدة"),
        f"- العائد/المخاطرة إلى الهدف الأول: {plan['first_target_rr'] if plan['first_target_rr'] is not None else 'غير متاح'}",
        f"- الحالة: {plan['setup_status']}",
        f"- التفعيل: {plan['confirmation_required']}",
        f"- الإبطال: {plan['invalidation']}"
    ]
    return lines


def build_telegram_report(result: dict) -> str:
    market = result["market"]
    quality = result["data_validation"]
    path = result["price_path"]
    lines = [
        f"📊 {ENGINE_NAME} | {SYMBOL} | v{VERSION}",
        f"السعر المرجعي: {fmt_price(result['current_price'])}",
        f"القرار: {result['decision']}",
        f"الاتجاه العام: {market['directional_bias']} | توافق الفريمات: {market['timeframe_alignment_score']}",
        f"تصويت الفريمات: {market['timeframe_votes']}",
        f"سلامة البيانات: {quality['status']}"
    ]
    if quality["blockers"]:
        lines.append("⛔ موانع التحليل: " + "؛ ".join(quality["blockers"][:8]))
    if quality["warnings"]:
        lines.append("⚠️ تحذيرات البيانات: " + "؛ ".join(quality["warnings"][:8]))
    lines.extend(["", "📍 أقرب مناطق الانعكاس"])
    for side, label in (("buy", "شراء"), ("sell", "بيع")):
        group = result["reversal_zones"][side]
        lines.append(f"{label}:")
        for zone in group["nearest"][:3]:
            lines.append(
                f"- {fmt_price(zone['low'])}–{fmt_price(zone['high'])} | قوة {zone['score']}/100 "
                f"| بُعد {zone['distance_atr']} ATR | {zone['strength_label']} | {','.join(zone['timeframes'])}"
            )
        if not group["nearest"]:
            lines.append("- لا توجد منطقة صالحة مكتشفة.")
        lines.append(f"مناطق قوية بعيدة: {len(group['strong_distant'])}")
    lines.extend(["", "🎯 خطط التداول المشروطة"])
    lines.extend(format_plan("خطة شراء", result["entry_analysis"]["buy_plan"]))
    lines.extend(format_plan("خطة بيع", result["entry_analysis"]["sell_plan"]))
    lines.extend([
        "", "🧭 السيناريو الصاعد",
        f"- شرط التفعيل: {path['bullish_scenario']['trigger']}",
        f"- المسار المحتمل: {path['bullish_scenario']['possible_path']}",
        f"- الإبطال: {path['bullish_scenario']['invalidation']}",
        "", "🧭 السيناريو الهابط",
        f"- شرط التفعيل: {path['bearish_scenario']['trigger']}",
        f"- المسار المحتمل: {path['bearish_scenario']['possible_path']}",
        f"- الإبطال: {path['bearish_scenario']['invalidation']}",
        "", "💧 السيولة",
        f"- أقرب سيولة أعلى: {fmt_price(result['liquidity']['nearest_above']['price']) if result['liquidity']['nearest_above'] else 'غير مرصودة'}",
        f"- أقرب سيولة أسفل: {fmt_price(result['liquidity']['nearest_below']['price']) if result['liquidity']['nearest_below'] else 'غير مرصودة'}",
        f"- آخر سحب سيولة: {result['liquidity']['recent_sweep']}",
        "", "تنبيه: المناطق تقديرية من بيانات الشموع وليست ضمانًا للانعكاس. لا توجد صفقة مسموح بها تلقائيًا."
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Flask API
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return jsonify({
        "status": "ok", "engine": ENGINE_NAME, "version": VERSION,
        "symbol": SYMBOL, "timeframes": TIMEFRAMES,
        "max_candles_per_timeframe": MAX_CANDLES,
        "minimum_complete_candles": MIN_COMPLETE_CANDLES,
        "preferred_complete_candles": PREFERRED_COMPLETE_CANDLES,
        "configured_source_timezone": os.getenv("SOURCE_TIMEZONE") or None,
        "source_timezones_by_timeframe": {
            tf: os.getenv("SOURCE_TIMEZONE_" + tf.upper()) or DEFAULT_SOURCE_TIMEZONES.get(tf)
            for tf in TIMEFRAMES
        },
        "candle_timestamp_type": CANDLE_TIMESTAMP_TYPE,
        "timestamp_policy": "epoch or timezone-offset timestamps accepted; naive timestamps use the documented per-timeframe XAU/USD source timezone unless explicitly overridden"
    })


@app.post("/analyze")
def analyze():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "invalid_or_missing_json_body"}), 400
        frames, errors, notes, diagnostics = normalize_payload(payload)
        quality = data_quality_report(frames, errors, notes, diagnostics)
        if quality["analysis_blocked"]:
            response = {
                "engine": ENGINE_NAME, "version": VERSION,
                "decision": "DATA_INVALID",
                "data_validation": quality,
                "message": "Analysis stopped because the candle data failed validation. Correct the source timestamps/data and resend; the engine will not shift timestamps or invent candles."
            }
            return jsonify(response), 422
        result = analyze_market(frames, quality)
        result["telegram_report"] = build_telegram_report(result)
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Unhandled analysis error")
        return jsonify({
            "engine": ENGINE_NAME, "version": VERSION,
            "error": "internal_analysis_error",
            "detail": str(exc) if app.debug else "Unexpected processing error. Check server logs."
        }), 500


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "not_found"}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "method_not_allowed"}), 405


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
