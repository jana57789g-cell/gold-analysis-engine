import os
import math
import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Dict, List, Optional, Tuple

# Vercel WSGI entrypoint: a module-level Flask instance named exactly `app`.
from flask import Flask, request, jsonify

app = Flask(__name__)
__all__ = ["app"]

VERSION = "7.3.0"
ENGINE_NAME = "Gold Analysis Engine"
SYMBOL = "XAU/USD"
TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]
TF_SECONDS = {
    "15min": 900, "1h": 3600, "4h": 14400,
    "1day": 86400, "1week": 604800, "1month": 2678400
}
MAX_CANDLES = 300
MIN_COMPLETE_CANDLES = int(os.getenv("MIN_COMPLETE_CANDLES", "25"))
if not 1 <= MIN_COMPLETE_CANDLES <= MAX_CANDLES:
    raise ValueError("MIN_COMPLETE_CANDLES must be between 1 and MAX_CANDLES")
PREFERRED_COMPLETE_CANDLES = 30
FUTURE_TOLERANCE_SECONDS = 60
LIVE_QUOTE_REQUIRED = False
LIVE_QUOTE_MAX_AGE_SECONDS = int(os.getenv("LIVE_QUOTE_MAX_AGE_SECONDS", "180"))
if LIVE_QUOTE_MAX_AGE_SECONDS < 1:
    raise ValueError("LIVE_QUOTE_MAX_AGE_SECONDS must be positive")
CANDLE_TIMESTAMP_TYPE = os.getenv("CANDLE_TIMESTAMP_TYPE", "open").strip().lower()
if CANDLE_TIMESTAMP_TYPE not in {"open", "close"}:
    raise ValueError("CANDLE_TIMESTAMP_TYPE must be 'open' or 'close'")

# A candidate must be reasonably close to price and have a meaningful zone score.
# These are analytical filters, not calibrated probabilities or execution advice.
MAX_PLAN_DISTANCE_ATR = float(os.getenv("MAX_PLAN_DISTANCE_ATR", "2.0"))
MIN_PLAN_ZONE_SCORE = float(os.getenv("MIN_PLAN_ZONE_SCORE", "70"))
if MAX_PLAN_DISTANCE_ATR <= 0:
    raise ValueError("MAX_PLAN_DISTANCE_ATR must be positive")
if not 0 <= MIN_PLAN_ZONE_SCORE <= 100:
    raise ValueError("MIN_PLAN_ZONE_SCORE must be between 0 and 100")

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


DEFAULT_SOURCE_TIMEZONES = {
    "15min": "UTC", "1h": "UTC", "4h": "UTC",
    "1day": "Australia/Sydney", "1week": "Australia/Sydney",
    "1month": "Australia/Sydney",
}


def find_source_timezone(payload: Any, timeframe: str,
                         timeframe_payload: Any = None) -> Optional[str]:
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
    import calendar
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def candle_end_time(candle_dt: datetime, timeframe: str) -> datetime:
    if CANDLE_TIMESTAMP_TYPE == "close":
        return candle_dt
    if timeframe == "1month":
        return _month_after(candle_dt)
    from datetime import timedelta
    return candle_dt + timedelta(seconds=TF_SECONDS[timeframe])


def extract_live_quote(payload: Any, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    candidates = []
    quote_keys = {"live_quote", "quote", "current_quote", "realtime_quote"}

    def walk(obj: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(obj, list):
            for item in obj[:10]:
                walk(item, depth + 1)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() in quote_keys:
                    candidates.append(value)
                elif isinstance(value, (dict, list)):
                    walk(value, depth + 1)

    walk(payload)
    if not candidates and isinstance(payload, dict) and "live_price" in payload:
        candidates.append({
            "price": payload.get("live_price"),
            "timestamp": payload.get("live_price_timestamp")
        })
    if not candidates:
        return {"valid": False, "price": None, "timestamp": None,
                "age_seconds": None, "reason": "LIVE_QUOTE_MISSING"}

    raw = candidates[0]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = raw[0]
    if not isinstance(raw, dict):
        return {"valid": False, "price": None, "timestamp": None,
                "age_seconds": None, "reason": "LIVE_QUOTE_MUST_INCLUDE_PRICE_AND_TIMESTAMP"}

    raw_price = raw.get("price", raw.get("value", raw.get("close")))
    raw_time = raw.get("timestamp", raw.get("datetime", raw.get("time", raw.get("updated_at"))))
    try:
        price = number(raw_price, "live_quote_price")
        if price <= 0:
            raise ValueError("non_positive_live_quote_price")
        quote_dt = parse_time(raw_time, None)
        age = (now - quote_dt).total_seconds()
        if age < -FUTURE_TOLERANCE_SECONDS:
            return {"valid": False, "price": price, "timestamp": quote_dt.isoformat(),
                    "age_seconds": round(age, 1), "reason": "LIVE_QUOTE_TIMESTAMP_IN_FUTURE"}
        if age > LIVE_QUOTE_MAX_AGE_SECONDS:
            return {"valid": False, "price": price, "timestamp": quote_dt.isoformat(),
                    "age_seconds": round(age, 1), "reason": "LIVE_QUOTE_STALE"}
        return {"valid": True, "price": price, "timestamp": quote_dt.isoformat(),
                "age_seconds": round(max(0.0, age), 1), "reason": None}
    except (ValueError, TypeError, OverflowError) as exc:
        return {"valid": False, "price": None, "timestamp": None,
                "age_seconds": None, "reason": str(exc)}


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
# Data quality
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
            if tf == "15min":
                blockers.append(f"{tf}:no_usable_candles")
            else:
                warnings.append(f"{tf}:no_usable_candles_excluded_from_analysis")
                degraded = True

    primary_count = len(frames.get("15min", []))
    if primary_count < MIN_COMPLETE_CANDLES:
        blockers.append(f"15min:insufficient_primary_candles:{primary_count}<{MIN_COMPLETE_CANDLES}")
    for tf in TIMEFRAMES:
        count = len(frames.get(tf, []))
        if count < MIN_COMPLETE_CANDLES and tf != "15min":
            warnings.append(f"{tf}:excluded_from_analysis_insufficient_candles:{count}<{MIN_COMPLETE_CANDLES}")
            degraded = True
    for err in errors:
        warnings.append("input_issue_reviewed:" + err)
        degraded = True
    valid = primary_count >= MIN_COMPLETE_CANDLES
    status = "INVALID" if not valid else "DEGRADED" if degraded or warnings else "VALID"
    available = [tf for tf in TIMEFRAMES if len(frames.get(tf, [])) >= MIN_COMPLETE_CANDLES]
    return {
        "status": status,
        "analysis_blocked": not valid,
        "max_candles_per_timeframe": MAX_CANDLES,
        "minimum_complete_candles": MIN_COMPLETE_CANDLES,
        "preferred_complete_candles": PREFERRED_COMPLETE_CANDLES,
        "source_timezones_by_timeframe": {tf: diagnostics.get(tf, {}).get("source_timezone") for tf in TIMEFRAMES},
        "candle_timestamp_type": CANDLE_TIMESTAMP_TYPE,
        "live_quote_required": LIVE_QUOTE_REQUIRED,
        "live_quote_max_age_seconds": LIVE_QUOTE_MAX_AGE_SECONDS,
        "timeframes": report,
        "available_timeframes": available,
        "excluded_timeframes": [tf for tf in TIMEFRAMES if tf not in available],
        "errors": errors,
        "warnings": list(dict.fromkeys(warnings)),
        "blockers": blockers,
        "engine_health": {
            "status": "DEGRADED" if valid and (degraded or warnings) else "HEALTHY" if valid else "BLOCKED",
            "usable_timeframes": len(available),
            "expected_timeframes": len(TIMEFRAMES),
            "primary_reference_available": primary_count >= MIN_COMPLETE_CANDLES,
            "policy": "continue_with_valid_timeframes; 15min_reference_required",
            "data_mutation_performed": False
        },
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
    last_high = highs[-1] if highs else None
    last_low = lows[-1] if lows else None
    previous_close = candles[-2]["close"] if len(candles) >= 2 else None
    close = candles[-1]["close"] if candles else None
    crossed_up = bool(last_high and previous_close is not None and
                      previous_close <= last_high["price"] < close)
    crossed_down = bool(last_low and previous_close is not None and
                        previous_close >= last_low["price"] > close)
    bos = ("BULLISH_BOS" if crossed_up and not crossed_down else
           "BEARISH_BOS" if crossed_down and not crossed_up else None)
    choch = ("BULLISH_CHOCH" if direction == "BEARISH" and bos == "BULLISH_BOS" else
             "BEARISH_CHOCH" if direction == "BULLISH" and bos == "BEARISH_BOS" else None)
    return {
        "direction": direction, "bos": bos, "choch": choch,
        "swing_high": last_high, "swing_low": last_low,
        "higher_high": bool(hh), "higher_low": bool(hl),
        "lower_high": bool(lh), "lower_low": bool(ll),
        "pivot_high_count": len(highs), "pivot_low_count": len(lows),
        "break_event_is_latest_candle": bool(bos)
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
# Reversal zones
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
        if side == "BUY" and mid >= current:
            continue
        if side == "SELL" and mid <= current:
            continue
        impulse_score = min(1.0, impulse / max(volatility*2.2, 1e-9))
        freshness_horizon = {
            "15min": 48, "1h": 72, "4h": 60,
            "1day": 45, "1week": 26, "1month": 12
        }.get(tf, 48)
        freshness = max(0.0, 1 - age / freshness_horizon)
        test_score = {0: 1.0, 1: 0.92, 2: 0.72, 3: 0.45}.get(tests, 0)
        narrow = max(0.0, 1 - width/max(volatility*1.8, 1e-9))
        reaction_index = min(i + 4, len(candles) - 1)
        reaction = rejection_strength(candles[reaction_index], side) if reaction_index > i else 0
        score = 100*(0.38*impulse_score + 0.22*freshness +
                     0.22*test_score + 0.10*narrow + 0.08*reaction)
        distance = abs(current-mid)
        zones.append({
            "side": side, "timeframe": tf, "low": round(lo, 5),
            "high": round(hi, 5), "mid": round(mid, 5),
            "score": round(score, 1), "age_candles": age, "tests": tests,
            "source_atr": round(volatility, 5),
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
        match = None
        for candidate in kept:
            overlap = min(zone["high"], candidate["high"]) - max(zone["low"], candidate["low"])
            smaller_width = min(zone["high"] - zone["low"], candidate["high"] - candidate["low"])
            if smaller_width > 0 and overlap / smaller_width >= 0.35:
                match = candidate
                break
        if match:
            tfs = set(match["timeframes"])
            is_new_timeframe = zone["timeframe"] not in tfs
            tfs.add(zone["timeframe"])
            match["timeframes"] = sorted(tfs)
            match["score"] = round(min(98.0, max(match["score"], zone["score"]) +
                                       (2.0 if is_new_timeframe else 0.0)), 1)
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
                "source_atr": zone.get("source_atr"),
                "status": zone["status"]
            })
    return sorted(kept, key=lambda z: (z["distance"], -z["score"]))


def zone_collections(zones: List[dict], volatility: float) -> dict:
    near_limit = max(volatility * 2.0, 0.0)
    active_limit = max(volatility * 4.0, 0.0)
    near = [z for z in zones if z["distance"] <= near_limit]
    active = [z for z in zones if z["distance"] <= active_limit]
    far = [z for z in zones if z["distance"] > near_limit]
    return {
        "nearest": active[:5],
        "nearby": near[:8],
        "distant": far[:8],
        "strong_nearby": [z for z in near if z["score"] >= 75][:5],
        "strong_distant": [z for z in far if z["score"] >= 75][:5],
        "active_distance_limit_atr": 4.0
    }


# ---------------------------------------------------------------------------
# Scenario, path and trade-plan analysis
# ---------------------------------------------------------------------------

def candle_body_ratio(candle: dict) -> float:
    span = max(candle["high"] - candle["low"], 1e-9)
    return min(1.0, abs(candle["close"] - candle["open"]) / span)


def displacement_score(candles: List[dict], volatility: float, side: str) -> float:
    if len(candles) < 3 or volatility <= 0:
        return 0.0
    sample = candles[-3:]
    directional = [c for c in sample if
                   (c["close"] > c["open"] if side == "BUY" else c["close"] < c["open"])]
    if not directional:
        return 0.0
    largest = max(abs(c["close"] - c["open"]) for c in directional)
    return round(min(1.0, largest / max(volatility * 0.85, 1e-9)), 3)


def confirmation_evidence(candles: List[dict], zone: dict, side: str,
                          volatility: float, liquidity: dict,
                          directional_bias: str = "NEUTRAL") -> dict:
    recent = candles[-5:]
    touch_offsets = [i for i, c in enumerate(recent)
                     if c["low"] <= zone["high"] and c["high"] >= zone["low"]]
    touched = bool(touch_offsets)
    bars_since_touch = (len(recent) - 1 - touch_offsets[-1]) if touched else None
    last = candles[-1]
    reaction_window = recent[touch_offsets[-1]:] if touched else [last]
    rejection = max(rejection_strength(c, side) for c in reaction_window)
    disp = displacement_score(candles, volatility, side)
    expected_bos = "BULLISH_BOS" if side == "BUY" else "BEARISH_BOS"
    expected_choch = "BULLISH_CHOCH" if side == "BUY" else "BEARISH_CHOCH"
    recent_structure = [(end, structure_summary(candles[:end]))
                        for end in range(max(2, len(candles)-2), len(candles)+1)]
    bos_events = [(end, item) for end, item in recent_structure if item.get("bos") == expected_bos]
    choch_events = [(end, item) for end, item in recent_structure if item.get("choch") == expected_choch]
    bos_confirmed = bool(bos_events)
    choch_confirmed = bool(choch_events)
    event_matches = choch_events or bos_events
    recent_event_end, recent_event = event_matches[-1] if event_matches else (None, {})
    sweep = liquidity.get("recent_sweep") or {}
    expected_sweep = "SELL_SIDE_SWEPT" if side == "BUY" else "BUY_SIDE_SWEPT"
    sweep_confirmed = sweep.get("side") == expected_sweep
    rejection_ok = rejection >= 0.55
    displacement_ok = disp >= 0.55
    bias_aligned = directional_bias == ("BULLISH" if side == "BUY" else "BEARISH")
    reversal_confirmed = bool(touched and rejection_ok and displacement_ok and
                              sweep_confirmed and choch_confirmed)
    continuation_confirmed = bool(touched and rejection_ok and displacement_ok and
                                  bos_confirmed and bias_aligned)
    checks = {
        "zone_touched_recently": touched,
        "rejection_candle": rejection_ok,
        "directional_displacement": displacement_ok,
        "fresh_bos": bos_confirmed,
        "fresh_choch": choch_confirmed,
        "liquidity_sweep": sweep_confirmed,
        "higher_timeframe_alignment": bias_aligned
    }
    weights = {
        "zone_touched_recently": 0.10, "rejection_candle": 0.15,
        "directional_displacement": 0.15, "fresh_bos": 0.15,
        "fresh_choch": 0.15, "liquidity_sweep": 0.15,
        "higher_timeframe_alignment": 0.15
    }
    score = sum(weights[k] for k, passed in checks.items() if passed)
    setup_type = "REVERSAL" if reversal_confirmed else (
        "CONTINUATION" if continuation_confirmed else "UNCONFIRMED")
    return {
        "checks": checks,
        "confirmation_score": round(score, 2),
        "score_note": "Technical evidence score only; not a win probability.",
        "rejection_strength": rejection,
        "displacement_strength": disp,
        "bars_since_last_zone_touch": bars_since_touch,
        "structure_event": recent_event.get("bos") or recent_event.get("choch"),
        "structure_event_age_candles": (len(candles) - recent_event_end) if recent_event_end else None,
        "recent_liquidity_sweep": sweep if sweep else None,
        "setup_type": setup_type,
        "reversal_confirmed": reversal_confirmed,
        "continuation_confirmed": continuation_confirmed,
        "confirmed": reversal_confirmed or continuation_confirmed
    }


def select_targets(side: str, entry: float, risk: float, liquidity: dict,
                   opposing_zones: List[dict], volatility: float) -> Tuple[List[float], str]:
    """Use both 15m and higher-timeframe pools; HTF pools retain their source."""
    primary_key = "buy_side_liquidity" if side == "BUY" else "sell_side_liquidity"
    candidates = []
    pools = list(liquidity.get(primary_key, []))
    for pool in liquidity.get("higher_timeframe_pools", []):
        expected_type = "EQUAL_HIGHS" if side == "BUY" else "EQUAL_LOWS"
        if pool.get("type") == expected_type:
            pools.append(pool)
    seen_prices = set()
    for pool in pools:
        price = pool.get("price")
        if price is None:
            continue
        price = float(price)
        is_ahead = price > entry if side == "BUY" else price < entry
        if is_ahead and round(price, 5) not in seen_prices:
            candidates.append((price, "OBSERVED_LIQUIDITY_HTF" if pool.get("timeframe") else "OBSERVED_LIQUIDITY"))
            seen_prices.add(round(price, 5))
    for zone in opposing_zones:
        boundary = zone["low"] if side == "BUY" else zone["high"]
        if boundary > entry if side == "BUY" else boundary < entry:
            candidates.append((boundary, "OPPOSING_ZONE_BOUNDARY"))
    candidates.sort(key=lambda x: x[0], reverse=(side == "SELL"))
    selected = []
    for price, source in candidates:
        distance = abs(price - entry)
        if distance < max(risk * 1.35, volatility * 0.25):
            continue
        if selected and abs(price - selected[-1][0]) < max(volatility * 0.20, 1e-8):
            continue
        selected.append((price, source))
        if len(selected) == 3:
            break
    return [x[0] for x in selected], (selected[0][1] if selected else "NO_VALID_OBSERVED_TARGET")


def make_trade_plan(zone: Optional[dict], side: str, current: float,
                    volatility: float, liquidity: dict,
                    data_blocked: bool, primary_candles: List[dict],
                    opposing_zones: List[dict], directional_bias: str) -> Optional[dict]:
    if not zone or volatility <= 0:
        return None
    if zone.get("score", 0) < MIN_PLAN_ZONE_SCORE:
        return None
    distance_atr = abs(current - zone["mid"]) / volatility
    if distance_atr > MAX_PLAN_DISTANCE_ATR:
        return None
    low, high, entry = zone["low"], zone["high"], zone["mid"]
    if high <= low or (high - low) > volatility * 2.2:
        return None

    evidence = confirmation_evidence(primary_candles, zone, side, volatility, liquidity, directional_bias)
    expected_bias = "BULLISH" if side == "BUY" else "BEARISH"
    opposite_bias = "BEARISH" if side == "BUY" else "BULLISH"
    aligned = directional_bias == expected_bias
    # A counter-trend candidate is retained only when a full reversal sequence exists.
    if directional_bias == opposite_bias and not evidence["reversal_confirmed"]:
        return None

    stop = low - volatility * 0.18 if side == "BUY" else high + volatility * 0.18
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    targets, target_source = select_targets(side, entry, risk, liquidity,
                                            opposing_zones, volatility)
    if not targets:
        return None
    rr = abs(targets[0] - entry) / risk
    if rr < 1.35:
        return None

    current_inside = low <= current <= high
    touched = evidence["checks"]["zone_touched_recently"]
    if data_blocked:
        status = "DATA_REVIEW_REQUIRED"
    elif not touched:
        status = "WAITING_FOR_ZONE"
    elif not evidence["confirmed"]:
        status = "WAITING_FOR_CONFIRMATION"
    else:
        status = "CONFIRMED_CONDITIONALLY_NOT_AUTO_EXECUTABLE"

    return {
        "side": side,
        "setup_type": evidence["setup_type"],
        "entry_label": "المنقطع",
        "entry_zone": {"low": low, "high": high},
        "entry_reference": entry,
        "distance_from_current": round(abs(current-entry), 5),
        "distance_atr": round(distance_atr, 2),
        "current_price_inside_zone": current_inside,
        "stop_loss": round(stop, 5),
        "targets": [{"name": f"TP{i+1}", "price": round(price, 5)}
                    for i, price in enumerate(targets)],
        "first_target_rr": round(rr, 2),
        "target_method": target_source,
        "target_count": len(targets),
        "zone_score": zone["score"], "zone_strength": zone["strength_label"],
        "zone_timeframes": zone["timeframes"],
        "entry_basis": {
            "zone_age_candles": zone.get("age_candles"),
            "zone_tests": zone.get("tests"),
            "impulse_strength": zone.get("impulse_strength"),
            "reaction_strength": zone.get("reaction_strength"),
            "confluence_timeframes": zone["timeframes"],
            "directional_alignment": aligned,
            "confirmation": evidence
        },
        "confirmation_required": (
            "انعكاس: سحب سيولة ثم CHoCH حديث ورفض وإزاحة من المنطقة."
            if evidence["setup_type"] == "REVERSAL" else
            "استمرار: توافق الاتجاه الأعلى ثم BOS حديث وإعادة اختبار ورفض وإزاحة."
            if evidence["setup_type"] == "CONTINUATION" else
            "انتظار لمس المنطقة واكتمال شروط الانعكاس أو الاستمرار."
        ),
        "setup_status": status,
        "trade_allowed_now": False,
        "plan_mode": "CONDITIONAL_ANALYTICAL_ONLY",
        "invalidation": "إلغاء السيناريو عند إغلاق واضح خلف حد الإبطال؛ لا توسّع وقف الخسارة بعد الدخول.",
        "structure_context": "تُراجع بنية 15 دقيقة والساعة مع عدم السماح لإشارة صغيرة بإلغاء اتجاه الفريمات الأعلى."
    }


def build_price_path(current: float, volatility: float, structure: dict,
                     liquidity: dict, zones_buy: List[dict],
                     zones_sell: List[dict]) -> dict:
    upside = [p["price"] for p in liquidity["buy_side_liquidity"] if p["price"] > current]
    downside = [p["price"] for p in liquidity["sell_side_liquidity"] if p["price"] < current]
    for pool in liquidity.get("higher_timeframe_pools", []):
        if pool.get("type") == "EQUAL_HIGHS" and pool["price"] > current:
            upside.append(pool["price"])
        elif pool.get("type") == "EQUAL_LOWS" and pool["price"] < current:
            downside.append(pool["price"])
    nearest_buy_zone = min(zones_buy, key=lambda z: z["distance"]) if zones_buy else None
    nearest_sell_zone = min(zones_sell, key=lambda z: z["distance"]) if zones_sell else None
    upside_target = min(upside) if upside else (
        structure["swing_high"]["price"] if structure.get("swing_high") and structure["swing_high"]["price"] > current else None
    )
    downside_target = max(downside) if downside else (
        structure["swing_low"]["price"] if structure.get("swing_low") and structure["swing_low"]["price"] < current else None
    )
    bullish_steps, bearish_steps = [], []
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



def build_scenario_ladder(current: float, liquidity: dict,
                          buy_zones: List[dict], sell_zones: List[dict],
                          per_tf: Dict[str, dict], price_path: dict) -> dict:
    """Descriptive, conditional market map for research and paper review only."""
    pools = list(liquidity.get("higher_timeframe_pools", []))
    pools += list(liquidity.get("equal_highs", []))
    pools += list(liquidity.get("equal_lows", []))
    unique = {}
    for pool in pools:
        try:
            price = round(float(pool["price"]), 5)
        except (KeyError, TypeError, ValueError):
            continue
        key = (price, pool.get("type"))
        unique.setdefault(key, {**pool, "price": price})

    above = sorted((x for x in unique.values() if x["price"] > current),
                   key=lambda x: x["price"])[:6]
    below = sorted((x for x in unique.values() if x["price"] < current),
                   key=lambda x: x["price"], reverse=True)[:6]

    def zone_view(zones: List[dict]) -> List[dict]:
        out = []
        for z in sorted(zones, key=lambda x: (x.get("distance", 1e99), -x.get("score", 0)))[:6]:
            out.append({
                "low": z.get("low"), "high": z.get("high"),
                "mid": z.get("mid"), "distance": z.get("distance"),
                "distance_atr": z.get("distance_atr"),
                "score": z.get("score"), "strength": z.get("strength_label"),
                "timeframes": z.get("timeframes", []),
                "tests": z.get("tests"), "age_candles": z.get("age_candles"),
                "status": z.get("status", "UNCONFIRMED")
            })
        return out

    structure = {}
    for tf in TIMEFRAMES:
        item = per_tf.get(tf)
        if not item:
            structure[tf] = {"available": False}
            continue
        st = item.get("structure", {})
        structure[tf] = {
            "available": True,
            "direction": st.get("direction"),
            "bos": st.get("bos"), "choch": st.get("choch"),
            "swing_high": (st.get("swing_high") or {}).get("price"),
            "swing_low": (st.get("swing_low") or {}).get("price"),
            "latest_close": item.get("latest_close"),
            "atr": item.get("atr")
        }

    return {
        "mode": "CONDITIONAL_RESEARCH_AND_PAPER_REVIEW_ONLY",
        "current_reference": round(current, 5),
        "upward_observed_levels": above,
        "downward_observed_levels": below,
        "demand_zones_to_monitor": zone_view(buy_zones),
        "supply_zones_to_monitor": zone_view(sell_zones),
        "timeframe_structure": structure,
        "bullish_condition": price_path.get("bullish_scenario", {}).get("trigger"),
        "bullish_invalidation": price_path.get("bullish_scenario", {}).get("invalidation"),
        "bearish_condition": price_path.get("bearish_scenario", {}).get("trigger"),
        "bearish_invalidation": price_path.get("bearish_scenario", {}).get("invalidation"),
        "interpretation": (
            "المستويات مناطق مراقبة مستنتجة من الشموع وليست أوامر دخول أو ضمانًا للحركة. "
            "يُشترط انتظار إغلاق وتأكيد هيكلي؛ وقد تتغير الخريطة مع وصول شموع جديدة."
        )
    }


def build_screening_summary(current: float, volatility: float,
                            buy_zones: List[dict], sell_zones: List[dict]) -> dict:
    all_zones = [("BUY", z) for z in buy_zones] + [("SELL", z) for z in sell_zones]
    within_2 = [x for x in all_zones if x[1].get("distance_atr", 999) <= 2.0]
    within_4 = [x for x in all_zones if x[1].get("distance_atr", 999) <= 4.0]
    score_pass = [x for x in all_zones if x[1].get("score", 0) >= MIN_PLAN_ZONE_SCORE]
    both = [x for x in all_zones if x[1].get("distance_atr", 999) <= MAX_PLAN_DISTANCE_ATR
            and x[1].get("score", 0) >= MIN_PLAN_ZONE_SCORE]
    return {
        "buy_zone_count": len(buy_zones), "sell_zone_count": len(sell_zones),
        "zones_within_2_atr": len(within_2), "zones_within_4_atr": len(within_4),
        "zones_meeting_score_threshold": len(score_pass),
        "zones_meeting_current_plan_distance_and_score": len(both),
        "current_distance_filter_atr": MAX_PLAN_DISTANCE_ATR,
        "current_minimum_score": MIN_PLAN_ZONE_SCORE,
        "nearest_buy_zone_distance_atr": min((z.get("distance_atr", 1e99) for z in buy_zones), default=None),
        "nearest_sell_zone_distance_atr": min((z.get("distance_atr", 1e99) for z in sell_zones), default=None),
        "explanation": (
            "عدد المناطق المعروضة لا يعني وجود إعداد مؤكد. تُعرض المناطق القريبة والبعيدة للسياق، "
            "ويُفصل بين قرب المنطقة وجودة المنطقة والتأكيد الهيكلي."
        )
    }

def _plan_rank(plan: dict) -> tuple:
    confirmation = plan.get("entry_basis", {}).get("confirmation", {})
    evidence_score = confirmation.get("confirmation_score", 0)
    aligned = bool(plan.get("entry_basis", {}).get("directional_alignment"))
    tf_count = len(plan.get("zone_timeframes", []))
    # Confirmed evidence first, then zone quality, confluence and proximity.
    return (
        bool(confirmation.get("confirmed")),
        evidence_score,
        aligned,
        plan.get("zone_score", 0),
        tf_count,
        -plan.get("distance_atr", 999)
    )


def analyze_market(frames: Dict[str, List[dict]], quality: dict,
                   live_quote: Optional[dict] = None) -> dict:
    primary = frames["15min"]
    quote_valid = bool(live_quote and live_quote.get("valid"))
    current = live_quote["price"] if quote_valid else primary[-1]["close"]
    per_tf, votes = {}, {"BULLISH": 0, "BEARISH": 0, "RANGE": 0}
    weights = {"1month": 5, "1week": 4, "1day": 3, "4h": 2, "1h": 1, "15min": 0}
    weighted_votes = {"BULLISH": 0, "BEARISH": 0, "RANGE": 0}
    all_zones = {"BUY": [], "SELL": []}
    for tf in TIMEFRAMES:
        candles = frames.get(tf, [])
        if len(candles) < MIN_COMPLETE_CANDLES:
            continue
        structure = structure_summary(candles)
        volatility = atr(candles)
        weight = weights[tf]
        per_tf[tf] = {
            "direction": structure["direction"],
            "bos": structure["bos"], "choch": structure["choch"],
            "atr": round(volatility, 5),
            "latest_close": round(candles[-1]["close"], 5),
            "latest_datetime": candles[-1]["datetime"],
            "structure": structure,
            "direction_weight": weight
        }
        votes[structure["direction"]] += 1
        weighted_votes[structure["direction"]] += weight
        for side in ("BUY", "SELL"):
            all_zones[side].extend(build_zones(candles, current, side, tf))

    current_vol = atr(primary)
    available_weights = {tf: weights[tf] for tf in per_tf}
    total_weight = sum(available_weights.values())
    winning_bias = max(weighted_votes, key=weighted_votes.get)
    weighted_alignment = round(weighted_votes[winning_bias] / total_weight, 3) if total_weight else 0.0
    bias = winning_bias if weighted_alignment >= 0.52 else "NEUTRAL"
    liquidity = liquidity_map(primary, current, current_vol)

    htf_pools = []
    for tf in ("1h", "4h", "1day", "1week", "1month"):
        if tf not in per_tf:
            continue
        tf_liq = liquidity_map(frames[tf], current, atr(frames[tf]))
        for key in ("buy_side_liquidity", "sell_side_liquidity"):
            for pool in tf_liq[key]:
                htf_pools.append({**pool, "timeframe": tf})
    liquidity["higher_timeframe_pools"] = htf_pools[:60]

    buy_zones = combine_zone_confluence(all_zones["BUY"])
    sell_zones = combine_zone_confluence(all_zones["SELL"])
    for side_zones in (buy_zones, sell_zones):
        for zone in side_zones:
            zone["mid"] = round((zone["low"] + zone["high"]) / 2, 5)
            zone["distance"] = round(abs(current - zone["mid"]), 5)
            zone["distance_atr"] = round(zone["distance"] / current_vol, 2) if current_vol else None
            zone["width_atr"] = round((zone["high"] - zone["low"]) / current_vol, 2) if current_vol else None
            tf_count = len(set(zone.get("timeframes", [])))
            zone["score"] = round(min(99.0, zone["score"] + min(8, max(0, tf_count-1)*2)), 1)
            zone["strength_label"] = "STRONG" if zone["score"] >= 78 else "MODERATE" if zone["score"] >= 65 else "WEAK"

    buy_zones = [z for z in buy_zones if z.get("width_atr") is not None and z["width_atr"] <= 2.2]
    sell_zones = [z for z in sell_zones if z.get("width_atr") is not None and z["width_atr"] <= 2.2]
    buy_zones.sort(key=lambda z: (-z["score"], z["distance"]))
    sell_zones.sort(key=lambda z: (-z["score"], z["distance"]))
    buy_groups = zone_collections(sorted(buy_zones, key=lambda z: (z["distance"], -z["score"])), current_vol)
    sell_groups = zone_collections(sorted(sell_zones, key=lambda z: (z["distance"], -z["score"])), current_vol)

    # Generate conditional analytical candidates, then retain only the single
    # strongest candidate. A missing/weak setup remains NO_TRADE / WAIT.
    buy_candidates = sorted(
        [z for z in buy_zones if z.get("distance_atr") is not None
         and z["distance_atr"] <= MAX_PLAN_DISTANCE_ATR],
        key=lambda z: (-z["score"], z["distance"])
    )
    sell_candidates = sorted(
        [z for z in sell_zones if z.get("distance_atr") is not None
         and z["distance_atr"] <= MAX_PLAN_DISTANCE_ATR],
        key=lambda z: (-z["score"], z["distance"])
    )
    buy_plans = [
        make_trade_plan(z, "BUY", current, current_vol, liquidity,
                        quality["analysis_blocked"], primary, sell_zones, bias)
        for z in buy_candidates[:8]
    ]
    sell_plans = [
        make_trade_plan(z, "SELL", current, current_vol, liquidity,
                        quality["analysis_blocked"], primary, buy_zones, bias)
        for z in sell_candidates[:8]
    ]
    candidate_plans = [p for p in buy_plans + sell_plans if p is not None]
    candidate_plans.sort(key=_plan_rank, reverse=True)
    best_plan = candidate_plans[0] if candidate_plans else None
    buy_plan = best_plan if best_plan and best_plan["side"] == "BUY" else None
    sell_plan = best_plan if best_plan and best_plan["side"] == "SELL" else None

    path = build_price_path(current, current_vol, per_tf["15min"]["structure"],
                            liquidity, buy_zones, sell_zones)
    if quality["analysis_blocked"]:
        decision = "DATA_INVALID"
    elif best_plan and best_plan["entry_basis"]["confirmation"]["confirmed"]:
        decision = "CONDITIONAL_SETUP_CONFIRMED"
    elif best_plan:
        decision = "WAIT_FOR_CONFIRMATION" if best_plan["setup_status"] != "WAITING_FOR_ZONE" else "WAIT_FOR_SETUP"
    elif bias == "NEUTRAL":
        decision = "WAIT_FOR_CONFIRMATION"
    else:
        decision = "NO_TRADE"

    return {
        "engine": ENGINE_NAME, "version": VERSION, "symbol": SYMBOL,
        "decision": decision, "current_price": round(current, 5),
        "price_source": "LIVE_QUOTE" if quote_valid else "LAST_15MIN_CANDLE_CLOSE_NOT_LIVE",
        "price_basis_note": ("السعر المباشر المستخدم للتحليل." if quote_valid else
                             "تم استخدام آخر إغلاق مكتمل لفريم 15 دقيقة كسعر مرجعي؛ ليس سعر تنفيذ مباشر."),
        "live_quote_required": False,
        "live_quote_max_age_seconds": LIVE_QUOTE_MAX_AGE_SECONDS,
        "live_quote": live_quote or {"valid": False, "price": None, "timestamp": None,
                                      "age_seconds": None, "reason": "LIVE_QUOTE_MISSING"},
        "market": {
            "directional_bias": bias,
            "bias_method": "HTF hierarchy: monthly > weekly > daily > 4h > 1h; 15m is execution/confirmation only",
            "timeframe_votes": votes,
            "weighted_timeframe_votes": weighted_votes,
            "timeframe_weights": weights,
            "timeframe_alignment_score": weighted_alignment,
            "confidence_note": "Alignment score measures weighted rule agreement; it is not a win probability."
        },
        "timeframe_analysis": per_tf,
        "market_structure": {tf: item["structure"] for tf, item in per_tf.items()},
        "liquidity": liquidity,
        "reversal_zones": {
            "buy": buy_groups, "sell": sell_groups,
            "all_buy_zones": sorted(buy_zones, key=lambda z: (-z["score"], z["distance"]))[:30],
            "all_sell_zones": sorted(sell_zones, key=lambda z: (-z["score"], z["distance"]))[:30]
        },
        "scenario_ladder": build_scenario_ladder(current, liquidity, buy_zones, sell_zones, per_tf, path),
        "screening_summary": build_screening_summary(current, current_vol, buy_zones, sell_zones),
        "entry_analysis": {
            "buy_plan": buy_plan, "sell_plan": sell_plan,
            "best_plan": best_plan,
            "candidate_count_before_selection": len(candidate_plans),
            "filters": {
                "maximum_distance_atr": MAX_PLAN_DISTANCE_ATR,
                "minimum_zone_score": MIN_PLAN_ZONE_SCORE,
                "one_best_candidate_only": True
            },
            "note": "Conditional analytical candidate only. trade_allowed_now is always false; verify with historical out-of-sample testing."
        },
        "price_path": path,
        "data_validation": quality,
        "risk_notice": "Research and paper-testing only. No automatic execution; no setup is guaranteed and no score is a win probability."
    }


# ---------------------------------------------------------------------------
# Human-readable report for Telegram / AI formatting node
# ---------------------------------------------------------------------------

def fmt_price(value: Any) -> str:
    return "غير متاح" if value is None else f"{float(value):,.2f}"


def format_plan(name: str, plan: Optional[dict]) -> List[str]:
    if not plan:
        return [f"{name}: لا توجد فرصة تحليلية قريبة مستوفية للشروط حاليًا — NO TRADE."]
    zone = plan["entry_zone"]
    lines = [
        f"{name} ({'انعكاس' if plan.get('setup_type') == 'REVERSAL' else 'استمرار' if plan.get('setup_type') == 'CONTINUATION' else 'غير مكتمل'}):",
        f"- المنقطع: من {fmt_price(zone['low'])} إلى {fmt_price(zone['high'])}",
        f"- السعر المرجعي التحليلي: {fmt_price(plan['entry_reference'])}",
        f"- البعد عن السعر المرجعي: {plan['distance_atr']} ATR",
        f"- وقف الإبطال النظري للاختبار: {fmt_price(plan['stop_loss'])}",
        f"- مستويات القياس: " + (" | ".join(f"{x['name']} {fmt_price(x['price'])}" for x in plan["targets"])
                                  if plan["targets"] else "لا توجد مستويات كافية"),
        f"- نسبة المسافة إلى مستوى القياس الأول: {plan['first_target_rr']}",
        f"- الحالة: {plan['setup_status']}",
        f"- تقييم المنطقة: {plan['zone_score']}/100 | الفريمات: {','.join(plan['zone_timeframes'])}",
        f"- تقييم الأدلة الفنية: {plan.get('entry_basis', {}).get('confirmation', {}).get('confirmation_score', 0)} (ليس احتمال نجاح)",
        f"- التأكيد المطلوب: {plan['confirmation_required']}",
        f"- الإبطال: {plan['invalidation']}",
        "- الوضع: تحليل/اختبار افتراضي فقط — لا يوجد إذن تنفيذ."
    ]
    return lines


def build_telegram_report(result: dict) -> str:
    market = result["market"]
    quality = result["data_validation"]
    path = result["price_path"]
    decision_labels = {
        "DATA_INVALID": "توقف — بيانات غير صالحة",
        "WAIT_FOR_CONFIRMATION": "انتظار تأكيد",
        "CONDITIONAL_SETUP_CONFIRMED": "إعداد تحليلي مكتمل — اختبار افتراضي فقط",
        "WAIT_FOR_SETUP": "انتظار تكوّن الإعداد",
        "NO_TRADE": "لا توجد فرصة مستوفية — NO TRADE"
    }
    direction_labels = {"BULLISH": "صاعد", "BEARISH": "هابط", "RANGE": "عرضي", "NEUTRAL": "محايد"}

    def votes_text(votes: dict) -> str:
        return " | ".join(
            f"{direction_labels.get(k, k)}: {v}"
            for k, v in (("BULLISH", votes.get("BULLISH", 0)),
                         ("BEARISH", votes.get("BEARISH", 0)),
                         ("RANGE", votes.get("RANGE", 0)))
        )

    def path_text(items: list) -> str:
        if not items:
            return "لا توجد مستويات مستهدفة مرصودة حاليًا."
        rendered = []
        for item in items:
            if "level" in item:
                reason = item.get("reason", "مستوى سيولة مرصود")
                reason_ar = "سيولة بيع مرصودة" if "sell_side" in reason else "مستوى مرصود"
                rendered.append(f"{fmt_price(item['level'])} ({reason_ar})")
            elif "zone" in item and len(item["zone"]) == 2:
                rendered.append(f"منطقة {fmt_price(item['zone'][0])}–{fmt_price(item['zone'][1])}")
        return " ← ".join(rendered) if rendered else "لا توجد مستويات مستهدفة مرصودة حاليًا."

    decision = decision_labels.get(result.get("decision"), "قيد الانتظار")
    votes = market.get("timeframe_votes", {})
    weighted = market.get("weighted_timeframe_votes", {})
    lines = [
        f"📊 {ENGINE_NAME} | {SYMBOL} | v{VERSION}",
        f"السعر المرجعي: {fmt_price(result['current_price'])} ({'سعر مباشر' if result.get('price_source') == 'LIVE_QUOTE' else 'آخر إغلاق 15 دقيقة — ليس سعرًا مباشرًا'})",
        f"القرار: {decision}",
        f"الاتجاه العام: {direction_labels.get(market['directional_bias'], market['directional_bias'])} | توافق الفريمات: {market['timeframe_alignment_score']}",
        f"تصويت الفريمات: {votes_text(votes)}",
        f"التصويت الموزون HTF: {votes_text(weighted)}",
        f"سلامة البيانات: {quality['status']}"
    ]
    if result.get("price_source") != "LIVE_QUOTE":
        lines.append("ℹ️ التحليل مبني على آخر إغلاق مكتمل لفريم 15 دقيقة؛ قد يكون السعر المرجعي تغيّر.")
    health = quality.get("engine_health", {})
    lines.append(f"سلامة المحرك: {health.get('status', 'غير معروف')} | الفريمات المستخدمة: {health.get('usable_timeframes', 0)}/{health.get('expected_timeframes', len(TIMEFRAMES))}")
    if quality.get("excluded_timeframes"):
        lines.append("⚠️ فريمات مستبعدة: " + ", ".join(quality["excluded_timeframes"]))
    if quality.get("blockers"):
        lines.append("⛔ سبب تعذر التحليل: " + "؛ ".join(quality["blockers"][:8]))
    if quality.get("warnings"):
        lines.append("⚠️ تحذيرات البيانات: " + "؛ ".join(quality["warnings"][:8]))
    lines.extend(["", "🧭 قراءة الفريمات الستة"])
    for tf in TIMEFRAMES:
        item = result.get("timeframe_analysis", {}).get(tf, {})
        st = item.get("structure", {})
        if not item:
            lines.append(f"- {tf}: غير متاح")
            continue
        lines.append(
            f"- {tf}: الاتجاه {direction_labels.get(st.get('direction'), st.get('direction', 'غير محدد'))}"
            f" | BOS: {st.get('bos') or 'لا يوجد'} | CHoCH: {st.get('choch') or 'لا يوجد'}"
            f" | قمة: {fmt_price((st.get('swing_high') or {}).get('price'))}"
            f" | قاع: {fmt_price((st.get('swing_low') or {}).get('price'))}"
        )

    ladder = result.get("scenario_ladder", {})
    lines.extend(["", "🪜 خريطة الحركة الشرطية — مستويات مراقبة وليست أوامر"])
    for title, key in (("مستويات أعلى السعر", "upward_observed_levels"),
                       ("مستويات أسفل السعر", "downward_observed_levels")):
        lines.append(title + ":")
        levels = ladder.get(key, [])
        if not levels:
            lines.append("- لا توجد تجمعات مرصودة في هذا الاتجاه ضمن البيانات الحالية.")
        for pool in levels[:6]:
            lines.append(
                f"- {fmt_price(pool.get('price'))} | {pool.get('type', 'LIQUIDITY')}"
                f" | الفريم: {pool.get('timeframe', '15min/مجمع')} | اللمسات: {pool.get('touches', 0)}"
            )

    for title, key in (("مناطق طلب للمراقبة", "demand_zones_to_monitor"),
                       ("مناطق عرض للمراقبة", "supply_zones_to_monitor")):
        lines.append(title + ":")
        zones = ladder.get(key, [])
        if not zones:
            lines.append("- لا توجد مناطق صالحة مستخرجة.")
        for z in zones[:6]:
            lines.append(
                f"- {fmt_price(z.get('low'))}–{fmt_price(z.get('high'))}"
                f" | التقييم {z.get('score')}/100 | البعد {z.get('distance_atr')} ATR"
                f" | الفريمات: {','.join(z.get('timeframes', [])) or 'غير محدد'}"
                f" | الاختبارات: {z.get('tests')} | {z.get('strength', 'غير محدد')}"
            )

    screening = result.get("screening_summary", {})
    lines.extend([
        "", "🔎 لماذا لم يظهر مرشح؟",
        f"- مناطق شراء: {screening.get('buy_zone_count', 0)} | مناطق بيع: {screening.get('sell_zone_count', 0)}",
        f"- مناطق ضمن 2 ATR: {screening.get('zones_within_2_atr', 0)} | ضمن 4 ATR: {screening.get('zones_within_4_atr', 0)}",
        f"- مناطق حققت حد التقييم: {screening.get('zones_meeting_score_threshold', 0)}",
        f"- مناطق حققت شرطي المسافة والتقييم الحاليين: {screening.get('zones_meeting_current_plan_distance_and_score', 0)}",
        f"- أقرب شراء: {screening.get('nearest_buy_zone_distance_atr')} ATR | أقرب بيع: {screening.get('nearest_sell_zone_distance_atr')} ATR"
    ])

    lines.extend(["", "📍 أقرب مناطق الانعكاس"])
    for side, label in (("buy", "شراء"), ("sell", "بيع")):
        group = result["reversal_zones"][side]
        lines.append(f"{label}:")
        for zone in group["nearest"][:3]:
            lines.append(
                f"- {fmt_price(zone['low'])}–{fmt_price(zone['high'])} | القوة {zone['score']}/100 "
                f"| البُعد {zone['distance_atr']} ATR | {zone['strength_label']} | {','.join(zone['timeframes'])}"
            )
        if not group["nearest"]:
            lines.append("- لا توجد منطقة قريبة مستوفية للشروط.")
        lines.append(f"مناطق قوية بعيدة (للسياق فقط): {len(group['strong_distant'])}")

    lines.extend(["", "🎯 أفضل إعداد تحليلي واحد"])
    best = result.get("entry_analysis", {}).get("best_plan")
    lines.extend(format_plan("أفضل مرشح", best))
    bull = path["bullish_scenario"]
    bear = path["bearish_scenario"]
    lines.extend([
        "", "🧭 السيناريو الصاعد",
        "- شرط التفعيل: تحوّل هيكلي صاعد على 15 دقيقة أو الساعة، ثم إغلاق فوق أقرب قمة محلية مؤكدة.",
        f"- المسار المحتمل: {path_text(bull['possible_path'])}",
        "- الإبطال: إغلاق حاسم أسفل آخر قاع مؤكد على 15 دقيقة.",
        "", "🧭 السيناريو الهابط",
        "- شرط التفعيل: تحوّل هيكلي هابط على 15 دقيقة أو الساعة، ثم إغلاق أسفل أقرب قاع محلي مؤكد.",
        f"- المسار المحتمل: {path_text(bear['possible_path'])}",
        "- الإبطال: إغلاق حاسم أعلى آخر قمة مؤكدة على 15 دقيقة.",
        "", "💧 السيولة",
        f"- أقرب سيولة أعلى: {fmt_price(result['liquidity']['nearest_above']['price']) if result['liquidity']['nearest_above'] else 'غير مرصودة'}",
        f"- أقرب سيولة أسفل: {fmt_price(result['liquidity']['nearest_below']['price']) if result['liquidity']['nearest_below'] else 'غير مرصودة'}",
        f"- تجمعات HTF المرصودة: {len(result['liquidity'].get('higher_timeframe_pools', []))}",
        f"- آخر سحب سيولة: {('لا يوجد سحب مؤكد حديث' if not result['liquidity'].get('recent_sweep') else 'تم رصد سحب سيولة حديث')}",
        "", "تفسير السيناريو الصاعد:",
        f"- الشرط: {bull.get('trigger')}",
        f"- الإبطال: {bull.get('invalidation')}",
        "تفسير السيناريو الهابط:",
        f"- الشرط: {bear.get('trigger')}",
        f"- الإبطال: {bear.get('invalidation')}",
        "", "تنبيه: النتائج خريطة تحليلية واختبارية فقط؛ ليست توصية تنفيذ أو ضمانًا للنتيجة."
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
        "live_quote_required": LIVE_QUOTE_REQUIRED,
        "live_quote_max_age_seconds": LIVE_QUOTE_MAX_AGE_SECONDS,
        "max_plan_distance_atr": MAX_PLAN_DISTANCE_ATR,
        "minimum_plan_zone_score": MIN_PLAN_ZONE_SCORE,
        "timestamp_policy": "epoch or timezone-offset timestamps accepted; naive timestamps use configured per-timeframe IANA timezone",
        "engine_health_monitor": "enabled",
        "partial_timeframe_policy": "15min reference required; other timeframes may be excluded with warnings",
        "plan_policy": "conditional_analysis_only; no_auto_execution; six_timeframe_scenario_ladder"
    })


@app.post("/analyze")
def analyze():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "invalid_or_missing_json_body"}), 400
        live_quote = extract_live_quote(payload)
        frames, errors, notes, diagnostics = normalize_payload(payload)
        quality = data_quality_report(frames, errors, notes, diagnostics)
        if quality["analysis_blocked"]:
            response = {
                "engine": ENGINE_NAME, "version": VERSION,
                "decision": "DATA_INVALID",
                "data_validation": quality,
                "message": "Analysis stopped because the candle data failed validation. Correct source timestamps/data and resend; the engine will not shift timestamps or invent candles."
            }
            return jsonify(response), 422
        result = analyze_market(frames, quality, live_quote)
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
