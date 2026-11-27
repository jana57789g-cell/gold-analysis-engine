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
    lh = len(rh) >= 2 and rh[-1]["price"] < rh[-2]["pri
