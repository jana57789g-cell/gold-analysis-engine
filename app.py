import os
import math
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify

VERSION = "5.2.0"
ENGINE_NAME = "Gold Analysis Engine"
SYMBOL = "XAU/USD"
TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]
TF_SECONDS = {
    "15min": 900, "1h": 3600, "4h": 14400,
    "1day": 86400, "1week": 604800, "1month": 2678400
}
MAX_CANDLES = 300
MIN_COMPLETE_CANDLES = 30
FUTURE_TOLERANCE_SECONDS = 60

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


def find_source_timezone(payload: Any) -> str:
    """Read an explicit IANA timezone from payload metadata or environment.

    `exchange_local` is a timestamp convention label, NOT a timezone name,
    so it is deliberately not treated as a timezone.
    """
    env_zone = os.getenv("SOURCE_TIMEZONE", "").strip()
    if env_zone:
        return env_zone
    keys = {"source_timezone", "exchange_timezone", "timezone", "tz"}
    found = []
    def walk(obj: Any, depth: int = 0) -> None:
        if depth > 6:
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
    walk(payload)
    unique = list(dict.fromkeys(found))
    if len(unique) > 1:
        raise ValueError("conflicting_source_timezones:" + ",".join(unique))
    return unique[0] if unique else None


def normalize_payload(payload: Any) -> Tuple[Dict[str, List[dict]], List[str], List[str]]:
    errors: List[str] = []
    notes: List[str] = []
    try:
        source_timezone = find_source_timezone(payload)
        if source_timezone:
            ZoneInfo(source_timezone)
            notes.append(f"naive_timestamps_interpreted_in_timezone:{source_timezone}")
        else:
            notes.append("timezone_not_assumed:naive_timestamps_will_be_rejected")
    except (ValueError, ZoneInfoNotFoundError) as exc:
        return {}, [str(exc)], notes
    raw_frames = unwrap_payload(payload)
    if not isinstance(raw_frames, dict):
        return {}, ["timeframes_must_be_object"], notes

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

        parsed: List[dict] = []
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
                # Future open timestamps are data errors, not merely incomplete
                # candles. Keep an explicit blocker even if that row is excluded.
                now_ts = datetime.now(timezone.utc).timestamp()
                if dt.timestamp() > now_ts + FUTURE_TOLERANCE_SECONDS:
                    errors.append(f"{tf}[{idx}]:future_timestamp_by_{int(dt.timestamp() - now_ts)}_seconds")
                # A candle is not complete until its interval has elapsed.
                # This only excludes it; it never changes OHLC or its timestamp.
                candle_end = dt.timestamp() + TF_SECONDS[tf]
                time_incomplete = candle_end > now_ts + FUTURE_TOLERANCE_SECONDS
                parsed.append({
                    "datetime": dt.isoformat(), "_ts": dt.timestamp(),
                    "open": o, "high": h, "low": lo, "close": c,
                    "volume": vol, "is_incomplete": bool(incomplete or time_incomplete)
                })
            except Exception as exc:
                errors.append(f"{tf}[{idx}]:{str(exc)}")

        parsed.sort(key=lambda x: x["_ts"])
        seen = set()
        unique = []
        for candle in parsed:
            ts = candle["_ts"]
            if ts in seen:
                errors.append(f"{tf}:duplicate_timestamp:{candle['datetime']}")
                continue
            seen.add(ts)
            unique.append(candle)

        complete = [x for x in unique if not x["is_incomplete"]]
        if len(complete) < MIN_COMPLETE_CANDLES:
            errors.append(f"{tf}:insufficient_complete_candles:{len(complete)}<{MIN_COMPLETE_CANDLES}")
        frames[tf] = complete

    return frames, errors, notes


# ---------------------------------------------------------------------------
# Data quality: future timestamps are a hard block, not a soft warning.
# Datetimes are assumed to be candle OPEN times, unless the provider contract
# says otherwise. The engine never shifts or fabricates timestamps.
# ---------------------------------------------------------------------------

def data_quality_report(frames: Dict[str, List[dict]], errors: List[str],
                        notes: List[str], now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    now_ts = now.timestamp()
    report, warnings, blockers = {}, [], []
    for tf in TIMEFRAMES:
        candles = frames.get(tf, [])
        latest = candles[-1] if candles else None
        if latest:
            future = latest["_ts"] - now_ts
            candle_end = latest["_ts"] + TF_SECONDS[tf]
            age_after_close = max(0.0, now_ts - candle_end)
            if future > FUTURE_TOLERANCE_SECONDS:
                blockers.append(f"{tf}:future_timestamp_by_{int(future)}_seconds")
            # A timeframe can legitimately have an older last completed candle.
            stale_limit = max(TF_SECONDS[tf] * 2.5, 86400 if tf == "1month" else 0)
            if age_after_close > stale_limit:
                warnings.append(f"{tf}:latest_completed_candle_is_stale")
            report[tf] = {
                "complete_count": len(candles),
                "latest_datetime": latest["datetime"],
                "latest_close": round(latest["close"], 5),
                "timestamp_assumption": "candle_open_time; explicit offset/epoch or configured IANA timezone required",
                "future_seconds": round(max(0.0, future), 1),
                "age_after_candle_close_seconds": round(age_after_close, 1)
            }
        else:
            report[tf] = {"complete_count": 0, "latest_datetime": None,
                          "latest_close": None, "future_seconds": None,
                          "age_after_candle_close_seconds": None}
            blockers.append(f"{tf}:no_usable_candles")

    if errors:
        blockers.extend(errors)
    valid = not blockers and all(len(frames.get(tf, [])) >= MIN_COMPLETE_CANDLES for tf in TIMEFRAMES)
    return {
        "status": "VALID" if valid else "INVALID",
        "analysis_blocked": not valid,
        "max_candles_per_timeframe": MAX_CANDLES,
        "source_timezone": next((n.split(":", 1)[1] for n in notes if n.startswith("naive_timestamps_interpreted_in_timezone:")), None),
        "timeframes": report,
        "errors": errors,
        "warnings": warnings,
        "blockers": blockers,
        "normalization_notes": notes,
        "self_healing": {
            "applied": ["explicit-offset/epoch UTC conversion", "chronological sorting", "safe numeric parsing"],
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
        test_score = {0: 1.0, 
