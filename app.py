import math
import os
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request

app = Flask(__name__)

VERSION = "3.1.0"
ENGINE_NAME = "Gold Analysis Engine"
SYMBOL = "XAU/USD"

TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week", "1month"]
INTERVAL_MINUTES = {
    "15min": 15,
    "1h": 60,
    "4h": 240,
    "1day": 1440,
    "1week": 10080,
    "1month": None,
}
MIN_COMPLETE_CANDLES = 10
MAX_CANDLES = 300
EPSILON = 1e-9


# ----------------------------- basic utilities -----------------------------

def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def dt_key(dt: datetime) -> float:
    # Naive timestamps are compared as exchange-local wall-clock values.
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).timestamp()
    return dt.timestamp()


def round_price(value: Optional[float]) -> Optional[float]:
    return round(float(value), 5) if value is not None and math.isfinite(value) else None


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ----------------------------- input validation ----------------------------

def normalize_payload(payload: Any) -> Tuple[Optional[dict], List[str]]:
    errors = []
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            return None, ["يجب أن تكون القائمة محتوية على عنصر JSON واحد فقط."]
        payload = payload[0]
    if not isinstance(payload, dict):
        return None, ["جسم الطلب يجب أن يكون JSON object أو قائمة بعنصر واحد."]
    report = payload.get("report", payload)
    if isinstance(report, list):
        if len(report) != 1 or not isinstance(report[0], dict):
            return None, ["report داخل القائمة يجب أن يحتوي على عنصر واحد."]
        report = report[0]
    if not isinstance(report, dict):
        return None, ["حقل report يجب أن يكون object."]
    tfs = report.get("timeframes")
    if not isinstance(tfs, dict):
        return None, ["حقل timeframes مفقود أو ليس object."]
    missing = [tf for tf in TIMEFRAMES if tf not in tfs]
    extra = [tf for tf in tfs if tf not in TIMEFRAMES]
    if missing:
        errors.append("الفريمات المفقودة: " + ", ".join(missing))
    if extra:
        errors.append("فريمات غير مدعومة: " + ", ".join(map(str, extra)))
    return report, errors


def validate_timeframe(tf: str, section: Any) -> dict:
    result = {
        "timeframe": tf,
        "status": "INVALID",
        "errors": [],
        "warnings": [],
        "input_count": 0,
        "valid_count": 0,
        "complete_count": 0,
        "incomplete_count": 0,
        "duplicate_count": 0,
        "invalid_count": 0,
        "out_of_order": False,
        "timestamp_basis": None,
        "timestamp_mode": None,
        "latest_datetime": None,
        "latest_close": None,
        "candles": [],
    }
    if not isinstance(section, dict):
        result["errors"].append("قسم الفريم ليس object.")
        return result

    basis = section.get("timestamp_basis")
    result["timestamp_basis"] = basis if isinstance(basis, str) else None
    raw_candles = section.get("candles")
    if not isinstance(raw_candles, list):
        result["errors"].append("حقل candles مفقود أو ليس list.")
        return result
    result["input_count"] = len(raw_candles)
    if len(raw_candles) > MAX_CANDLES:
        result["errors"].append(f"عدد الشموع أكبر من الحد المسموح ({MAX_CANDLES}).")
        return result

    parsed = []
    seen = set()
    modes = set()
    for index, candle in enumerate(raw_candles):
        if not isinstance(candle, dict):
            result["invalid_count"] += 1
            result["errors"].append(f"العنصر رقم {index} ليس شمعة object.")
            continue

        dt = parse_datetime(candle.get("datetime"))
        if dt is None:
            result["invalid_count"] += 1
            result["errors"].append(f"تاريخ غير صالح في الشمعة رقم {index}.")
            continue
        modes.add("aware" if dt.tzinfo is not None else "naive")
        key = dt.isoformat()
        if key in seen:
            result["duplicate_count"] += 1
            result["errors"].append(f"تاريخ مكرر: {candle.get('datetime')}")
            continue
        seen.add(key)

        names = ("open", "high", "low", "close")
        if not all(finite_number(candle.get(name)) for name in names):
            result["invalid_count"] += 1
            result["errors"].append(f"قيم OHLC غير رقمية أو غير محدودة في الشمعة رقم {index}.")
            continue
        o, h, l, c = (float(candle[name]) for name in names)
        if min(o, h, l, c) <= 0:
            result["invalid_count"] += 1
            result["errors"].append(f"سعر صفر أو سالب في الشمعة رقم {index}.")
            continue
        if h + EPSILON < max(o, c, l) or l - EPSILON > min(o, c, h):
            result["invalid_count"] += 1
            result["errors"].append(f"تركيب OHLC غير منطقي في الشمعة رقم {index}.")
            continue

        if "is_incomplete" not in candle:
            result["warnings"].append(f"حقل is_incomplete مفقود في الشمعة رقم {index}؛ اعتُبرت غير مكتملة احتياطيًا.")
            incomplete = True
        else:
            incomplete = candle.get("is_incomplete")
            if not isinstance(incomplete, bool):
                result["warnings"].append(f"is_incomplete ليست boolean في الشمعة رقم {index}؛ اعتُبرت غير مكتملة.")
                incomplete = True

        parsed.append({
            "datetime": dt,
            "datetime_text": candle.get("datetime"),
            "open": o, "high": h, "low": l, "close": c,
            "is_incomplete": incomplete,
        })

    if len(modes) > 1:
        result["errors"].append("الفريم يحتوي على تواريخ timezone-aware وأخرى بدون timezone.")
    result["timestamp_mode"] = next(iter(modes)) if len(modes) == 1 else ("mixed" if modes else None)

    if not parsed:
        result["errors"].append("لا توجد شموع صالحة للتحليل.")
        return result

    original_keys = [dt_key(x["datetime"]) for x in parsed]
    ascending = all(original_keys[i] > original_keys[i - 1] for i in range(1, len(original_keys)))
    descending = all(original_keys[i] < original_keys[i - 1] for i in range(1, len(original_keys)))
    result["out_of_order"] = not ascending
    result["input_order"] = "ASCENDING" if ascending else ("DESCENDING" if descending else "MIXED")
    if descending:
        result["warnings"].append("الشموع واردة من الأحدث إلى الأقدم؛ تم عكسها داخليًا بالترتيب الزمني الصحيح.")
    elif not ascending:
        result["errors"].append("ترتيب الشموع مختلط وغير متتابع؛ أرسلها بترتيب زمني ثابت لتجنب تحليل بيانات مشوشة.")

    parsed.sort(key=lambda x: dt_key(x["datetime"]))
    complete = [x for x in parsed if not x["is_incomplete"]]
    incomplete = [x for x in parsed if x["is_incomplete"]]
    result["valid_count"] = len(parsed)
    result["complete_count"] = len(complete)
    result["incomplete_count"] = len(incomplete)

    if len(complete) < MIN_COMPLETE_CANDLES:
        result["errors"].append(
            f"عدد الشموع المكتملة {len(complete)} أقل من الحد الأدنى {MIN_COMPLETE_CANDLES}."
        )
    if result["duplicate_count"]:
        result["errors"].append("وجود تواريخ مكررة يمنع اعتماد الفريم.")
    if len(modes) > 1:
        result["errors"].append("خلط التوقيت المحلي والتوقيت ذي المنطقة الزمنية يمنع اعتماد الفريم.")

    # Analyze completed candles only. Incomplete current candle is reported, never used.
    complete.sort(key=lambda x: dt_key(x["datetime"]))
    if complete:
        result["latest_datetime"] = complete[-1]["datetime"].isoformat()
        result["latest_close"] = round_price(complete[-1]["close"])
        if result["timestamp_mode"] == "aware" or basis == "UTC":
            latest_epoch = dt_key(complete[-1]["datetime"])
            now_epoch = datetime.now(timezone.utc).timestamp()
            if latest_epoch > now_epoch + 2 * 60 * 60:
                result["errors"].append(
                    "آخر شمعة مكتملة مؤرخة في المستقبل بأكثر من ساعتين؛ راجع ساعة المصدر أو المنطقة الزمنية."
                )

    if result["timestamp_mode"] == "naive":
        if basis not in ("UTC", "exchange_local"):
            result["warnings"].append(
                "التواريخ بلا timezone؛ timestamp_basis يجب أن يكون UTC أو exchange_local."
            )
            result["timestamp_reliable"] = False
        else:
            result["timestamp_reliable"] = basis == "UTC"
            if basis == "exchange_local":
                result["warnings"].append(
                    "التوقيت exchange_local متسق داخليًا لكن المنطقة الزمنية الفعلية غير مثبتة."
                )
    elif result["timestamp_mode"] == "aware":
        result["timestamp_reliable"] = True
        offsets = {x["datetime"].utcoffset() for x in parsed}
        if len(offsets) > 1:
            result["warnings"].append("تغير UTC offset داخل الفريم؛ راجع إعدادات مصدر البيانات.")
    else:
        result["timestamp_reliable"] = False

    # Detect large gaps without treating weekends/holidays as invalid data.
    interval = INTERVAL_MINUTES[tf]
    if interval and len(complete) >= 2:
        gaps = []
        for i in range(1, len(complete)):
            delta = (dt_key(complete[i]["datetime"]) - dt_key(complete[i - 1]["datetime"])) / 60
            if delta > interval * 2.5:
                gaps.append(round(delta / interval, 1))
        if gaps:
            result["warnings"].append(
                f"تم رصد {len(gaps)} فجوة زمنية أكبر من 2.5 شمعة؛ قد تكون عطلات/انقطاع مصدر."
            )
            result["largest_gap_candles"] = max(gaps)

    result["candles"] = complete
    if not result["errors"]:
        result["status"] = "VALID_WITH_WARNINGS" if result["warnings"] else "VALID"
    return result


def validate_all_timeframes(report: dict) -> dict:
    results = {}
    for tf in TIMEFRAMES:
        results[tf] = validate_timeframe(tf, report.get("timeframes", {}).get(tf))

    usable = [tf for tf in TIMEFRAMES if results[tf]["status"] in ("VALID", "VALID_WITH_WARNINGS")]
    global_errors = []
    if len(usable) != len(TIMEFRAMES):
        global_errors.append("واحد أو أكثر من الفريمات غير صالح؛ تم إيقاف السيناريوهات الكاملة.")

    # Check timestamps are comparable across frames.
    modes = {results[tf]["timestamp_mode"] for tf in usable if results[tf]["timestamp_mode"]}
    if len(modes) > 1:
        global_errors.append("أنماط التوقيت مختلفة بين الفريمات؛ لا يمكن مقارنة التوقيتات بثقة.")

    latest_keys = []
    for tf in usable:
        candles = results[tf]["candles"]
        if candles:
            latest_keys.append((tf, dt_key(candles[-1]["datetime"])))
    if latest_keys:
        newest = max(v for _, v in latest_keys)
        for tf, key in latest_keys:
            # A completed candle on a slower frame naturally closes less frequently.
            tolerance = (INTERVAL_MINUTES[tf] or 43200) * 2.5 * 60
            if newest - key > tolerance:
                results[tf]["warnings"].append("آخر شمعة مكتملة قديمة مقارنة بباقي الفريمات.")

    timestamp_reliable = bool(usable) and all(
        results[tf].get("timestamp_reliable", False) for tf in usable
    ) and not global_errors
    return {
        "status": "VALID" if len(usable) == len(TIMEFRAMES) and not global_errors else "DATA_ERROR",
        "usable_timeframes": usable,
        "timeframes": {
            tf: {k: v for k, v in results[tf].items() if k != "candles"}
            for tf in TIMEFRAMES
        },
        "global_errors": global_errors,
        "timestamp_reliable": timestamp_reliable,
        "timestamp_note": (
            "كل التواريخ ذات timezone واضح ومتسق."
            if timestamp_reliable else
            "التوقيت غير مثبت بالكامل؛ راجع timestamp_basis والمنطقة الزمنية في المصدر."
        ),
    }


# ----------------------------- market analysis ----------------------------

def true_ranges(candles: List[dict]) -> List[float]:
    values = []
    for i, c in enumerate(candles):
        if i == 0:
            values.append(c["high"] - c["low"])
        else:
            prev = candles[i - 1]["close"]
            values.append(max(
                c["high"] - c["low"],
                abs(c["high"] - prev),
                abs(c["low"] - prev),
            ))
    return values


def atr(candles: List[dict], period: int = 14) -> float:
    ranges = true_ranges(candles)
    sample = ranges[-period:]
    return mean(sample) if sample else 0.0


def pivots(candles: List[dict], side: str, left: int = 2, right: int = 2) -> List[dict]:
    out = []
    key = "high" if side == "high" else "low"
    start = max(left, len(candles) - 80)
    end = len(candles) - right
    for i in range(start, end):
        value = candles[i][key]
        window = candles[i - left:i + right + 1]
        if side == "high":
            if value >= max(x["high"] for x in window) and any(
                value > x["high"] for j, x in enumerate(window) if j != left
            ):
                out.append({"price": value, "index": i, "datetime": candles[i]["datetime"].isoformat()})
        else:
            if value <= min(x["low"] for x in window) and any(
                value < x["low"] for j, x in enumerate(window) if j != left
            ):
                out.append({"price": value, "index": i, "datetime": candles[i]["datetime"].isoformat()})
    return out


def trend_label(candles: List[dict]) -> str:
    closes = [x["close"] for x in candles]
    if len(closes) < 6:
        return "UNKNOWN"
    fast = mean(closes[-5:])
    slow = mean(closes[-min(15, len(closes)):])
    first = closes[-min(10, len(closes))]
    last = closes[-1]
    if fast > slow and last > first:
        return "BULLISH"
    if fast < slow and last < first:
        return "BEARISH"
    return "RANGE"


def cluster_levels(levels: List[dict], tolerance: float) -> List[dict]:
    if not levels:
        return []
    levels = sorted(levels, key=lambda x: x["price"])
    clusters = []
    for item in levels:
        if not clusters or abs(item["price"] - mean(x["price"] for x in clusters[-1]["items"])) > tolerance:
            clusters.append({"items": [item]})
        else:
            clusters[-1]["items"].append(item)
    result = []
    for group in clusters:
        items = group["items"]
        result.append({
            "price": mean(x["price"] for x in items),
            "touches": len(items),
            "timeframes": sorted(set(x["timeframe"] for x in items)),
            "source_types": sorted(set(x["source"] for x in items)),
            "latest_index": max(x.get("index", 0) for x in items),
        })
    return result


def collect_levels(tf_data: dict, side: str) -> List[dict]:
    """Collect confirmed swing levels plus explicit, rule-defined FVG/OB references."""
    found = []
    for tf in TIMEFRAMES:
        candles = tf_data[tf]["candles"]
        for p in pivots(candles, side):
            found.append({
                "price": p["price"], "index": p["index"],
                "datetime": p["datetime"], "timeframe": tf,
                "source": "SWING_HIGH" if side == "high" else "SWING_LOW",
            })
        local_atr = atr(candles)
        min_gap = max(local_atr * 0.05, 0.10)
        # Three-candle fair-value gaps. Only completed candles are supplied here.
        for i in range(2, len(candles)):
            a, c = candles[i - 2], candles[i]
            if side == "high" and a["low"] - c["high"] >= min_gap:
                found.append({
                    "price": (a["low"] + c["high"]) / 2, "index": i,
                    "datetime": c["datetime"].isoformat(), "timeframe": tf,
                    "source": "FVG_BEARISH",
                })
            elif side == "low" and c["low"] - a["high"] >= min_gap:
                found.append({
                    "price": (c["low"] + a["high"]) / 2, "index": i,
                    "datetime": c["datetime"].isoformat(), "timeframe": tf,
                    "source": "FVG_BULLISH",
                })
        # Basic displacement order block: last opposite candle before a >=0.9 ATR break.
        body_threshold = max(local_atr * 0.9, 0.10)
        for i in range(1, len(candles)):
            prev, cur = candles[i - 1], candles[i]
            body = abs(cur["close"] - cur["open"])
            if side == "high":
                valid_ob = (prev["close"] > prev["open"] and cur["close"] < cur["open"]
                            and cur["close"] < prev["low"] and body >= body_threshold)
                source = "ORDER_BLOCK_BEARISH"
            else:
                valid_ob = (prev["close"] < prev["open"] and cur["close"] > cur["open"]
                            and cur["close"] > prev["high"] and body >= body_threshold)
                source = "ORDER_BLOCK_BULLISH"
            if valid_ob:
                found.append({
                    "price": (prev["open"] + prev["close"]) / 2, "index": i - 1,
                    "datetime": prev["datetime"].isoformat(), "timeframe": tf,
                    "source": source,
                })
    return found


def rejection_score(candles: List[dict], level: float, side: str, tolerance: float) -> int:
    score = 0
    recent = candles[-8:]
    for c in recent:
        if side == "SELL":
            touched = c["high"] >= level - tolerance and c["high"] <= level + tolerance * 2
            rejected = c["close"] < c["open"] and (c["high"] - max(c["open"], c["close"])) > (c["high"] - c["low"]) * 0.25
        else:
            touched = c["low"] <= level + tolerance and c["low"] >= level - tolerance * 2
            rejected = c["close"] > c["open"] and (min(c["open"], c["close"]) - c["low"]) > (c["high"] - c["low"]) * 0.25
        if touched and rejected:
            score += 5
    return min(score, 15)


def build_zones(tf_data: dict, current: float, side: str, base_atr: float) -> List[dict]:
    raw = collect_levels(tf_data, "high" if side == "SELL" else "low")
    if not raw:
        return []
    tolerance = max(base_atr * 0.35, current * 0.00035, 0.15)
    clusters = cluster_levels(raw, tolerance)
    zones = []
    for cluster in clusters:
        level = cluster["price"]
        # Sell zones are above current; buy zones below current.
        if side == "SELL" and level <= current:
            continue
        if side == "BUY" and level >= current:
            continue
        distance = abs(level - current)
        width = max(base_atr * 0.22, current * 0.00025, 0.10)
        low, high = level - width, level + width
        rejection = 0
        for tf in cluster["timeframes"]:
            rejection = max(rejection, rejection_score(tf_data[tf]["candles"], level, side, tolerance))
        pattern_sources = sum(1 for x in cluster["source_types"] if x.startswith(("FVG_", "ORDER_BLOCK_")))
        score = 10 + min(cluster["touches"] * 8, 32) + min(len(cluster["timeframes"]) * 12, 36) + min(pattern_sources * 4, 12) + rejection
        # A strong label requires independent multi-timeframe evidence, not a single pivot.
        score = int(clamp(score, 0, 95))
        if score >= 70 and len(cluster["timeframes"]) >= 2 and cluster["touches"] >= 2:
            strength = "STRONG_CONFLUENCE"
        elif score >= 50 and (len(cluster["timeframes"]) >= 2 or cluster["touches"] >= 3):
            strength = "MODERATE_CONFLUENCE"
        else:
            strength = "LIMITED_CONFLUENCE"

        risk_buffer = max(base_atr * 0.25, width * 0.5, 0.10)
        if side == "SELL":
            stop = high + risk_buffer
            risk = stop - level
            # Zone-specific targets: calculated from this zone's own risk, not current-price ATR.
            raw_targets = [level - risk * rr for rr in (1.0, 1.5, 2.0, 3.0)]
            targets = [x for x in raw_targets if x < level]
            targets = sorted(set(targets), reverse=True)[:4]
            invalidation = round_price(high + risk_buffer)
        else:
            stop = low - risk_buffer
            risk = level - stop
            raw_targets = [level + risk * rr for rr in (1.0, 1.5, 2.0, 3.0)]
            targets = sorted(set(x for x in raw_targets if x > level))[:4]
            invalidation = round_price(low - risk_buffer)

        if risk <= 0:
            continue
        zones.append({
            "direction": side,
            "zone": {"low": round_price(low), "high": round_price(high), "midpoint": round_price(level)},
            "reference_price": round_price(level),
            "distance_from_current": round_price(distance),
            "distance_atr": round(distance / max(base_atr, EPSILON), 2),
            "score": score,
            "strength": strength,
            "touches": cluster["touches"],
            "timeframes": cluster["timeframes"],
            "sources": cluster["source_types"],
            "evidence_quality": "تلاقي متعدد الفريمات/اللمسات" if strength == "STRONG_CONFLUENCE" else "أدلة محدودة أو متوسطة؛ لا تُعامل كمنطقة دفاع مؤكدة",
            "method_note": "المصادر المعروضة هي Swing أو FVG ثلاثي الشموع أو Order Block أساسي بعد شمعة اندفاع وكسر؛ التعريفات آلية ومبسطة وليست ضمانًا مؤسسيًا.",
            "stop_loss": round_price(stop),
            "risk_distance": round_price(risk),
            "targets": {f"tp{i+1}": round_price(v) for i, v in enumerate(targets)},
            "target_method": "مستويات مستقلة لكل منطقة محسوبة من مخاطرة المنطقة نفسها (1R/1.5R/2R/3R)، وليست إسقاط ATR موحدًا من السعر الحالي.",
            "confirmation": (
                "انتظر وصول السعر للمنطقة ثم رفضًا هابطًا وإغلاق 15min أسفل قاع شمعة الرفض."
                if side == "SELL" else
                "انتظر وصول السعر للمنطقة ثم رفضًا صاعدًا وإغلاق 15min أعلى قمة شمعة الرفض."
            ),
            "invalidation_level": invalidation,
            "invalidation_rule": (
                "إغلاق 15min أعلى حد المنطقة والستوب يلغي سيناريو البيع."
                if side == "SELL" else
                "إغلاق 15min أسفل حد المنطقة والستوب يلغي سيناريو الشراء."
            ),
            "status": "WAIT_FOR_RETEST",
        })
    zones.sort(key=lambda z: (z["distance_from_current"], -z["score"]))
    return zones[:12]


def liquidity_map(tf_data: dict, current: float, base_atr: float) -> dict:
    highs = collect_levels(tf_data, "high")
    lows = collect_levels(tf_data, "low")
    tolerance = max(base_atr * 0.20, current * 0.0002, 0.10)
    high_clusters = cluster_levels(highs, tolerance)
    low_clusters = cluster_levels(lows, tolerance)

    def pack(clusters, side):
        chosen = []
        for c in clusters:
            if c["touches"] >= 2:
                chosen.append({
                    "price": round_price(c["price"]),
                    "touches": c["touches"],
                    "timeframes": c["timeframes"],
                    "distance": round_price(abs(c["price"] - current)),
                    "type": "EQUAL_HIGHS" if side == "high" else "EQUAL_LOWS",
                })
        chosen.sort(key=lambda x: x["distance"])
        return chosen[:10]

    buy_stops = pack([x for x in high_clusters if x["price"] > current], "high")
    sell_stops = pack([x for x in low_clusters if x["price"] < current], "low")
    return {
        "above_price": buy_stops,
        "below_price": sell_stops,
        "note": "مناطق سيولة محتملة مستنتجة من قمم/قيعان متقاربة؛ ليست أوامر معلومة أو مضمونة.",
    }


def structure_summary(tf_data: dict) -> dict:
    result = {}
    for tf in TIMEFRAMES:
        candles = tf_data[tf]["candles"]
        result[tf] = {
            "trend": trend_label(candles),
            "latest_close": round_price(candles[-1]["close"]),
            "atr": round_price(atr(candles)),
            "completed_candles": len(candles),
        }
    return result


def make_scenario(side: str, zones: List[dict], current: float, trend_data: dict) -> dict:
    nearest = zones[:12]
    if not nearest:
        return {
            "direction": side, "status": "NO_VALID_ZONE",
            "message": "لا توجد منطقة انعكاس صالحة فوق/تحت السعر الحالي ضمن البيانات المتاحة.",
            "zones": [], "confluence_score": 0,
        }
    lead = nearest[0]
    agreeing = sum(1 for v in trend_data.values() if v == ("BEARISH" if side == "SELL" else "BULLISH"))
    opposing = sum(1 for v in trend_data.values() if v == ("BULLISH" if side == "SELL" else "BEARISH"))
    score = int(clamp(lead["score"] + agreeing * 2 - opposing, 0, 99))
    return {
        "direction": side,
        "status": "CONDITIONAL",
        "confluence_score": score,
        "score_is_probability": False,
        "scenario_description": (
            "إذا واصل السعر الصعود، راقب مناطق المقاومة التالية لفرص بيع بعد ظهور رفض وتأكيد."
            if side == "SELL" else
            "إذا واصل السعر الهبوط، راقب مناطق الدعم التالية لفرص شراء بعد ظهور ارتداد وتأكيد."
        ),
        "nearest_zone": lead,
        "zones_nearest_first": nearest,
        "trend_alignment": {
            "agreeing_timeframes": agreeing,
            "opposing_timeframes": opposing,
        },
    }


def cross_timeframe_price_check(tf_data: dict) -> dict:
    """Flag large close discrepancies; different candle close times are expected."""
    fast = tf_data["15min"]["candles"][-1]["close"]
    fast_atr = max(atr(tf_data["15min"]["candles"]), 0.01)
    checks = []
    warnings = []
    for tf in TIMEFRAMES:
        close = tf_data[tf]["candles"][-1]["close"]
        diff = abs(close - fast)
        allowance = max(3.0 * fast_atr, 2.0 * atr(tf_data[tf]["candles"]), fast * 0.003)
        suspicious = tf != "15min" and diff > allowance
        checks.append({
            "timeframe": tf, "latest_close": round_price(close),
            "difference_from_15min": round_price(diff),
            "review_threshold": round_price(allowance),
            "suspicious_difference": suspicious,
        })
        if suspicious:
            warnings.append(f"إغلاق {tf} بعيد بشكل غير معتاد عن 15min؛ راجع تزامن البيانات أو مصدر السعر.")
    return {"status": "REVIEW_REQUIRED" if warnings else "CONSISTENT_ENOUGH", "checks": checks, "warnings": warnings}


def analyze_market(tf_data: dict, validation: dict) -> dict:
    base = tf_data["15min"]["candles"]
    current = base[-1]["close"]
    base_atr = atr(base)
    if base_atr <= 0:
        raise ValueError("تعذر حساب ATR من شموع 15min.")

    trends = {tf: trend_label(tf_data[tf]["candles"]) for tf in TIMEFRAMES}
    sell_zones = build_zones(tf_data, current, "SELL", base_atr)
    buy_zones = build_zones(tf_data, current, "BUY", base_atr)
    sell = make_scenario("SELL", sell_zones, current, trends)
    buy = make_scenario("BUY", buy_zones, current, trends)
    liquidity = liquidity_map(tf_data, current, base_atr)
    consistency = cross_timeframe_price_check(tf_data)

    if sell.get("confluence_score", 0) > buy.get("confluence_score", 0):
        leading = "SELL"
    elif buy.get("confluence_score", 0) > sell.get("confluence_score", 0):
        leading = "BUY"
    else:
        leading = "NEUTRAL"

    near_limit = base_atr * 3.0
    nearest_sell = next((z for z in sell_zones if z["distance_from_current"] <= near_limit), None)
    nearest_buy = next((z for z in buy_zones if z["distance_from_current"] <= near_limit), None)

    return {
        "current_price": round_price(current),
        "price_source": "آخر إغلاق مكتمل على فريم 15min",
        "as_of": base[-1]["datetime"].isoformat(),
        "leading_scenario": leading,
        "decision": "WAIT_FOR_CONFIRMATION",
        "market_structure": {
            "timeframes": trends,
            "summary": structure_summary(tf_data),
        },
        "scenarios": {
            "if_price_rises": sell,
            "if_price_falls": buy,
        },
        "nearest_strong_levels": {
            "sell_zone_within_3_atr": next((z for z in sell_zones if z["distance_from_current"] <= near_limit and z["strength"] == "STRONG_CONFLUENCE"), None),
            "buy_zone_within_3_atr": next((z for z in buy_zones if z["distance_from_current"] <= near_limit and z["strength"] == "STRONG_CONFLUENCE"), None),
            "note": "هذا القسم يعرض المناطق التي اجتازت شروط القوة فقط. المناطق الأضعف تظهر منفصلة في السيناريوهات ولا تُسمى قوية.",
        },
        "all_zones_summary": {
            "sell_strong_count": sum(z["strength"] == "STRONG_CONFLUENCE" for z in sell_zones),
            "buy_strong_count": sum(z["strength"] == "STRONG_CONFLUENCE" for z in buy_zones),
            "sell_zones_count": len(sell_zones), "buy_zones_count": len(buy_zones),
            "note": "قد لا توجد منطقة قوية حاليًا؛ لا يتم رفع تصنيف منطقة ضعيفة لإجبار المحرك على إعطاء إشارة."
        },
        "cross_timeframe_price_check": consistency,
        "hidden_liquidity": liquidity,
        "volatility": {
            "atr_15min": round_price(base_atr),
            "one_atr_down": round_price(current - base_atr),
            "one_atr_up": round_price(current + base_atr),
            "note": "ATR مقياس تذبذب وليس هدفًا أو توقعًا.",
        },
        "rules": {
            "entry": "لا دخول بمجرد لمس المنطقة؛ انتظر رفضًا واضحًا وإغلاق شمعة 15min يؤكد الاتجاه.",
            "risk": "الستوب خلف حد المنطقة؛ لا توسّع الستوب بعد الدخول.",
            "targets": "الأهداف الحسابية إرشادية؛ راجع مستويات السيولة المقابلة قبل اعتمادها.",
        },
        "limitations": [
            "درجة التلاقي ليست احتمال نجاح ولا نسبة ربح.",
            "المناطق الفنية قد تُخترق؛ لا توجد منطقة مضمونة الانعكاس.",
            "البيانات المحدودة إلى 30 شمعة لكل فريم تقلل دقة البنية التاريخية.",
            "تعريفات FVG وOrder Block آلية ومبسطة؛ راجع تكوين كل نموذج قبل الاعتماد عليه.",
            "اختلاف إغلاقات الفريمات الكبيرة قد يكون بسبب توقيت الإغلاق، لكن الاختلاف الكبير يُعلّم للمراجعة ولا يُعامل كحقيقة متزامنة.",
        ],
    }


# -------------------------------- API routes -------------------------------

@app.get("/")
def home():
    return jsonify({
        "engine": ENGINE_NAME,
        "version": VERSION,
        "symbol": SYMBOL,
        "status": "online",
        "endpoint": "POST /analyze",
        "features": ["Strict OHLC validation", "Descending candle-order support", "Ranked BUY/SELL swing reaction zones", "Per-zone risk-based targets", "Liquidity pools", "Cross-timeframe price consistency checks"],
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "version": VERSION})


@app.post("/analyze")
def analyze():
    payload = request.get_json(silent=True)
    report, payload_errors = normalize_payload(payload)
    if report is None or payload_errors:
        return jsonify({
            "engine": ENGINE_NAME, "version": VERSION,
            "decision": "DATA_ERROR",
            "errors": payload_errors or ["بنية البيانات غير صالحة."],
        }), 400

    validation = validate_all_timeframes(report)
    response = {
        "engine": ENGINE_NAME,
        "version": VERSION,
        "symbol": report.get("summary", {}).get("symbol", SYMBOL)
            if isinstance(report.get("summary", {}), dict) else SYMBOL,
        "decision": "DATA_ERROR",
        "data_validation": validation,
        "market_analysis": None,
        "warnings": [],
    }

    if validation["status"] != "VALID":
        response["warnings"].append(
            "لم يتم إنشاء سيناريوهات تداول لأن فحص البيانات لم يجتز جميع الفريمات."
        )
        return jsonify(response), 200

    tf_data = {
        tf: validate_timeframe(tf, report["timeframes"][tf])
        for tf in TIMEFRAMES
    }

    # Extra self-check: all frames must have enough complete candles and valid OHLC.
    for tf in TIMEFRAMES:
        if len(tf_data[tf]["candles"]) < MIN_COMPLETE_CANDLES:
            response["decision"] = "DATA_ERROR"
            response["warnings"].append(f"الفريم {tf} لا يحتوي على شموع كافية.")
            return jsonify(response), 200

    try:
        market = analyze_market(tf_data, validation)
        response["decision"] = market["decision"]
        response["market_analysis"] = market
        if not validation["timestamp_reliable"]:
            response["warnings"].append(
                "التحليل الفني محسوب، لكن توقيت البيانات غير مثبت بالكامل؛ لا تعتمد على as_of كتوقيت UTC."
            )
    except Exception as exc:
        response["decision"] = "ANALYSIS_ERROR"
        response["warnings"].append("حدث خطأ أثناء التحليل وتم إيقاف النتيجة بدل إخراج مستويات غير مكتملة.")
        response["internal_error"] = str(exc)

    return jsonify(response), 200


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found", "available": ["/", "/health", "/analyze"]}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed", "expected": "POST /analyze"}), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
