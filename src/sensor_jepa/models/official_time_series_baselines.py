from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BaselineAvailability:
    aeon_available: bool
    sktime_available: bool
    official_minirocket_available: bool
    official_multirocket_available: bool
    ts2vec_official_available: bool
    fallback_enabled: bool = True


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _can_import(path: str, name: str) -> bool:
    try:
        module = __import__(path, fromlist=[name])
        getattr(module, name)
        return True
    except Exception:
        return False


def check_official_baseline_availability() -> BaselineAvailability:
    aeon = _has_module("aeon")
    sktime = _has_module("sktime")
    ts2vec = _has_module("ts2vec")
    mini = False
    multi = False
    if aeon:
        mini = mini or _can_import("aeon.transformations.collection.convolution_based", "MiniRocket")
        multi = multi or _can_import("aeon.transformations.collection.convolution_based", "MultiRocket")
    if sktime:
        mini = mini or _can_import("sktime.transformations.panel.rocket", "MiniRocket")
        multi = multi or _can_import("sktime.transformations.panel.rocket", "MiniRocketMultivariate")
    return BaselineAvailability(
        aeon_available=aeon,
        sktime_available=sktime,
        official_minirocket_available=mini,
        official_multirocket_available=multi,
        ts2vec_official_available=ts2vec,
        fallback_enabled=True,
    )


def baseline_metadata_for_name(model_name: str, model_family: str = "", notes: str = "") -> dict[str, object]:
    name = model_name.lower()
    family = model_family.lower()
    notes_l = notes.lower()
    is_official = "official" in name or family.endswith("_official") or "official" in notes_l
    fallback = "lite" in name or "proxy" in name or "fallback" in family or "fallback" in notes_l
    available = True
    if "minirocket" in name and is_official:
        available = check_official_baseline_availability().official_minirocket_available
    elif ("multirocket" in name or "minirocket_multivariate" in name) and is_official:
        available = check_official_baseline_availability().official_multirocket_available
    elif "ts2vec" in name and is_official:
        available = check_official_baseline_availability().ts2vec_official_available
    backend = "unknown"
    if "aeon" in notes_l:
        backend = "aeon"
    elif "sktime" in notes_l:
        backend = "sktime"
    elif "ts2vec" in name:
        backend = "ts2vec_proxy" if fallback else "ts2vec"
    elif "rocket" in name:
        backend = "rocket_lite" if fallback else "rocket_official"
    return {
        "baseline_is_official": bool(is_official and not fallback),
        "baseline_backend": backend,
        "baseline_available": bool(available),
        "fallback_used": bool(fallback),
        "sota_claim_eligible": bool(is_official and available and not fallback),
    }


def availability_as_dict() -> dict[str, bool]:
    return asdict(check_official_baseline_availability())


def diagnostic_lines() -> list[str]:
    availability = check_official_baseline_availability()
    rows = asdict(availability)
    return [f"{key}: {str(value).lower()}" for key, value in rows.items()]
