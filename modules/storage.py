import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import streamlit as st


DATASET_FOLDER = "datasets"


def _config():
    try:
        cfg = st.secrets["supabase"]
        url = str(cfg["url"]).strip().rstrip("/")
        key = str(cfg["secret_key"]).strip()
        bucket = str(cfg.get("bucket", "osas-data")).strip()
    except Exception:
        return None

    if not url or not key or not bucket:
        return None
    return url, key, bucket


def is_configured():
    return _config() is not None


def _safe_filename(name):
    name = str(name or "survey_dataset.csv").strip()
    name = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not name.lower().endswith(".csv"):
        name += ".csv"
    stem = name[:-4]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (stem or "survey_dataset") + ".csv"


def _headers(key, content_type=None):
    # New sb_secret_* keys are opaque API keys, not JWTs.
    # Send them on apikey only. Legacy service_role JWTs can also use Bearer.
    headers = {"apikey": key}
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _request(method, endpoint, key, data=None, content_type=None, extra_headers=None):
    headers = _headers(key, content_type)
    if extra_headers:
        headers.update(extra_headers)

    req = Request(endpoint, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as response:
            return response.read(), dict(response.headers)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            message = parsed.get("message") or parsed.get("error") or body
        except Exception:
            message = body
        raise RuntimeError(f"Storage request failed ({e.code}): {message}") from None
    except URLError as e:
        raise RuntimeError(f"Storage connection failed: {e.reason}") from None


def save_dataset(raw_bytes, original_name):
    if not raw_bytes:
        raise ValueError("The uploaded CSV is empty.")

    cfg = _config()
    if cfg is None:
        raise RuntimeError("Private dataset storage is not configured.")

    url, key, bucket = cfg
    safe_name = _safe_filename(original_name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(raw_bytes).hexdigest()[:10]
    object_name = f"{stamp}_{digest}_{safe_name}"
    object_path = f"{DATASET_FOLDER}/{object_name}"

    endpoint = (
        f"{url}/storage/v1/object/"
        f"{quote(bucket, safe='')}/{quote(object_path, safe='/')}"
    )
    _request(
        "POST",
        endpoint,
        key,
        data=raw_bytes,
        content_type="text/csv",
        extra_headers={
            "x-upsert": "false",
            "Cache-Control": "no-store",
        },
    )

    return {
        "object_path": object_path,
        "filename": safe_name,
        "uploaded_at": stamp,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _list_latest():
    cfg = _config()
    if cfg is None:
        raise RuntimeError("Private dataset storage is not configured.")

    url, key, bucket = cfg
    endpoint = f"{url}/storage/v1/object/list/{quote(bucket, safe='')}"
    payload = json.dumps({
        "prefix": DATASET_FOLDER,
        "limit": 100,
        "offset": 0,
        "sortBy": {"column": "name", "order": "desc"},
    }).encode("utf-8")

    body, _ = _request(
        "POST",
        endpoint,
        key,
        data=payload,
        content_type="application/json",
        extra_headers={"Cache-Control": "no-store"},
    )
    items = json.loads(body.decode("utf-8"))
    files = [
        item for item in items
        if isinstance(item, dict) and item.get("id") and item.get("name")
    ]
    if not files:
        return None
    return files[0]


def load_latest_dataset():
    cfg = _config()
    if cfg is None:
        raise RuntimeError("Private dataset storage is not configured.")

    url, key, bucket = cfg
    item = _list_latest()
    if not item:
        return None

    object_name = item["name"]
    object_path = f"{DATASET_FOLDER}/{object_name}"
    endpoint = (
        f"{url}/storage/v1/object/authenticated/"
        f"{quote(bucket, safe='')}/{quote(object_path, safe='/')}"
    )
    raw, _ = _request(
        "GET",
        endpoint,
        key,
        extra_headers={"Cache-Control": "no-store"},
    )

    if not raw:
        return None

    # Object names are timestamp_hash_originalfilename.csv.
    parts = object_name.split("_", 2)
    display_name = parts[2] if len(parts) == 3 else object_name

    return {
        "raw": raw,
        "filename": display_name,
        "object_path": object_path,
        "updated_at": item.get("updated_at") or item.get("created_at"),
    }
