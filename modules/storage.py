import hashlib
import io
import re
from datetime import datetime, timezone

import streamlit as st
from supabase import create_client


DATASET_FOLDER = "datasets"
KEEP_RECENT_FILES = 12


def _config():
    try:
        cfg = st.secrets["supabase"]
        url = str(cfg["url"]).strip()
        key = str(cfg["secret_key"]).strip()
        bucket = str(cfg.get("bucket", "osas-data")).strip()
    except Exception:
        return None

    if not url or not key or not bucket:
        return None
    return url, key, bucket


def is_configured():
    return _config() is not None


def _client():
    cfg = _config()
    if cfg is None:
        raise RuntimeError("Private dataset storage is not configured.")
    url, key, bucket = cfg
    return create_client(url, key), bucket


def _safe_filename(name):
    name = str(name or "survey_dataset.csv").strip()
    name = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not name.lower().endswith(".csv"):
        name += ".csv"
    stem = name[:-4]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (stem or "survey_dataset") + ".csv"


def save_dataset(raw_bytes, original_name):
    if not raw_bytes:
        raise ValueError("The uploaded CSV is empty.")

    client, bucket = _client()
    safe_name = _safe_filename(original_name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(raw_bytes).hexdigest()[:10]
    object_name = f"{stamp}_{digest}_{safe_name}"
    object_path = f"{DATASET_FOLDER}/{object_name}"

    client.storage.from_(bucket).upload(
        path=object_path,
        file=io.BytesIO(raw_bytes),
        file_options={
            "content-type": "text/csv",
            "cache-control": "0",
            "upsert": "false",
        },
    )

    # Keep a small history of recent uploads and remove older versions.
    try:
        files = client.storage.from_(bucket).list(
            DATASET_FOLDER,
            {
                "limit": 100,
                "offset": 0,
                "sortBy": {"column": "name", "order": "desc"},
            },
        )
        real_files = [item for item in files if item.get("id") and item.get("name")]
        old = real_files[KEEP_RECENT_FILES:]
        if old:
            client.storage.from_(bucket).remove(
                [f"{DATASET_FOLDER}/{item['name']}" for item in old]
            )
    except Exception:
        # Cleanup failure must never invalidate a successful upload.
        pass

    return {
        "object_path": object_path,
        "filename": safe_name,
        "uploaded_at": stamp,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def load_latest_dataset():
    client, bucket = _client()
    files = client.storage.from_(bucket).list(
        DATASET_FOLDER,
        {
            "limit": 1,
            "offset": 0,
            "sortBy": {"column": "name", "order": "desc"},
        },
    )
    files = [item for item in files if item.get("id") and item.get("name")]
    if not files:
        return None

    item = files[0]
    object_name = item["name"]
    object_path = f"{DATASET_FOLDER}/{object_name}"
    raw = client.storage.from_(bucket).download(object_path)
    if not raw:
        return None

    # Object names are: timestamp_hash_originalfilename.csv
    parts = object_name.split("_", 2)
    display_name = parts[2] if len(parts) == 3 else object_name

    return {
        "raw": bytes(raw),
        "filename": display_name,
        "object_path": object_path,
        "updated_at": item.get("updated_at") or item.get("created_at"),
    }
