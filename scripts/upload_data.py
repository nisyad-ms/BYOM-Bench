#!/usr/bin/env python
"""Upload outputs/ folder to Azure Blob Storage.

Usage:
    python scripts/upload_data.py
    python scripts/upload_data.py --path outputs/2026-02-02_1414
    python scripts/upload_data.py --prefix v1/
    python scripts/upload_data.py --overwrite
"""

import argparse
import time
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

ACCOUNT_URL = "https://memorypublicdata001wus2.blob.core.windows.net"
CONTAINER_NAME = "internal-benchmark"


def upload_directory(local_path: Path, prefix: str, overwrite: bool) -> None:
    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(account_url=ACCOUNT_URL, credential=credential)
    container = blob_service.get_container_client(CONTAINER_NAME)

    files = [f for f in local_path.rglob("*") if f.is_file()]
    if not files:
        print(f"No files found in {local_path}")
        return

    print(f"Found {len(files)} files to upload from {local_path}")

    uploaded = 0
    skipped = 0
    for filepath in sorted(files):
        blob_name = prefix + str(filepath.relative_to(local_path))
        blob_client = container.get_blob_client(blob_name)

        if not overwrite:
            try:
                blob_client.get_blob_properties()
                skipped += 1
                continue
            except Exception:
                pass

        print(f"  Uploading: {blob_name}")
        with open(filepath, "rb") as f:
            blob_client.upload_blob(f, overwrite=overwrite)
        uploaded += 1

    print(f"\nDone. Uploaded: {uploaded}, Skipped (already exists): {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Upload outputs to Azure Blob Storage")
    parser.add_argument("--path", type=str, default="outputs", help="Local directory to upload (default: outputs/)")
    parser.add_argument("--prefix", type=str, default="", help="Blob name prefix (e.g., 'v1/')")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing blobs (default: skip)")
    args = parser.parse_args()

    local_path = Path(args.path)
    if not local_path.is_dir():
        print(f"Directory not found: {local_path}")
        return

    upload_directory(local_path, args.prefix, args.overwrite)


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.1f}s")
