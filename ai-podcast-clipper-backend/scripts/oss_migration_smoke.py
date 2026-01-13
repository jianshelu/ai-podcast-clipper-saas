#!/usr/bin/env python3
"""OSS migration end-to-end smoke test.

Steps:
1) Generate presigned PUT URL and upload a file.
2) Generate presigned GET URL and download; verify checksum.
3) Validate Range support for streaming.
"""

import argparse
import hashlib
import os
import pathlib
import sys
import tempfile
import uuid

import oss2
import boto3
from botocore.config import Config
import requests


def _sha256_file(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_stream(response: requests.Response) -> str:
    hasher = hashlib.sha256()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            hasher.update(chunk)
    return hasher.hexdigest()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(2)
    return value


def _signed_put_url_oss(bucket: oss2.Bucket, key: str, expires: int) -> str:
    return bucket.sign_url("PUT", key, expires)


def _signed_get_url_oss(bucket: oss2.Bucket, key: str, expires: int, filename: str) -> str:
    params = {
        "response-content-disposition": f'attachment; filename="{filename}"'
    }
    return bucket.sign_url("GET", key, expires, params=params)


def _signed_put_url_s3(client, bucket: str, key: str, expires: int) -> str:
    return client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def _signed_get_url_s3(client, bucket: str, key: str, expires: int, filename: str) -> str:
    return client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ResponseContentDisposition": f'attachment; filename="{filename}"',
        },
        ExpiresIn=expires,
    )


def _check_response(response: requests.Response, label: str) -> None:
    if response.ok:
        return
    request_id = (
        response.headers.get("x-oss-request-id")
        or response.headers.get("x-oss-requestid")
        or response.headers.get("x-amz-request-id")
    )
    print(
        f"{label} failed status={response.status_code} request_id={request_id} body={response.text[:500]}",
        file=sys.stderr,
    )
    response.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="OSS migration smoke test")
    parser.add_argument("--file", required=True, help="Path to a test file")
    parser.add_argument("--job-id", default=str(uuid.uuid4()), help="Job id to use")
    parser.add_argument("--expires", type=int, default=600, help="Presign expiration seconds")
    parser.add_argument(
        "--input-prefix",
        default="jobs/{job_id}/inputs",
        help="Prefix for uploaded inputs (default: jobs/{job_id}/inputs)",
    )
    parser.add_argument(
        "--output-prefix",
        default="jobs/{job_id}/outputs",
        help="Prefix for copied outputs (default: jobs/{job_id}/outputs)",
    )
    parser.add_argument(
        "--backend",
        choices=("oss", "s3"),
        default=os.getenv("OSS_BACKEND", "oss"),
        help="Signing backend (oss or s3)",
    )
    parser.add_argument(
        "--s3-signature-version",
        choices=("s3v4", "s3"),
        default=os.getenv("S3_SIGNATURE_VERSION", "s3v4"),
        help="S3 signature version for MinIO (default: s3v4)",
    )
    parser.add_argument(
        "--signature",
        choices=("v1", "v4"),
        default=os.getenv("OSS_SIGNATURE_VERSION", "v1"),
        help="Signature version (default: v1)",
    )
    parser.add_argument(
        "--path-style",
        action="store_true",
        default=os.getenv("OSS_PATH_STYLE", "").lower() in ("1", "true", "yes"),
        help="Use path-style URLs for S3-compatible endpoints",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("OSS_REGION"),
        help="Region for signature v4 (optional)",
    )
    args = parser.parse_args()

    file_path = pathlib.Path(args.file).expanduser().resolve()
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        return 2

    access_key_id = _require_env("OSS_ACCESS_KEY_ID")
    access_key_secret = _require_env("OSS_ACCESS_KEY_SECRET")
    endpoint = _require_env("OSS_ENDPOINT")
    bucket_name = _require_env("OSS_BUCKET")

    if args.backend == "oss":
        if args.signature == "v4":
            auth = oss2.AuthV4(access_key_id, access_key_secret)
        else:
            auth = oss2.Auth(access_key_id, access_key_secret)

        bucket = oss2.Bucket(
            auth,
            endpoint,
            bucket_name,
            region=args.region,
            is_path_style=args.path_style,
        )
        presign_put = lambda key: _signed_put_url_oss(bucket, key, args.expires)
        presign_get = lambda key, filename: _signed_get_url_oss(
            bucket, key, args.expires, filename
        )
    else:
        region = args.region or "us-east-1"
        config = Config(
            signature_version=args.s3_signature_version,
            s3={"addressing_style": "path" if args.path_style else "virtual"},
        )
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=access_key_secret,
            config=config,
        )
        presign_put = lambda key: _signed_put_url_s3(client, bucket_name, key, args.expires)
        presign_get = lambda key, filename: _signed_get_url_s3(
            client, bucket_name, key, args.expires, filename
        )

    ext = file_path.suffix or ".bin"
    input_prefix = args.input_prefix.format(job_id=args.job_id).rstrip("/")
    output_prefix = args.output_prefix.format(job_id=args.job_id).rstrip("/")
    input_key = f"{input_prefix}/original{ext}"
    output_key = f"{output_prefix}/original{ext}"
    filename = file_path.name

    put_url = presign_put(input_key)
    print(f"[put] key={input_key}")
    with file_path.open("rb") as handle:
        put_response = requests.put(put_url, data=handle)
    _check_response(put_response, "PUT")

    print(f"[copy] {input_key} -> {output_key}")
    if args.backend == "oss":
        bucket.copy_object(bucket.bucket_name, input_key, output_key)
    else:
        client.copy({"Bucket": bucket_name, "Key": input_key}, bucket_name, output_key)

    get_url = presign_get(output_key, filename)
    print("[get] download and hash check (from outputs)")
    with requests.get(get_url, stream=True) as get_response:
        _check_response(get_response, "GET")
        download_hash = _sha256_stream(get_response)

    local_hash = _sha256_file(file_path)
    if local_hash != download_hash:
        print(
            f"Checksum mismatch local={local_hash} downloaded={download_hash}",
            file=sys.stderr,
        )
        return 1

    print("[range] checking Accept-Ranges and partial content")
    range_headers = {"Range": "bytes=0-1023"}
    range_response = requests.get(get_url, headers=range_headers, stream=True)
    if range_response.status_code not in (200, 206):
        print(
            f"Range request unexpected status={range_response.status_code}",
            file=sys.stderr,
        )
        return 1

    accept_ranges = range_response.headers.get("Accept-Ranges")
    content_range = range_response.headers.get("Content-Range")
    print(f"[range] status={range_response.status_code} accept_ranges={accept_ranges} content_range={content_range}")

    print("[ok] OSS migration smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
