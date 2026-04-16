import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Optional

try:
    import boto3
except Exception:  # pragma: no cover - optional dependency
    boto3 = None


@dataclass
class StorageConfig:
    mode: str = "LOCAL_DISK"
    local_root: str = "./static/uploads"
    bucket: Optional[str] = None
    endpoint_url: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None


class StorageManager:
    """Storage lane manager for local disk or S3-compatible providers."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.mode = (config.mode or "LOCAL_DISK").upper()
        os.makedirs(self.config.local_root, exist_ok=True)
        self._s3 = None

        if self.mode == "S3_COMPATIBLE":
            if boto3 is None:
                raise RuntimeError("S3_COMPATIBLE mode requires boto3 to be installed")
            self._s3 = boto3.client(
                "s3",
                endpoint_url=config.endpoint_url,
                region_name=config.region,
                aws_access_key_id=config.access_key_id,
                aws_secret_access_key=config.secret_access_key,
            )

    def _unique_name(self, original_name: str) -> str:
        stem, ext = os.path.splitext(original_name)
        safe_stem = "".join(c for c in stem if c.isalnum() or c in {"-", "_"})[:80] or "file"
        return f"{safe_stem}_{uuid.uuid4().hex[:10]}{ext.lower()}"

    def save_upload(self, file_obj, filename: str) -> tuple[str, str]:
        """Save inbound Flask upload and return (object_key, local_path)."""
        object_key = filename
        local_path = os.path.join(self.config.local_root, object_key)
        file_obj.save(local_path)

        if self.mode == "S3_COMPATIBLE":
            self.store_file(local_path, object_key)

        return object_key, local_path

    def store_file(self, local_path: str, object_key: Optional[str] = None) -> str:
        key = object_key or os.path.basename(local_path)
        if self.mode == "LOCAL_DISK":
            target = os.path.join(self.config.local_root, key)
            if os.path.abspath(target) != os.path.abspath(local_path):
                shutil.copy2(local_path, target)
            return key

        if self._s3 is None or not self.config.bucket:
            raise RuntimeError("S3_COMPATIBLE mode requires bucket and credentials")

        self._s3.upload_file(local_path, self.config.bucket, key)
        return key

    def generate_private_url(self, object_key: str, expires_seconds: int = 300) -> str:
        if self.mode == "LOCAL_DISK":
            return f"/uploads/{object_key}"

        if self._s3 is None or not self.config.bucket:
            raise RuntimeError("S3_COMPATIBLE mode requires bucket and credentials")

        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.config.bucket, "Key": object_key},
            ExpiresIn=expires_seconds,
        )
