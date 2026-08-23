import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import boto3
import pandas as pd


def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


@dataclass
class DataSource:
    s3_client: Optional[object] = None

    @staticmethod
    def from_paths(*paths: str) -> "DataSource":
        use_s3 = any(is_s3_uri(p) for p in paths if isinstance(p, str))
        return DataSource(s3_client=boto3.client("s3") if use_s3 else None)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        if is_s3_uri(path):
            if self.s3_client is None:
                raise RuntimeError("S3 path provided but s3_client is None.")
            u = urlparse(path)
            if u.scheme != "s3" or not u.netloc or not u.path:
                raise ValueError(f"Not a valid s3 uri: {path}")
            bucket = u.netloc
            key = u.path.lstrip("/")
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read().decode(encoding)
        else:
            with open(path, "r", encoding=encoding) as f:
                return f.read()

    def read_json(self, path: str):
        return json.loads(self.read_text(path))

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        if is_s3_uri(path):
            if self.s3_client is None:
                raise RuntimeError("S3 path provided but s3_client is None.")
            u = urlparse(path)
            if u.scheme != "s3" or not u.netloc or not u.path:
                raise ValueError(f"Not a valid s3 uri: {path}")
            bucket = u.netloc
            key = u.path.lstrip("/")
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return pd.read_csv(obj["Body"], **kwargs)
        else:
            return pd.read_csv(path, **kwargs)
