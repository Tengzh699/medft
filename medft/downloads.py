from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import requests


def error_chain(error: BaseException) -> str:
    details, seen = [], set()
    while error is not None and id(error) not in seen and len(details) < 6:
        seen.add(id(error))
        details.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__
    return "\n  ".join(details)


def _verified(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    sha256 = entry.get("lfs", {}).get("sha256")
    digest = hashlib.sha256() if sha256 else hashlib.sha1()
    if not sha256:
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == (sha256 or entry["blobId"])


def _http_file(session: requests.Session, url: str, destination: Path, entry: dict) -> Path:
    """Publish only verified bytes; keep interrupted transfers in a managed .part file."""
    if _verified(destination, entry):
        print(f"已校验本地缓存：{destination.name}", file=sys.stderr, flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    size = entry["size"]
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset == size and _verified(partial, entry):
        partial.replace(destination)
        return destination
    if offset >= size:
        # This file is an incomplete/invalid transfer belonging to this pinned URL.
        partial.write_bytes(b"")
        offset = 0
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"HTTP下载 {destination.name}：{offset / 1024**2:.1f}/{size / 1024**2:.1f} MiB", file=sys.stderr, flush=True)
    with session.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise RuntimeError("下载响应经过压缩，无法可靠核对字节范围。")
        if response.status_code == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
                raise RuntimeError("服务器返回的Content-Range与续传位置或文件大小不符，已保留原.part文件。")
            mode = "ab" if offset else "wb"
        elif response.status_code == 200:
            # A server may ignore Range; start over instead of appending a full file.
            mode, offset = "wb", 0
        else:
            raise RuntimeError(f"下载响应应为200或206，实际为{response.status_code}。")
        advertised = response.headers.get("Content-Length")
        if advertised is not None and int(advertised) != size - offset:
            raise RuntimeError("下载响应长度与固定版本文件大小不符。")
        received, next_progress = offset, offset + 16 * 1024**2
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if not chunk:
                    continue
                if received + len(chunk) > size:
                    raise RuntimeError("收到的数据超过固定版本文件大小，下载已停止。")
                stream.write(chunk)
                received += len(chunk)
                if received >= next_progress:
                    print(f"  {received / 1024**2:.1f}/{size / 1024**2:.1f} MiB", file=sys.stderr, flush=True)
                    next_progress = received + 16 * 1024**2
    if not _verified(partial, entry):
        raise RuntimeError(f"{destination.name}未通过上游文件大小/哈希校验；已保留.part文件，重跑可续传或重下。")
    partial.replace(destination)
    return destination


def download_http_files(*, repo: str, revision: str, filenames: tuple[str, ...],
                        cache_dir: Path, endpoint: str, no_proxy: bool = False) -> tuple[dict[str, Path], dict]:
    """Use normal GET requests, without depending on Hub-specific HEAD headers."""
    endpoint = endpoint.rstrip("/")
    if not endpoint.startswith(("https://", "http://")):
        raise ValueError("endpoint须为完整的HTTP/HTTPS地址。")
    with requests.Session() as session:
        # Only this session changes; system proxy settings and environment stay intact.
        session.trust_env = not no_proxy
        with session.get(f"{endpoint}/api/datasets/{repo}/revision/{revision}",
                         params={"blobs": "true"}, timeout=(15, 60)) as response:
            response.raise_for_status()
            metadata = response.json()
        if metadata.get("sha") != revision or metadata.get("id") != repo:
            raise ValueError("服务器返回的数据集或提交版本不符。")
        entries = {item["rfilename"]: item for item in metadata.get("siblings", [])}
        paths, hashes = {}, {}
        root = cache_dir / "http" / ("datasets--" + repo.replace("/", "--")) / revision
        for filename in filenames:
            entry = entries.get(filename, {})
            digest = entry.get("lfs", {}).get("sha256") or entry.get("blobId", "")
            algorithm = "sha256" if entry.get("lfs") else "git_blob_sha1"
            if (type(entry.get("size")) is not int or entry["size"] < 1
                    or not re.fullmatch(r"[0-9a-f]{64}" if algorithm == "sha256" else r"[0-9a-f]{40}", digest)):
                raise ValueError(f"服务器未提供{filename}的有效大小和上游哈希。")
            paths[filename] = _http_file(
                session, f"{endpoint}/datasets/{repo}/resolve/{revision}/{filename}", root / filename, entry,
            )
            hashes[filename] = {"algorithm": algorithm, "value": digest, "bytes": entry["size"]}
    return paths, hashes
