import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from medft.config import load_config
from medft.corpus import PUBLIC_FILE, PUBLIC_REPO, PUBLIC_REVISION, download_public_data
from medft.downloads import download_http_files


class HttpDownloadTests(unittest.TestCase):
    """Exercise real HTTP GET/Range requests against a tiny local server, no external network."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        card = b"---\nlicense: apache-2.0\n---\nLocal test fixture.\n"
        self.files = {PUBLIC_FILE: b'{"questions":[["question"]],"answers":["answer"]}\n' * 100, "README.md": card}
        self.metadata = {
            "id": PUBLIC_REPO, "sha": PUBLIC_REVISION,
            "siblings": [
                {"rfilename": PUBLIC_FILE, "size": len(self.files[PUBLIC_FILE]),
                 "lfs": {"sha256": hashlib.sha256(self.files[PUBLIC_FILE]).hexdigest()}},
                {"rfilename": "README.md", "size": len(card),
                 "blobId": hashlib.sha1(f"blob {len(card)}\0".encode() + card).hexdigest()},
            ],
        }
        self.state = {"ignore_range": False, "bad_range": False, "tamper": False, "requests": []}
        state, files, metadata = self.state, self.files, self.metadata

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                # This server intentionally lacks the HEAD behavior expected by the Hub SDK.
                self.send_error(405)

            def do_GET(self):
                path = urlsplit(self.path).path
                state["requests"].append((path, self.headers.get("Range")))
                status, content_range = 200, None
                if path.startswith("/api/datasets/"):
                    body = json.dumps(metadata).encode()
                else:
                    filename = path.rsplit("/", 1)[-1]
                    if filename not in files:
                        self.send_error(404)
                        return
                    body = files[filename]
                    if state["tamper"]:
                        body = b"X" + body[1:]
                    requested_range = self.headers.get("Range")
                    if requested_range and not state["ignore_range"]:
                        start = int(requested_range.removeprefix("bytes=").removesuffix("-"))
                        content_range = f"bytes {start + int(state['bad_range'])}-{len(body) - 1}/{len(body)}"
                        body, status = body[start:], 206
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                if content_range:
                    self.send_header("Content-Range", content_range)
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def download(self):
        return download_http_files(
            repo=PUBLIC_REPO, revision=PUBLIC_REVISION, filenames=(PUBLIC_FILE, "README.md"),
            cache_dir=self.cache, endpoint=self.endpoint, no_proxy=True,
        )

    def partial(self, data):
        destination = self.cache / "http" / ("datasets--" + PUBLIC_REPO.replace("/", "--")) / PUBLIC_REVISION / PUBLIC_FILE
        destination.parent.mkdir(parents=True)
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(data)
        return destination, part

    def test_get_download_verifies_both_hash_types_and_writes_usable_manifest(self):
        config = load_config("configs/v1_expanded.yaml")
        config["data_generation"]["public"].update(cache_dir=str(self.cache), manifest=str(self.cache / "manifest.json"))
        result = download_public_data(config, endpoint=self.endpoint, transport="http", no_proxy=True)
        self.assertEqual(Path(result["path"]).read_bytes(), self.files[PUBLIC_FILE])
        self.assertEqual(result["download_transport"], "http")
        self.assertEqual(result["upstream_hashes"][PUBLIC_FILE]["algorithm"], "sha256")
        self.assertEqual(result["upstream_hashes"]["README.md"]["algorithm"], "git_blob_sha1")
        self.assertTrue((self.cache / "manifest.json").is_file())

    def test_partial_resumes_at_exact_offset_and_verified_cache_is_reused(self):
        destination, part = self.partial(self.files[PUBLIC_FILE][:37])
        self.download()
        self.assertEqual(destination.read_bytes(), self.files[PUBLIC_FILE])
        self.assertFalse(part.exists())
        file_requests = [item for item in self.state["requests"] if item[0].endswith(PUBLIC_FILE)]
        self.assertEqual(file_requests[0][1], "bytes=37-")
        self.download()
        self.assertEqual(len([item for item in self.state["requests"] if item[0].endswith(PUBLIC_FILE)]), 1)

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        destination, part = self.partial(self.files[PUBLIC_FILE][:37])
        self.state["ignore_range"] = True
        self.download()
        self.assertEqual(destination.read_bytes(), self.files[PUBLIC_FILE])
        self.assertFalse(part.exists())

    def test_bad_range_does_not_modify_existing_partial(self):
        destination, part = self.partial(self.files[PUBLIC_FILE][:37])
        self.state["bad_range"] = True
        with self.assertRaisesRegex(RuntimeError, "Content-Range"):
            self.download()
        self.assertEqual(part.read_bytes(), self.files[PUBLIC_FILE][:37])
        self.assertFalse(destination.exists())

    def test_tampered_bytes_are_not_published_and_next_run_can_recover(self):
        self.state["tamper"] = True
        with self.assertRaisesRegex(RuntimeError, "哈希"):
            self.download()
        root = self.cache / "http" / ("datasets--" + PUBLIC_REPO.replace("/", "--")) / PUBLIC_REVISION
        self.assertFalse((root / PUBLIC_FILE).exists())
        self.assertTrue((root / (PUBLIC_FILE + ".part")).exists())
        self.state["tamper"] = False
        paths, _ = self.download()
        self.assertEqual(paths[PUBLIC_FILE].read_bytes(), self.files[PUBLIC_FILE])

    def test_wrong_revision_is_rejected_before_fetching_files(self):
        self.metadata["sha"] = "wrong-revision"
        with self.assertRaisesRegex(ValueError, "版本"):
            self.download()
        self.assertFalse(any(path.endswith(PUBLIC_FILE) for path, _ in self.state["requests"]))


if __name__ == "__main__":
    unittest.main()
