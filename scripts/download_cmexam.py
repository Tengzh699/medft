"""Download only pinned official data and retain source terms and attribution."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import requests
from medft.config import file_hash, write_json, utc_now
from scripts.prepare_cmexam import FILES, REVISION


def main():
    out = ROOT / "data/raw/cmexam"
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/williamliujl/CMExam/{REVISION}/"
    session = requests.Session()
    receipt = {"repository":"https://github.com/williamliujl/CMExam", "revision":REVISION,
               "retrieved_at":utc_now(), "files":{}}
    for name, expected in FILES.items():
        target = out/name
        url = base+"data/"+name
        if not target.exists():
            temporary = out/(name+".part")
            with session.get(url, stream=True, timeout=(20,120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk)
            if file_hash(temporary) != expected:
                raise ValueError("Source hash mismatch; .part preserved: "+name)
            temporary.replace(target)
        if file_hash(target) != expected:
            raise ValueError("Existing source changed: "+name)
        receipt['files'][name] = {"url":url, "sha256":expected, "bytes":target.stat().st_size}
        print(name+" verified", flush=True)
    for name in ['README.md', 'LICENSE']:
        target = out/name
        if not target.exists():
            response = session.get(base+name, timeout=(20,60))
            response.raise_for_status()
            target.write_bytes(response.content)
        receipt['files'][name] = {"url":base+name, "sha256":file_hash(target)}
    write_json(out/'download-receipt.json', receipt)


if __name__ == '__main__':
    main()
