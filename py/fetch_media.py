"""Download real PMC supplementary media (from a PigMed `real-media.jsonl` inventory) for GPU media benchmarks.

  python py/fetch_media.py --inventory real-media.jsonl --out DIR --kind video --n 100 --max-mb 40

Writes DIR/<kind>/<n>.<ext> and DIR/<kind>.jsonl rows {"id", "path", "kind", "source", "bytes"} (CC licences as in the
inventory); skips files over --max-mb (HEAD request first) and files that fail to download.
"""
import argparse
import json
from pathlib import Path

import requests


def https(url: str) -> str:
    if url.startswith("s3://"):
        bucket, _, key = url[5:].partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=["video", "audio"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-mb", type=float, default=40)
    a = ap.parse_args()
    d = a.out / a.kind
    d.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    rows = []
    for line in open(a.inventory):
        r = json.loads(line)
        if r.get("kind") != a.kind or not r.get("license", "").startswith("CC"):
            continue
        url = https(r["url"])
        try:
            h = s.head(url, timeout=30)
            size = int(h.headers.get("Content-Length", 0))
            if h.status_code != 200 or size == 0 or size > a.max_mb * 2**20:
                continue
            dest = d / f"{len(rows):04d}{r.get('ext', '')}"
            with s.get(url, timeout=300, stream=True) as g:
                g.raise_for_status()
                with open(dest, "wb") as f:
                    for c in g.iter_content(1 << 20):
                        f.write(c)
        except Exception:
            continue
        rows.append({"id": f"{a.kind}-{len(rows):04d}", "path": f"{a.kind}/{dest.name}", "kind": a.kind,
                     "source": f"{r.get('pmcid')} {r['url']} ({r.get('license')})", "bytes": size})
        if len(rows) >= a.n:
            break
    with open(a.out / f"{a.kind}.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)
    print(json.dumps({"kind": a.kind, "n": len(rows), "mb": round(sum(x["bytes"] for x in rows) / 2**20, 1)}))


if __name__ == "__main__":
    main()
