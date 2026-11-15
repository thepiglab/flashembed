"""Video preprocessing diagnosis: flashembed's ffprobe/ffmpeg path (src/media.rs, replicated here with the same
commands) vs the card runtime's torchcodec path (transformers read_video_torchcodec + EmbeddingGemma2VideoProcessor
.sample_frames: fps 1, max 32 frames, uniform overflow).

  python py/video_diag.py --ffmpeg FFMPEG --ffprobe FFPROBE video1 [video2 ...]

Per video: both metadata (frames, fps, duration), both sampled index lists, and per sampled frame the max / mean
absolute pixel difference between the two decoders (frames compared at the same index position).
"""
import argparse
import json
import subprocess

import numpy as np


def probe(ffprobe, path):
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_frames,avg_frame_rate,width,height,time_base:packet=pts,dts,duration,flags",
                          "-of", "json", path], capture_output=True, text=True).stdout
    v = json.loads(out)
    s = v["streams"][0]
    tn, td = (int(x) for x in s.get("time_base", "0/1").split("/"))
    secs = lambda x: x * tn / td
    pts, lo, hi = [], None, None
    for p in v.get("packets", []):
        ts = p.get("pts", p.get("dts"))
        if ts is None or (len(p.get("flags", "")) > 1 and p["flags"][1] == "D"):
            continue
        ts = int(ts)
        pts.append(ts)
        lo = ts if lo is None else min(lo, ts)
        hi = ts + int(p.get("duration", 0)) if hi is None else max(hi, ts + int(p.get("duration", 0)))
    pts.sort()
    d = secs(hi) - secs(lo)
    return {"frames": len(pts), "fps": len(pts) / d, "duration": d, "w": s["width"], "h": s["height"], "pts": pts,
            "tb": (tn, td)}


def sample(frames, fps, duration, max_frames=32):
    step = fps / 1.0
    n = max(1, int(duration * 1.0))
    idx = [min(frames - 1, int(i * step)) for i in range(n)]
    if len(idx) > max_frames:
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, max_frames, dtype=int)]
    return idx


def ours(ffmpeg, path, m, idx):
    want = sorted(set(m["pts"][i] for i in idx))
    prev = "if(isnan(prev_selected_pts),-1e18,prev_selected_pts)"
    expr = "+".join(f"gte(pts,{t})*lt({prev},{t})" for t in want)
    r = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", "-copyts", "-i", path, "-map", "0:v:0",
                        "-vf", f"select='{expr}',showinfo", "-fps_mode", "passthrough", "-sws_flags", "bilinear", "-f",
                        "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    pts = [int(l.split("pts:")[1].split()[0]) for l in r.stderr.decode(errors="replace").splitlines()
           if "showinfo" in l and " n:" in l]
    fsz = m["w"] * m["h"] * 3
    got = np.frombuffer(r.stdout[: len(r.stdout) // fsz * fsz], np.uint8).reshape(-1, m["h"], m["w"], 3)
    out = []
    for i in idx:
        k = next((j for j, p in enumerate(pts[: len(got)]) if p >= m["pts"][i]), None)
        out.append(None if k is None else got[k])
    return pts, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--ffprobe", required=True)
    ap.add_argument("videos", nargs="+")
    a = ap.parse_args()
    from torchcodec.decoders import VideoDecoder

    for path in a.videos:
        m = probe(a.ffprobe, path)
        i_ours = sample(m["frames"], m["fps"], m["duration"])
        d = VideoDecoder(path, seek_mode="exact", num_ffmpeg_threads=0)
        md = d.metadata
        i_hf = sample(md.num_frames, md.average_fps, md.duration_seconds)
        fb = d.get_frames_at(indices=i_hf)
        hf = fb.data.permute(0, 2, 3, 1).numpy()
        hf_pts = [round(float(x), 5) for x in fb.pts_seconds[:8]]
        our_pts = [round(m["pts"][i] * m["tb"][0] / m["tb"][1], 5) for i in i_ours[:8]]
        allf, fo = ours(a.ffmpeg, path, m, i_ours)
        diffs = []
        for k, (x, y) in enumerate(zip(fo, hf)):
            if x is None or x.shape != y.shape:
                diffs.append(None)
                continue
            dd = np.abs(x.astype(int) - y.astype(int))
            diffs.append((int(dd.max()), round(float(dd.mean()), 3)))
        where = None
        exact = {k: repr(getattr(md, k, None)) for k in ("num_frames", "average_fps", "duration_seconds",
                 "begin_stream_seconds", "end_stream_seconds", "num_frames_from_content", "num_frames_from_header",
                 "average_fps_from_header", "duration_seconds_from_header", "begin_stream_seconds_from_content",
                 "end_stream_seconds_from_content")}
        print(json.dumps({"video": path.split("/")[-1], "torchcodec_exact": exact, "ours_exact": repr(m["fps"]) + " " + repr(m["duration"]),
                          "ours_meta": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items() if k != "pts"},
                          "pts_first": m["pts"][:6], "hf_pts_s": hf_pts, "our_pts_s": our_pts,
                          "torchcodec_meta": {"frames": md.num_frames, "fps": round(md.average_fps, 6),
                                              "duration": round(md.duration_seconds, 6),
                                              "begin": md.begin_stream_seconds, "end": md.end_stream_seconds},
                          "ours_selected_pts": allf[:8],
                          "idx_same": i_ours == i_hf, "idx_ours": i_ours[:12], "idx_hf": i_hf[:12],
                          "pixel_diff_first": diffs[:8]}))


if __name__ == "__main__":
    main()
