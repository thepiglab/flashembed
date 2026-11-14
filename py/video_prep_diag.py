"""Where does a video's preprocessing differ between the card runtime and flashembed, after decoding?

flashembed resizes each decoded frame like an image at 140 soft tokens (the image processor's path). This script runs
the card runtime's video processor on the file and, on the very same decoded frames (torchcodec, as the processor
uses), the image processor at max_soft_tokens=140, and prints shapes and pixel differences per frame.

  python py/video_prep_diag.py SNAPSHOT video1 [video2 ...]
"""
import sys

import numpy as np


def main():
    snap, paths = sys.argv[1], sys.argv[2:]
    from transformers import AutoProcessor
    from torchcodec.decoders import VideoDecoder

    proc = AutoProcessor.from_pretrained(snap)
    vp, ip = proc.video_processor, proc.image_processor
    print("video processor:", {k: v for k, v in vp.to_dict().items() if not k.startswith("_")})
    for p in paths:
        out = vp(videos=[p], return_tensors="np")
        pv = out["pixel_values_videos"]
        keys = {k: getattr(v, "shape", v) for k, v in out.items()}
        d = VideoDecoder(p, seek_mode="exact", num_ffmpeg_threads=0)
        md = d.metadata
        n, fps, dur = md.num_frames, md.average_fps, md.duration_seconds
        idx = [min(n - 1, int(i * fps)) for i in range(max(1, int(dur)))]
        if len(idx) > 32:
            idx = [idx[i] for i in np.linspace(0, len(idx) - 1, 32, dtype=int)]
        sel = d.get_frames_at(indices=idx).data.numpy()  # [n, 3, H, W] uint8
        print(p.split("/")[-1], "video out", keys, "frames", sel.shape, "sampled", idx[:12])
        for k, f in enumerate(sel[:4]):
            im = ip(images=[f], max_soft_tokens=140, return_tensors="np")
            a = im["pixel_values"][0]
            b = pv[0][k] if pv.ndim >= 4 else pv[k]
            n = min(len(a), len(b))
            d = np.abs(a[:n].astype(np.float64) - b[:n].astype(np.float64))
            print(f"  frame {k}: image-proc {a.shape} video-proc {b.shape} max|d|*255={d.max() * 255:.2f} "
                  f"mean|d|*255={d.mean() * 255:.4f}")


if __name__ == "__main__":
    main()
