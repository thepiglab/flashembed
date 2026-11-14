# Quantization: per-run tables

Generated with `flashembed qreport` from `slurm/qbench.sbatch` outputs (2026-11-13). Each block is one batch job: all of
its runtimes ran back to back on one node with the same 16 CPUs and inputs (999 PMC text chunks, 200 figures);
rep 0 over the full set, rep 1 re-timed the first 128-256 items. Summary and method: [QUANTIZATION.md](QUANTIZATION.md).

## Round 1: ONNX Runtime schemes, all CPUs

ORT dynamic / static / weight-only int8 and int4 vs fp32 flashembed and the model-card Python runtime.

**neomme image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 3.45 | 0.43x | 1.00x | 4281 | - | - | - | - |
| flashembed fp32 | 8.31 | 1.00x | 2.33x | 3427 | - | - | - | 1.00000 |
| flashembed dyn8 | 12.75 | 1.54x | 3.58x | 2978 | 0.83166 / 0.84045 / 0.922764 | 0.551 | 0.840 | 0.83166 |
| flashembed dyn8rr | 12.43 | 1.51x | 3.52x | 2983 | 0.85576 / 0.85718 / 0.928095 | 0.547 | 0.840 | 0.85576 |
| flashembed nb4-a4 | 5.80 | 0.70x | 1.63x | 2938 | 0.86115 / 0.88418 / 0.951277 | 0.666 | 0.915 | 0.86115 |
| flashembed nb8-a4 | 5.97 | 0.72x | 1.68x | 3018 | 0.99586 / 0.99837 / 0.999668 | 0.956 | 1.000 | 0.99586 |
| flashembed qdq8 | 12.49 | 1.51x | 3.52x | 2968 | 0.83728 / 0.84732 / 0.921720 | 0.528 | 0.810 | 0.83728 |
| flashembed qdq8-avg | 12.86 | 1.54x | 3.59x | 2968 | 0.85256 / 0.87074 / 0.936018 | 0.577 | 0.845 | 0.85256 |

**neomme text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 3.02 | 0.40x | 1.00x | 25295 | - | - | - | - |
| flashembed fp32 | 8.96 | 1.00x | 2.57x | 9924 | - | - | - | 1.00000 |
| flashembed dyn8 | 13.46 | 1.51x | 3.88x | 9614 | 0.91500 / 0.92721 / 0.967921 | 0.871 | 1.000 | 0.91500 |
| flashembed dyn8rr | 13.75 | 1.49x | 3.85x | 9612 | 0.92800 / 0.93907 / 0.976163 | 0.868 | 0.999 | 0.92800 |
| flashembed nb4-a4 | 6.16 | 0.70x | 1.78x | 9626 | 0.89870 / 0.94061 / 0.962197 | 0.718 | 0.990 | 0.89870 |
| flashembed nb8-a4 | 6.40 | 0.71x | 1.83x | 9732 | 0.99894 / 0.99967 / 0.999873 | 0.974 | 1.000 | 0.99894 |
| flashembed qdq8 | 13.25 | 1.53x | 3.91x | 9678 | 0.91518 / 0.92379 / 0.944095 | 0.858 | 0.999 | 0.91518 |
| flashembed qdq8-avg | 14.00 | 1.52x | 3.92x | 9675 | 0.96209 / 0.96888 / 0.979431 | 0.905 | 0.999 | 0.96209 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 5 (EPYC 9655P) | dyn8 | 199 | 0.307 / 0.276 | 0.411 / 0.365 | -0.046 [-0.086, -0.006] | 0.286 |
| neomme | Zen 5 (EPYC 9655P) | dyn8rr | 199 | 0.307 / 0.261 | 0.411 / 0.348 | -0.063 [-0.103, -0.022] | 0.246 |
| neomme | Zen 5 (EPYC 9655P) | nb4-a4 | 199 | 0.307 / 0.211 | 0.411 / 0.311 | -0.099 [-0.138, -0.061] | 0.221 |
| neomme | Zen 5 (EPYC 9655P) | nb8-a4 | 199 | 0.307 / 0.317 | 0.411 / 0.414 | +0.003 [-0.004, +0.012] | 0.598 |
| neomme | Zen 5 (EPYC 9655P) | qdq8 | 199 | 0.307 / 0.261 | 0.411 / 0.356 | -0.054 [-0.096, -0.013] | 0.216 |
| neomme | Zen 5 (EPYC 9655P) | qdq8-avg | 199 | 0.307 / 0.251 | 0.411 / 0.354 | -0.056 [-0.094, -0.019] | 0.256 |

**gemma image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.79 | 0.47x | 1.00x | 3932 | - | - | - | - |
| flashembed fp32 | 1.66 | 1.00x | 2.12x | 2600 | - | - | - | 1.00000 |
| flashembed dyn8 | 2.14 | 1.30x | 2.75x | 1815 | 0.81094 / 0.84666 / 0.933918 | 0.469 | 0.800 | 0.81094 |
| flashembed dyn8-novis | 1.71 | 1.02x | 2.17x | 2248 | 0.94613 / 0.95115 / 0.968067 | 0.739 | 0.980 | 0.94613 |
| flashembed dyn8rr | 2.16 | 1.29x | 2.74x | 1800 | 0.78558 / 0.83737 / 0.930818 | 0.477 | 0.790 | 0.78558 |
| flashembed nb4-a4 | 1.18 | 0.71x | 1.50x | 1758 | 0.94537 / 0.94817 / 0.972658 | 0.752 | 0.990 | 0.94537 |
| flashembed nb8-a4 | 1.21 | 0.73x | 1.54x | 1900 | 0.99745 / 0.99817 / 0.999749 | 0.965 | 1.000 | 0.99744 |
| flashembed qdq8 | 2.19 | 1.33x | 2.81x | 1862 | 0.83620 / 0.84953 / 0.929117 | 0.506 | 0.800 | 0.83620 |
| flashembed qdq8-avg | 2.22 | 1.33x | 2.82x | 1861 | 0.87279 / 0.89404 / 0.965464 | 0.603 | 0.885 | 0.87279 |

**gemma text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 2.05 | 0.28x | 1.00x | 7197 | - | - | - | - |
| flashembed fp32 | 9.32 | 1.00x | 3.71x | 5612 | - | - | - | 1.00000 |
| flashembed dyn8 | 11.08 | 1.19x | 4.41x | 5282 | 0.92967 / 0.94542 / 0.975218 | 0.601 | 0.935 | 0.92967 |
| flashembed dyn8-novis | 10.81 | 1.16x | 4.29x | 5308 | 0.92967 / 0.94542 / 0.975218 | 0.601 | 0.935 | 0.92967 |
| flashembed dyn8rr | 10.73 | 1.16x | 4.31x | 5274 | 0.93532 / 0.94721 / 0.974472 | 0.601 | 0.939 | 0.93532 |
| flashembed nb4-a4 | 6.01 | 0.64x | 2.37x | 5287 | 0.93537 / 0.95670 / 0.974628 | 0.782 | 0.992 | 0.93537 |
| flashembed nb8-a4 | 6.13 | 0.65x | 2.41x | 5409 | 0.99843 / 0.99959 / 0.999829 | 0.972 | 1.000 | 0.99843 |
| flashembed qdq8 | 10.74 | 1.19x | 4.38x | 5217 | 0.90187 / 0.92559 / 0.948828 | 0.573 | 0.919 | 0.90187 |
| flashembed qdq8-avg | 10.38 | 1.16x | 4.27x | 5214 | 0.94235 / 0.96620 / 0.977880 | 0.751 | 0.990 | 0.94235 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 5 (EPYC 9655P) | dyn8 | 199 | 0.764 / 0.804 | 0.835 / 0.864 | +0.029 [-0.014, +0.073] | 0.688 |
| gemma | Zen 5 (EPYC 9655P) | dyn8-novis | 199 | 0.764 / 0.724 | 0.835 / 0.801 | -0.034 [-0.059, -0.011] | 0.759 |
| gemma | Zen 5 (EPYC 9655P) | dyn8rr | 199 | 0.764 / 0.804 | 0.835 / 0.856 | +0.021 [-0.022, +0.067] | 0.714 |
| gemma | Zen 5 (EPYC 9655P) | nb4-a4 | 199 | 0.764 / 0.814 | 0.835 / 0.869 | +0.035 [+0.010, +0.060] | 0.794 |
| gemma | Zen 5 (EPYC 9655P) | nb8-a4 | 199 | 0.764 / 0.769 | 0.835 / 0.841 | +0.006 [-0.005, +0.019] | 0.910 |
| gemma | Zen 5 (EPYC 9655P) | qdq8 | 199 | 0.764 / 0.784 | 0.835 / 0.841 | +0.006 [-0.032, +0.044] | 0.693 |
| gemma | Zen 5 (EPYC 9655P) | qdq8-avg | 199 | 0.764 / 0.804 | 0.835 / 0.847 | +0.012 [-0.026, +0.048] | 0.724 |

**neomme image, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 2.58 | 0.77x | 1.00x | 4205 | - | - | - | - |
| flashembed fp32 | 3.66 | 1.00x | 1.30x | 3421 | - | - | - | 1.00000 |
| flashembed dyn8 | 5.10 | 1.38x | 1.80x | 2972 | 0.23866 / 0.25119 / 0.712992 | 0.160 | 0.175 | 0.23866 |
| flashembed dyn8rr | 4.74 | 1.33x | 1.73x | 2960 | 0.86223 / 0.87001 / 0.926873 | 0.552 | 0.840 | 0.86223 |
| flashembed nb4-a4 | 2.68 | 0.73x | 0.95x | 2938 | 0.86299 / 0.88357 / 0.951589 | 0.664 | 0.915 | 0.86299 |
| flashembed nb8-a4 | 2.22 | 0.62x | 0.80x | 3016 | 0.99679 / 0.99838 / 0.999683 | 0.955 | 1.000 | 0.99679 |
| flashembed qdq8 | 5.06 | 1.41x | 1.83x | 2965 | 0.75805 / 0.77929 / 0.874058 | 0.439 | 0.635 | 0.75805 |
| flashembed qdq8-avg | 5.03 | 1.43x | 1.86x | 2940 | 0.33677 / 0.53615 / 0.664518 | 0.091 | 0.115 | 0.33677 |

**neomme text, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 2.40 | 0.66x | 1.00x | 25269 | - | - | - | - |
| flashembed fp32 | 4.26 | 1.00x | 1.55x | 9923 | - | - | - | 1.00000 |
| flashembed dyn8 | 5.69 | 1.34x | 2.09x | 9610 | 0.90289 / 0.92115 / 0.964952 | 0.841 | 0.999 | 0.90289 |
| flashembed dyn8rr | 5.80 | 1.35x | 2.10x | 9598 | 0.92747 / 0.93966 / 0.976348 | 0.864 | 0.999 | 0.92747 |
| flashembed nb4-a4 | 2.93 | 0.68x | 1.05x | 9639 | 0.89896 / 0.94059 / 0.962097 | 0.721 | 0.990 | 0.89896 |
| flashembed nb8-a4 | 2.47 | 0.58x | 0.90x | 9730 | 0.99931 / 0.99962 / 0.999872 | 0.975 | 1.000 | 0.99931 |
| flashembed qdq8 | 6.03 | 1.42x | 2.21x | 9677 | 0.90542 / 0.91520 / 0.939321 | 0.822 | 0.999 | 0.90542 |
| flashembed qdq8-avg | 5.95 | 1.40x | 2.17x | 9664 | 0.93898 / 0.95853 / 0.971751 | 0.810 | 0.997 | 0.93898 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 3 (EPYC 75F3) | dyn8 | 199 | 0.307 / 0.146 | 0.411 / 0.199 | -0.212 [-0.271, -0.153] | 0.101 |
| neomme | Zen 3 (EPYC 75F3) | dyn8rr | 199 | 0.307 / 0.256 | 0.411 / 0.347 | -0.064 [-0.105, -0.022] | 0.246 |
| neomme | Zen 3 (EPYC 75F3) | nb4-a4 | 199 | 0.307 / 0.206 | 0.411 / 0.310 | -0.101 [-0.138, -0.063] | 0.226 |
| neomme | Zen 3 (EPYC 75F3) | nb8-a4 | 199 | 0.307 / 0.312 | 0.411 / 0.413 | +0.002 [-0.004, +0.009] | 0.598 |
| neomme | Zen 3 (EPYC 75F3) | qdq8 | 199 | 0.307 / 0.246 | 0.411 / 0.331 | -0.080 [-0.132, -0.030] | 0.201 |
| neomme | Zen 3 (EPYC 75F3) | qdq8-avg | 199 | 0.307 / 0.020 | 0.411 / 0.051 | -0.360 [-0.421, -0.301] | 0.015 |

**gemma image, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.61 | 0.73x | 1.00x | 3878 | - | - | - | - |
| flashembed fp32 | 0.86 | 1.00x | 1.37x | 2616 | - | - | - | 1.00000 |
| flashembed dyn8 | 1.09 | 1.27x | 1.74x | 1796 | 0.57346 / 0.58309 / 0.663656 | 0.129 | 0.215 | 0.57346 |
| flashembed dyn8-novis | 0.88 | 1.02x | 1.39x | 2196 | 0.53201 / 0.56113 / 0.671451 | 0.186 | 0.270 | 0.53201 |
| flashembed dyn8rr | 1.10 | 1.27x | 1.73x | 1785 | 0.78121 / 0.83577 / 0.933092 | 0.475 | 0.790 | 0.78121 |
| flashembed nb4-a4 | 0.67 | 0.77x | 1.06x | 1772 | 0.94575 / 0.94884 / 0.972633 | 0.756 | 0.990 | 0.94575 |
| flashembed nb8-a4 | 0.58 | 0.66x | 0.90x | 1908 | 0.99766 / 0.99797 / 0.999746 | 0.959 | 1.000 | 0.99766 |
| flashembed qdq8 | 1.11 | 1.29x | 1.76x | 1868 | 0.50893 / 0.53622 / 0.604864 | 0.369 | 0.635 | 0.50894 |
| flashembed qdq8-avg | 1.13 | 1.29x | 1.76x | 1851 | 0.53543 / 0.54961 / 0.621358 | 0.382 | 0.590 | 0.53544 |

**gemma text, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 1.69 | 0.45x | 1.00x | 7080 | - | - | - | - |
| flashembed fp32 | 5.04 | 1.00x | 2.38x | 5599 | - | - | - | 1.00000 |
| flashembed dyn8 | 5.96 | 1.20x | 2.84x | 5261 | 0.43865 / 0.47128 / 0.603355 | 0.213 | 0.341 | 0.43865 |
| flashembed dyn8-novis | 6.11 | 1.20x | 2.87x | 5278 | 0.43865 / 0.47128 / 0.603355 | 0.213 | 0.341 | 0.43865 |
| flashembed dyn8rr | 6.02 | 1.22x | 2.89x | 5354 | 0.93475 / 0.94642 / 0.974812 | 0.596 | 0.942 | 0.93475 |
| flashembed nb4-a4 | 3.51 | 0.70x | 1.67x | 5292 | 0.93612 / 0.95686 / 0.974656 | 0.782 | 0.993 | 0.93612 |
| flashembed nb8-a4 | 3.11 | 0.61x | 1.47x | 5350 | 0.99598 / 0.99954 / 0.999830 | 0.972 | 1.000 | 0.99598 |
| flashembed qdq8 | 6.12 | 1.23x | 2.93x | 5212 | 0.44380 / 0.47571 / 0.584017 | 0.416 | 0.691 | 0.44380 |
| flashembed qdq8-avg | 6.21 | 1.22x | 2.92x | 5204 | 0.47955 / 0.51448 / 0.615166 | 0.417 | 0.692 | 0.47955 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 3 (EPYC 75F3) | dyn8 | 199 | 0.764 / 0.065 | 0.835 / 0.117 | -0.718 [-0.768, -0.669] | 0.070 |
| gemma | Zen 3 (EPYC 75F3) | dyn8-novis | 199 | 0.764 / 0.030 | 0.835 / 0.072 | -0.763 [-0.809, -0.714] | 0.035 |
| gemma | Zen 3 (EPYC 75F3) | dyn8rr | 199 | 0.764 / 0.794 | 0.835 / 0.851 | +0.016 [-0.026, +0.058] | 0.704 |
| gemma | Zen 3 (EPYC 75F3) | nb4-a4 | 199 | 0.764 / 0.814 | 0.835 / 0.869 | +0.035 [+0.011, +0.060] | 0.794 |
| gemma | Zen 3 (EPYC 75F3) | nb8-a4 | 199 | 0.764 / 0.759 | 0.835 / 0.834 | -0.001 [-0.010, +0.008] | 0.899 |
| gemma | Zen 3 (EPYC 75F3) | qdq8 | 199 | 0.764 / 0.633 | 0.835 / 0.701 | -0.134 [-0.194, -0.078] | 0.553 |
| gemma | Zen 3 (EPYC 75F3) | qdq8-avg | 199 | 0.764 / 0.457 | 0.835 / 0.548 | -0.287 [-0.345, -0.229] | 0.427 |

**neomme image, Zen 2 (EPYC 7702)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 1.61 | 0.60x | 1.00x | 4189 | - | - | - | - |
| flashembed fp32 | 2.65 | 1.00x | 1.67x | 3424 | - | - | - | 1.00000 |
| flashembed dyn8 | 2.96 | 1.13x | 1.88x | 2945 | 0.23866 / 0.25119 / 0.712992 | 0.160 | 0.175 | 0.23866 |
| flashembed dyn8rr | 3.00 | 1.13x | 1.88x | 2955 | 0.86223 / 0.87001 / 0.926873 | 0.552 | 0.840 | 0.86223 |
| flashembed nb4-a4 | 2.00 | 0.76x | 1.27x | 2933 | 0.86299 / 0.88357 / 0.951589 | 0.664 | 0.915 | 0.86299 |
| flashembed nb8-a4 | 1.44 | 0.55x | 0.92x | 3033 | 0.99679 / 0.99838 / 0.999683 | 0.955 | 1.000 | 0.99679 |
| flashembed qdq8 | 2.94 | 1.13x | 1.89x | 2957 | 0.75805 / 0.77929 / 0.874058 | 0.439 | 0.635 | 0.75805 |
| flashembed qdq8-avg | 3.04 | 1.16x | 1.93x | 2937 | 0.33677 / 0.53615 / 0.664518 | 0.091 | 0.115 | 0.33677 |

**neomme text, Zen 2 (EPYC 7702)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 1.40 | 0.53x | 1.00x | 25328 | - | - | - | - |
| flashembed fp32 | 2.99 | 1.00x | 1.90x | 9917 | - | - | - | 1.00000 |
| flashembed dyn8 | 3.42 | 1.12x | 2.14x | 9613 | 0.90289 / 0.92115 / 0.964952 | 0.841 | 0.999 | 0.90289 |
| flashembed dyn8rr | 3.44 | 1.13x | 2.15x | 9600 | 0.92747 / 0.93966 / 0.976348 | 0.864 | 0.999 | 0.92747 |
| flashembed nb4-a4 | 2.17 | 0.74x | 1.40x | 9636 | 0.89896 / 0.94059 / 0.962097 | 0.721 | 0.990 | 0.89896 |
| flashembed nb8-a4 | 1.40 | 0.50x | 0.94x | 9714 | 0.99931 / 0.99962 / 0.999872 | 0.975 | 1.000 | 0.99931 |
| flashembed qdq8 | 2.85 | 1.04x | 1.95x | 9670 | 0.90542 / 0.91520 / 0.939321 | 0.822 | 0.999 | 0.90542 |
| flashembed qdq8-avg | 2.98 | 1.06x | 2.00x | 9675 | 0.93898 / 0.95853 / 0.971751 | 0.810 | 0.997 | 0.93898 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 2 (EPYC 7702) | dyn8 | 199 | 0.307 / 0.146 | 0.411 / 0.199 | -0.212 [-0.271, -0.153] | 0.101 |
| neomme | Zen 2 (EPYC 7702) | dyn8rr | 199 | 0.307 / 0.256 | 0.411 / 0.347 | -0.064 [-0.105, -0.022] | 0.246 |
| neomme | Zen 2 (EPYC 7702) | nb4-a4 | 199 | 0.307 / 0.206 | 0.411 / 0.310 | -0.101 [-0.138, -0.063] | 0.226 |
| neomme | Zen 2 (EPYC 7702) | nb8-a4 | 199 | 0.307 / 0.312 | 0.411 / 0.413 | +0.002 [-0.004, +0.009] | 0.598 |
| neomme | Zen 2 (EPYC 7702) | qdq8 | 199 | 0.307 / 0.246 | 0.411 / 0.331 | -0.080 [-0.132, -0.030] | 0.201 |
| neomme | Zen 2 (EPYC 7702) | qdq8-avg | 199 | 0.307 / 0.020 | 0.411 / 0.051 | -0.360 [-0.421, -0.301] | 0.015 |

**gemma image, Zen 2 (EPYC 7702)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.34 | 0.57x | 1.00x | 3914 | - | - | - | - |
| flashembed fp32 | 0.62 | 1.00x | 1.75x | 2617 | - | - | - | 1.00000 |
| flashembed dyn8 | 0.63 | 1.04x | 1.81x | 1776 | 0.57346 / 0.58309 / 0.663656 | 0.129 | 0.215 | 0.57346 |
| flashembed dyn8-novis | 0.60 | 1.01x | 1.76x | 2218 | 0.53201 / 0.56113 / 0.671451 | 0.186 | 0.270 | 0.53201 |
| flashembed dyn8rr | 0.64 | 1.05x | 1.83x | 1787 | 0.78121 / 0.83577 / 0.933092 | 0.475 | 0.790 | 0.78121 |
| flashembed nb4-a4 | 0.45 | 0.72x | 1.26x | 1766 | 0.94575 / 0.94884 / 0.972633 | 0.756 | 0.990 | 0.94575 |
| flashembed nb8-a4 | 0.34 | 0.57x | 0.99x | 1913 | 0.99766 / 0.99797 / 0.999746 | 0.959 | 1.000 | 0.99766 |
| flashembed qdq8 | 0.65 | 1.06x | 1.85x | 1856 | 0.50893 / 0.53622 / 0.604864 | 0.369 | 0.635 | 0.50894 |
| flashembed qdq8-avg | 0.67 | 1.09x | 1.90x | 1866 | 0.53543 / 0.54961 / 0.621358 | 0.382 | 0.590 | 0.53544 |

**gemma text, Zen 2 (EPYC 7702)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.92 | 0.39x | 1.00x | 7085 | - | - | - | - |
| flashembed fp32 | 3.07 | 1.00x | 2.72x | 5668 | - | - | - | 1.00000 |
| flashembed dyn8 | 3.31 | 1.07x | 2.92x | 5305 | 0.43865 / 0.47128 / 0.603355 | 0.213 | 0.341 | 0.43865 |
| flashembed dyn8-novis | 3.38 | 1.10x | 2.99x | 5298 | 0.43865 / 0.47128 / 0.603355 | 0.213 | 0.341 | 0.43865 |
| flashembed dyn8rr | 3.31 | 1.08x | 2.93x | 5268 | 0.93475 / 0.94642 / 0.974812 | 0.596 | 0.942 | 0.93475 |
| flashembed nb4-a4 | 2.16 | 0.69x | 1.89x | 5319 | 0.93612 / 0.95686 / 0.974656 | 0.782 | 0.993 | 0.93612 |
| flashembed nb8-a4 | 1.72 | 0.56x | 1.52x | 5345 | 0.99598 / 0.99954 / 0.999830 | 0.972 | 1.000 | 0.99598 |
| flashembed qdq8 | 3.26 | 1.08x | 2.92x | 5220 | 0.44380 / 0.47571 / 0.584017 | 0.416 | 0.691 | 0.44380 |
| flashembed qdq8-avg | 3.37 | 1.10x | 2.99x | 5222 | 0.47955 / 0.51448 / 0.615166 | 0.417 | 0.692 | 0.47955 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 2 (EPYC 7702) | dyn8 | 199 | 0.764 / 0.065 | 0.835 / 0.117 | -0.718 [-0.768, -0.669] | 0.070 |
| gemma | Zen 2 (EPYC 7702) | dyn8-novis | 199 | 0.764 / 0.030 | 0.835 / 0.072 | -0.763 [-0.809, -0.714] | 0.035 |
| gemma | Zen 2 (EPYC 7702) | dyn8rr | 199 | 0.764 / 0.794 | 0.835 / 0.851 | +0.016 [-0.026, +0.058] | 0.704 |
| gemma | Zen 2 (EPYC 7702) | nb4-a4 | 199 | 0.764 / 0.814 | 0.835 / 0.869 | +0.035 [+0.011, +0.060] | 0.794 |
| gemma | Zen 2 (EPYC 7702) | nb8-a4 | 199 | 0.764 / 0.759 | 0.835 / 0.834 | -0.001 [-0.010, +0.008] | 0.899 |
| gemma | Zen 2 (EPYC 7702) | qdq8 | 199 | 0.764 / 0.633 | 0.835 / 0.701 | -0.134 [-0.194, -0.078] | 0.553 |
| gemma | Zen 2 (EPYC 7702) | qdq8-avg | 199 | 0.764 / 0.457 | 0.835 / 0.548 | -0.287 [-0.345, -0.229] | 0.427 |

## Layer sensitivity (Zen 5)

`dyn8` with one class of Linear layers kept fp32 (`-kdown` = MLP down_proj, `-kgu` = gate/up, `-ko` = attention output, `-kqkv`, `-mlp` = only MLPs int8, `-sel` = embeddings / heads / first and last layer fp32).

**neomme image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 8.08 | 1.00x | - | 3419 | - | - | - | - |
| flashembed dyn8-kdown | 10.07 | 1.25x | - | 3050 | 0.96836 / 0.98623 / 0.997771 | 0.877 | 1.000 | - |
| flashembed dyn8-kgu | 9.92 | 1.23x | - | 3178 | 0.84471 / 0.86298 / 0.925722 | 0.554 | 0.845 | - |
| flashembed dyn8-ko | 10.83 | 1.34x | - | 3043 | 0.83971 / 0.85381 / 0.930210 | 0.559 | 0.815 | - |
| flashembed dyn8-kqkv | 10.98 | 1.36x | - | 3042 | 0.84217 / 0.85062 / 0.923236 | 0.532 | 0.815 | - |
| flashembed dyn8-mlp | 10.15 | 1.26x | - | 3147 | 0.83285 / 0.85907 / 0.928048 | 0.580 | 0.835 | - |
| flashembed dyn8-sel | 11.08 | 1.37x | - | 2952 | 0.83631 / 0.85114 / 0.919107 | 0.563 | 0.810 | - |

**neomme text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 8.08 | 1.00x | - | 9924 | - | - | - | - |
| flashembed dyn8-kdown | 10.62 | 1.31x | - | 9632 | 0.99413 / 0.99838 / 0.999532 | 0.947 | 1.000 | - |
| flashembed dyn8-kgu | 10.45 | 1.29x | - | 9840 | 0.91793 / 0.93029 / 0.968812 | 0.871 | 0.999 | - |
| flashembed dyn8-ko | 11.02 | 1.36x | - | 9698 | 0.91923 / 0.92856 / 0.970012 | 0.877 | 0.999 | - |
| flashembed dyn8-kqkv | 11.76 | 1.46x | - | 9693 | 0.91451 / 0.92907 / 0.968735 | 0.873 | 1.000 | - |
| flashembed dyn8-mlp | 10.49 | 1.30x | - | 9794 | 0.91655 / 0.93260 / 0.969657 | 0.880 | 1.000 | - |
| flashembed dyn8-sel | 11.73 | 1.45x | - | 9515 | 0.90658 / 0.91626 / 0.962711 | 0.875 | 1.000 | - |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 5 (EPYC 9655P) | dyn8-kdown | 199 | 0.307 / 0.312 | 0.411 / 0.412 | +0.001 [-0.013, +0.017] | 0.452 |
| neomme | Zen 5 (EPYC 9655P) | dyn8-kgu | 199 | 0.307 / 0.286 | 0.411 / 0.370 | -0.041 [-0.078, -0.002] | 0.251 |
| neomme | Zen 5 (EPYC 9655P) | dyn8-ko | 199 | 0.307 / 0.276 | 0.411 / 0.368 | -0.043 [-0.081, -0.003] | 0.246 |
| neomme | Zen 5 (EPYC 9655P) | dyn8-kqkv | 199 | 0.307 / 0.261 | 0.411 / 0.357 | -0.054 [-0.092, -0.016] | 0.266 |
| neomme | Zen 5 (EPYC 9655P) | dyn8-mlp | 199 | 0.307 / 0.266 | 0.411 / 0.364 | -0.047 [-0.082, -0.013] | 0.281 |
| neomme | Zen 5 (EPYC 9655P) | dyn8-sel | 199 | 0.307 / 0.266 | 0.411 / 0.364 | -0.047 [-0.083, -0.010] | 0.261 |

**gemma image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 1.39 | 1.00x | - | 2601 | - | - | - | - |
| flashembed dyn8-kdown | 1.73 | 1.24x | - | 1946 | 0.96626 / 0.97253 / 0.994983 | 0.802 | 0.995 | - |
| flashembed dyn8-kgu | 1.57 | 1.13x | - | 2161 | 0.83514 / 0.84165 / 0.933089 | 0.486 | 0.790 | - |
| flashembed dyn8-ko | 1.81 | 1.30x | - | 1866 | 0.83527 / 0.83865 / 0.932943 | 0.472 | 0.795 | - |
| flashembed dyn8-kqkv | 1.70 | 1.22x | - | 1957 | 0.80473 / 0.85539 / 0.935839 | 0.484 | 0.790 | - |
| flashembed dyn8-mlp | 1.69 | 1.21x | - | 2090 | 0.79800 / 0.84519 / 0.936474 | 0.492 | 0.795 | - |
| flashembed dyn8-sel | 1.75 | 1.26x | - | 1887 | 0.85444 / 0.87581 / 0.952574 | 0.535 | 0.875 | - |

**gemma text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 7.59 | 1.00x | - | 5605 | - | - | - | - |
| flashembed dyn8-kdown | 9.40 | 1.24x | - | 5356 | 0.97280 / 0.98832 / 0.996760 | 0.853 | 1.000 | - |
| flashembed dyn8-kgu | 8.94 | 1.18x | - | 5432 | 0.93512 / 0.94728 / 0.975319 | 0.612 | 0.953 | - |
| flashembed dyn8-ko | 9.55 | 1.26x | - | 5307 | 0.94072 / 0.94646 / 0.975435 | 0.601 | 0.947 | - |
| flashembed dyn8-kqkv | 9.41 | 1.24x | - | 5357 | 0.93667 / 0.94647 / 0.975919 | 0.611 | 0.953 | - |
| flashembed dyn8-mlp | 8.87 | 1.17x | - | 5460 | 0.94313 / 0.95562 / 0.978670 | 0.710 | 0.988 | - |
| flashembed dyn8-sel | 9.64 | 1.27x | - | 5310 | 0.95196 / 0.96097 / 0.982294 | 0.760 | 0.996 | - |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 5 (EPYC 9655P) | dyn8-kdown | 199 | 0.764 / 0.824 | 0.835 / 0.879 | +0.045 [+0.018, +0.072] | 0.764 |
| gemma | Zen 5 (EPYC 9655P) | dyn8-kgu | 199 | 0.764 / 0.794 | 0.835 / 0.857 | +0.022 [-0.021, +0.065] | 0.688 |
| gemma | Zen 5 (EPYC 9655P) | dyn8-ko | 199 | 0.764 / 0.829 | 0.835 / 0.872 | +0.038 [-0.003, +0.078] | 0.724 |
| gemma | Zen 5 (EPYC 9655P) | dyn8-kqkv | 199 | 0.764 / 0.799 | 0.835 / 0.859 | +0.024 [-0.015, +0.064] | 0.709 |
| gemma | Zen 5 (EPYC 9655P) | dyn8-mlp | 199 | 0.764 / 0.789 | 0.835 / 0.858 | +0.023 [-0.015, +0.063] | 0.693 |
| gemma | Zen 5 (EPYC 9655P) | dyn8-sel | 199 | 0.764 / 0.774 | 0.835 / 0.846 | +0.011 [-0.031, +0.055] | 0.673 |

## Rust QMatMul (q8blk), final kernel

`q8blk` = K-block 64, `q8blk-b32` = K-block 32, `-f` = same-input MatMuls fused (QMatMul + Split).

**neomme image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 7.80 | 1.00x | - | 3434 | - | - | - | - |
| flashembed q8blk | 12.90 | 1.61x | - | 2995 | 0.99679 / 0.99742 / 0.999543 | 0.947 | 1.000 | - |
| flashembed q8blk-b32 | 12.22 | 1.56x | - | 3019 | 0.99658 / 0.99821 / 0.999677 | 0.949 | 1.000 | - |

**neomme text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 8.14 | 1.00x | - | 9922 | - | - | - | - |
| flashembed q8blk | 12.85 | 1.60x | - | 9642 | 0.99839 / 0.99958 / 0.999833 | 0.971 | 1.000 | - |
| flashembed q8blk-b32 | 12.59 | 1.57x | - | 9704 | 0.99951 / 0.99968 / 0.999873 | 0.976 | 1.000 | - |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 5 (EPYC 9655P) | q8blk | 199 | 0.307 / 0.296 | 0.411 / 0.403 | -0.008 [-0.015, -0.001] | 0.538 |
| neomme | Zen 5 (EPYC 9655P) | q8blk-b32 | 199 | 0.307 / 0.307 | 0.411 / 0.411 | -0.000 [-0.008, +0.008] | 0.588 |

**gemma image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 1.81 | 1.00x | - | 2620 | - | - | - | - |
| flashembed q8blk | 2.38 | 1.31x | - | 1912 | 0.99531 / 0.99681 / 0.999643 | 0.945 | 1.000 | - |
| flashembed q8blk-b32 | 2.34 | 1.28x | - | 1933 | 0.99770 / 0.99811 / 0.999758 | 0.963 | 1.000 | - |

**gemma text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 10.29 | 1.00x | - | 5598 | - | - | - | - |
| flashembed q8blk | 12.68 | 1.24x | - | 5373 | 0.99337 / 0.99935 / 0.999751 | 0.968 | 1.000 | - |
| flashembed q8blk-b32 | 12.11 | 1.21x | - | 5407 | 0.99811 / 0.99957 / 0.999829 | 0.974 | 1.000 | - |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 5 (EPYC 9655P) | q8blk | 199 | 0.764 / 0.759 | 0.835 / 0.833 | -0.001 [-0.013, +0.011] | 0.884 |
| gemma | Zen 5 (EPYC 9655P) | q8blk-b32 | 199 | 0.764 / 0.764 | 0.835 / 0.836 | +0.001 [-0.009, +0.012] | 0.915 |

**neomme image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 8.64 | 1.00x | - | 3439 | - | - | - | - |
| flashembed q8blk | 13.80 | 1.59x | - | 2990 | 0.99679 / 0.99742 / 0.999543 | 0.947 | 1.000 | - |
| flashembed q8blk-b32-f | 12.86 | 1.48x | - | 3104 | 0.99658 / 0.99821 / 0.999677 | 0.949 | 1.000 | - |
| flashembed q8blk-f | 13.24 | 1.53x | - | 3101 | 0.99679 / 0.99742 / 0.999543 | 0.947 | 1.000 | - |

**neomme text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 9.04 | 1.00x | - | 9920 | - | - | - | - |
| flashembed q8blk | 13.98 | 1.59x | - | 9633 | 0.99839 / 0.99958 / 0.999833 | 0.971 | 1.000 | - |
| flashembed q8blk-b32-f | 13.24 | 1.49x | - | 9970 | 0.99951 / 0.99968 / 0.999873 | 0.976 | 1.000 | - |
| flashembed q8blk-f | 13.77 | 1.52x | - | 9921 | 0.99839 / 0.99958 / 0.999833 | 0.971 | 1.000 | - |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 5 (EPYC 9655P) | q8blk | 199 | 0.307 / 0.296 | 0.411 / 0.403 | -0.008 [-0.015, -0.001] | 0.538 |
| neomme | Zen 5 (EPYC 9655P) | q8blk-b32-f | 199 | 0.307 / 0.307 | 0.411 / 0.411 | -0.000 [-0.008, +0.008] | 0.588 |
| neomme | Zen 5 (EPYC 9655P) | q8blk-f | 199 | 0.307 / 0.296 | 0.411 / 0.403 | -0.008 [-0.015, -0.001] | 0.538 |

**gemma image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 1.28 | 1.00x | - | 2611 | - | - | - | - |
| flashembed q8blk | 1.74 | 1.33x | - | 1892 | 0.99531 / 0.99681 / 0.999643 | 0.945 | 1.000 | - |
| flashembed q8blk-b32-f | 1.30 | 1.02x | - | 2159 | 0.99770 / 0.99811 / 0.999758 | 0.963 | 1.000 | - |
| flashembed q8blk-f | 1.34 | 1.01x | - | 2139 | 0.99531 / 0.99681 / 0.999643 | 0.945 | 1.000 | - |

**gemma text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| flashembed fp32 | 7.03 | 1.00x | - | 5609 | - | - | - | - |
| flashembed q8blk | 9.04 | 1.35x | - | 5361 | 0.99337 / 0.99935 / 0.999751 | 0.968 | 1.000 | - |
| flashembed q8blk-b32-f | 7.32 | 1.13x | - | 5796 | 0.99811 / 0.99957 / 0.999829 | 0.974 | 1.000 | - |
| flashembed q8blk-f | 7.83 | 1.07x | - | 5767 | 0.99337 / 0.99935 / 0.999751 | 0.968 | 1.000 | - |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 5 (EPYC 9655P) | q8blk | 199 | 0.764 / 0.759 | 0.835 / 0.833 | -0.001 [-0.013, +0.011] | 0.884 |
| gemma | Zen 5 (EPYC 9655P) | q8blk-b32-f | 199 | 0.764 / 0.764 | 0.835 / 0.836 | +0.001 [-0.009, +0.012] | 0.915 |
| gemma | Zen 5 (EPYC 9655P) | q8blk-f | 199 | 0.764 / 0.759 | 0.835 / 0.833 | -0.001 [-0.013, +0.011] | 0.884 |

**neomme image, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 2.14 | 0.71x | 1.00x | 4056 | - | - | - | - |
| flashembed fp32 | 3.56 | 1.00x | 1.44x | 3443 | - | - | - | 1.00000 |
| flashembed dyn8rr | 4.94 | 1.45x | 2.07x | 2938 | 0.86223 / 0.87001 / 0.926873 | 0.552 | 0.840 | 0.86223 |
| flashembed q8blk | 5.32 | 1.57x | 2.25x | 2970 | 0.98808 / 0.99400 / 0.998882 | 0.915 | 1.000 | 0.98808 |
| flashembed q8blk-b32 | 4.88 | 1.45x | 2.07x | 3004 | 0.99136 / 0.99662 / 0.999272 | 0.927 | 1.000 | 0.99136 |

**neomme text, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 2.07 | 0.61x | 1.00x | 25270 | - | - | - | - |
| flashembed fp32 | 3.94 | 1.00x | 1.68x | 9926 | - | - | - | 1.00000 |
| flashembed dyn8rr | 5.60 | 1.40x | 2.36x | 9608 | 0.92747 / 0.93966 / 0.976348 | 0.864 | 0.999 | 0.92747 |
| flashembed q8blk | 5.77 | 1.44x | 2.42x | 9655 | 0.99591 / 0.99884 / 0.999679 | 0.951 | 1.000 | 0.99591 |
| flashembed q8blk-b32 | 5.48 | 1.37x | 2.30x | 9704 | 0.99820 / 0.99926 / 0.999771 | 0.958 | 1.000 | 0.99820 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 3 (EPYC 75F3) | dyn8rr | 199 | 0.307 / 0.256 | 0.411 / 0.347 | -0.064 [-0.105, -0.022] | 0.246 |
| neomme | Zen 3 (EPYC 75F3) | q8blk | 199 | 0.307 / 0.296 | 0.411 / 0.400 | -0.011 [-0.022, -0.002] | 0.558 |
| neomme | Zen 3 (EPYC 75F3) | q8blk-b32 | 199 | 0.307 / 0.307 | 0.411 / 0.410 | -0.000 [-0.012, +0.011] | 0.497 |

**gemma image, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.74 | 0.82x | 1.00x | 3945 | - | - | - | - |
| flashembed fp32 | 0.89 | 1.00x | 1.22x | 2620 | - | - | - | 1.00000 |
| flashembed dyn8rr | 1.10 | 1.25x | 1.52x | 1796 | 0.78121 / 0.83577 / 0.933092 | 0.475 | 0.790 | 0.78121 |
| flashembed q8blk | 1.11 | 1.25x | 1.52x | 1875 | 0.97165 / 0.97989 / 0.997877 | 0.838 | 0.990 | 0.97165 |
| flashembed q8blk-b32 | 1.07 | 1.21x | 1.47x | 1926 | 0.97603 / 0.98641 / 0.998420 | 0.860 | 1.000 | 0.97603 |

**gemma text, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 1.78 | 0.49x | 1.00x | 7016 | - | - | - | - |
| flashembed fp32 | 4.87 | 1.00x | 2.19x | 5594 | - | - | - | 1.00000 |
| flashembed dyn8rr | 6.08 | 1.28x | 2.79x | 5278 | 0.93475 / 0.94642 / 0.974812 | 0.596 | 0.942 | 0.93475 |
| flashembed q8blk | 6.18 | 1.27x | 2.78x | 5373 | 0.99670 / 0.99821 / 0.999322 | 0.945 | 1.000 | 0.99670 |
| flashembed q8blk-b32 | 5.80 | 1.23x | 2.66x | 5335 | 0.99645 / 0.99891 / 0.999576 | 0.958 | 1.000 | 0.99645 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 3 (EPYC 75F3) | dyn8rr | 199 | 0.764 / 0.794 | 0.835 / 0.851 | +0.016 [-0.026, +0.058] | 0.704 |
| gemma | Zen 3 (EPYC 75F3) | q8blk | 199 | 0.764 / 0.819 | 0.835 / 0.866 | +0.032 [+0.010, +0.055] | 0.819 |
| gemma | Zen 3 (EPYC 75F3) | q8blk-b32 | 199 | 0.764 / 0.819 | 0.835 / 0.867 | +0.032 [+0.012, +0.054] | 0.839 |

**neomme image, Zen 2 (EPYC 7702)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.63 | 0.33x | 1.00x | 4196 | - | - | - | - |
| flashembed fp32 | 2.51 | 1.00x | 3.19x | 3439 | - | - | - | 1.00000 |
| flashembed dyn8rr | 3.01 | 1.17x | 3.74x | 2950 | 0.86223 / 0.87001 / 0.926873 | 0.552 | 0.840 | 0.86223 |
| flashembed q8blk | 2.95 | 1.16x | 3.72x | 2998 | 0.98808 / 0.99400 / 0.998882 | 0.915 | 1.000 | 0.98808 |
| flashembed q8blk-b32 | 2.98 | 1.18x | 3.76x | 3001 | 0.99136 / 0.99662 / 0.999272 | 0.927 | 1.000 | 0.99136 |

**neomme text, Zen 2 (EPYC 7702)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 1.13 | 0.41x | 1.00x | 25280 | - | - | - | - |
| flashembed fp32 | 2.76 | 1.00x | 2.45x | 9924 | - | - | - | 1.00000 |
| flashembed dyn8rr | 3.07 | 1.11x | 2.72x | 9608 | 0.92747 / 0.93966 / 0.976348 | 0.864 | 0.999 | 0.92747 |
| flashembed q8blk | 3.18 | 1.14x | 2.78x | 9628 | 0.99591 / 0.99884 / 0.999679 | 0.951 | 1.000 | 0.99591 |
| flashembed q8blk-b32 | 2.93 | 1.05x | 2.57x | 9684 | 0.99820 / 0.99926 / 0.999771 | 0.958 | 1.000 | 0.99820 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 2 (EPYC 7702) | dyn8rr | 199 | 0.307 / 0.256 | 0.411 / 0.347 | -0.064 [-0.105, -0.022] | 0.246 |
| neomme | Zen 2 (EPYC 7702) | q8blk | 199 | 0.307 / 0.296 | 0.411 / 0.400 | -0.011 [-0.022, -0.002] | 0.558 |
| neomme | Zen 2 (EPYC 7702) | q8blk-b32 | 199 | 0.307 / 0.307 | 0.411 / 0.410 | -0.000 [-0.012, +0.011] | 0.497 |

**gemma image, Zen 2 (EPYC 7702)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.28 | 0.56x | 1.00x | 3945 | - | - | - | - |
| flashembed fp32 | 0.55 | 1.00x | 1.79x | 2628 | - | - | - | 1.00000 |
| flashembed dyn8rr | 0.53 | 1.03x | 1.84x | 1784 | 0.78121 / 0.83577 / 0.933092 | 0.475 | 0.790 | 0.78121 |
| flashembed q8blk | 0.57 | 1.12x | 1.99x | 1975 | 0.97165 / 0.97989 / 0.997877 | 0.838 | 0.990 | 0.97165 |
| flashembed q8blk-b32 | 0.57 | 1.10x | 1.96x | 1923 | 0.97603 / 0.98641 / 0.998420 | 0.860 | 1.000 | 0.97603 |

**gemma text, Zen 2 (EPYC 7702)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.89 | 0.33x | 1.00x | 7225 | - | - | - | - |
| flashembed fp32 | 2.77 | 1.00x | 3.01x | 5615 | - | - | - | 1.00000 |
| flashembed dyn8rr | 2.95 | 1.00x | 3.02x | 5282 | 0.93475 / 0.94642 / 0.974812 | 0.596 | 0.942 | 0.93475 |
| flashembed q8blk | 3.07 | 1.10x | 3.33x | 5362 | 0.99670 / 0.99821 / 0.999322 | 0.945 | 1.000 | 0.99670 |
| flashembed q8blk-b32 | 3.04 | 1.06x | 3.19x | 5394 | 0.99645 / 0.99891 / 0.999576 | 0.958 | 1.000 | 0.99645 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 2 (EPYC 7702) | dyn8rr | 199 | 0.764 / 0.794 | 0.835 / 0.851 | +0.016 [-0.026, +0.058] | 0.704 |
| gemma | Zen 2 (EPYC 7702) | q8blk | 199 | 0.764 / 0.819 | 0.835 / 0.866 | +0.032 [+0.010, +0.055] | 0.819 |
| gemma | Zen 2 (EPYC 7702) | q8blk-b32 | 199 | 0.764 / 0.819 | 0.835 / 0.867 | +0.032 [+0.012, +0.054] | 0.839 |

## Rust QMatMul, first version (scalar activation quantizer)

Superseded by the AVX2 quantizer above; kept for the record.

**neomme image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 3.34 | 0.48x | 1.00x | 4123 | - | - | - | - |
| flashembed fp32 | 7.31 | 1.00x | 2.07x | 3431 | - | - | - | 1.00000 |
| flashembed dyn8rr | 10.80 | 1.49x | 3.09x | 2986 | 0.85576 / 0.85718 / 0.928095 | 0.547 | 0.840 | 0.85576 |
| flashembed q8blk | 9.92 | 1.39x | 2.86x | 2995 | 0.99706 / 0.99718 / 0.999532 | 0.948 | 1.000 | 0.99706 |
| flashembed q8blk-b32 | 9.59 | 1.35x | 2.78x | 3009 | 0.99636 / 0.99831 / 0.999672 | 0.950 | 1.000 | 0.99636 |

**neomme text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 2.83 | 0.46x | 1.00x | 25281 | - | - | - | - |
| flashembed fp32 | 7.34 | 1.00x | 2.23x | 9923 | - | - | - | 1.00000 |
| flashembed dyn8rr | 10.62 | 1.53x | 3.37x | 9611 | 0.92800 / 0.93907 / 0.976163 | 0.868 | 0.999 | 0.92800 |
| flashembed q8blk | 10.35 | 1.41x | 3.14x | 9653 | 0.99903 / 0.99959 / 0.999833 | 0.971 | 1.000 | 0.99903 |
| flashembed q8blk-b32 | 9.58 | 1.37x | 3.02x | 9694 | 0.99949 / 0.99967 / 0.999873 | 0.976 | 1.000 | 0.99949 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 5 (EPYC 9655P) | dyn8rr | 199 | 0.307 / 0.261 | 0.411 / 0.348 | -0.063 [-0.103, -0.022] | 0.246 |
| neomme | Zen 5 (EPYC 9655P) | q8blk | 199 | 0.307 / 0.296 | 0.411 / 0.404 | -0.006 [-0.017, +0.003] | 0.583 |
| neomme | Zen 5 (EPYC 9655P) | q8blk-b32 | 199 | 0.307 / 0.307 | 0.411 / 0.410 | -0.001 [-0.009, +0.007] | 0.623 |

**gemma image, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.85 | 0.49x | 1.00x | 3879 | - | - | - | - |
| flashembed fp32 | 1.78 | 1.00x | 2.05x | 2609 | - | - | - | 1.00000 |
| flashembed dyn8rr | 2.34 | 1.31x | 2.67x | 1810 | 0.78558 / 0.83737 / 0.930818 | 0.477 | 0.790 | 0.78558 |
| flashembed q8blk | 2.13 | 1.19x | 2.44x | 1875 | 0.99569 / 0.99721 / 0.999648 | 0.949 | 1.000 | 0.99568 |
| flashembed q8blk-b32 | 2.06 | 1.16x | 2.37x | 1926 | 0.99664 / 0.99820 / 0.999750 | 0.960 | 1.000 | 0.99664 |

**gemma text, Zen 5 (EPYC 9655P)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 2.13 | 0.25x | 1.00x | 7160 | - | - | - | - |
| flashembed fp32 | 10.28 | 1.00x | 4.10x | 5619 | - | - | - | 1.00000 |
| flashembed dyn8rr | 12.91 | 1.20x | 4.97x | 5281 | 0.93532 / 0.94721 / 0.974472 | 0.601 | 0.939 | 0.93532 |
| flashembed q8blk | 10.57 | 1.02x | 4.19x | 5358 | 0.99646 / 0.99939 / 0.999751 | 0.968 | 1.000 | 0.99646 |
| flashembed q8blk-b32 | 10.44 | 0.99x | 4.09x | 5453 | 0.99718 / 0.99956 / 0.999830 | 0.973 | 1.000 | 0.99718 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 5 (EPYC 9655P) | dyn8rr | 199 | 0.764 / 0.804 | 0.835 / 0.856 | +0.021 [-0.022, +0.067] | 0.714 |
| gemma | Zen 5 (EPYC 9655P) | q8blk | 199 | 0.764 / 0.754 | 0.835 / 0.832 | -0.003 [-0.014, +0.007] | 0.899 |
| gemma | Zen 5 (EPYC 9655P) | q8blk-b32 | 199 | 0.764 / 0.764 | 0.835 / 0.837 | +0.002 [-0.009, +0.012] | 0.905 |

**neomme image, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 2.72 | 0.80x | 1.00x | 4017 | - | - | - | - |
| flashembed fp32 | 3.69 | 1.00x | 1.26x | 3428 | - | - | - | 1.00000 |
| flashembed dyn8rr | 5.12 | 1.41x | 1.78x | 2977 | 0.86223 / 0.87001 / 0.926873 | 0.552 | 0.840 | 0.86223 |
| flashembed q8blk | 4.63 | 1.29x | 1.63x | 3005 | 0.98869 / 0.99404 / 0.998845 | 0.924 | 1.000 | 0.98869 |
| flashembed q8blk-b32 | 4.60 | 1.27x | 1.61x | 3002 | 0.99050 / 0.99528 / 0.999260 | 0.930 | 1.000 | 0.99050 |

**neomme text, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 2.32 | 0.66x | 1.00x | 25318 | - | - | - | - |
| flashembed fp32 | 4.27 | 1.00x | 1.56x | 9928 | - | - | - | 1.00000 |
| flashembed dyn8rr | 5.73 | 1.34x | 2.08x | 9612 | 0.92747 / 0.93966 / 0.976348 | 0.864 | 0.999 | 0.92747 |
| flashembed q8blk | 5.23 | 1.19x | 1.87x | 9647 | 0.99465 / 0.99889 / 0.999675 | 0.952 | 1.000 | 0.99465 |
| flashembed q8blk-b32 | 4.90 | 1.12x | 1.76x | 9699 | 0.99675 / 0.99909 / 0.999774 | 0.958 | 1.000 | 0.99675 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| neomme | Zen 3 (EPYC 75F3) | dyn8rr | 199 | 0.307 / 0.256 | 0.411 / 0.347 | -0.064 [-0.105, -0.022] | 0.246 |
| neomme | Zen 3 (EPYC 75F3) | q8blk | 199 | 0.307 / 0.296 | 0.411 / 0.402 | -0.009 [-0.024, +0.005] | 0.518 |
| neomme | Zen 3 (EPYC 75F3) | q8blk-b32 | 199 | 0.307 / 0.307 | 0.411 / 0.408 | -0.003 [-0.016, +0.010] | 0.543 |

**gemma image, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 0.50 | 0.72x | 1.00x | 3853 | - | - | - | - |
| flashembed fp32 | 0.71 | 1.00x | 1.40x | 2630 | - | - | - | 1.00000 |
| flashembed dyn8rr | 0.81 | 1.17x | 1.64x | 1803 | 0.78121 / 0.83577 / 0.933092 | 0.475 | 0.790 | 0.78121 |
| flashembed q8blk | 0.78 | 1.12x | 1.56x | 1898 | 0.97141 / 0.97869 / 0.997906 | 0.842 | 1.000 | 0.97141 |
| flashembed q8blk-b32 | 0.74 | 1.03x | 1.45x | 1929 | 0.97602 / 0.98640 / 0.998408 | 0.861 | 1.000 | 0.97602 |

**gemma text, Zen 3 (EPYC 75F3)** (16 threads, batch 4)

| runtime | items/s | vs fp32 flashembed | vs Python | peak RSS MB | dense cos vs fp32: min / p1 / median | top-10 overlap | fp32 top-1 in top 10 | dense cos vs Python: min |
|---|---|---|---|---|---|---|---|---|
| Python (model card, fp32) | 1.25 | 0.47x | 1.00x | 7113 | - | - | - | - |
| flashembed fp32 | 3.87 | 1.00x | 2.38x | 5627 | - | - | - | 1.00000 |
| flashembed dyn8rr | 4.51 | 1.27x | 2.94x | 5304 | 0.93475 / 0.94642 / 0.974812 | 0.596 | 0.942 | 0.93475 |
| flashembed q8blk | 4.02 | 1.09x | 2.55x | 5355 | 0.99611 / 0.99809 / 0.999320 | 0.945 | 1.000 | 0.99611 |
| flashembed q8blk-b32 | 3.87 | 1.05x | 2.46x | 5404 | 0.99679 / 0.99881 / 0.999571 | 0.955 | 1.000 | 0.99679 |

**Caption -> figure retrieval on the held-out set** (each caption queries all figures of the set)

| model | CPU | variant | queries | R@1 fp32 / variant | MRR fp32 / variant | MRR variant - fp32 [95 % CI] | same rank as fp32 |
|---|---|---|---|---|---|---|---|
| gemma | Zen 3 (EPYC 75F3) | dyn8rr | 199 | 0.764 / 0.794 | 0.835 / 0.851 | +0.016 [-0.026, +0.058] | 0.704 |
| gemma | Zen 3 (EPYC 75F3) | q8blk | 199 | 0.764 / 0.814 | 0.835 / 0.864 | +0.029 [+0.008, +0.052] | 0.824 |
| gemma | Zen 3 (EPYC 75F3) | q8blk-b32 | 199 | 0.764 / 0.824 | 0.835 / 0.869 | +0.034 [+0.012, +0.058] | 0.819 |

## Graph sizes

| variant | gemma_lm | gemma_vision_dyn | neomme_text | neomme_image |
|---|---|---|---|---|
| fp32 | 1088 | 674 | 1021 | 1054 |
| dyn8 | 679 | 219 | 459 | 467 |
| dyn8-kdown | 754 | 332 | 646 | 660 |
| dyn8-kgu | 830 | 445 | 699 | 726 |
| dyn8-ko | 723 | 247 | 566 | 574 |
| dyn8-kqkv | 761 | 303 | 539 | 547 |
| dyn8-mlp | 862 | 335 | 647 | 680 |
| dyn8-novis | 679 | fp32 | 459 | 467 |
| dyn8-sel | 733 | 278 | 526 | 559 |
| dyn8rr | 679 | 219 | 459 | 467 |
| dyn8rr-sel | 733 | 278 | 526 | 559 |
| nb4-a1 | 627 | 161 | 388 | 393 |
| nb4-a4 | 627 | 161 | 388 | 393 |
| nb8-a1 | 695 | 237 | 481 | 491 |
| nb8-a4 | 695 | 237 | 481 | 491 |
| nb8-a4-novis | 695 | fp32 | 481 | 491 |
| q8blk | 695 | 237 | 481 | 491 |
| q8blk-b32 | 712 | 256 | 505 | 515 |
| q8blk-b32-f | 712 | 256 | 505 | 515 |
| q8blk-f | 695 | 237 | 481 | 491 |
| qdq8 | 679 | 219 | 459 | 467 |
| qdq8-avg | 679 | 219 | 459 | 467 |
| qdq8-sel | 734 | 278 | 526 | 559 |
