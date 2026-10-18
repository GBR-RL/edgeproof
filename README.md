# edgeproof

[![CI](https://github.com/GBR-RL/edgeproof/actions/workflows/ci.yml/badge.svg)](https://github.com/GBR-RL/edgeproof/actions/workflows/ci.yml)
[![Performance gate](https://github.com/GBR-RL/edgeproof/actions/workflows/perf.yml/badge.svg)](https://github.com/GBR-RL/edgeproof/actions/workflows/perf.yml)
[![C++ runner](https://github.com/GBR-RL/edgeproof/actions/workflows/cpp.yml/badge.svg)](https://github.com/GBR-RL/edgeproof/actions/workflows/cpp.yml)

What accuracy does it cost to make an industrial inspection model fast on the hardware at the
line, and which deployment route pays off? edgeproof trains defect detectors and segmenters on
VisA and takes them through every common route to production: ONNX Runtime and OpenVINO on x86
and ARM64 CPUs, INT8 after training (PTQ) and quantization-aware training (QAT), TensorRT on an
NVIDIA GPU, and a C++ runner. Every variant is measured the same way: latency of one image, and
accuracy through the runtime itself on 960 test images. A CI gate rebuilds the main variants on
x86-64 and ARM64 runners for every pull request and fails when accuracy or latency regress.

The code picks a GPU automatically (CUDA or Apple MPS, else the CPU) and runs on either. Models
were trained and the GPU numbers measured on one NVIDIA H100 of the RWTH CLAIX-2023 cluster; the
CPU numbers come from a Xeon node of the same cluster and from GitHub's x86-64 and ARM64 runners.

## What to deploy where

YOLO26s detection at 1024 px, one image at a time, 4 threads on the CPUs; details and every
other variant below.

| Target | Route | p50 | mAP50 | Why |
|---|---|--:|--:|---|
| Xeon with AMX | OpenVINO BF16 | 30 ms | 0.412 | as fast as INT8, no calibration set; INT8 (31 ms, 0.415) if the model must be 3.6× smaller |
| Other x86-64 CPUs | OpenVINO INT8 | 64 to 210 ms | 0.415 | ONNX Runtime INT8 only on CPUs with VNNI: without it, it scores 0.016 |
| ARM64 (Neoverse-N2) | ONNX Runtime INT8, head in float | 145 ms | 0.397 | 2.4× over FP32; OpenVINO is nine times slower there |
| NVIDIA H100 | TensorRT FP16 | 1.17 ms | 0.421 | 3.4× over PyTorch at FP32 accuracy; INT8 saves under 10% and costs mAP50 |

On every target: keep the box decoding in float when quantizing (`--float-decode`), and prefer
the NMS head where its post-processing is affordable (+0.02 mAP50, +6 points of recall).

## Setup

- **Data.** [VisA](https://registry.opendata.aws/visa/) (12 industrial parts, CC BY 4.0), split
  per part into train, validation and test: 1,440 / 480 / 960 images. The test split holds 240
  defective images with 570 defect regions, and 720 defect-free images to measure false alarms.
  One class, "defect": boxes for detection, polygons for segmentation.
- **Models.** YOLO26s, whose head can skip NMS, and YOLO11s, each for detection and
  segmentation, at 1024 px. 100 epochs from COCO weights, 9 to 19 GPU-minutes per model on the
  H100 ([`slurm/train.sbatch`](slurm/train.sbatch)). The trained detectors are attached to the
  [models-v1 release](https://github.com/GBR-RL/edgeproof/releases/tag/models-v1).
- **Accuracy.** Box mAP50 and mAP50-95 (and mask mAP50) on the test split, plus the line's view:
  each image scored by its most confident detection, the threshold set so that 1% of the
  defect-free images would be rejected, and the share of defective images caught at that
  threshold. One defective image is 0.4 points of this recall, so differences of a few points
  are noise.
- **CPU latency.** Batch 1, 1024×1024, 4 threads, the network alone (pre- and post-processing are
  separate), median and 99th percentile of 100 runs after 10 warm-up runs. Every variant measured
  in one job on one otherwise idle node with an Intel Xeon Platinum 8468
  ([`slurm/latency.sbatch`](slurm/latency.sbatch)).
- **GPU latency.** The engine alone with its input already in GPU memory, batch 1, median of 1000
  runs ([`src/edgeproof/gpu.py`](src/edgeproof/gpu.py)).

## Results on an x86 CPU

YOLO26s detection, NMS-free head, from training framework to INT8 (Xeon Platinum 8468, 4 threads):

| Route | p50 ms | p99 ms | vs PyTorch | mAP50 | mAP50-95 | Recall @ 1% false alarms | Size MB |
|---|--:|--:|--:|--:|--:|--:|--:|
| PyTorch FP32 | 172 | 178 | 1.0× | 0.423 | 0.223 | 0.79 | 20.4 |
| ONNX Runtime FP32 | 120 | 125 | 1.4× | 0.415 | 0.222 | 0.78 | 38.4 |
| OpenVINO FP32 | 93 | 97 | 1.8× | 0.415 | 0.222 | 0.78 | 38.5 |
| ONNX Runtime INT8 (percentile, decoding in float) | 64 | 65 | 2.7× | 0.409 | 0.212 | 0.78 | 10.7 |
| **OpenVINO BF16 (AMX)** | **30** | **34** | **5.8×** | **0.412** | **0.221** | **0.79** | **38.5** |
| **OpenVINO INT8 (NNCF performance)** | **31** | **35** | **5.5×** | **0.415** | **0.213** | **0.77** | **10.6** |

All 18 FP32, 6 BF16 and 49 INT8 variants, segmentation included, are in
[docs/results.md](docs/results.md).

- **On a Xeon with AMX, BF16 is as fast as INT8 and needs no calibration.** OpenVINO in bfloat16
  runs every model 3.1 to 3.4× faster than in FP32, the same speed as INT8, within 0.003 mAP50
  and 0.003 mask mAP50 of FP32 and without a calibration set. INT8 still makes the model 3.6×
  smaller, and it needs no AMX.
- **INT8 breaks where the head turns features into coordinates.** Quantizing the whole network
  fails in two ways. ONNX Runtime puts box coordinates (0 to 1024 px) and confidences (0 to 1)
  under one INT8 output scale of about 4, so every confidence rounds to 0 and all eight fully
  quantized variants detect nothing. OpenVINO's quantizer (NNCF) keeps the last operations in
  float, but in YOLO11 it still quantizes the decoded boxes in grid cells, before the
  multiplication by the stride, where one INT8 step is several pixels: mAP50 falls from 0.390 to
  0.243 while recall, which depends on the confidences alone, stays at 0.78. Keeping only the
  decoding in float (`--float-decode`: the head's convolutions stay INT8) fixes both: YOLO11 on
  OpenVINO 0.386 at 28 ms, ONNX Runtime with entropy calibration 0.390 instead of 0, and it is
  faster than keeping the whole head in float (64 instead of 71 ms on ONNX Runtime). YOLO26's
  decoding already stays in float under NNCF, so there full INT8 costs no mAP50.
- **The NMS head is the better choice for INT8 too.** YOLO26's NMS-free head loses 2 points of
  mAP50 and 6 points of recall at 1% false alarms against the same model's NMS head (0.415 vs
  0.436, 0.78 vs 0.84), and the network is as fast with either. With the decoding in float, the
  NMS head keeps its lead in INT8: 0.433 on ONNX Runtime (61 ms), 0.424 on OpenVINO (31 ms), at
  0.82 recall.
- **Segmentation quantizes as well as detection, given the same care.** YOLO26s-seg in full INT8
  on OpenVINO loses 0.013 mask mAP50 at 42 ms (3.4× over FP32). YOLO11s-seg loses 0.080 in full
  INT8 and 0.023 with the decoding in float, at 38 ms.
- **128 calibration images are enough.** 32 images cost YOLO26 on OpenVINO 0.008 mAP50 and
  3 points of recall. 512 images move mAP50 by -0.006 to +0.001 against 128 across four
  configurations, no more than the spread between calibration methods.
- **Exporting costs nothing.** Evaluated on the same square input with the same head, PyTorch
  gives YOLO26s 0.4151 mAP50 and 0.2220 mAP50-95, the exports 0.4151 and 0.2220. The PyTorch row
  above scores 0.008 higher only because Ultralytics evaluates PyTorch models on a 960×1024
  input, while the exports use a fixed 1024×1024 square.

## Results on an NVIDIA H100

TensorRT 11 engines built through NVIDIA ModelOpt ([`src/edgeproof/gpu.py`](src/edgeproof/gpu.py)),
batch 1, the engine alone. p50 in ms and mAP50 on the 960 test images:

| Route | YOLO26s NMS-free | | YOLO26s NMS | | YOLO11s NMS | |
|---|--:|--:|--:|--:|--:|--:|
| | **ms** | **mAP50** | **ms** | **mAP50** | **ms** | **mAP50** |
| PyTorch FP32 | 3.97 | 0.422 | 3.88 | 0.437 | 3.25 | 0.389 |
| TensorRT FP32 | 1.63 | 0.415 | - | - | 1.48 | 0.390 |
| **TensorRT FP16** | **1.17** | **0.421** | **1.15** | **0.436** | **1.07** | **0.388** |
| TensorRT INT8, calibrated (PTQ) | 1.07 | 0.398 | 1.07 | 0.428 | 1.01 | 0.371 |
| PTQ of the model fine-tuned 10 more epochs | 1.07 | 0.391 | - | - | 0.98 | 0.366 |
| INT8 after 10 epochs of QAT | 1.33 | 0.402 | 1.36 | 0.428 | 1.21 | 0.390 |

Every row, with mAP50-95, recall and engine sizes, is in
[docs/results.md](docs/results.md#nvidia-h100).

- **FP16 is the route on this GPU.** It runs 3.0 to 3.7× faster than PyTorch, within 0.001 mAP50
  of PyTorch FP32, and the same holds for segmentation: mask mAP50 0.414 against 0.414 for
  YOLO26s-seg, 0.413 against 0.410 for YOLO11s-seg. At batch 1 INT8 saves at most 0.1 ms on
  top, under 10%.
- **INT8 after training costs more on TensorRT than on the CPU.** ModelOpt's calibration on 512
  training images loses 0.008 to 0.023 mAP50 against FP16, while NNCF's INT8 for OpenVINO loses
  nothing on the same YOLO26s.
- **Quantization-aware training recovers that loss.** Against the fair control, PTQ of a model
  fine-tuned in float for the same 10 epochs, QAT gains 0.011 mAP50 on YOLO26s and 0.024 on
  YOLO11s, where it matches the FP32 engine (0.390). But its engines are slower than PTQ's of
  the same model, and slower than FP16 (1.21 to 1.36 ms), and 1.5 to 1.8 times as large. QAT
  pays off where INT8 is required, such as on a smaller accelerator without fast FP16; on the
  H100, FP16 is faster and more accurate.
- Recall at 1% false alarms moves more between INT8 engines than mAP50 (0.67 to 0.84 against
  0.75 to 0.84 in FP16); compare INT8 engines by mAP50. On the NMS-free head PyTorch scores 0.007
  above the engines for the reason given above (a 960×1024 input against the export's square).
  TensorRT's FP32 engine may compute in TF32, which is why FP16 can land slightly above it.

## ARM64 and the CI performance gate

[`edgeproof gate`](src/edgeproof/gate.py) rebuilds the variants in
[`perf/gate.yaml`](perf/gate.yaml) from the released checkpoints, times them and evaluates them on
the 960 test images. It fails when mAP50 or recall drop below the cluster's reference (by more
than 0.01 and 0.03), when an INT8 variant is not a minimum factor faster than FP32 in the same
run, or when FP32 exceeds a generous budget for the CPU architecture. The
[performance workflow](.github/workflows/perf.yml) runs it on GitHub's 4-vCPU x86-64 and ARM64
runners for every pull request that touches the code. YOLO26s, NMS-free head, 4 threads, p50 ms
and mAP50 on the four runner CPUs seen:

| Variant | x86-64 Xeon 8573C | x86-64 Zen 4 (EPYC 9V74) | x86-64 Zen 3 (EPYC 7763) | ARM64 (Neoverse-N2) |
|---|--:|--:|--:|--:|
| ONNX Runtime FP32 | 220 ms, 0.415 | 271 ms, 0.415 | 365 ms, 0.415 | 345 ms, 0.420 |
| OpenVINO FP32 | 193 ms, 0.415 | 272 ms, 0.415 | 346 ms, 0.415 | 3,182 ms, 0.420 |
| ONNX Runtime INT8, head in float | 122 ms, 0.396 | 163 ms, 0.396 | 245 ms, **0.016** | 145 ms, 0.397 |
| OpenVINO INT8 | 64 ms, 0.415 | 98 ms, 0.414 | 210 ms, 0.417 | - |

- **ONNX Runtime INT8 breaks on x86 CPUs without VNNI.** On the Zen 3 runner the same INT8 model
  that scores 0.396 on Zen 4 and 0.399 on the cluster's Xeon detects almost nothing (0.016
  mAP50). ONNX Runtime's U8S8 kernels use `VPMADDUBSW` on x86 CPUs without VNNI, whose 16-bit
  intermediate sums can saturate; its documentation recommends 7-bit weights (`reduce_range`) or
  U8U8 there (not measured here). An INT8 model validated on one x86 machine is not validated
  for the fleet. The gate checks this variant only on runners with VNNI and says so on others.
- **On ARM64, ONNX Runtime is the runtime, and INT8 pays off more than on x86.** OpenVINO
  2026.4 runs the FP32 model nine times slower than ONNX Runtime on the Neoverse-N2, so the gate
  runs OpenVINO on x86-64 only. ONNX Runtime INT8 gets 2.4× over FP32 there, against 1.7× on Zen
  4, and has no saturation problem (ARM's dot-product instructions accumulate in 32 bits).
- **The same recipe gives slightly different models per architecture.** ONNX Runtime FP32
  scores 0.420 mAP50 on ARM64 and 0.415 on x86-64; INT8 catches 0.74 of the defective images
  on ARM64 against 0.78. Small numeric differences in kernels and resizing move borderline
  detections. The gate therefore checks against a lower bound with a tolerance, and the ARM64
  INT8 variant has references of its own.
- **Absolute latency budgets do not work on shared runners.** The x86-64 runner was a Xeon, a
  Zen 4 or a Zen 3 from one run to the next, and OpenVINO INT8 took 64, 98 or 210 ms. The gate
  therefore asks for a speed-up over FP32 measured in the same run (1.3× for INT8 on x86-64; the
  runners reach 1.6 to 3.0×), which only fails when quantization stops working.

## A C++ runner

[`cpp/`](cpp) runs the exported models the way a line application would, through the ONNX Runtime
C++ API: read, letterbox with OpenCV, infer, decode (YOLO26's NMS-free output or NMS for YOLO11),
scale back. The geometry and the decoding are unit-tested on their own. CI builds it on x86-64
and ARM64, runs it on the 960 test images and compares every image's detections with Ultralytics
running the same ONNX model ([`cpp/parity.py`](cpp/parity.py)). Per-image medians on the same
runners, ONNX Runtime FP32, 4 threads:

| Runner | Model | Identical detections | Python pre / infer / post ms | C++ pre / infer / post ms | End to end |
|---|---|--:|--:|--:|--:|
| x86-64, Xeon Platinum 8573C | YOLO26s, NMS-free | 960 / 960 | 7.0 / 232 / 0.3 | 4.4 / 226 / 0.0 | -3.6% |
| x86-64, Xeon Platinum 8573C | YOLO11s, NMS | 960 / 960 | 7.3 / 246 / 0.5 | 4.4 / 242 / 0.0 | -2.8% |
| ARM64, Neoverse-N2 | YOLO26s, NMS-free | 949 / 960 | 3.8 / 356 / 0.3 | 3.1 / 357 / 0.0 | -0.1% |
| ARM64, Neoverse-N2 | YOLO11s, NMS | 949 / 960 | 3.4 / 369 / 0.6 | 2.8 / 366 / 0.0 | -1.0% |

- **The detections match.** On x86-64 the C++ runner reproduces Ultralytics exactly (boxes within
  0.001 px). On ARM64 the system's OpenCV 4.6 and opencv-python 5.0 round the bilinear resize
  differently by one grey level (JPEG decoding is identical); that moves a borderline detection
  on 11 of 960 images.
- **C++ buys almost nothing here.** It cuts pre-processing by a third on x86-64 and decoding to
  nothing, but the network is over 95% of the time, so the whole pipeline gets 0.1 to 3.6% faster.
  For a model of this size, quantization and the choice of runtime decide the latency, not the
  language around them; C++ matters for deployment (no Python on the device), not for speed.

## Pitfalls found on the way

Each of these silently produced wrong or misleading numbers before it was fixed:

- **Ultralytics runs YOLO26 with its NMS head unless told otherwise.** `val` and `predict` default
  to `nms=None`, which picks the one-to-many head plus NMS, while an export with `nms=False` keeps
  the one-to-one head. Comparing the two looks like an export losing 2 points of mAP50; with
  the same head and the same square input, PyTorch and the exports agree to the fourth decimal.
  The head is a parameter here and part of every result row.
- **OpenVINO computes in bfloat16 on CPUs with AMX.** On the Xeon 8468 the CPU plugin's default
  inference precision is bf16, so "FP32" OpenVINO numbers from such a machine are bf16 unless
  `INFERENCE_PRECISION_HINT` is set. Here FP32 sets it, and BF16 is a precision of its own.
- **ONNX Runtime sizes its thread pool for the whole machine, not the Slurm job.** Inside an
  8-core job on a 96-core node it started about 120 threads and validation slowed tenfold. Its
  idle workers also spin between runs and starved the Python pre- and post-processing on the same
  cores (3 ms per image became up to 270 ms). ModelOpt's INT8 calibration for TensorRT runs
  through ONNX Runtime on the CPU and hits the same oversubscription, inside session options of
  its own: an INT8 export took 33 minutes while the GPU waited, and 4 to 7 minutes once those
  sessions were pinned to the job's cores too.
- **ONNX Runtime's `CalibMaxIntermediateOutputs` discards data** (version 1.31). Meant to bound the
  min-max calibrator's memory, it clears each full chunk without computing its ranges: with a
  multiple of the chunk size nothing is left and calibration fails, with any other count it
  calibrates on the remainder only. Calibration here feeds chunks itself; the ranges equal a
  single pass exactly.
- **Keeping the "head" in float with ONNX Runtime changes nothing for its sigmoids.** ONNX
  Runtime never quantizes `Sigmoid`, so excluding the head's sigmoids from quantization gives
  exactly the same model; what matters is the decoding after them.
- **TensorRT 11 has no precision flags any more.** Networks are strongly typed: FP16 and INT8 must
  be in the ONNX graph (ModelOpt AutoCast, Q/DQ nodes) before the build, and the INT8 calibrator
  API is gone.
- **Ultralytics runs a list of image paths as one batch.** `predict([...960 paths...])` sent all
  test images through the network at once and needed 12 GB for the first convolution.
- **ONNX Runtime writes a shape-inferred copy next to the model it quantizes.** Variants quantized
  in parallel from one export deleted each other's copy; each now quantizes its own copy.

## Reproduce

```bash
pip install -e ".[runtimes,train,dev]"
edgeproof data                                   # download VisA, split, write YOLO labels
edgeproof train --family yolo26 --task detect    # GPU if available
edgeproof bench --name yolo26s-det-1024 --head e2e
edgeproof ptq --name yolo26s-det-1024 --runtime openvino --method performance
edgeproof finetune --name yolo26s-det-1024 --qat # quantization-aware, needs CUDA
edgeproof gpu --name yolo26s-det-1024-qat --precision int8   # TensorRT, needs CUDA
edgeproof gate                                   # the CI performance gate, locally
edgeproof report                                 # the tables, from results/
```

The C++ runner builds with CMake (OpenCV and a C++20 compiler; ONNX Runtime is downloaded):

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/edgeproof-run --model yolo26s-det-1024.onnx --images data/yolo/detect/images/test
```

On a Slurm cluster, [`slurm/`](slurm) has the jobs used here: training and the GPU work on one
GPU behind a quota check, CPU jobs for benchmarks and quantization, and the whole-node latency
run over [`slurm/variants.txt`](slurm/variants.txt). The raw result rows are in
[`results/`](results).
