# edgeproof

[![CI](https://github.com/GBR-RL/edgeproof/actions/workflows/ci.yml/badge.svg)](https://github.com/GBR-RL/edgeproof/actions/workflows/ci.yml)

What accuracy does it cost to make an industrial inspection model fast on the hardware at the
line, and which deployment route pays off? edgeproof trains defect detectors and segmenters on
VisA, exports them to ONNX Runtime and OpenVINO, quantizes them to INT8 after training, and
measures every variant the same way: latency of one image on a fixed number of CPU threads, and
accuracy through the runtime itself on 960 test images.

The code picks a GPU automatically (CUDA or Apple MPS, else the CPU) and runs on either. The
models were trained on one NVIDIA H100 of the RWTH CLAIX-2023 cluster; the deployment numbers
below are measured on CPU, the hardware an inspection station usually has.

## Setup

- **Data.** [VisA](https://registry.opendata.aws/visa/) (12 industrial parts, CC BY 4.0), split
  per part into train, validation and test: 1,440 / 480 / 960 images. The test split holds 240
  defective images with 570 defect regions, and 720 defect-free images to measure false alarms.
  One class, "defect": boxes for detection, polygons for segmentation.
- **Models.** YOLO26s, whose head can skip NMS, and YOLO11s, each for detection and
  segmentation, at 1024 px. 100 epochs from COCO weights, 9 to 19 GPU-minutes per model on the
  H100 ([`slurm/train.sbatch`](slurm/train.sbatch)).
- **Accuracy.** Box mAP50 and mAP50-95 (and mask mAP50) on the test split, plus the line's view:
  each image scored by its most confident detection, the threshold set so that 1% of the
  defect-free images would be rejected, and the share of defective images caught at that
  threshold. One defective image is 0.4 points of this recall, so differences of a few points
  are noise.
- **Latency.** Batch 1, 1024×1024, 4 threads, the network alone (pre- and post-processing are
  separate), median and 99th percentile of 100 runs after 10 warm-up runs. Every variant measured
  in one job on one otherwise idle node with an Intel Xeon Platinum 8468
  ([`slurm/latency.sbatch`](slurm/latency.sbatch)). All runtimes compute in FP32 unless the row
  says INT8.

## Results

YOLO26s detection, NMS-free head, the routes from training framework to INT8:

| Route | p50 ms | p99 ms | vs PyTorch | mAP50 | mAP50-95 | Recall @ 1% false alarms | Size MB |
|---|--:|--:|--:|--:|--:|--:|--:|
| PyTorch FP32 | 172 | 178 | 1.0× | 0.423 | 0.223 | 0.79 | 20.4 |
| ONNX Runtime FP32 | 120 | 125 | 1.4× | 0.415 | 0.222 | 0.78 | 38.4 |
| OpenVINO FP32 | 93 | 97 | 1.8× | 0.415 | 0.222 | 0.78 | 38.5 |
| ONNX Runtime INT8 (percentile, head in float) | 71 | 77 | 2.4× | 0.417 | 0.217 | 0.78 | 11.9 |
| **OpenVINO INT8 (NNCF performance)** | **31** | **35** | **5.5×** | **0.415** | **0.213** | **0.77** | **10.6** |

All 18 FP32 and 24 INT8 variants, segmentation included, are in
[docs/results.md](docs/results.md).

- **OpenVINO INT8 is the route that pays off on an x86 CPU.** It is 5.5× faster than PyTorch and
  3.0× faster than OpenVINO FP32. mAP50 is unchanged, mAP50-95 drops by 0.009, recall at 1% false
  alarms by one point (within noise), and the model is 3.6× smaller. INT8 through ONNX Runtime
  gets 1.7× over its FP32 and stays 2.3× slower than OpenVINO INT8 on this CPU.
- **Exporting costs nothing.** Evaluated on the same square input with the same head, PyTorch
  gives YOLO26s 0.4151 mAP50 and 0.2220 mAP50-95, the exports 0.4151 and 0.2220. The PyTorch row
  above scores 0.008 higher only because Ultralytics evaluates PyTorch models on a 960×1024
  input, while the exports use a fixed 1024×1024 square; on the other models the gap is within
  ±0.004.
- **YOLO26's NMS-free head costs accuracy and saves almost no time here.** Against the model's
  own NMS head it loses 2 points of mAP50 (0.415 vs 0.436 through the exports) and 6 points of
  recall at 1% false alarms (0.78 vs 0.84). The network is as fast with either head (93 ms on
  OpenVINO), and NMS for one class takes 0.8 ms of post-processing against 0.3 ms without it.
- **Fully quantized, ONNX Runtime INT8 detects nothing.** All eight variants that quantize the
  head score exactly 0, for both models and every calibration method. The output tensor carries
  box coordinates (0 to 1024 px) and confidences (0 to 1) under one INT8 scale of about 4, so
  every confidence rounds to 0. With the head kept in float (1.4 to 2.5 MB larger), entropy and
  percentile calibration land within 0.01 mAP50 of FP32; min-max with per-tensor weights loses
  0.058 on YOLO26.
- **The architecture decides whether full INT8 works.** OpenVINO's quantizer (NNCF) leaves the
  last operations in float. In YOLO26 its nearest quantization point is ten operations before
  the output, so the box decoding stays in float and full INT8 with the performance preset costs
  no mAP50. In YOLO11 it
  quantizes the decoded box coordinates, in grid cells before the multiplication by the stride,
  where one INT8 step becomes several pixels. Its mAP50 falls from 0.390 to 0.243 while recall
  at 1% false alarms, which depends on the confidences alone, stays at 0.78. Keeping YOLO11's
  head in float restores FP32 accuracy exactly, at 37 instead of 28 ms.
- **Segmentation follows the same ranking.** OpenVINO FP32 runs YOLO26s-seg in 143 ms and
  YOLO11s-seg in 136 ms, against 256 and 226 ms in PyTorch, at equal mask mAP.

## Pitfalls found on the way

Each of these silently produced wrong or misleading numbers before it was fixed:

- **Ultralytics runs YOLO26 with its NMS head unless told otherwise.** `val` and `predict` default
  to `nms=None`, which picks the one-to-many head plus NMS, while an export with `nms=False` keeps
  the one-to-one head. Comparing the two looks like an export losing 2 points of mAP50; with
  the same head and the same square input, PyTorch and the exports agree to the fourth decimal.
  The head is a parameter here and part of every result row.
- **OpenVINO computes in bfloat16 on CPUs with AMX.** On the Xeon 8468 the CPU plugin's default
  inference precision is bf16, so "FP32" OpenVINO numbers from such a machine are bf16 unless
  `INFERENCE_PRECISION_HINT` is set.
- **ONNX Runtime sizes its thread pool for the whole machine, not the Slurm job.** Inside an
  8-core job on a 96-core node it started about 120 threads and validation slowed tenfold. Its
  idle workers also spin between runs and starved the Python pre- and post-processing on the same
  cores (3 ms per image became up to 270 ms).
- **ONNX Runtime's `CalibMaxIntermediateOutputs` discards data** (version 1.31). Meant to bound the
  min-max calibrator's memory, it clears each full chunk without computing its ranges: with a
  multiple of the chunk size nothing is left and calibration fails, with any other count it
  calibrates on the remainder only. Calibration here feeds chunks itself; the ranges equal a
  single pass exactly.
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
edgeproof report                                 # the tables above, from results/
```

On a Slurm cluster, [`slurm/`](slurm) has the jobs used here: training on one GPU behind a
quota check, CPU jobs for benchmarks and quantization, and the whole-node latency run over
[`slurm/variants.txt`](slurm/variants.txt). The raw result rows are in [`results/`](results).

Still to come: quantization-aware training and TensorRT on the H100, ARM64 CPUs, a performance
gate in CI, and a C++ runner.
