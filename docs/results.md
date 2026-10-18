# Results

Every measured variant, generated with `edgeproof report` from [`results/accuracy.jsonl`](../results/accuracy.jsonl) (accuracy through each runtime on the 960 test images), [`results/latency.jsonl`](../results/latency.jsonl) (all CPU variants in one run on an otherwise idle node: Intel Xeon Platinum 8468, 4 threads, batch 1, 1024x1024, the network alone) and [`results/gpu.jsonl`](../results/gpu.jsonl) (one NVIDIA H100). Recall is image-level, at the threshold that rejects 1% of the 720 defect-free test images. Heads: NMS-free is YOLO26's one-to-one head, NMS the one-to-many head followed by NMS.

## FP32 on each runtime

| Model | Head | Runtime | p50 ms | p99 ms | mAP50 | mAP50-95 | Mask mAP50 | Recall @ 1% false alarms | Size MB |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| yolo26s-det-1024 | NMS-free | PyTorch | 172 | 178 | 0.423 | 0.223 | - | 0.79 | 20.4 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | 120 | 125 | 0.415 | 0.222 | - | 0.78 | 38.4 |
| yolo26s-det-1024 | NMS-free | OpenVINO | 93 | 97 | 0.415 | 0.222 | - | 0.78 | 38.5 |
| yolo26s-det-1024 | NMS | PyTorch | 170 | 174 | 0.437 | 0.235 | - | 0.84 | 20.4 |
| yolo26s-det-1024 | NMS | ONNX Runtime | 108 | 120 | 0.436 | 0.234 | - | 0.84 | 38.4 |
| yolo26s-det-1024 | NMS | OpenVINO | 93 | 102 | 0.436 | 0.234 | - | 0.84 | 38.5 |
| yolo11s-det-1024 | NMS | PyTorch | 163 | 169 | 0.389 | 0.204 | - | 0.73 | 19.3 |
| yolo11s-det-1024 | NMS | ONNX Runtime | 132 | 134 | 0.390 | 0.204 | - | 0.75 | 38.2 |
| yolo11s-det-1024 | NMS | OpenVINO | 94 | 104 | 0.390 | 0.204 | - | 0.75 | 38.2 |
| yolo26s-seg-1024 | NMS-free | PyTorch | 256 | 263 | 0.433 | 0.221 | 0.414 | 0.82 | 23.4 |
| yolo26s-seg-1024 | NMS-free | ONNX Runtime | 188 | 192 | 0.436 | 0.220 | 0.411 | 0.81 | 42.0 |
| yolo26s-seg-1024 | NMS-free | OpenVINO | 143 | 154 | 0.436 | 0.220 | 0.411 | 0.81 | 42.1 |
| yolo26s-seg-1024 | NMS | PyTorch | 250 | 255 | 0.459 | 0.240 | 0.421 | 0.80 | 23.4 |
| yolo26s-seg-1024 | NMS | ONNX Runtime | 189 | 201 | 0.456 | 0.241 | 0.423 | 0.79 | 42.0 |
| yolo26s-seg-1024 | NMS | OpenVINO | 141 | 151 | 0.456 | 0.241 | 0.423 | 0.79 | 42.1 |
| yolo11s-seg-1024 | NMS | PyTorch | 226 | 234 | 0.427 | 0.230 | 0.410 | 0.83 | 20.6 |
| yolo11s-seg-1024 | NMS | ONNX Runtime | 190 | 201 | 0.431 | 0.229 | 0.410 | 0.83 | 40.8 |
| yolo11s-seg-1024 | NMS | OpenVINO | 136 | 142 | 0.431 | 0.229 | 0.410 | 0.83 | 40.9 |

## BF16 on OpenVINO

OpenVINO's CPU plugin computing in bfloat16 on the Xeon's AMX units, from the same FP32 model (`--precision bf16`). Speed-up and differences are against OpenVINO FP32.

| Model | Head | FP32 p50 ms | BF16 p50 ms | Speed-up | mAP50 (vs FP32) | mAP50-95 (vs FP32) | Mask mAP50 (vs FP32) | Recall @ 1% false alarms |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| yolo26s-det-1024 | NMS-free | 93 | 30 | 3.1x | 0.412 (-0.003) | 0.221 (-0.001) | - | 0.79 |
| yolo26s-det-1024 | NMS | 93 | 29 | 3.2x | 0.435 (-0.001) | 0.235 (+0.000) | - | 0.83 |
| yolo11s-det-1024 | NMS | 94 | 28 | 3.3x | 0.389 (-0.001) | 0.204 (+0.000) | - | 0.74 |
| yolo26s-seg-1024 | NMS-free | 143 | 43 | 3.3x | 0.434 (-0.001) | 0.220 (+0.000) | 0.409 (-0.002) | 0.80 |
| yolo26s-seg-1024 | NMS | 141 | 42 | 3.3x | 0.458 (+0.002) | 0.241 (+0.000) | 0.423 (+0.000) | 0.78 |
| yolo11s-seg-1024 | NMS | 136 | 40 | 3.4x | 0.428 (-0.002) | 0.228 (+0.000) | 0.407 (-0.003) | 0.83 |

## INT8 after training

128 calibration images from the training split unless noted. ONNX Runtime uses QDQ with UInt8 activations and Int8 weights, per channel unless noted; OpenVINO uses NNCF presets (performance: symmetric, mixed: asymmetric activations). Head ops: "INT8" quantizes the whole network (NNCF still keeps its last operations in float), "float" keeps the detection head unquantized, "INT8, decoding in float" quantizes the head's convolutions and keeps the operations that turn them into boxes and scores in float. Speed-up and differences are against FP32 on the same model, head and runtime.

| Model | Head | Runtime | Calibration | Head ops | p50 ms | Speed-up | mAP50 (vs FP32) | mAP50-95 (vs FP32) | Mask mAP50 (vs FP32) | Recall @ 1% false alarms | Size MB |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| yolo26s-det-1024 | NMS-free | ONNX Runtime | entropy | INT8 | 62 | 1.9x | 0.000 (-0.415) | 0.000 (-0.222) | - | 0.00 | 10.5 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | entropy | float | 71 | 1.7x | 0.406 (-0.009) | 0.217 (-0.005) | - | 0.80 | 11.9 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | minmax | INT8 | 60 | 2.0x | 0.000 (-0.415) | 0.000 (-0.222) | - | 0.00 | 10.5 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | minmax | float | 72 | 1.7x | 0.399 (-0.016) | 0.213 (-0.008) | - | 0.77 | 11.9 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | minmax, per-tensor | INT8 | 62 | 2.0x | 0.000 (-0.415) | 0.000 (-0.222) | - | 0.00 | 10.2 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | minmax, per-tensor | float | 72 | 1.7x | 0.357 (-0.058) | 0.189 (-0.033) | - | 0.61 | 11.7 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | percentile, 32 images | float | 73 | 1.6x | 0.411 (-0.004) | 0.223 (+0.001) | - | 0.78 | 11.9 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | percentile | INT8 | 61 | 2.0x | 0.000 (-0.415) | 0.000 (-0.222) | - | 0.00 | 10.5 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | percentile | float | 71 | 1.7x | 0.417 (+0.002) | 0.217 (-0.005) | - | 0.78 | 11.9 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | percentile | INT8, decoding in float | 64 | 1.9x | 0.409 (-0.007) | 0.212 (-0.010) | - | 0.78 | 10.7 |
| yolo26s-det-1024 | NMS-free | ONNX Runtime | percentile, 512 images | float | 72 | 1.7x | 0.411 (-0.004) | 0.220 (-0.002) | - | 0.79 | 11.9 |
| yolo26s-det-1024 | NMS-free | OpenVINO | mixed | INT8 | 35 | 2.7x | 0.406 (-0.009) | 0.204 (-0.018) | - | 0.78 | 10.6 |
| yolo26s-det-1024 | NMS-free | OpenVINO | mixed | float | 39 | 2.4x | 0.415 (+0.000) | 0.213 (-0.009) | - | 0.76 | 11.9 |
| yolo26s-det-1024 | NMS-free | OpenVINO | performance, 32 images | INT8 | 31 | 3.0x | 0.408 (-0.008) | 0.214 (-0.008) | - | 0.74 | 10.6 |
| yolo26s-det-1024 | NMS-free | OpenVINO | performance | INT8 | 31 | 3.0x | 0.415 (+0.000) | 0.213 (-0.009) | - | 0.77 | 10.6 |
| yolo26s-det-1024 | NMS-free | OpenVINO | performance | float | 37 | 2.5x | 0.419 (+0.004) | 0.214 (-0.008) | - | 0.75 | 11.9 |
| yolo26s-det-1024 | NMS-free | OpenVINO | performance | INT8, decoding in float | 31 | 3.0x | 0.415 (+0.000) | 0.213 (-0.009) | - | 0.77 | 10.6 |
| yolo26s-det-1024 | NMS-free | OpenVINO | performance, 512 images | INT8 | 31 | 3.0x | 0.416 (+0.001) | 0.215 (-0.007) | - | 0.77 | 10.6 |
| yolo26s-det-1024 | NMS | ONNX Runtime | percentile | float | 71 | 1.5x | 0.432 (-0.005) | 0.232 (-0.002) | - | 0.82 | 11.9 |
| yolo26s-det-1024 | NMS | ONNX Runtime | percentile | INT8, decoding in float | 61 | 1.8x | 0.433 (-0.003) | 0.229 (-0.005) | - | 0.82 | 10.6 |
| yolo26s-det-1024 | NMS | OpenVINO | performance | INT8 | 31 | 3.0x | 0.346 (-0.091) | 0.143 (-0.092) | - | 0.82 | 10.6 |
| yolo26s-det-1024 | NMS | OpenVINO | performance | INT8, decoding in float | 31 | 3.0x | 0.424 (-0.012) | 0.227 (-0.008) | - | 0.82 | 10.6 |
| yolo11s-det-1024 | NMS | ONNX Runtime | entropy, 32 images | float | 81 | 1.6x | 0.388 (-0.002) | 0.201 (-0.003) | - | 0.73 | 12.8 |
| yolo11s-det-1024 | NMS | ONNX Runtime | entropy | INT8 | 56 | 2.4x | 0.000 (-0.390) | 0.000 (-0.204) | - | 0.00 | 10.3 |
| yolo11s-det-1024 | NMS | ONNX Runtime | entropy | float | 80 | 1.6x | 0.386 (-0.004) | 0.203 (-0.001) | - | 0.79 | 12.8 |
| yolo11s-det-1024 | NMS | ONNX Runtime | entropy | INT8, decoding in float | 66 | 2.0x | 0.390 (+0.000) | 0.201 (-0.003) | - | 0.75 | 10.5 |
| yolo11s-det-1024 | NMS | ONNX Runtime | entropy, 512 images | float | 84 | 1.6x | 0.385 (-0.005) | 0.203 (+0.000) | - | 0.76 | 12.8 |
| yolo11s-det-1024 | NMS | ONNX Runtime | minmax | INT8 | 57 | 2.3x | 0.000 (-0.390) | 0.000 (-0.204) | - | 0.00 | 10.3 |
| yolo11s-det-1024 | NMS | ONNX Runtime | minmax | float | 80 | 1.6x | 0.381 (-0.008) | 0.196 (-0.008) | - | 0.71 | 12.8 |
| yolo11s-det-1024 | NMS | ONNX Runtime | minmax, per-tensor | INT8 | 57 | 2.3x | 0.000 (-0.390) | 0.000 (-0.204) | - | 0.00 | 10.1 |
| yolo11s-det-1024 | NMS | ONNX Runtime | minmax, per-tensor | float | 80 | 1.6x | 0.383 (-0.007) | 0.198 (-0.005) | - | 0.74 | 12.7 |
| yolo11s-det-1024 | NMS | ONNX Runtime | percentile | INT8 | 57 | 2.3x | 0.000 (-0.390) | 0.000 (-0.204) | - | 0.00 | 10.3 |
| yolo11s-det-1024 | NMS | ONNX Runtime | percentile | float | 80 | 1.6x | 0.380 (-0.010) | 0.204 (+0.000) | - | 0.75 | 12.8 |
| yolo11s-det-1024 | NMS | ONNX Runtime | percentile | INT8, decoding in float | 68 | 1.9x | 0.379 (-0.011) | 0.196 (-0.007) | - | 0.78 | 10.5 |
| yolo11s-det-1024 | NMS | OpenVINO | mixed | INT8 | 32 | 3.0x | 0.330 (-0.059) | 0.141 (-0.063) | - | 0.77 | 10.4 |
| yolo11s-det-1024 | NMS | OpenVINO | mixed | float | 40 | 2.4x | 0.385 (-0.005) | 0.203 (-0.001) | - | 0.76 | 12.8 |
| yolo11s-det-1024 | NMS | OpenVINO | performance, 32 images | float | 36 | 2.6x | 0.390 (+0.000) | 0.203 (-0.001) | - | 0.72 | 12.8 |
| yolo11s-det-1024 | NMS | OpenVINO | performance | INT8 | 28 | 3.3x | 0.243 (-0.147) | 0.094 (-0.110) | - | 0.78 | 10.4 |
| yolo11s-det-1024 | NMS | OpenVINO | performance | float | 37 | 2.5x | 0.391 (+0.001) | 0.204 (+0.000) | - | 0.78 | 12.8 |
| yolo11s-det-1024 | NMS | OpenVINO | performance | INT8, decoding in float | 28 | 3.3x | 0.386 (-0.004) | 0.193 (-0.011) | - | 0.78 | 10.4 |
| yolo11s-det-1024 | NMS | OpenVINO | performance, 512 images | float | 36 | 2.6x | 0.389 (-0.001) | 0.203 (-0.001) | - | 0.71 | 12.8 |
| yolo26s-seg-1024 | NMS-free | ONNX Runtime | percentile | float | 141 | 1.3x | 0.429 (-0.007) | 0.216 (-0.004) | 0.400 (-0.011) | 0.82 | 15.5 |
| yolo26s-seg-1024 | NMS-free | ONNX Runtime | percentile | INT8, decoding in float | 104 | 1.8x | 0.424 (-0.012) | 0.213 (-0.007) | 0.392 (-0.019) | 0.81 | 11.6 |
| yolo26s-seg-1024 | NMS-free | OpenVINO | performance | INT8 | 42 | 3.4x | 0.430 (-0.006) | 0.214 (-0.006) | 0.398 (-0.013) | 0.84 | 11.7 |
| yolo26s-seg-1024 | NMS-free | OpenVINO | performance | INT8, decoding in float | 42 | 3.4x | 0.430 (-0.006) | 0.214 (-0.006) | 0.398 (-0.013) | 0.84 | 11.7 |
| yolo11s-seg-1024 | NMS | ONNX Runtime | percentile | float | 141 | 1.3x | 0.418 (-0.012) | 0.223 (-0.005) | 0.402 (-0.008) | 0.79 | 15.4 |
| yolo11s-seg-1024 | NMS | ONNX Runtime | percentile | INT8, decoding in float | 104 | 1.8x | 0.414 (-0.017) | 0.214 (-0.015) | 0.392 (-0.018) | 0.79 | 11.2 |
| yolo11s-seg-1024 | NMS | OpenVINO | performance | INT8 | 39 | 3.5x | 0.318 (-0.113) | 0.133 (-0.096) | 0.330 (-0.080) | 0.81 | 11.2 |
| yolo11s-seg-1024 | NMS | OpenVINO | performance | INT8, decoding in float | 38 | 3.6x | 0.409 (-0.021) | 0.212 (-0.017) | 0.387 (-0.023) | 0.81 | 11.2 |

## NVIDIA H100

TensorRT 11 engines built by Ultralytics through NVIDIA ModelOpt, against PyTorch FP32 on the GPU. Latency is the engine alone with its input in GPU memory, batch 1, median of 1000 runs; PyTorch is timed the same way. INT8 engines quantize every convolution except the head's output convolutions and the DFL, and keep the rest in FP16. "+10 epochs" fine-tunes the baseline in float for 10 more epochs, "+10 epochs QAT" the same with quantization simulated in training; INT8 of a QAT model takes its trained ranges, of any other model ranges calibrated on 512 training images (PTQ).

| Model | Head | Training | Route | p50 ms | p99 ms | Speed-up | mAP50 (vs PyTorch) | mAP50-95 (vs PyTorch) | Mask mAP50 | Recall @ 1% false alarms | Size MB |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| yolo26s-det-1024 | NMS-free | baseline | PyTorch FP32 | 3.97 | 4.58 | 1.0x | 0.422 (+0.000) | 0.223 (+0.000) | - | 0.79 | 20.4 |
| yolo26s-det-1024 | NMS-free | baseline | TensorRT FP32 | 1.63 | 1.64 | 2.4x | 0.415 (-0.007) | 0.222 (-0.001) | - | 0.77 | 245.0 |
| yolo26s-det-1024 | NMS-free | baseline | TensorRT FP16 | 1.17 | 1.17 | 3.4x | 0.421 (-0.002) | 0.223 (+0.000) | - | 0.78 | 209.1 |
| yolo26s-det-1024 | NMS-free | baseline | TensorRT INT8 | 1.07 | 1.07 | 3.7x | 0.398 (-0.025) | 0.210 (-0.013) | - | 0.80 | 96.2 |
| yolo26s-det-1024 | NMS-free | +10 epochs | TensorRT INT8 | 1.07 | 1.08 | 3.7x | 0.391 (-0.032) | 0.201 (-0.022) | - | 0.67 | 96.1 |
| yolo26s-det-1024 | NMS-free | +10 epochs QAT | PyTorch, INT8 simulated | 46.68 | 49.93 | 0.1x | 0.412 (-0.010) | 0.218 (-0.005) | - | 0.76 | 20.8 |
| yolo26s-det-1024 | NMS-free | +10 epochs QAT | TensorRT INT8 | 1.33 | 1.34 | 3.0x | 0.402 (-0.021) | 0.217 (-0.006) | - | 0.72 | 173.2 |
| yolo26s-det-1024 | NMS | baseline | PyTorch FP32 | 3.88 | 4.37 | 1.0x | 0.437 (+0.000) | 0.235 (+0.000) | - | 0.84 | 20.4 |
| yolo26s-det-1024 | NMS | baseline | TensorRT FP16 | 1.15 | 1.16 | 3.4x | 0.436 (-0.001) | 0.236 (+0.001) | - | 0.84 | 223.5 |
| yolo26s-det-1024 | NMS | baseline | TensorRT INT8 | 1.07 | 1.08 | 3.6x | 0.428 (-0.009) | 0.229 (-0.006) | - | 0.84 | 112.3 |
| yolo26s-det-1024 | NMS | +10 epochs QAT | TensorRT INT8 | 1.36 | 1.37 | 2.9x | 0.428 (-0.009) | 0.234 (-0.001) | - | 0.76 | 191.9 |
| yolo11s-det-1024 | NMS | baseline | PyTorch FP32 | 3.25 | 3.74 | 1.0x | 0.389 (+0.000) | 0.204 (+0.000) | - | 0.73 | 19.3 |
| yolo11s-det-1024 | NMS | baseline | TensorRT FP32 | 1.48 | 1.50 | 2.2x | 0.390 (+0.001) | 0.204 (+0.000) | - | 0.75 | 265.4 |
| yolo11s-det-1024 | NMS | baseline | TensorRT FP16 | 1.07 | 1.08 | 3.0x | 0.388 (-0.001) | 0.202 (-0.002) | - | 0.75 | 223.3 |
| yolo11s-det-1024 | NMS | baseline | TensorRT INT8 | 1.01 | 1.02 | 3.2x | 0.371 (-0.018) | 0.192 (-0.012) | - | 0.76 | 130.8 |
| yolo11s-det-1024 | NMS | +10 epochs | TensorRT INT8 | 0.98 | 0.99 | 3.3x | 0.366 (-0.023) | 0.184 (-0.020) | - | 0.70 | 130.8 |
| yolo11s-det-1024 | NMS | +10 epochs QAT | PyTorch, INT8 simulated | 37.54 | 40.93 | 0.1x | 0.389 (+0.000) | 0.198 (-0.006) | - | 0.75 | 19.6 |
| yolo11s-det-1024 | NMS | +10 epochs QAT | TensorRT INT8 | 1.21 | 1.30 | 2.7x | 0.390 (+0.001) | 0.199 (-0.006) | - | 0.77 | 192.1 |
| yolo26s-seg-1024 | NMS-free | baseline | PyTorch FP32 | 5.15 | 5.65 | 1.0x | 0.433 (+0.000) | 0.221 (+0.000) | 0.414 | 0.82 | 23.4 |
| yolo26s-seg-1024 | NMS-free | baseline | TensorRT FP16 | 1.39 | 1.40 | 3.7x | 0.439 (+0.006) | 0.221 (+0.000) | 0.414 | 0.80 | 211.5 |
| yolo11s-seg-1024 | NMS | baseline | PyTorch FP32 | 4.18 | 4.54 | 1.0x | 0.427 (+0.000) | 0.230 (+0.000) | 0.410 | 0.83 | 20.6 |
| yolo11s-seg-1024 | NMS | baseline | TensorRT FP16 | 1.22 | 1.23 | 3.4x | 0.431 (+0.003) | 0.231 (+0.001) | 0.413 | 0.83 | 226.1 |
