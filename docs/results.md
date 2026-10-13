# Results

Every measured variant, generated with `edgeproof report` from [`results/accuracy.jsonl`](../results/accuracy.jsonl) (accuracy through each runtime on the 960 test images) and [`results/latency.jsonl`](../results/latency.jsonl) (all variants in one run on an otherwise idle node: Intel Xeon Platinum 8468, 4 threads, batch 1, 1024x1024, the network alone). Recall is image-level, at the threshold that rejects 1% of the 720 defect-free test images. Heads: NMS-free is YOLO26's one-to-one head, NMS the one-to-many head followed by NMS.

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

## INT8 after training

128 calibration images from the training split. ONNX Runtime uses QDQ with UInt8 activations and Int8 weights, per channel unless noted; OpenVINO uses NNCF presets (performance: symmetric, mixed: asymmetric activations). Head "float" keeps the detection head unquantized. Speed-up and differences are against FP32 on the same model, head and runtime.

| Model | Runtime | Calibration | Head | p50 ms | Speed-up | mAP50 (vs FP32) | mAP50-95 (vs FP32) | Recall @ 1% false alarms | Size MB |
|---|---|---|---|--:|--:|--:|--:|--:|--:|
| yolo26s-det-1024 | ONNX Runtime | minmax | INT8 | 60 | 2.0x | 0.000 (-0.415) | 0.000 (-0.222) | 0.00 | 10.5 |
| yolo26s-det-1024 | ONNX Runtime | minmax | float | 72 | 1.7x | 0.399 (-0.016) | 0.213 (-0.008) | 0.77 | 11.9 |
| yolo26s-det-1024 | ONNX Runtime | minmax, per-tensor | INT8 | 62 | 2.0x | 0.000 (-0.415) | 0.000 (-0.222) | 0.00 | 10.2 |
| yolo26s-det-1024 | ONNX Runtime | minmax, per-tensor | float | 72 | 1.7x | 0.357 (-0.058) | 0.189 (-0.033) | 0.61 | 11.7 |
| yolo26s-det-1024 | ONNX Runtime | entropy | INT8 | 62 | 1.9x | 0.000 (-0.415) | 0.000 (-0.222) | 0.00 | 10.5 |
| yolo26s-det-1024 | ONNX Runtime | entropy | float | 71 | 1.7x | 0.406 (-0.009) | 0.217 (-0.005) | 0.80 | 11.9 |
| yolo26s-det-1024 | ONNX Runtime | percentile | INT8 | 61 | 2.0x | 0.000 (-0.415) | 0.000 (-0.222) | 0.00 | 10.5 |
| yolo26s-det-1024 | ONNX Runtime | percentile | float | 71 | 1.7x | 0.417 (+0.002) | 0.217 (-0.005) | 0.78 | 11.9 |
| yolo26s-det-1024 | OpenVINO | performance | INT8 | 31 | 3.0x | 0.415 (+0.000) | 0.213 (-0.009) | 0.77 | 10.6 |
| yolo26s-det-1024 | OpenVINO | performance | float | 37 | 2.5x | 0.419 (+0.004) | 0.214 (-0.008) | 0.75 | 11.9 |
| yolo26s-det-1024 | OpenVINO | mixed | INT8 | 35 | 2.7x | 0.406 (-0.009) | 0.204 (-0.018) | 0.78 | 10.6 |
| yolo26s-det-1024 | OpenVINO | mixed | float | 39 | 2.4x | 0.415 (+0.000) | 0.213 (-0.009) | 0.76 | 11.9 |
| yolo11s-det-1024 | ONNX Runtime | minmax | INT8 | 57 | 2.3x | 0.000 (-0.390) | 0.000 (-0.204) | 0.00 | 10.3 |
| yolo11s-det-1024 | ONNX Runtime | minmax | float | 80 | 1.6x | 0.381 (-0.008) | 0.196 (-0.008) | 0.71 | 12.8 |
| yolo11s-det-1024 | ONNX Runtime | minmax, per-tensor | INT8 | 57 | 2.3x | 0.000 (-0.390) | 0.000 (-0.204) | 0.00 | 10.1 |
| yolo11s-det-1024 | ONNX Runtime | minmax, per-tensor | float | 80 | 1.6x | 0.383 (-0.007) | 0.198 (-0.005) | 0.74 | 12.7 |
| yolo11s-det-1024 | ONNX Runtime | entropy | INT8 | 56 | 2.4x | 0.000 (-0.390) | 0.000 (-0.204) | 0.00 | 10.3 |
| yolo11s-det-1024 | ONNX Runtime | entropy | float | 80 | 1.6x | 0.386 (-0.004) | 0.203 (-0.001) | 0.79 | 12.8 |
| yolo11s-det-1024 | ONNX Runtime | percentile | INT8 | 57 | 2.3x | 0.000 (-0.390) | 0.000 (-0.204) | 0.00 | 10.3 |
| yolo11s-det-1024 | ONNX Runtime | percentile | float | 80 | 1.6x | 0.380 (-0.010) | 0.204 (+0.000) | 0.75 | 12.8 |
| yolo11s-det-1024 | OpenVINO | performance | INT8 | 28 | 3.3x | 0.243 (-0.147) | 0.094 (-0.110) | 0.78 | 10.4 |
| yolo11s-det-1024 | OpenVINO | performance | float | 37 | 2.5x | 0.391 (+0.001) | 0.204 (+0.000) | 0.78 | 12.8 |
| yolo11s-det-1024 | OpenVINO | mixed | INT8 | 32 | 3.0x | 0.330 (-0.059) | 0.141 (-0.063) | 0.77 | 10.4 |
| yolo11s-det-1024 | OpenVINO | mixed | float | 40 | 2.4x | 0.385 (-0.005) | 0.203 (-0.001) | 0.76 | 12.8 |
