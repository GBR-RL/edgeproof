// edgeproof-run: runs an exported YOLO model (ONNX, FP32 or INT8 Q/DQ) on a folder of images
// through the ONNX Runtime C++ API, the way a line application would: read, letterbox, infer,
// decode, scale back. Writes one JSON line of detections per image and prints the median time of
// each stage, measured the same way as Ultralytics' per-stage speeds (image reading excluded).
//
// Usage: edgeproof-run --model m.onnx --images dir [--imgsz 1024] [--threads 4] [--conf 0.25]
//                      [--out detections.jsonl]

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "edgeproof/yolo.hpp"

namespace fs = std::filesystem;
using edgeproof::Detection;
using edgeproof::Letterbox;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
  std::string model;
  std::string images;
  std::string out = "detections.jsonl";
  int imgsz = 1024;
  int threads = 4;
  float conf = 0.25f;
};

Args parse(int argc, char** argv) {
  Args a;
  std::map<std::string, std::string> kv;
  for (int i = 1; i + 1 < argc; i += 2) kv[argv[i]] = argv[i + 1];
  if (!kv.contains("--model") || !kv.contains("--images")) {
    throw std::invalid_argument("usage: edgeproof-run --model m.onnx --images dir [--imgsz N] "
                                "[--threads N] [--conf X] [--out file]");
  }
  a.model = kv["--model"];
  a.images = kv["--images"];
  if (kv.contains("--out")) a.out = kv["--out"];
  if (kv.contains("--imgsz")) a.imgsz = std::stoi(kv["--imgsz"]);
  if (kv.contains("--threads")) a.threads = std::stoi(kv["--threads"]);
  if (kv.contains("--conf")) a.conf = std::stof(kv["--conf"]);
  return a;
}

// BGR image to the network input: letterboxed, RGB, CHW, float in [0, 1].
Letterbox preprocess(const cv::Mat& bgr, int size, std::vector<float>& blob) {
  const Letterbox lb = edgeproof::letterbox(bgr.cols, bgr.rows, size);
  cv::Mat resized = bgr;
  if (lb.resized_w != bgr.cols || lb.resized_h != bgr.rows) {
    cv::resize(bgr, resized, cv::Size(lb.resized_w, lb.resized_h), 0, 0, cv::INTER_LINEAR);
  }
  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, lb.top, size - lb.resized_h - lb.top, lb.left,
                     size - lb.resized_w - lb.left, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  const auto plane = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
  blob.resize(3 * plane);
  for (int y = 0; y < size; ++y) {
    const auto* row = padded.ptr<cv::Vec3b>(y);
    for (int x = 0; x < size; ++x) {
      const auto i = static_cast<std::size_t>(y) * static_cast<std::size_t>(size) +
                     static_cast<std::size_t>(x);
      blob[i] = static_cast<float>(row[x][2]) / 255.0f;
      blob[plane + i] = static_cast<float>(row[x][1]) / 255.0f;
      blob[2 * plane + i] = static_cast<float>(row[x][0]) / 255.0f;
    }
  }
  return lb;
}

double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

}  // namespace

int main(int argc, char** argv) try {
  const Args args = parse(argc, argv);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edgeproof");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(args.threads);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // Idle workers would otherwise spin between images and compete with pre- and post-processing.
  options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  Ort::Session session(env, args.model.c_str(), options);

  Ort::AllocatorWithDefaultOptions allocator;
  const std::string input_name = session.GetInputNameAllocated(0, allocator).get();
  const std::string output_name = session.GetOutputNameAllocated(0, allocator).get();
  const auto out_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (out_shape.size() != 3) throw std::runtime_error("expected a (1, a, b) detection output");
  // (1, 300, 6): the NMS-free head; (1, 4 + classes, anchors): candidates for NMS.
  const bool end2end = out_shape[2] == 6;
  const char* input_names[] = {input_name.c_str()};
  const char* output_names[] = {output_name.c_str()};
  const std::array<std::int64_t, 4> input_shape{1, 3, args.imgsz, args.imgsz};
  const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(args.images)) {
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) throw std::runtime_error("no images in " + args.images);

  std::ofstream out(args.out);
  std::vector<float> blob;
  std::vector<double> pre, infer, post, total;
  for (const fs::path& file : files) {
    const cv::Mat image = cv::imread(file.string(), cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("cannot read " + file.string());

    const auto t0 = Clock::now();
    const Letterbox lb = preprocess(image, args.imgsz, blob);
    const auto t1 = Clock::now();
    auto input = Ort::Value::CreateTensor<float>(memory, blob.data(), blob.size(),
                                                 input_shape.data(), input_shape.size());
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    const auto t2 = Clock::now();
    const float* data = outputs[0].GetTensorData<float>();
    const auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const std::span<const float> view(data, count);
    std::vector<Detection> dets =
        end2end ? edgeproof::decode_end2end(view, static_cast<std::size_t>(out_shape[1]), args.conf)
                : edgeproof::decode_nms(view, static_cast<std::size_t>(out_shape[2]), args.conf);
    edgeproof::to_original(dets, lb);
    const auto t3 = Clock::now();

    pre.push_back(ms(t0, t1));
    infer.push_back(ms(t1, t2));
    post.push_back(ms(t2, t3));
    total.push_back(ms(t0, t3));

    out << R"({"image": ")" << file.filename().string() << R"(", "boxes": [)";
    for (std::size_t i = 0; i < dets.size(); ++i) {
      const auto& d = dets[i];
      char buf[160];
      std::snprintf(buf, sizeof buf, "%s[%.3f, %.3f, %.3f, %.3f, %.5f]", i ? ", " : "",
                    static_cast<double>(d.box.x1), static_cast<double>(d.box.y1),
                    static_cast<double>(d.box.x2), static_cast<double>(d.box.y2),
                    static_cast<double>(d.score));
      out << buf;
    }
    out << "]}\n";
  }
  std::printf(
      "{\"images\": %zu, \"head\": \"%s\", \"threads\": %d, \"preprocess_ms\": %.2f, "
      "\"inference_ms\": %.2f, \"postprocess_ms\": %.2f, \"total_ms\": %.2f}\n",
      files.size(), end2end ? "e2e" : "nms", args.threads, median(pre), median(infer),
      median(post), median(total));
  return 0;
} catch (const std::exception& e) {
  std::cerr << "edgeproof-run: " << e.what() << '\n';
  return 1;
}
