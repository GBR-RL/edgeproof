#include "edgeproof/yolo.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace edgeproof {

namespace {

// Python's round(): halves to even, which std::lround does not do.
int py_round(double x) { return static_cast<int>(std::nearbyint(x)); }

}  // namespace

Letterbox letterbox(int width, int height, int input_size) {
  if (width <= 0 || height <= 0 || input_size <= 0) {
    throw std::invalid_argument("letterbox: sizes must be positive");
  }
  const double gain = std::min(static_cast<double>(input_size) / height,
                               static_cast<double>(input_size) / width);
  Letterbox lb{};
  lb.width = width;
  lb.height = height;
  lb.resized_w = py_round(width * gain);
  lb.resized_h = py_round(height * gain);
  // Ultralytics pads round(d - 0.1) before and round(d + 0.1) after: an odd remainder goes to
  // the right and bottom.
  lb.left = py_round((input_size - lb.resized_w) / 2.0 - 0.1);
  lb.top = py_round((input_size - lb.resized_h) / 2.0 - 0.1);
  return lb;
}

float iou(const Box& a, const Box& b) noexcept {
  const float w = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float h = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = w * h;
  const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<Detection> decode_end2end(std::span<const float> output, std::size_t rows,
                                      float conf) {
  if (output.size() < rows * 6) throw std::invalid_argument("decode_end2end: output too small");
  std::vector<Detection> out;
  for (std::size_t i = 0; i < rows; ++i) {
    const float* r = output.data() + i * 6;
    if (r[4] > conf) out.push_back({{r[0], r[1], r[2], r[3]}, r[4]});
  }
  return out;  // the head sorts by score already
}

std::vector<Detection> decode_nms(std::span<const float> output, std::size_t anchors, float conf,
                                  float iou_threshold, std::size_t max_det,
                                  std::size_t max_candidates) {
  if (output.size() < anchors * 5) throw std::invalid_argument("decode_nms: output too small");
  const float* cx = output.data();
  const float* cy = cx + anchors;
  const float* w = cy + anchors;
  const float* h = w + anchors;
  const float* score = h + anchors;
  std::vector<Detection> candidates;
  for (std::size_t i = 0; i < anchors; ++i) {
    if (score[i] <= conf) continue;
    const float hw = w[i] / 2.0f;
    const float hh = h[i] / 2.0f;
    candidates.push_back({{cx[i] - hw, cy[i] - hh, cx[i] + hw, cy[i] + hh}, score[i]});
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Detection& a, const Detection& b) { return a.score > b.score; });
  if (candidates.size() > max_candidates) candidates.resize(max_candidates);

  std::vector<Detection> kept;
  std::vector<char> suppressed(candidates.size(), 0);
  for (std::size_t i = 0; i < candidates.size() && kept.size() < max_det; ++i) {
    if (suppressed[i] != 0) continue;
    kept.push_back(candidates[i]);
    for (std::size_t j = i + 1; j < candidates.size(); ++j) {
      if (suppressed[j] == 0 && iou(candidates[i].box, candidates[j].box) > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  return kept;
}

void to_original(std::vector<Detection>& detections, const Letterbox& lb) noexcept {
  // Per axis, from the rounded size, as Ultralytics' scale_boxes does.
  const auto gx = static_cast<float>(lb.resized_w) / static_cast<float>(lb.width);
  const auto gy = static_cast<float>(lb.resized_h) / static_cast<float>(lb.height);
  const auto w = static_cast<float>(lb.width);
  const auto h = static_cast<float>(lb.height);
  const auto l = static_cast<float>(lb.left);
  const auto t = static_cast<float>(lb.top);
  for (Detection& d : detections) {
    d.box = {std::clamp((d.box.x1 - l) / gx, 0.0f, w), std::clamp((d.box.y1 - t) / gy, 0.0f, h),
             std::clamp((d.box.x2 - l) / gx, 0.0f, w), std::clamp((d.box.y2 - t) / gy, 0.0f, h)};
  }
}

}  // namespace edgeproof
