// Pre- and post-processing of the exported YOLO models, without OpenCV or ONNX Runtime, so the
// geometry and the decoding are testable on their own. Mirrors what Ultralytics does in Python:
// letterbox to a square with gray padding, then either the NMS-free head's top-k detections
// (YOLO26) or the raw candidates through non-maximum suppression (YOLO11, YOLO26's NMS head).
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace edgeproof {

struct Box {
  float x1, y1, x2, y2;
};

struct Detection {
  Box box;
  float score;
};

// Where the original image lands inside the square network input.
struct Letterbox {
  int width, height;          // original image
  int resized_w, resized_h;   // after scaling
  int left, top;              // padding before the image
};

Letterbox letterbox(int width, int height, int input_size);

float iou(const Box& a, const Box& b) noexcept;

// YOLO26's one-to-one head: (max_det, 6) rows of x1, y1, x2, y2, score, class in input pixels.
std::vector<Detection> decode_end2end(std::span<const float> output, std::size_t rows,
                                      float conf);

// The one-to-many head: (4 + classes, anchors) channels-first, boxes as centre, width, height.
// Single class here, as in this study. Candidates above ``conf`` go through NMS.
std::vector<Detection> decode_nms(std::span<const float> output, std::size_t anchors, float conf,
                                  float iou_threshold = 0.7f, std::size_t max_det = 300,
                                  std::size_t max_candidates = 30000);

// Input pixels back to the original image, clipped to it.
void to_original(std::vector<Detection>& detections, const Letterbox& lb) noexcept;

}  // namespace edgeproof
