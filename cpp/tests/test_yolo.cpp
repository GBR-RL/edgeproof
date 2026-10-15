// Geometry and decoding against values worked out by hand, and against Ultralytics' rounding.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "edgeproof/yolo.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what, int line) {
  if (!ok) {
    std::fprintf(stderr, "line %d: %s\n", line, what);
    ++failures;
  }
}

#define CHECK(x) check((x), #x, __LINE__)

bool near(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol; }

void letterbox_matches_ultralytics() {
  // VisA's 1500x1000 images: gain 1024/1500, 1024x683, 170 rows above and 171 below.
  const auto lb = edgeproof::letterbox(1500, 1000, 1024);
  CHECK(lb.resized_w == 1024 && lb.resized_h == 683);
  CHECK(lb.left == 0 && lb.top == 170);
  // A square image fills the input exactly.
  const auto sq = edgeproof::letterbox(512, 512, 1024);
  CHECK(sq.resized_w == 1024 && sq.left == 0 && sq.top == 0);
  // Python rounds halves to even: 8x5 into 4 scales to 4x2.5, which Python makes 4x2 (not 4x3),
  // padded round(0.9) = 1 row above.
  const auto tie = edgeproof::letterbox(8, 5, 4);
  CHECK(tie.resized_w == 4 && tie.resized_h == 2 && tie.top == 1);
}

void boxes_go_back_to_the_original() {
  const auto lb = edgeproof::letterbox(1500, 1000, 1024);
  std::vector<edgeproof::Detection> d{{{0.0f, 170.0f, 1024.0f, 853.0f}, 0.9f},
                                      {{-5.0f, 100.0f, 2000.0f, 200.0f}, 0.5f}};
  edgeproof::to_original(d, lb);
  CHECK(near(d[0].box.x1, 0.0f) && near(d[0].box.y1, 0.0f));
  CHECK(near(d[0].box.x2, 1500.0f) && near(d[0].box.y2, 1000.0f, 1e-2f));
  CHECK(d[1].box.x1 == 0.0f && d[1].box.y1 == 0.0f && d[1].box.x2 == 1500.0f);  // clipped
}

void end2end_keeps_rows_above_the_threshold() {
  const std::vector<float> out{10, 10, 20, 20, 0.9f, 0,  //
                               30, 30, 40, 40, 0.2f, 0,  //
                               50, 50, 60, 60, 0.25f, 0};
  const auto d = edgeproof::decode_end2end(out, 3, 0.25f);
  CHECK(d.size() == 1 && near(d[0].score, 0.9f) && near(d[0].box.x2, 20.0f));
}

void nms_suppresses_overlaps_and_keeps_order() {
  // Channels first: cx, cy, w, h, score for 4 anchors.
  const std::vector<float> out{
      50, 52, 200, 50,   // cx
      50, 50, 200, 50,   // cy
      20, 20, 10, 20,    // w
      20, 20, 10, 20,    // h
      0.8f, 0.9f, 0.7f, 0.1f,
  };
  const auto d = edgeproof::decode_nms(out, 4, 0.25f);
  CHECK(d.size() == 2);                     // anchor 0 overlaps anchor 1, anchor 3 is below conf
  CHECK(near(d[0].score, 0.9f) && near(d[0].box.x1, 42.0f));
  CHECK(near(d[1].score, 0.7f) && near(d[1].box.x1, 195.0f));
  CHECK(edgeproof::decode_nms(out, 4, 0.25f, 0.95f).size() == 3);  // looser IoU keeps both
}

}  // namespace

int main() {
  letterbox_matches_ultralytics();
  boxes_go_back_to_the_original();
  end2end_keeps_rows_above_the_threshold();
  nms_suppresses_overlaps_and_keeps_order();
  if (failures == 0) std::puts("all checks passed");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
