#pragma once

#include <opencv2/opencv.hpp>

cv::Mat
    compute_powertower(
        int rows, int cols, float range_left, float range_right, float range_top, float range_bottom,
        int max_iterations = 200, int threshold = 100
    );