#include <opencv2/opencv.hpp>
#include <complex>

cv::Mat compute_powertower(int rows, int cols, float range_left, float range_right, float range_top, float range_bottom,
                           int height = 200, int threshold = 100) {
    auto powertower_formula = [&height, &threshold](const std::complex<float>& c) -> uchar {
        std::complex<float> z = c;
        for (int i = 0; i < height; ++i) {
            z = std::pow(c, z);
            if (std::abs(z) > threshold) { return (uchar)255; }
        }
        return (uchar)0;
    };

    float scaleX = cols / (range_right - range_left);
    float scaleY = rows / (range_bottom - range_top);

    cv::Mat powertower_set(rows, cols, CV_8UC1);
    uchar* data = powertower_set.data;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            int offset = y * cols;
            for (int x = 0; x < cols; ++x) {
                float real = x / scaleX + range_left;
                float imag = y / scaleY + range_top;
                std::complex<float> c(real, imag);
                data[offset + x] = powertower_formula(c);
            }
        }
    });


    return powertower_set;
}