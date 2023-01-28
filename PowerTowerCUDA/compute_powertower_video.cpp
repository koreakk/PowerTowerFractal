#include <opencv2/opencv.hpp>
#include "compute_powertower.h"

void compute_powertower_video(cv::VideoWriter& writer, int rows, int cols, int frame, int start_x, int start_y,
                              int zoom_factor, int max_iterations, int threshold)
{
    double px = 0, py = 0;

    for (int i = 0; i < frame; ++i) {
        double p = 2.0 * pow(zoom_factor, i);

        // Compute the range of complex numbers for the current frame
        double range_left = px - p, range_right = px + p;
        double range_top = py - p, range_bottom = py + p;

        // Compute the power tower image for the current range of complex numbers
        cv::Mat image = compute_powertower(rows, cols, range_left, range_right, range_top, range_bottom, max_iterations, threshold);

        // Write the image to the video file
        writer.write(image);

        // Update the center point of the zoom
        px += (1 - zoom_factor) * (start_x - px);
        py += (1 - zoom_factor) * (start_y - py);
    }
}
