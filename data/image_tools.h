#ifndef DATA_MAKE_HOMOGRAPHY_DATA__
#define DATA_MAKE_HOMOGRAPHY_DATA___


#include <iostream>
#include <random>

#include "opencv2/opencv.hpp"

const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);
const cv::Scalar MAG(255, 0, 255);

int randint(std::mt19937& gen, int min, int max);
void print_dim(cv::Mat& x);
void plot_pts(cv::Mat& img, std::vector<cv::Point>& pts);
void draw_poly(cv::Mat& img, std::vector<cv::Point>& pts, const cv::Scalar& color, int thickness=1);

class Patch {
    int x_max, y_max, patch_size, max_jitter;
    std::vector<cv::Point> corners;

  public:
    Patch(cv::Mat& img, int patch_size, int max_jitter);

    std::vector<cv::Point> get_corners(); 
    void random_shift(std::mt19937& gen);
    void random_skew(std::mt19937& gen);

};

#endif
