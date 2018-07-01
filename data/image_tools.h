#ifndef DATA_MAKE_HOMOGRAPHY_DATA__
#define DATA_MAKE_HOMOGRAPHY_DATA___


#include <iostream>
#include <random>

#include "opencv2/opencv.hpp"

const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);
const cv::Scalar MAG(255, 0, 255);

int randint(std::mt19937& gen, unsigned min, unsigned max);
void print_dim(cv::Mat& x);
void plot_pts(cv::Mat& img, std::vector<cv::Point2f>& pts);
void draw_poly(cv::Mat& img, std::vector<cv::Point2f>& pts, const cv::Scalar& color, int thickness=1);

class Patch {
  private:
    int max_x, max_y;

  public:
    int x_left, x_right, y_left, y_right;
    Patch(cv::Mat& img, int patch_size);

    void random_shift(void);
    void random_skew(void);

};

#endif
