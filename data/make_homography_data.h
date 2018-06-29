#ifndef DATA_MAKE_HOMOGRAPHY_DATA__
#define DATA_MAKE_HOMOGRAPHY_DATA___


#include <iostream>

class Patch {
  int max_x, max_y;
  public:
    int x_left, x_right, y_left, y_right;
    Patch(Mat& img, int patch_size) : max_x(img.cols), max_y(img.rows) {}

    void random_shift(){
      std::cout << max_x << std::endl;
    }

    void random_skew(){
      std::cout << max_y << std::endl;
    }
};


#endif
