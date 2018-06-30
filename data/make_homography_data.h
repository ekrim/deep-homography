#ifndef DATA_MAKE_HOMOGRAPHY_DATA__
#define DATA_MAKE_HOMOGRAPHY_DATA___


#include <iostream>

class Patch {
  private:
    int max_x, max_y;

  public:
    int x_left, x_right, y_left, y_right;
    Patch(Mat& img, int patch_size);

    void random_shift();
    void random_skew();

};


#endif
