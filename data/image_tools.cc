#include "image_tools.h"

#include <iostream>
#include <fstream>
#include <iostream>
#include <random>

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"


using std::cout;
using std::endl;
using std::vector;
using namespace cv;


// one random integer in [min, max]
int randint(std::mt19937& gen, int min, int max){
  return min + (gen() % (max-min+1));
}


void print_dim(Mat& x){
  cout << "image size: (" << x.rows << ", " << x.cols << ", " << x.channels();
  cout << ")" << endl;
}


void plot_pts(Mat& img, vector<Point2f>& pts){
  for (auto const& pt : pts){
    circle(img, pt, 3.0, BLUE, -1, 8);
  }
}


void draw_poly(Mat& img, vector<Point2f>& pts, const Scalar& color, int thickness){
  for (vector<int>::size_type i = 0; i != pts.size(); i++){
    int i_next = (i == pts.size()-1) ? 0 : i+1;  
    line(img, pts[i], pts[i_next], color, thickness);
  }
}


Patch::Patch(Mat& img, int patch_size, int max_jitter) : max_x(img.cols), max_y(img.rows), patch_size(patch_size), max_jitter(max_jitter) {
  if (patch_size > max_x || patch_size > max_y){
    throw "image not big enough for patch";
  }
  if (max_jitter + (patch_size
  corners[0] = 
}


void Patch::random_shift(std::mt19937& gen){
  int x_margin = (max_x - patch_size)/2;  
  int y_margin = (max_y - patch_size)/2;
  for (auto const& pt : corners){
    
  }

  int delta_x = get_delta(gen, x_margin);
  int delta_y = get_delta(gen, y_margin);

  shift_patch(delta_x, delta_y);
}


void Patch::random_skew(std::mt19937& gen){
  for (auto const& pt : corners){
    pt.x += randint(gen, -max_jitter, max_jitter);
    pt.x = pt.x < 0 ? 0 : pt.x;
    pt.x = pt.x >= x_max ? x_max : pt.x;
    pt.y += randint(gen, -max_jitter, max_jitter);
    pt.y = pt.y < 0 ? 0 : pt.y;
    pt.y = pt.y > y_max ? y_max : pt.y;
  }
}


// return the vector of corners
vector<Point2f> Patch::get_corners(){
  return corners;  
}


// shift the patch by a given amount
void Patch::shift_patch(int delta_x, int delta_y){
  for (auto const& pt : corners){
    pt.x += delta_x;
    pt.y += delta_y
  }
}


// given a desired available margin, compute the delta shift for the patch
int Patch::get_delta(std::mt19937& gen, int margin){ 
  // if the margin cant fit in the desired jitter, split the difference
  if (margin - max_jitter < 0){
    max_jitter = margin/2;
    margin /= 2;
  // otherwise, remove the space reserved for jitter 
  } else {
    margin -= max_jitter; 
  }
  int delta = randint(gen, -margin, margin);
  return delta;    
}
