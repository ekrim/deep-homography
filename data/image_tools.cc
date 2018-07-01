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


int randint(std::mt19937& gen, unsigned min, unsigned max){
  return min + (gen() % (max-min+1));
}


void print_dim(Mat& x){
  cout << "image size: (" << x.rows << ", " << x.cols << ", " << x.channels();
  cout << ")" << endl;
}

/**
void compute_shift(vector<int>& shift_vals, int patch, int img){ 
  shift_vals[1] = img - patch - shift_vals[0];
  if (img < patch_rows){
    throw "image not big enough for patch";
  } else if (shift_vals[1] < 0){
    shift_vals[0] = img - patch;
    shift_vals[1] = 0;
  }
}

void make_points(
    Mat& img, vector<Point2f>& pts1, vector<Point2f>& pts2, 
    std::mt19937& gen, 
    int patch_rows, int patch_cols, int max_jitter){

  vector<int> delta_x(2); 
  vector<int> delta_y(2);
  compute_shift(delta_x, patch/2, img.cols/2);
  compute_shift(delta_y, patch/2, img.rows/2);
  
  Point(randint(gen, 0, delta_x[1]) 
  

  pts1.push_back(Point2f(offset, offset));
  pts1.push_back(Point2f(img.cols-offset, offset));
  pts1.push_back(Point2f(img.cols-offset, img.rows-offset));
  pts1.push_back(Point2f(offset, img.rows-offset));
}


void make_points(Mat& img, vector<Point2f>& pts, int offset, std::mt19937& gen){
  vector<int> x_vec{ offset, img.cols-offset, img.cols-offset, offset};
  vector<int> y_vec{ offset, offset, img.rows-offset, img.rows-offset};
  
  int delta_x, delta_y;
  for (vector<int>::size_type i = 0; i != x_vec.size(); i++){
    delta_x = randint(gen, 0, 2*offset) - offset;
    delta_y = randint(gen, 0, 2*offset) - offset;
    pts.push_back(Point2f(x_vec[i]+delta_x, y_vec[i]+delta_y));
  }
}
*/

void plot_pts(Mat& img, vector<Point2f>& pts){
  for (auto const &pt : pts){
    circle(img, pt, 3.0, BLUE, -1, 8);
  }
}


void draw_poly(Mat& img, vector<Point2f>& pts, const Scalar& color, int thickness){
  for (vector<int>::size_type i = 0; i != pts.size(); i++){
    int i_next = (i == pts.size()-1) ? 0 : i+1;  
    line(img, pts[i], pts[i_next], color, thickness);
  }
}


Patch::Patch(Mat& img, int patch_size) : max_x(img.cols), max_y(img.rows) {}

void Patch::random_shift(void){
  cout << max_x << endl;
}

void Patch::random_skew(void){
  cout << max_y << endl;
}



