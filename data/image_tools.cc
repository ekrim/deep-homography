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


void plot_pts(Mat& img, vector<Point>& pts){
  for (auto const& pt : pts){
    circle(img, pt, 3.0, BLUE, -1, 8);
  }
}


void draw_poly(Mat& img, vector<Point>& pts, const Scalar& color, int thickness){
  for (vector<int>::size_type i = 0; i != pts.size(); i++){
    int i_next = (i == pts.size()-1) ? 0 : i+1;  
    line(img, pts[i], pts[i_next], color, thickness);
  }
}


Patch::Patch(Mat& img, int patch_size, int max_jitter) 
    : x_max(img.cols)
    , y_max(img.rows)
    , patch_size(patch_size)
    , max_jitter(max_jitter) {

  if (patch_size > x_max || patch_size > y_max){
    throw "image not big enough for patch";
  }
  
  if (max_jitter >= patch_size/2){
    throw "too much jitter!";
  }

  int x_margin = (x_max - patch_size)/2;
  int y_margin = (y_max - patch_size)/2;
  max_jitter = x_margin > max_jitter ? max_jitter : x_margin; 
  max_jitter = y_margin > max_jitter ? max_jitter : y_margin; 

  corners.push_back(Point(x_margin, y_margin));
  corners.push_back(Point(x_margin+patch_size, y_margin));
  corners.push_back(Point(x_margin+patch_size, y_margin+patch_size)); 
  corners.push_back(Point(x_margin, y_margin+patch_size));
}


void Patch::random_shift(std::mt19937& gen){
  int max_x_shift = corners[0].x - max_jitter;  
  int max_y_shift = corners[0].y - max_jitter;

  int delta_x = randint(gen, -max_x_shift, max_x_shift); 
  int delta_y = randint(gen, -max_y_shift, max_y_shift); 
  for (auto& pt : corners){
    pt.x += delta_x;
    pt.y += delta_y;
  }
}


void Patch::random_skew(std::mt19937& gen){
  for (auto& pt : corners){
    pt.x += randint(gen, -max_jitter, max_jitter);
    pt.x = pt.x < 0 ? 0 : pt.x;
    pt.x = pt.x >= x_max ? x_max : pt.x;
    pt.y += randint(gen, -max_jitter, max_jitter);
    pt.y = pt.y < 0 ? 0 : pt.y;
    pt.y = pt.y > y_max ? y_max : pt.y;
  }
}


// return the vector of corners
vector<Point> Patch::get_corners(){
  return corners;  
}
