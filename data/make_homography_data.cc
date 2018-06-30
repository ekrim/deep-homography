#include "make_homography_data.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <experimental/filesystem>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/videoio.hpp"

using std::cout;
using std::endl;
using std::vector;
using namespace cv;
using namespace cv::xfeatures2d;
namespace fs = std::experimental::filesystem;


const Scalar BLUE(255, 0, 0);
const Scalar GREEN(0, 255, 0);
const Scalar RED(0, 0, 255);
const Scalar MAG(255, 0, 255);


void readme(){
  cout << " Usage: ./make_homography_data <img_directory>" << endl;
}


int randint(std::mt19937& gen, unsigned min, unsigned max){
  return min + (gen() % (max-min+1));
}


void print_dim(Mat& x){
  cout << "image size: (" << x.rows << ", " << x.cols << ", " << x.channels();
  cout << ")" << endl;
}


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


void plot_pts(Mat& img, vector<Point2f>& pts){
  circle(img, pts[0], 3.0, BLUE, -1, 8);
  circle(img, pts[1], 3.0, GREEN, -1, 8);
  circle(img, pts[2], 3.0, RED, -1, 8);
  circle(img, pts[3], 3.0, MAG, -1, 8);
}


void draw_poly(Mat& img, vector<Point2f>& pts, const Scalar& color, int thickness=1){
  for (vector<int>::size_type i = 0; i != pts.size(); i++){
    int i_next = (i == pts.size()-1) ? 0 : i+1;  
    line(img, pts[i], pts[i_next], color, thickness);
  }
}


class Patch {
  int max_x, max_y;
  public:
    int x_left, x_right, y_left, y_right;
    Patch(Mat& img, int patch_size) : max_x(img.cols), max_y(img.rows) {
    }

    void random_shift(){
      cout << max_x << endl;
    }

    void random_skew(){
      cout << max_y << endl;
    }
}
  


int main(int argc, char** argv )
{
  cout << "OpenCV Version: " << CV_MAJOR_VERSION << ".";
  cout << CV_MINOR_VERSION << endl;
  
  int offset = 80;

  bool show_plots;

  if (argc == 2){
    show_plots = false;
  } else if (argc == 3){
    show_plots = (bool)atoi(argv[2]);
  } else {
    readme(); return -1;
  } 
 
  int cnt = 0; 
  for (auto& f_it: fs::directory_iterator(argv[1])){
   
    char new_file[50];
    sprintf(new_file, "../synth_data/%09d.jpg", cnt);
    cout << new_file << endl;

    std::string img_file = f_it.path().string();
    cout << img_file << endl;
    Mat img = imread( img_file, CV_LOAD_IMAGE_COLOR);
    print_dim(img);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    vector<Point2f> pts1, pts2;
    make_points(img, pts1, offset);
    make_points(img, pts2, offset, generator);
    cout << pts2 << endl;

    Mat h = findHomography(pts1, pts2).inv();
     
    Mat img_new;
    warpPerspective(img, img_new, h, img.size());

    if (show_plots){
      plot_pts(img, pts1);
      plot_pts(img, pts2);
      draw_poly(img, pts1, RED);
      draw_poly(img, pts2, BLUE);
    
      imshow("Source image", img);
      imshow("Warped source image", img_new);
    }

    int width = pts1[1].x - pts1[0].x;
    int height = pts1[2].y - pts1[1].y;
 
    Mat roi = Mat(img, Rect(pts1[0].x, pts1[0].y, width, height)).clone();
    Mat roi_gray(roi);
    cvtColor(roi, roi_gray, CV_RGB2GRAY);

    if (show_plots){
      imshow("Source rect", roi_gray);
    }
    imwrite(new_file, roi);

    Mat roi_new = Mat(img_new, Rect(pts1[0].x, pts1[0].y, width, height)).clone();
    Mat roi_new_gray(roi_new);
    cvtColor(roi_new, roi_new_gray, CV_RGB2GRAY);
    if (show_plots){
      imshow("Warped rect", roi_new_gray);
      waitKey(0);
    }
    cnt++;
  }
  return 0;
}
