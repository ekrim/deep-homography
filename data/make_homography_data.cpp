#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>
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
using namespace cv;
using namespace cv::xfeatures2d;


void readme(){
  cout << " Usage: ./make_homography_data <img1>" << endl;
}

int randint(std::mt19937 &gen, unsigned min, unsigned max){
  return min + (gen() % (max-min+1));
}

void print_dim(Mat &x){
  cout << "image size: (" << x.rows << ", " << x.cols << ", " << x.channels();
  cout << ")" << endl;
}

void make_points(Mat &img, std::vector<Point2f> &pts, int offset){
  pts.push_back(Point2f(offset, offset));
  pts.push_back(Point2f(img.cols-offset, offset));
  pts.push_back(Point2f(img.cols-offset, img.rows-offset));
  pts.push_back(Point2f(offset, img.rows-offset));
}

void make_points(Mat &img, std::vector<Point2f> &pts, int offset, std::mt19937 &gen){
  std::vector<int> x_vec{ offset, img.cols-offset, img.cols-offset, offset};
  std::vector<int> y_vec{ offset, offset, img.rows-offset, img.rows-offset};
  
  for (std::vector<int>::size_type i = 0; i != x_vec.size(); i++){
    int delta_x = randint(gen, 0, 2*offset) - offset;
    int delta_y = randint(gen, 0, 2*offset) - offset;
    pts.push_back(Point2f(x_vec[i]+delta_x, y_vec[i]+delta_y));
  }
}

void plot_pts(Mat &img, std::vector<Point2f> &pts){
  circle(img, pts[0], 3.0, Scalar(255, 0, 0), -1, 8);
  circle(img, pts[1], 3.0, Scalar(0, 255, 0), -1, 8);
  circle(img, pts[2], 3.0, Scalar(0, 0, 255), -1, 8);
  circle(img, pts[3], 3.0, Scalar(255, 0, 255), -1, 8);
}

int main(int argc, char** argv )
{
  cout << "OpenCV Version: " << CV_MAJOR_VERSION << ".";
  cout << CV_MINOR_VERSION << endl;

  if( argc != 2){
    readme(); return -1;
  }

  Mat img = imread( argv[1], CV_LOAD_IMAGE_COLOR);
  print_dim(img);

  int offset = 30;
  unsigned seed = 137;
  std::mt19937 generator(seed);
  std::vector<Point2f> pts1;
  std::vector<Point2f> pts2;
  make_points(img, pts1, offset);
  make_points(img, pts2, offset, generator);
  cout << pts2 << endl;

  Mat h = findHomography(pts1, pts2).inv();
   
  Mat img_new;
  warpPerspective(img, img_new, h, img.size());

  plot_pts(img, pts1);
  plot_pts(img, pts2);

  imshow("Source image", img);
  imshow("Warped source image", img_new);

  int width = pts1[1].x - pts1[0].x;
  int height = pts1[2].y - pts1[1].y;
 
  Mat roi = Mat(img, Rect(pts1[0].x, pts1[0].y, width, height)).clone();
  imshow("Source rect", roi);

  Mat roi_new = Mat(img_new, Rect(pts1[0].x, pts1[0].y, width, height)).clone();
  imshow("Warped rect", roi_new);
  waitKey(0);
  return 0;
}
