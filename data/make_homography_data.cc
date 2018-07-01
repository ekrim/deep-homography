#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <experimental/filesystem>

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "image_tools.h"


using std::cout;
using std::endl;
using std::vector;

using namespace cv;
namespace fs = std::experimental::filesystem;


void readme(){
  cout << " Usage: ./make_homography_data <img_directory>" << endl;
}


int main(int argc, char** argv ){
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
    pts1.push_back(Point2f(50,50));
    pts1.push_back(Point2f(400,50));
    pts1.push_back(Point2f(400,400));
    pts1.push_back(Point2f(50,400));

    pts2.push_back(Point2f(60,60));
    pts2.push_back(Point2f(410,60));
    pts2.push_back(Point2f(410,410));
    pts2.push_back(Point2f(60,410));

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
