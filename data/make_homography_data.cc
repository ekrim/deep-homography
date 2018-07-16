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
  cout << " Usage: ./make_homography_data <img_directory> <int_patch_size> <int_max_jitter> <bool_show_plots>" << endl;
}


int main(int argc, char** argv ){
  std::string dir_name(argv[1]);
  int patch_size = atoi(argv[2]);
  int max_jitter = atoi(argv[3]);
  
  cout << "OpenCV Version: " << CV_MAJOR_VERSION << ".";
  cout << CV_MINOR_VERSION << endl;
  
  bool show_plots = false;
  if (argc == 5){
    show_plots = (bool)atoi(argv[4]);
  } else if (argc > 5 || argc < 4){
    readme(); return -1;
  } 
 
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed);
 
  int cnt = 0; 
  char f_roi_orig[50];
  char f_roi_warp[50];
  char f_number[9];
  std::ofstream f_labels("../label_file.txt");
  for (auto const& f_it: fs::directory_iterator(dir_name)){
     
    // reading the file
    std::string img_file = f_it.path().string();
    //cout << img_file << endl;
    Mat img = imread( img_file, CV_LOAD_IMAGE_COLOR);
    //print_dim(img);

    if (img.rows > 256 && img.cols > 256){
    // the new files
    sprintf(f_number, "%09d", cnt);
    sprintf(f_roi_orig, "../synth_data/%09d_orig.jpg", cnt);
    sprintf(f_roi_warp, "../synth_data/%09d_warp.jpg", cnt);
    //cout << f_roi_orig << endl;
    f_labels << f_number << ";";

    // patch and jittered patch
    Patch patch(img, patch_size, max_jitter);
    patch.random_shift(gen);
    vector<Point2f> pts1 = patch.get_corners();
    patch.random_skew(gen);
    vector<Point2f> pts2 = patch.get_corners();

    Mat h = findHomography(pts1, pts2).inv();
     
    // save the label data
    for (int i_pts = 0; i_pts < pts1.size(); ++i_pts){
      f_labels << pts2[i_pts].x - pts1[i_pts].x << ",";
      f_labels << pts2[i_pts].y - pts1[i_pts].y << ",";
    }
    f_labels << endl;

    // apply the transformation
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
 
    // convert the original roi to grayscale
    Mat roi = Mat(img, Rect(pts1[0].x, pts1[0].y, width, height)).clone();
    Mat roi_gray(roi);
    cvtColor(roi, roi_gray, CV_RGB2GRAY);

    if (show_plots){
      imshow("Source rect", roi_gray);
    }
    imwrite(f_roi_orig, roi_gray);

    // convert the warped roi to grayscale
    Mat roi_new = Mat(img_new, Rect(pts1[0].x, pts1[0].y, width, height)).clone();
    Mat roi_new_gray(roi_new);
    cvtColor(roi_new, roi_new_gray, CV_RGB2GRAY);
    if (show_plots){
      imshow("Warped rect", roi_new_gray);
      waitKey(0);
    }
    imwrite(f_roi_warp, roi_new_gray);
    cout << cnt << endl;
    cnt++;
    }
  }

  f_labels.close();
  return 0;
}
