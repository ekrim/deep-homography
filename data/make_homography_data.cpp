#include <stdio.h>
#include <iostream>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme(){
    std::cout << " Usage: ./DisplayImage <img1>" << std::endl;
}

void BrightnessAndContrast(Mat &I, Mat &new_image, float alpha, float beta){

    for( int y = 0; y < I.rows; y++){
        for( int x = 0; x < I.cols; x++){
            for( int c = 0; c < I.channels(); c++){
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(alpha*I.at<Vec3b>(y, x)[c] + beta);
            }
        }
    }
}

int main(int argc, char** argv )
{
    std::cout << "OpenCV Version: " << CV_MAJOR_VERSION << ".";
    std::cout << CV_MINOR_VERSION << std::endl;

    if( argc != 2){
        readme(); return -1;
    }

    Mat img = imread( argv[1], CV_LOAD_IMAGE_COLOR);
    std::cout << img.rows << ", " << img.cols << ", " << img.channels() << std::endl;
    std::cout << img.type() << std::endl;

    std::vector<Point2f> pts1;
    pts1.push_back(Point2f(50, 50));
    pts1.push_back(Point2f(500, 50));
    pts1.push_back(Point2f(500, 300));
    pts1.push_back(Point2f(50, 300));
   
    std::vector<Point2f> pts2;
    pts2.push_back(Point2f(10, 10));
    pts2.push_back(Point2f(630, 0));
    pts2.push_back(Point2f(390, 250));
    pts2.push_back(Point2f(70, 210));

    Mat h = findHomography(pts1, pts2).inv();
     
    Mat img_new;
    warpPerspective(img, img_new, h, img.size());

    circle(img, pts1[0], 2.0, Scalar(255, 0, 0), -1, 8);
    circle(img, pts1[1], 2.0, Scalar(0, 255, 0), -1, 8);
    circle(img, pts1[2], 2.0, Scalar(0, 0, 255), -1, 8);
    circle(img, pts1[3], 2.0, Scalar(255, 0, 255), -1, 8);

    //namedWindow( "display window", WINDOW_AUTOSIZE);
    circle(img, pts2[0], 3.0, Scalar(255, 0, 0), 1, 8);
    circle(img, pts2[1], 3.0, Scalar(0, 255, 0), 1, 8);
    circle(img, pts2[2], 3.0, Scalar(0, 0, 255), 1, 8);
    circle(img, pts2[3], 3.0, Scalar(255, 0, 255), 1, 8);

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
