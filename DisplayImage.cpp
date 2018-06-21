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
     
    circle(img, Point(0, 0), 10.0, Scalar(0, 0, 255), -1, 8);
    //namedWindow( "display window", WINDOW_AUTOSIZE);
    imshow("Keypoints 1", img);
    
    waitKey(0);
    return 0;
}
