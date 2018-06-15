#include <stdio.h>
#include <iostream>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme(){
    std::cout << " Usage: ./DisplayImage <img1> <img2>" << std::endl;
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

    if( argc != 3 ){
        readme(); return -1;
    }

    Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    
    int min_hessian = 400;
    Ptr<SURF> detector = SURF::create( min_hessian);

    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector->detect( img_1, keypoints_1);
    detector->detect( img_2, keypoints_2);

    Mat img_keypoints_1; Mat img_keypoints_2;

    drawKeyPoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    imshow("Keypoints 1", img_keypoints_1);
    imshow("Keypoints 2", img_keypoints_2);
    
    waitKey(0);
    return 0;
}
