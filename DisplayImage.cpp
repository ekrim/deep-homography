#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;


void brightness_and_contrast(Mat &I, Mat &new_image, float alpha, float beta){

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
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );
    Mat new_image = Mat::zeros(image.size(), image.type());
    brightness_and_contrast(image, new_image, 2.2, 50.0);

    std::cout << "number of rows: " << image.rows << std::endl;
    std::cout << "number of cols: " << image.cols << std::endl;
    std::cout << "number of channels: " << image.channels() << std::endl;

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    imshow("Display New Image", new_image);

    waitKey(0);

    return 0;
}
