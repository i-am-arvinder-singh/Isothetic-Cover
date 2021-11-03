#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include <opencv2/imgcodecs.hpp>

// using namespace std;


void show_image(cv::Mat &image){

    imshow("Image: ",image);

    cv::waitKey(0);

}

void show_dims(cv::Mat &image){
    std::cout<<"Rows: "<<image.rows<<std::endl;
    std::cout<<"Cols: "<<image.cols<<std::endl;
}

int main(){

    std::string test_pic = "./test_pic.jpeg";
    std::string yinyangGrayPath = "./test_pic_gray.jpeg";

    cv::Mat image = cv::imread(test_pic);
    cv::Mat grayImage;

    cv::cvtColor(image,grayImage,cv::COLOR_BGR2GRAY);

    show_image(image);
    
    show_image(grayImage);

    if(!image.data) {
        std::cout<< "No image"<<std::endl;
        return 0;
    }

    cv::Mat channels[3];

    cv::split(image,channels);


    show_image(channels[0]);
    show_image(channels[1]);
    show_image(channels[2]);

    show_dims(channels[0]);



}
