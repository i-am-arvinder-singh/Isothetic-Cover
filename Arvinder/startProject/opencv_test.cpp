#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>

// using namespace std;

void show_image(cv::Mat &image){

    imshow("Image: ",image);

    cv::waitKey(0);

}

void show_dims(cv::Mat &image){
    std::cout<<"Rows: "<<image.rows<<std::endl;
    std::cout<<"Cols: "<<image.cols<<std::endl;
}

bool are_all_pixels_white(cv::Mat &image){
    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            if(i==0 || j==0 || i==(int)image.rows-1 || j==(int)image.cols-1)
                continue;
            cv::Scalar pixel = image.at<uchar>(j, i);
            // std::cout<<"---> "<<pixel.val[0]<<std::endl;
            int intensity = pixel.val[0];
            int val = (intensity==255?0:1);
            if(val==1)
                return false;
            // if(pixel.val[0]==255)
            //     return false;
        }
        // std::cout<<std::endl;
    }
    return true;
}

/*

    Inner and Outer Cover contruction algorithm:

    https://github.com/rahul-chowdhury/image-processing-project-3

*/

int main(){

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);std::
    cout.tie(NULL);


    auto start1 = std::chrono::high_resolution_clock::now();

    std::string test_pic = "./nike.png";
    std::string yinyangGrayPath = "./test_pic_gray.jpeg";

    cv::Mat image = cv::imread(test_pic);
    cv::Mat grayImage, binaryImage;

    cv::cvtColor(image,grayImage,cv::COLOR_BGR2GRAY);

    // show_image(image);

    // show_image(grayImage);

    if(!image.data) {
        std::cout<< "No image"<<std::endl;
        return 0;
    }

    // show_image(image);

    cv::Mat channels[3];

    cv::split(image,channels);

    cv::threshold(grayImage,binaryImage,100,255,cv::THRESH_BINARY);

    // show_image(binaryImage);

    int GRID_SIZE = 5;

    int height = image.size().height;
    int width = image.size().width;

    // cv::Mat back_to_rgb;

    // cv::cvtColor(binaryImage,back_to_rgb,cv::COLOR_GRAY2BGR);
    // int g_ = g;
    // for (int i = 0; i<height; i += g_){
    //     cv::line(back_to_rgb, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255, 0, 0));
    //     // g_++;
    // }
    // g_ = g;
    // for (int i = 0; i<width; i += g_){
    //     cv::line(back_to_rgb, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 0, 0));
    //     // g_++;
    // }

    std::vector<cv::Rect> mCells;

    int BLOCK_HEIGHT = (height  )/GRID_SIZE;
    int BLOCK_WIDTH = (width )/GRID_SIZE;

    std::cout<<"BLOCK_HEIGHT : "<<BLOCK_HEIGHT<<std::endl;
    std::cout<<"BLOCK_WIDTH : "<<BLOCK_WIDTH<<std::endl;

    std::vector<std::vector<cv::Mat>> block_matrix(BLOCK_HEIGHT,std::vector<cv::Mat>(BLOCK_WIDTH));

    for (int y = 0; y < height - GRID_SIZE; y += GRID_SIZE) {
        for (int x = 0; x < width - GRID_SIZE; x += GRID_SIZE) {
            int k = x*y + x;
            cv::Rect grid_rect(x, y, GRID_SIZE, GRID_SIZE);
            // std::cout << grid_rect<< std::endl;
            mCells.push_back(grid_rect);
            rectangle(binaryImage, grid_rect, cv::Scalar(0, 255, 0), 1);
            int i_ = y/GRID_SIZE;
            int j_ = x/GRID_SIZE;
            block_matrix[i_][j_] = binaryImage(grid_rect);
            // cv::imshow("src", binaryImage);
            // cv::imshow(cv::format("grid%d",k), binaryImage(grid_rect));
            // cv::waitKey(0);
        }
    }

    std::vector<std::vector<bool>> is_pixel_present(BLOCK_HEIGHT,std::vector<bool>(BLOCK_WIDTH));

    std::cout<<"******************************"<<std::endl;

    for(int i=0;i<BLOCK_HEIGHT;i++){
        for(int j=0;j<BLOCK_WIDTH;j++){
            // cv::imshow(cv::format("grid"), block_matrix[i][j]);
            // cv::waitKey(0);
            // std::cout<<" ### "<<i<<" "<<j<<std::endl;
            if(are_all_pixels_white(block_matrix[i][j])){
                is_pixel_present[i][j] = 0;
            }
            else{
                is_pixel_present[i][j] = 1;
            }
            std::cout<<is_pixel_present[i][j];
        }
        std::cout<<std::endl;
    }

    std::cout<<"******************************"<<std::endl;


	auto stop1 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop1-start1);

    std::cout << "Time: " << duration . count() / 1000 << std::endl;

    // for(int i=0;i<width;i+=g)
    //     for(int j=0;j<height;j+=g)
    //         image.at<cv::Vec3b>(i,j) = cv::Scalar(10,10,10); 
        
    // show_image(back_to_rgb);

    // show_image(channels[0]);
    // show_image(channels[1]);
    // show_image(channels[2]);

    // show_dims(channels[0]);



}
