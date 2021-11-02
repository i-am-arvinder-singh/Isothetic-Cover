#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;

/*

To Run this on Mac M1 command:

arch -x86_64 g++ -c opencv_test.cpp -o cv -I/usr/local/Cellar/opencv/4.5.3_2/include/opencv4

*/

int main(){

    string yinyangPath = "./yinyang.jpeg";
    string yinyangGrayPath = "./yinyangGray.jpeg";

    cv::Mat image = cv::imread(yinyangPath,0);

    cout<<"Here"<<endl;

    if(!image.data) {
        cout<< "No image"<<endl;
        return 0;
    }

    cout<<"hello world"<<endl;

    imshow("This is a color image: ",image);

    cv::waitKey(0);

    // cout<<"Hello World!"<<endl;

}
