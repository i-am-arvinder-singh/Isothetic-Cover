
/*
######################################################################################################################
#                                                                                                                    #
#                                                                                                                    #
#                                                 Isothetic Cover Algorithm (TIPS)                                   #
#                                                    Implemented By:                                                 #
#                                                   1. Arvinder Singh (i-am-arvinder-singh)                          #
#                                                   2. Dhruv Tyagi    (dhruvtyagi)                                   #
#                                                                                                                    #
#                                                                                                                    #
######################################################################################################################
*/

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>

using namespace std;

#define INNER 1



void show_image(cv::Mat &image){

    imshow("Image: ",image);

    cv::waitKey(0);

}

void show_dims(cv::Mat &image){
    std::cout<<"Rows: "<<image.rows<<std::endl;
    std::cout<<"Cols: "<<image.cols<<std::endl;
}

bool are_all_pixels_white(cv::Mat &image){
    // std::cout<<"###########"<<std::endl;
    // std::cout<<"Rows:==> "<<image.rows<<std::endl;
    // std::cout<<"Cols:==> "<<image.cols<<std::endl;
    // std::cout<<"###########"<<std::endl;
    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            if(i==0 || j==0 || i==(int)image.rows-1 || j==(int)image.cols-1)
                continue;
            cv::Scalar pixel = image.at<uchar>(j, i);
            // std::cout<<"---> "<<pixel.val[0]<<std::endl;
            int intensity = pixel.val[0];
            int val = (intensity==255?0:1);

            
            if(INNER==1){
                if(val==0)
                    return true;
            }
            else{
                if(val==1)
                    return false;
            }
            // if(pixel.val[0]==255)
            //     return false;
        }
        // std::cout<<std::endl;
    }
    // return true;
    if(INNER==1)
        return false;
    else
        return true;
}

/*

    Inner and Outer Cover contruction algorithm:

    https://github.com/rahul-chowdhury/image-processing-project-3

*/

int main(){

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::cout.tie(NULL);


    auto start1 = std::chrono::high_resolution_clock::now();

    std::string test_pic = "./test1.png";
    // std::string yinyangGrayPath = "./test_pic_gray.jpeg";

    cv::Mat image = cv::imread(test_pic);
    cv::Mat grayImage, binaryImage;

    cv::cvtColor(image,grayImage,cv::COLOR_BGR2GRAY);

    if(!image.data) {
        std::cout<< "No image"<<std::endl;
        return 0;
    }

    // show_image(image);

    // show_image(grayImage);

    // show_image(image);

    cv::Mat channels[3];

    cv::split(image,channels);

    cv::threshold(grayImage,binaryImage,40,255,cv::THRESH_BINARY);

    cv::Mat binaryImage_copy = binaryImage.clone();

    // show_image(binaryImage_copy);

    int GRID_SIZE = 10;

    int height = image.size().height;
    int width = image.size().width;

    cv::Mat back_to_rgb;

    // cv::cvtColor(binaryImage,back_to_rgb,cv::COLOR_GRAY2BGR);
    // int g_ = GRID_SIZE;
    // for (int i = 0; i<height; i += g_){
    //     cv::line(back_to_rgb, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255, 0, 0));
    //     // g_++;
    // }
    // g_ = GRID_SIZE;
    // for (int i = 0; i<width; i += g_){
    //     cv::line(back_to_rgb, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 0, 0));
    //     // g_++;
    // }

    // show_image(back_to_rgb);


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

    // show_image(binaryImage);

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
    cv::cvtColor(binaryImage_copy,back_to_rgb,cv::COLOR_GRAY2BGR);


    

    // ####################################### Actual Algo Implementation Starts Here ######################################### //

    vector<vector<bool>> is_pixel_taken(height/GRID_SIZE+2,vector<bool>(width/GRID_SIZE+2,false));


    std::function<int(int,int)> find_type_C = [&](int h, int w){
        int block_i = h/GRID_SIZE;
        int block_j = w/GRID_SIZE;
        int sum = 0;
        for(int i=-1;i<=0;i++){
            for(int j=-1;j<=0;j++){
                sum+=is_pixel_present[i+block_i][j+block_j];////////
            }
        }
        return sum;
    };

    // cout<<"----------------------$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"<<endl;
    for(int i=GRID_SIZE;i<height-GRID_SIZE;i+=GRID_SIZE){
        for(int j=GRID_SIZE;j<width-GRID_SIZE;j+=GRID_SIZE){
            // cout<<":::::::::::::::::::+=======>>> "<<i<<" "<<j<<endl;
            cout<<find_type_C(i,j);
        }
        cout<<endl;
    }
    

    std::function<std::pair<int,int>(int,int)> find_start_point_OIC = [&](int a, int b){
        // cout<<"///////////////////+=======>>> "<<a<<" "<<b<<endl;
        for(int w=b;w<width-GRID_SIZE;w+=GRID_SIZE){
            // if(h==660){
            //     cout<<":::::::::::::::::::+=======>>> "<<h<<" "<<w<<endl;
            // }
            if(is_pixel_taken[a/GRID_SIZE][w/GRID_SIZE])
                continue;
            auto type = find_type_C(a,w);
            if(type==1)
                return make_pair(a,w);
        }
        for(int h=a+GRID_SIZE;h<height-GRID_SIZE;h+=GRID_SIZE){
            for(int w=GRID_SIZE;w<width-GRID_SIZE;w+=GRID_SIZE){
                // if(h==660){
                //     cout<<":::::::::::::::::::+=======>>> "<<h<<" "<<w<<endl;
                // }
                if(is_pixel_taken[h/GRID_SIZE][w/GRID_SIZE])
                    continue;
                auto type = find_type_C(h,w);
                if(type==1)
                    return make_pair(h,w);
            }
        }
        return make_pair(-1,-1);
    };

    /*
    -------
    | 1| 2|
    -------
    | 4| 3|
    -------
    */

    std::function<int(pair<int,int>)> find_K_in_C1 = [&](pair<int,int> p){
        int i = p.first;
        int j = p.second;
        i/=GRID_SIZE;
        j/=GRID_SIZE;
        int val = find_type_C(p.first,p.second);
        assert(val==1);
        if(is_pixel_present[i-1][j-1])
            return 1;
        else if(is_pixel_present[i-1][j])
            return 2;
        else if(is_pixel_present[i][j-1])
            return 4;
        else if(is_pixel_present[i][j])
            return 3;
        assert(false);
        return -1;
    };


    std::function<int(pair<int,int>)> find_K_in_C3 = [&](pair<int,int> p){
        int i = p.first;
        int j = p.second;
        i/=GRID_SIZE;
        j/=GRID_SIZE;
        int val = find_type_C(p.first,p.second);
        assert(val==3);
        if(!is_pixel_present[i-1][j-1])
            return 1;
        else if(!is_pixel_present[i-1][j])
            return 2;
        else if(!is_pixel_present[i][j-1])
            return 4;
        else if(!is_pixel_present[i][j])
            return 3;
        assert(false);
        return -1;
    };





    /*

    ^
    |    ====>  1

    - >  ====>  2

    |
    V    ====>  3

    < -  ====>  4
        

    */

    std::function<int(pair<int,int>,pair<int,int>)> find_dir = [&](pair<int,int> p_prev, pair<int,int> p_cur){//1 2
        int x_prev = p_prev.first;
        int y_prev = p_prev.second;

        int x_cur = p_cur.first;
        int y_cur = p_cur.second;

        if(x_prev==x_cur){
            if(y_cur<y_prev){
                return 1;
            }
            else{
                return 3;
            }
        }
        else{
            if(x_prev<x_cur){
                return 2;
            }
            else
                return 4;
        }

    };

    for(int a=GRID_SIZE;a<height;a+=GRID_SIZE){

        bool see = false;
        for(int b=GRID_SIZE;b<width;b+=GRID_SIZE){

            auto start_pixel_position = find_start_point_OIC(a,b);

            if(start_pixel_position.first==-1 and start_pixel_position.second==-1){
                see = true;
                break;
            }

            cout<<"====> "<<start_pixel_position.first<<" "<<start_pixel_position.second<<endl;


            int orientation_of_1 = find_K_in_C1(start_pixel_position);

            vector<int> direction_vector;
            vector<pair<int,int>> point_list;

            pair<int,int> p_next;

            if(orientation_of_1==1){//starting anti clockwise

                p_next = {start_pixel_position.first-GRID_SIZE,start_pixel_position.second};
                direction_vector.push_back(1);

            }
            else if(orientation_of_1==2){

                p_next = {start_pixel_position.first,start_pixel_position.second+GRID_SIZE};
                direction_vector.push_back(2);

            }
            else if(orientation_of_1==3){

                p_next = {start_pixel_position.first+GRID_SIZE,start_pixel_position.second};
                direction_vector.push_back(3);

            }
            else{

                p_next = {start_pixel_position.first,start_pixel_position.second-GRID_SIZE};
                direction_vector.push_back(4);

            }

            point_list.push_back(start_pixel_position);

            while(p_next!=start_pixel_position){

                auto p_cur = p_next;

                point_list.push_back(p_cur);

                int type = find_type_C(p_cur.first,p_cur.second);

                assert(type!=0);

                if(type==1){
                    int orientation_of_1 = find_K_in_C1(p_cur);
                    if(orientation_of_1==1){//starting anti clockwise

                        p_next = {p_cur.first-GRID_SIZE,p_cur.second};
                        direction_vector.push_back(1);

                    }
                    else if(orientation_of_1==2){

                        p_next = {p_cur.first,p_cur.second+GRID_SIZE};
                        direction_vector.push_back(2);

                    }
                    else if(orientation_of_1==3){

                        p_next = {p_cur.first+GRID_SIZE,p_cur.second};
                        direction_vector.push_back(3);

                    }
                    else{

                        p_next = {p_cur.first,p_cur.second-GRID_SIZE};
                        direction_vector.push_back(4);

                    }
                }
                else if(type==2){
                    int cur_i = p_cur.first;
                    int cur_j = p_cur.second;
                    cur_i/=GRID_SIZE;
                    cur_j/=GRID_SIZE;
                    if(is_pixel_present[cur_i][cur_j] and is_pixel_present[cur_i-1][cur_j-1]){
                        int prev_direction = direction_vector.back();
                        if(prev_direction==1){
                            p_next = {p_cur.first,p_cur.second+GRID_SIZE};
                            direction_vector.push_back(2);
                        }
                        else if(prev_direction==2){
                            p_next = {p_cur.first-GRID_SIZE,p_cur.second};
                            direction_vector.push_back(1);
                        }
                        else if(prev_direction==3){
                            p_next = {p_cur.first,p_cur.second-GRID_SIZE};
                            direction_vector.push_back(4);
                        }
                        else if(prev_direction==4){
                            p_next = {p_cur.first+GRID_SIZE,p_cur.second};
                            direction_vector.push_back(3);
                        }
                        else
                            assert(false);
                    }
                    else if(is_pixel_present[cur_i-1][cur_j] and is_pixel_present[cur_i][cur_j-1]){
                        int prev_direction = direction_vector.back();
                        if(prev_direction==1){
                            p_next = {p_cur.first,p_cur.second-GRID_SIZE};
                            direction_vector.push_back(4);
                        }
                        else if(prev_direction==2){
                            p_next = {p_cur.first+GRID_SIZE,p_cur.second};
                            direction_vector.push_back(3);
                        }
                        else if(prev_direction==3){
                            p_next = {p_cur.first,p_cur.second+GRID_SIZE};
                            direction_vector.push_back(2);
                        }
                        else if(prev_direction==4){
                            p_next = {p_cur.first-GRID_SIZE,p_cur.second};
                            direction_vector.push_back(1);
                        }
                        else
                            assert(false);
                    }
                    else{

                        int prev_dir = direction_vector.back();
                        if(prev_dir==1){
                            p_next = {p_cur.first-GRID_SIZE,p_cur.second};
                        }
                        else if(prev_dir==2){
                            p_next = {p_cur.first,p_cur.second+GRID_SIZE};
                        }
                        else if(prev_dir==3){
                            p_next = {p_cur.first+GRID_SIZE,p_cur.second};
                        }
                        else if(prev_dir==4){
                            p_next = {p_cur.first,p_cur.second-GRID_SIZE};
                        }
                        else{
                            assert(false);
                        }
                        direction_vector.push_back(prev_dir);

                    }
                }
                else if(type==3){   
                    
                    int orientation_of_3 = find_K_in_C3(p_cur);
                    int prev_direction = direction_vector.back();
                    if(orientation_of_3==1){//starting anti clockwise
                        assert(prev_direction!=1 and prev_direction!=4);
                        if(prev_direction==2){

                            p_next = {p_cur.first-GRID_SIZE,p_cur.second};
                            direction_vector.push_back(1);
                            
                        }
                        else if(prev_direction==3){

                            p_next = {p_cur.first,p_cur.second-GRID_SIZE};
                            direction_vector.push_back(4);

                        }
                        else{
                            assert(false);
                        }

                    }
                    else if(orientation_of_3==2){

                        assert(prev_direction!=1 and prev_direction!=2);
                        if(prev_direction==3){

                            p_next = {p_cur.first,p_cur.second+GRID_SIZE};
                            direction_vector.push_back(2);
                            
                        }
                        else if(prev_direction==4){

                            p_next = {p_cur.first-GRID_SIZE,p_cur.second};
                            direction_vector.push_back(1);

                        }
                        else{
                            assert(false);
                        }

                    }
                    else if(orientation_of_3==3){

                        assert(prev_direction!=2 and prev_direction!=3);
                        if(prev_direction==1){

                            p_next = {p_cur.first,p_cur.second+GRID_SIZE};
                            direction_vector.push_back(2);
                            
                        }
                        else if(prev_direction==4){

                            p_next = {p_cur.first+GRID_SIZE,p_cur.second};
                            direction_vector.push_back(3);

                        }
                        else{
                            assert(false);
                        }

                    }
                    else if(orientation_of_3==4){

                        assert(prev_direction!=3 and prev_direction!=4);
                        if(prev_direction==1){

                            p_next = {p_cur.first,p_cur.second-GRID_SIZE};
                            direction_vector.push_back(4);
                            
                        }
                        else if(prev_direction==2){

                            p_next = {p_cur.first+GRID_SIZE,p_cur.second};
                            direction_vector.push_back(3);

                        }
                        else{
                            assert(false);
                        }

                    }
                    else{
                        assert(false);
                    }

                }
                else if(type==4){
                    assert(false);
                }
                else{
                    assert(false);
                }

            }
            point_list.push_back(start_pixel_position);
            // show_image(binaryImage_copy);
            for(int i=1;i<point_list.size();i++){
                int x1 = point_list[i].first;
                int y1 = point_list[i].second;

                int x2 = point_list[i-1].first;
                int y2 = point_list[i-1].second;
                cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 255, 0));
            }

            for(auto p:point_list){
                is_pixel_taken[p.first/GRID_SIZE][p.second/GRID_SIZE] = true;
            }

            cout<<"here"<<endl;

            // show_image(back_to_rgb);

            a = start_pixel_position.first;
            b = start_pixel_position.second;

            // b+=GRID_SIZE;

            // if(a>height)
            //     break;
            // if(b>width)
            //     continue;


        }
        if(see)
            break;
    }

    

    // show_image(back_to_rgb);

    bool is_saved = cv::imwrite("./OIC_cover_butterfly.jpeg",back_to_rgb);

    if(!is_saved){
        cout<<"Save Unsuccessful."<<endl;
        return 0;
    }










    // ####################################### Actual Algo Implementation Ends Here ######################################### //


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
