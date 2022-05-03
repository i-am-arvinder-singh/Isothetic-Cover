
/*
######################################################################################################################
#                                                                                                                    #
#                                                                                                                    #
#                                                 Isothetic Cover Algorithm (TIPS)                                   #
#                                                                                                                    #
#                                                                                                                    #
######################################################################################################################
*/

#include <iostream>
#include <string>
// #include <bits/stdc++.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
namespace fs = std::__fs::filesystem;

// #include <tesseract/baseapi.h>

using namespace std;

// #define INNER 0
#define IMPROVED 1
#define LINE_GEN_STRICTNESS 3
#define CNT_LINE_AFTER 2
#define ERROR_TOLERANCE 110// in percent
#define THRESHOLD 0.6
// #define PRE_PROCESS 1

int GRID_SIZE = 1;
int PRE_PROCESS = 0;
int INNER = 0;
int smaller_gs = 5;

int line_detection(cv::Mat &, vector<pair<int,int>> &, vector<int> &);

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

bool are_all_pixels_white_improved(cv::Mat &image, int start_i, int start_j){
    // show_image(image);
    // show_dims(image);
    start_i*=GRID_SIZE;
    start_j*=GRID_SIZE;
    // cout<<"-----> "<<start_i<<" "<<start_j<<endl;
    // cout<<"***********************************"<<endl;
    // for(int i=start_i;i<start_i+GRID_SIZE and i<image.rows;i++){
    //     for(int j=start_j;j<start_j+GRID_SIZE and i<image.cols;j++){
    //         cv::Scalar pixel = image.at<uchar>(i, j);
    //         // std::cout<<"---> "<<pixel.val[0]<<std::endl;
    //         int intensity = pixel.val[0];
    //         int val = (intensity==255?0:1);
    //         cout<<val;
    //     }
    //     cout<<endl;
    // }
    // cout<<"***********************************"<<endl;
    for(int i=start_i;i<start_i+GRID_SIZE and i<image.rows;i++){
        for(int j=start_j;j<start_j+GRID_SIZE and j<image.cols;j++){
            cv::Scalar pixel = image.at<uchar>(i, j);
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
            // cout<<val;
        }
        // cout<<endl;
    }
    if(INNER==1)
        return false;
    else
        return true;
}


/*
    if true -> rectillinear (rectangular)
    else    -> irregular
*/

bool analyse_shape(vector<int> &dir){
    
    
    long long ret_err = LLONG_MAX;

    int n = dir.size();
    int cnt = 0;
    // cout<<"1111 HERERERERER "<<n<<endl;
    for(int len3_ = 1;len3_<(n/2);len3_++){
        cnt = 0;
        int len3 = len3_;
        int len2 = (n - (2*len3))/2;
        int len1 = len3;
        int len4 = len2;

        int len2_ = len2;
        int len1_ = len1;
        int len4_ = len4;

        double arr[5] = {};

        int i = 0;
        while(i<n and (len3>0)){
            if(dir[i]!=3)
                cnt++;
            len3--;
            i++;
        }
        arr[3] = ((double)cnt)/(len3_);
        cnt=0;
        while(i<n and (len2>0)){
            if(dir[i]!=2)
                cnt++;
            len2--;
            i++;
        }
        arr[2] = ((double)cnt)/(len2_);
        cnt=0;
        while(i<n and (len1>0)){
            if(dir[i]!=1)
                cnt++;
            len1--;
            i++;
        }
        arr[1] = ((double)cnt)/(len1_);
        cnt=0;
        while(i<n and (len4>0)){
            if(dir[i]!=4)
                cnt++;
            len4--;
            i++;
        }
        arr[4] = ((double)cnt)/(len4_);
        double total = 0;
        // for(int i=1;i<=4;i++)
        //     total+=arr[i];

        total+=(len1_*arr[1]);
        total+=(len2_*arr[2]);
        total+=(len3_*arr[3]);
        total+=(len4_*arr[4]);

        double err_now = total/(len1_+len2_+len3_+len4_);
        // double err_now = ((double)cnt)/n;
        ret_err = min(ret_err,(long long)(err_now*n));
    }
    // cout<<"2222 HERERERERER"<<endl;
    // cout<<"RETT errr : "<<ret_err<<endl;
    if(ret_err>ERROR_TOLERANCE and (n)>10)
        return false;
    return true;


}

double polygon_area(vector<pair<int,int>> &cover_points){
    double area = 0.0;
    
    int n = cover_points.size();

    int j = n - 1;
    double min_x = INT_MAX, min_y = INT_MAX;
    double max_x = INT_MIN, max_y = INT_MIN;

    for (int i = 0; i < n; i++){
        min_x = min(min_x,(double)cover_points[i].first);
        min_y = min(min_y,(double)cover_points[i].second);
        max_x = max(max_x,(double)cover_points[i].first);
        max_y = max(max_y,(double)cover_points[i].second);
        area += (cover_points[j].first + cover_points[i].first) * (cover_points[j].second - cover_points[i].second);
        j = i;  
    }

    return abs(area)/2;
}

bool analyse_shape_improved(vector<pair<int,int>> &cover_points){
    
    double area = 0.0;
    
    int n = cover_points.size();

    int j = n - 1;
    double min_x = INT_MAX, min_y = INT_MAX;
    double max_x = INT_MIN, max_y = INT_MIN;

    for (int i = 0; i < n; i++){
        min_x = min(min_x,(double)cover_points[i].first);
        min_y = min(min_y,(double)cover_points[i].second);
        max_x = max(max_x,(double)cover_points[i].first);
        max_y = max(max_y,(double)cover_points[i].second);
        area += (cover_points[j].first + cover_points[i].first) * (cover_points[j].second - cover_points[i].second);
        j = i;  
    }



    double denom = 2.0*(max_x-min_x)*(max_y-min_y);


    cout<<"########################"<<endl;
    cout<<min_x<<" "<<min_y<<" "<<max_x<<" "<<max_y<<endl;
    cout<<area<<" "<<denom<<endl;
    cout<<"########################"<<endl;

    double ratio = abs(area)/denom;

    cout<<"RETT errr : "<<ratio<<endl;
    
    if(ratio>THRESHOLD)
        return true;
    return false;

}



vector<pair<vector<pair<int,int>>,vector<int>>> cover_gen(
    cv::Mat binaryImage_temp){
    
    cv::Mat binaryImage = binaryImage_temp.clone();


    cv::Mat binaryImage_copy = binaryImage.clone();
    cv::Mat binaryImage_copy_copy = binaryImage.clone();


    int height = binaryImage_copy_copy.size().height;
    int width = binaryImage_copy_copy.size().width;

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


    if(PRE_PROCESS==1){
        cv::Mat dst;
        cv::Mat elementKernel1 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3),cv::Point(-1,-1));
        cv::Mat elementKernel2 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4),cv::Point(-1,-1));
        cv::Mat elementKernel3 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(2,2),cv::Point(-1,-1));
        // cv::morphologyEx(binaryImage_copy,dst,cv::MORPH_CLOSE,elementKernel);
        int times = 5;
        cv::dilate(binaryImage_copy,binaryImage_copy,elementKernel1,cv::Point(-1,-1),1);
        while(times--){
            // cv::dilate(binaryImage_copy,dst,elementKernel3,cv::Point(-1,-1),1);
            cv::erode(binaryImage_copy,dst,elementKernel2,cv::Point(-1,-1),1);
            binaryImage_copy = dst.clone();
            binaryImage = dst.clone();
        }
        // show_image(binaryImage_copy);
    }


    std::vector<cv::Rect> mCells;

    int BLOCK_HEIGHT = (height  )/GRID_SIZE;
    int BLOCK_WIDTH = (width )/GRID_SIZE;

    // std::cout<<"BLOCK_HEIGHT : "<<BLOCK_HEIGHT<<std::endl;
    // std::cout<<"BLOCK_WIDTH : "<<BLOCK_WIDTH<<std::endl;

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
    // cout<<"hreeeeeee"<<endl;
    // show_image(binaryImage_copy);

    std::vector<std::vector<bool>> is_pixel_present(BLOCK_HEIGHT,std::vector<bool>(BLOCK_WIDTH));

    // std::cout<<"******************************"<<std::endl;

    for(int i=0;i<BLOCK_HEIGHT;i++){
        for(int j=0;j<BLOCK_WIDTH;j++){
            // cv::imshow(cv::format("grid"), block_matrix[i][j]);
            // cv::waitKey(0);
            // std::cout<<" ### "<<i<<" "<<j<<std::endl;
            if(INNER==1){
                if((IMPROVED==1?are_all_pixels_white_improved(binaryImage_copy,i,j):are_all_pixels_white(block_matrix[i][j]))){
                    is_pixel_present[i][j] = 0;
                }
                else{
                    is_pixel_present[i][j] = 1;
                }

            }
            else{

                if((IMPROVED==1?are_all_pixels_white_improved(binaryImage_copy,i,j):are_all_pixels_white(block_matrix[i][j]))){
                    is_pixel_present[i][j] = 0;
                }
                else{
                    is_pixel_present[i][j] = 1;
                }
            }
            // std::cout<<is_pixel_present[i][j];
        }
        // std::cout<<std::endl;
    }

    // std::cout<<"******************************"<<std::endl;
    cv::cvtColor(binaryImage_copy,back_to_rgb,cv::COLOR_GRAY2BGR);

    vector<pair<vector<pair<int,int>>,vector<int>>> return_value;

    // ####################################### Actual Algo Implementation Starts Here ######################################### //

    vector<vector<bool>> is_pixel_taken(height/GRID_SIZE+2,vector<bool>(width/GRID_SIZE+2,false));


    std::function<int(int,int)> find_type_C = [&](int h, int w){
        int block_i = h/GRID_SIZE;
        int block_j = w/GRID_SIZE;
        int sum = 0;
        for(int i=-1;i<=0;i++){
            for(int j=-1;j<=0;j++){
                // if(i+block_i<0 || j+block_j<0)
                //     continue;
                sum+=is_pixel_present[i+block_i][j+block_j];////////
            }
        }
        return sum;
    };

    // cout<<"----------------------$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"<<endl;
    // for(int i=GRID_SIZE;i<height-GRID_SIZE;i+=GRID_SIZE){
    //     for(int j=GRID_SIZE;j<width-GRID_SIZE;j+=GRID_SIZE){
    //         // cout<<":::::::::::::::::::+=======>>> "<<i<<" "<<j<<endl;
    //         cout<<find_type_C(i,j);
    //     }
    //     cout<<endl;
    // }
    

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

    int number_of_covers = 0;
    int graphic_area_cnt = 0;
    int possible_textual_area_cnt = 0;

    for(int a=GRID_SIZE;a<height;a+=GRID_SIZE){

        bool see = false;
        for(int b=GRID_SIZE;b<width;b+=GRID_SIZE){

            auto start_pixel_position = find_start_point_OIC(a,b);

            if(start_pixel_position.first==-1 and start_pixel_position.second==-1){
                see = true;
                break;
            }

            number_of_covers++;

            // cout<<"====> "<<start_pixel_position.first<<" "<<start_pixel_position.second<<endl;


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
            
            // cout<<"^^^^^^^^^^^^ FOR DEBUG ^^^^^^^^^^^^^::::: "<<number_of_covers<<endl;

            while(p_next!=start_pixel_position){

                auto p_cur = p_next;

                point_list.push_back(p_cur);

                // cout<<"^^^^^^^^^^^^ FOR DEBUG ^^^^^^^^^^^^^ pre ::::: "<<number_of_covers<<" | POINT:::  "<<p_cur.first<<" "<<p_cur.second<<endl;

                int type = find_type_C(p_cur.first,p_cur.second); /// BUG

                // cout<<"^^^^^^^^^^^^ FOR DEBUG ^^^^^^^^^^^^^ post ::::: "<<number_of_covers<<endl;

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
            
            /* 
                Traversal directions:
            */

            // cout<<"-------------------Traversal Direction Below Cover No."<<number_of_covers<<"------------------"<<endl;

            // for(auto ele:direction_vector){
            //     cout<<ele;
            // }
            // cout<<endl;
            // cout<<direction_vector.size()<<endl;
            // // Analyze shape
            // cout<<"******===>> "<<"here"<<endl;
            // // bool is_rectillinear = analyse_shape_regex(direction_vector);
            // cout<<"******===>> "<<"here"<<endl;
            // // cout<<direction_vector[35]<<" "<<direction_vector[108]<<endl;
            // cout<<endl<<"-------------------------------------"<<endl;

            return_value.push_back({point_list,direction_vector});



            // point_list.push_back(start_pixel_position);
            // show_image(binaryImage_copy);
            
            
            

            // line_detection(back_to_rgb,point_list,direction_vector);

            // for(int i=1;i<point_list.size();i++){
            //     int x1 = point_list[i].first;
            //     int y1 = point_list[i].second;

            //     int x2 = point_list[i-1].first;
            //     int y2 = point_list[i-1].second;
            //     // if(once){
            //     //     cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(0, 0, 255));    
            //     //     once = false;
            //     // }
            //     // else
            //     cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 0, 0));
            //     break;
            // }

            for(auto p:point_list){
                is_pixel_taken[p.first/GRID_SIZE][p.second/GRID_SIZE] = true;
            }

            // cout<<"here"<<endl;

            // show_image(back_to_rgb);

            a = start_pixel_position.first;
            b = start_pixel_position.second;

            // b+=GRID_SIZE;

            // if(a>height)
            //     break;
            // if(b>width)
            //     continue;

            // cout<<"^^^^^^^^^^^^ FOR DEBUG ^^^^^^^^^^^^^ "<<number_of_covers<<endl;

        }
        if(see)
            break;
    }

    cv::Mat temp;
    cv::cvtColor(binaryImage_copy_copy,temp,cv::COLOR_GRAY2BGR);

    // show_image(temp);

    for(auto &ele1 : return_value){
        ele1.first.push_back(ele1.first[0]);
        // cout<<"**********##########********"<<endl;
        // for(auto &vals:ele1.first)
        //     cout<<"====>>> ["<<vals.first<<" "<<vals.second<<"]"<<endl;
        // cout<<"**********##########********"<<endl;
        for(int i=1;i<ele1.first.size();i++){
            int x1 = ele1.first[i].first;
            int y1 = ele1.first[i].second;

            int x2 = ele1.first[i-1].first;
            int y2 = ele1.first[i-1].second;
            cv::line(temp,cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 0, 0));
            // show_image(graphic_img_copy);
        }
    }

    // show_image(temp);


    return return_value;


    // ####################################### Actual Algo Implementation Ends Here ######################################### //

}

int find_perimeter(vector<pair<int,int>> &point_list){
    int n = point_list.size() * GRID_SIZE;
    assert(n>1);
    return n;
}

int line_detection(cv::Mat &back_to_rgb, vector<pair<int,int>> &cover_point_list, vector<int> &direction_vector){
    // bool once = true;
    // for(int i=1;i<cover_point_list.size();i++){
    //     int x1 = cover_point_list[i].first;
    //     int y1 = cover_point_list[i].second;

    //     int x2 = cover_point_list[i-1].first;
    //     int y2 = cover_point_list[i-1].second;
    //     if(once){
    //         cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(0, 0, 255));    
    //         once = false;
    //     }
    //     else
    //         cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 255, 0));
    // }




    vector<int> sum[5];

    int n = direction_vector.size();

    for(int i=1;i<=4;i++){
        sum[i] = vector<int>(n);
    }

    for(int i=0;i<n;i++){
        for(int j=1;j<=4;j++){
            sum[j][i] = (direction_vector[i]==j);
        }
    }
    // cout<<"Upto here?"<<endl;
    
    for(int j=1;j<=4;j++)
        for(int i=1;i<n;i++)
            sum[j][i]+=sum[j][i-1];

    std::function<int(int,int,int)> sum_in_range = [&](int l, int r, int dir){
        if(l>r)
            return 0;
        return sum[dir][r] - ((l-1>=0)?sum[dir][l-1]:0);
    };

    vector<int> ids[5];

    for(int i=0;i<n;i++){
        for(int j=1;j<=4;j++){
            if(direction_vector[i]==j){
                ids[j].push_back(i);
            }
        }
    }

    vector<pair<int,int>> line_list;

    for(int j=1;j<=4;j++){
        if(ids[j].size()<=1)
            continue;// There is no line

        // cout<<"(((((((((test))))))))"<<endl;
        // cout<<j<<" || ";
        // for(auto ele:ids[j])
        //     cout<<ele<<" ";
        // cout<<endl;
        // cout<<"((((((((())))))))"<<endl;
        // continue;

        int m = ids[j].size();


        // int i = 0;
        // while(i<m){
        //     int prev = ids[j][i];
        //     int k = i+1;
        //     int left = ids[j][i];
        //     while(k<m and ids[j][k]==prev+1){
        //         prev = ids[j][k];
        //         k++;
        //     }
        //     if(k==m){
        //         line_list.push_back({left,ids[j][k-1]});
        //         break;
        //     }
        //     else{
        //         int next_alt = direction_vector[prev+1];

        //     }
        // }


        //////////////////////////////////// PREV IMPLEMENTATION ////////////////////////////////////

        for(int i=1;i<m;i++){
            int cur_left = ids[j][i-1];
            int cur_right = ids[j][i];
            if(cur_right-cur_left==1){
                //// Comment below and write continue for original implementation ////
                int w = i;
                while(ids[j][w]-ids[j][w-1]==1 and w<m){
                    w++;
                }
                w--;
                if (w-i+1>2)
                line_list.push_back({ids[j][i],ids[j][w]});
                // cout<<ids[j][i]<<" "<<ids[j][w]<<endl;
                i = w;
                //////////////////////////////////////////////////////////////////////
            }
            else{
                int lb = cur_left+1;
                int rb = cur_right-1;
                int dir_val_in_range = -1;
                int no_of_dir_in_range = 0;
                for(int k=1;k<=4;k++){
                    if(k==j)
                        continue;
                    int see = sum_in_range(lb,rb,k);
                    if(see==rb-lb+1){
                        dir_val_in_range = k;
                        no_of_dir_in_range = see;
                        break;
                    }
                }
                

                if(dir_val_in_range!=-1){
                    int w = i+1;
                    int line_left = i-1;
                    int cnt = 1;
                    while(w<m){
                        cur_left = ids[j][w-1];
                        cur_right = ids[j][w];
                        lb = cur_left+1;
                        rb = cur_right-1;
                        int see = sum_in_range(lb,rb,dir_val_in_range);
                        // cout<<"&&&&&&====> "<<j<<" || "<<see<<" "<<w<<" "<<no_of_dir_in_range<<endl;
                        if(abs(see-no_of_dir_in_range)<=LINE_GEN_STRICTNESS and see==rb-lb+1){
                            w++;
                            cnt++;
                        }
                        else{
                            // cout<<"$$$$$$$$$$$ "<<see<<" "<<no_of_dir_in_range<<" "<<dir_val_in_range<<endl;
                            break;
                        }
                    }
                    // cout<<"^^^^^^^^^^^^^ "<<i-1<<" "<<
                    w--;
                    i = w;
                    if(cnt>CNT_LINE_AFTER){
                        // cout<<"-------->>>> "<<j<<" == "<<ids[j][line_left]<<" "<<ids[j][w]<<endl;
                        line_list.push_back({ids[j][line_left],ids[j][w]});
                    }
                }
                else
                    continue;

            }
        }

        //////////////////////////////////// PREV IMPLEMENTATION ////////////////////////////////////
        
    }

    sort(line_list.begin(),line_list.end());
    vector<pair<int,int>> new_line_list;

    if(!line_list.empty()){
        int now_right = line_list[0].second;
        new_line_list.push_back(line_list[0]);
        for(int i=1;i<line_list.size();i++){
            if(line_list[i].first>now_right){
                new_line_list.push_back(line_list[i]);
                now_right = new_line_list.back().second;
            }
        }
    }

    // cout<<"@@@@@@@ ST LINES::::: "<<new_line_list.size()<<endl;

    int cnt_color_lines = 0;

    vector<pair<int,int>> improved_line_list;

    
    for(int i=0;i<new_line_list.size();i++){
        if(i==0){
            improved_line_list.push_back(new_line_list[i]);
        }
        else if(abs(new_line_list[i-1].second-new_line_list[i].first)>1){
            improved_line_list.push_back({new_line_list[i-1].second, new_line_list[i].first});
        }
        improved_line_list.push_back(new_line_list[i]);
    }

    for(auto &[l,r]:improved_line_list){
        // cout<<"####### ==> "<<l<<" "<<r<<endl;
        /*
            Generate some color:
        */
       cnt_color_lines++;

        int b_ = rand()%255;
        int g_ = rand()%255;
        int r_ = rand()%255;

        for(int i=l;i<r;i++){
            int x1 = cover_point_list[i+1].first;
            int y1 = cover_point_list[i+1].second;

            int x2 = cover_point_list[i].first;
            int y2 = cover_point_list[i].second;
            cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(b_, g_, r_), 3); 
            // cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(0, 0, 255)); 
        }
    }

    // show_image(back_to_rgb);



    return cnt_color_lines;

}

vector< vector<vector<pair<int,int>>> > cover_mapping(vector<pair<vector<pair<int,int>>,vector<int>>> covers, cv::Mat binaryImage){


    

    int prev_gs = GRID_SIZE;

    GRID_SIZE = smaller_gs;
    PRE_PROCESS = 0;
    INNER = 0;

    struct vert_lines{
        int cover_no;
        int y_low, y_high;
        int x_val;
    };

    int total_big_covers = covers.size();

    vector< vector<vector<pair<int,int>>> > ans(total_big_covers);

    vector<vert_lines> vertical_lines;

    int cnt = 0;

    for(auto &p:covers){// (y,x)
        int n = p.first.size();
        for(int i=0;i<n;i++){
            pair<int,int> now = p.first[i];
            pair<int,int> next = p.first[(i+1)%n];
            if(now.second == next.second){
                vert_lines v;
                v.cover_no = cnt;
                v.y_low = min(now.first,next.first);
                v.y_high = max(now.first,next.first);
                v.x_val = now.second;
                vertical_lines.push_back(v);
            }
        }
        cnt++;
    }

    cv::Mat binaryImage_copy = binaryImage.clone();

    // show_image(binaryImage_copy);

    vector< pair< vector<pair<int,int>>,vector<int> > > covers_small_grid = cover_gen(binaryImage);

    // show_image(binaryImage);

    sort(vertical_lines.begin(),vertical_lines.end(),[&](vert_lines v1, vert_lines v2){
        return v1.y_low < v2.y_low;
    });

    int m = vertical_lines.size();

    cnt = 1;

    for(auto &p:covers_small_grid){

        int n = p.first.size();
        int y = p.first[0].first;
        int x = p.first[0].second;

        std::function<bool(int)> isLying = [&](int id){
            if(vertical_lines[id].y_low<=y and y<=vertical_lines[id].y_high){
                return true;
            }
            return false;
        };

        int i = 0;
        int j = m - 1;
        
        int k = -1;

        // cout<<"************"<<endl;

        // for(auto ele:vertical_lines){
        //     cout<<ele.y_low<<" "<<ele.y_high<<endl;
        // }
        
        // cout<<endl<<"************"<<endl;
        // cout<<"#############"<<endl;
        // cout<<y<<endl;
        // cout<<endl;
        // cout<<endl<<"#############"<<endl;

        while(i<=j){
            int mid = (i+j)/2;
            int low = vertical_lines[mid].y_low;
            int high = vertical_lines[mid].y_high;
            // cout<<mid<<endl;
            if(isLying(mid)){
                // Process
                // cout<<endl<<"hehehehhehehheh";
                // exit(-1);
                k = mid;
                vector<vert_lines> temp;
                temp.push_back(vertical_lines[mid]);

                // Expand from middle
                int left = mid - 1;
                while(left>=0 and isLying(left)){
                    temp.push_back(vertical_lines[left]);
                    left--;
                }

                int right = mid + 1;
                while(right<m and isLying(right)){
                    temp.push_back(vertical_lines[right]);
                    right++;
                }

                sort(temp.begin(),temp.end(),[&](vert_lines v1, vert_lines v2){
                    return abs(x-v1.x_val) < abs(x-v2.x_val);
                });

                int perimeter = find_perimeter(p.first);
                if(perimeter>35)
                    ans[temp[0].cover_no].push_back(p.first);


                break;
            }
            else{
                if(y>high){
                    i = mid + 1;
                }
                else{
                    j = mid - 1;
                }
            }
        }

        // cout<<cnt<<endl;
        // exit();
        // cnt++;
        // assert(k!=-1);
        if(k==-1)
            continue;




    }

    cv::Mat back_to_rgb_copy;
    cv::cvtColor(binaryImage_copy,back_to_rgb_copy,cv::COLOR_GRAY2BGR);

    for(auto &ele:covers){

        ele.first.push_back(ele.first[0]);
        
        for(int i=1;i<ele.first.size();i++){
            int x1 = ele.first[i].first;
            int y1 = ele.first[i].second;

            int x2 = ele.first[i-1].first;
            int y2 = ele.first[i-1].second;
            cv::line(back_to_rgb_copy,cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(0, 0, 255), 3);
            // show_image(graphic_img_copy);
        }

    }
    
    // show_image(back_to_rgb_copy);

    // for(int i=0;i<total_big_covers;i++){
    //     int choice_b = rand()%256;
    //     int choice_g = rand()%256;
    //     int choice_r = rand()%256;

    //     for(auto &covs:ans[i]){
    //         covs.push_back(covs[0]);
    //         for(int i=1;i<covs.size();i++){
    //             int x1 = covs[i].first;
    //             int y1 = covs[i].second;

    //             int x2 = covs[i-1].first;
    //             int y2 = covs[i-1].second;
    //             cv::line(back_to_rgb_copy,cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(choice_b, choice_g, choice_r));
    //             // show_image(graphic_img_copy);
    //         }
    //     }

    // }

    /// EXPERIMENT ///

    int big_test_crop = 1;

    for(int i=0;i<total_big_covers;i++){

        int choice_b = rand()%256;
        int choice_g = rand()%256;
        int choice_r = rand()%256;

        choice_b = 255;
        choice_g = 0;
        choice_r = 0;

        

        int word_test_crops = 1;

        for(auto &covs:ans[i]){
            int max_y = INT_MIN;
            int max_x = INT_MIN;

            int min_y = INT_MAX;
            int min_x = INT_MAX;
            // int perimeter = find_perimeter(covs);
            
            // if(perimeter<35){
            //     continue;
            // }


            for(int i=1;i<covs.size();i++){
                max_y = max(covs[i].first, max_y);//y
                max_x = max(covs[i].second, max_x);//x

                min_y = min(covs[i].first, min_y);
                min_x = min(covs[i].second, min_x);

            }

            cv::Rect crop_region(min_x, min_y, max_x-min_x, max_y-min_y);

            cv::Mat crop = binaryImage_copy(crop_region);

            // show_image(crop);

            int thickness = 1;

            cv::line(back_to_rgb_copy,cv::Point(min_x, min_y), cv::Point(min_x, max_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);
            cv::line(back_to_rgb_copy,cv::Point(min_x, min_y), cv::Point(max_x, min_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);
            cv::line(back_to_rgb_copy,cv::Point(max_x, max_y), cv::Point(min_x, max_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);
            cv::line(back_to_rgb_copy,cv::Point(max_x, max_y), cv::Point(max_x, min_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);

            // cv::Mat crop_img(crop_region);



            // string out_name = "./word_test_imgs/" + to_string(big_test_crop) + "_" + to_string(word_test_crops)+".png";
            // cv::imwrite(out_name,crop);
            
            word_test_crops++;


        }
        big_test_crop++;

    }

    ////////////////
    cout<<"****** here"<<endl;
    // show_image(back_to_rgb_copy);
    string out_name = "./cover_identif_0003.png";
    cv::imwrite(out_name,back_to_rgb_copy);

    // cout<<endl<<"hehehehhehehheh";
    // show_image(binaryImage);
    exit(100);
    
    return ans;

}


bool is_EO(cv::Mat &binaryImage){

    // Ensure that the image is binary

    int cur_GRID_SIZE = GRID_SIZE;
    int cur_PRE_PROCESS = PRE_PROCESS;
    int cur_INNER = INNER;
    int cur_smaller_gs = smaller_gs;

    GRID_SIZE = 2;
    PRE_PROCESS = 0;
    INNER = 1;
    smaller_gs = 5;

    cv::Mat copy_image = binaryImage.clone();

    // show_image(binaryImage);

    cv::Mat back_to_rgb_copy;
    cv::cvtColor(copy_image,back_to_rgb_copy,cv::COLOR_GRAY2BGR);

    // show_image(binaryImage);



    vector<pair<vector<pair<int,int>>,vector<int>>> return_vec = cover_gen(binaryImage);

    // show_image(copy_image);

    double A = 0;
    for(auto &[cover_points, dir_vec]: return_vec){
        A+=polygon_area(cover_points); // Resolution dependent calculation
    }

    int cur_rows = copy_image.rows;
    int cur_cols = copy_image.cols;

    // cout<<cur_rows<<" "<<cur_cols<<endl;

    // exit(-1);

    float area_ratio = A/(cur_cols*cur_rows);

    // show_image(binaryImage);

    cout<<area_ratio<<endl;
    // exit(-1);


    // int thickness = 2;

    // int choice_b = 255;
    // int choice_g = 0;
    // int choice_r = 0;

    // for(auto &ele1 : return_vec){
    //     ele1.first.push_back(ele1.first[0]);
        
    //     int min_x = INT_MAX;
    //     int max_x = INT_MIN;
    //     int min_y = INT_MAX;
    //     int max_y = INT_MIN;
    //     for(int i=1;i<ele1.first.size();i++){


    //         int x1 = ele1.first[i].first;
    //         int y1 = ele1.first[i].second;

    //         int x2 = ele1.first[i-1].first;
    //         int y2 = ele1.first[i-1].second;


    //         // min_x = min({min_x, y1, y2});
    //         // max_x = max({max_x, y1, y2});

    //         // min_y = min({min_y, x1, x2});
    //         // max_y = max({max_y, x1, x2});
    //         cv::line(back_to_rgb_copy,cv::Point(y2, x2), cv::Point(y1, x1), cv::Scalar(choice_b, choice_g, choice_r), thickness);
    //         // cv::line(back_to_rgb_copy,cv::Point(x1, y2), cv::Point(max_x, min_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);
    //         // cv::line(back_to_rgb_copy,cv::Point(x2, y1), cv::Point(min_x, max_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);
    //         // cv::line(back_to_rgb_copy,cv::Point(x2, y2), cv::Point(max_x, min_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);
            
    //     }

    // }

    // show_image(back_to_rgb_copy);
    // exit(-1);


    GRID_SIZE = cur_GRID_SIZE;
    PRE_PROCESS = cur_PRE_PROCESS;
    INNER = cur_INNER;
    smaller_gs = cur_smaller_gs;



}


int main(){

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::cout.tie(NULL);

    srand(time(NULL));
    


    auto start1 = std::chrono::high_resolution_clock::now();



    /*
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                REMEMBER THAT TEST CASES MUST HAVE SOME THICK OUTER WHITE BOUNDARY
                        OTHERWISE THIS IMPLEMENTATION WILL GIVE SEG FAULT
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    */

    // vector<string> pic_names ={"0002","0003","0007","0008","0009","0012","0014","0016","doc_img1","doc_img2","doc_img5","doc_img9"};
    // vector<string> exts = {"jpg","jpg","jpg","jpg","jpg","jpg","jpg","jpg","png","png","png","png"};

    // vector<string> pic_names = {"./Images_for_project/hds", "./Images_for_project/egd2"};
    // vector<string> exts = {"jpg","jpg"};    

    vector<string> pic_names;
    vector<string> exts;   

    string path = "Drawings/Hand_Drawn";
    // string path = "Drawings/Architechtural_Plan";
    

    for (const auto & entry : fs::directory_iterator(path)){
        string path_name = entry.path();
        if(path_name.substr(path_name.length()-5)=="Store" || path_name.substr(path_name.length()-4)=="DS_S"){
            cout<<path_name<<endl;
            continue;

        }
        pic_names.push_back(path_name.substr(0,path_name.length()-4));
        exts.push_back(path_name.substr(path_name.length()-3,3));
    }

    // for(auto &ele1 : pic_names){
    //     cout<<ele1<<endl;
    // }
    // cout<<"--------"<<endl;
    // for(auto &ele2 : exts){
    //     cout<<ele2<<endl;
    // }
    // exit(-1);
        // std::cout << entry.path() << std::endl;
    
    // exit(99);

    assert(pic_names.size()==exts.size());

    float avg = 0;

    for(int i=0;i<pic_names.size();i++){

        // if(pic_names[i]!="./Images_for_project/hds4")
        //     continue;

        cout<<"NOW PROCESSING=====> "<<i<<" "<<pic_names[i]<<endl;

        string pic_name = pic_names[i];
        string ext = exts[i];
        std::string test_pic = "./"+ pic_name+"."+ext;
        // std::string yinyangGrayPath = "./test_pic_gray.jpeg";

        cv::Mat image = cv::imread(test_pic);
        cv::Mat grayImage, binaryImage;

        int border = 220;

        // cv::Mat temp = cv::Mat::zeros(cv::Size(image.cols+border,image.rows+border), CV_64FC1);
        cv::Mat temp(image.rows+border,image.cols+border, CV_8UC3, cv::Scalar(255,255,255));
        int new_dim_cols = image.cols+border;
        int new_dim_rows = image.rows+border;

        for(int i=0;i<temp.rows;i++){
            for(int j=0;j<temp.cols;j++){
                cv::Vec3b pixel = temp.at<cv::Vec3b>(cv::Point(j,i));
                pixel.val[0] = 255;
                pixel.val[1] = 255;
                pixel.val[2] = 255;
                temp.at<cv::Vec3b>(cv::Point(j,i)) = pixel;
            }
        }

        // show_image(temp);

        int shift = (border - 20)/2;

        for(int i=0;i<image.rows;i++){
            for(int j=0;j<image.cols;j++){
                cv::Vec3b pixel = image.at<cv::Vec3b>(cv::Point(j,i));
                temp.at<cv::Vec3b>(cv::Point(j+shift,i+shift)) = pixel;
            }
        }



        // show_image(temp);

        image = temp;

        // show_image(image);


        cv::cvtColor(image,grayImage,cv::COLOR_BGR2GRAY);


        if(!image.data) {
            std::cout<< "No image"<<std::endl;
            return 0;
        }

        cv::Mat channels[3];

        cv::split(image,channels);


        // show_image(grayImage);
        cv::threshold(grayImage,binaryImage,200,255,cv::THRESH_BINARY);//40
        

        cv::Mat binaryImage_copy_copy = binaryImage.clone();

        // show_image(binaryImage_copy);

        cv::Mat back_to_rgb_copy;
        cv::cvtColor(binaryImage_copy_copy,back_to_rgb_copy,cv::COLOR_GRAY2BGR);

        is_EO(binaryImage);

        show_image(binaryImage);

        // continue;


        
        // cv::dilate(binaryImage_copy,dst,elementKernel2,cv::Point(-1,-1),1);
        // cv::erode(dst,dst,elementKernel2,cv::Point(-1,-1),1);
        // show_image(dst);

        // cv::imwrite("./test5.jpeg",binaryImage_copy);



        // 5, 10, 15, 20

        // 10 test images => binary

        


        //////// COVER FINDING AND ANALYSIS /////////

        // show_image(binaryImage);


        // INNER = 1;
        // GRID_SIZE = 3;

        // cv::Mat elementKernel1 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3),cv::Point(-1,-1));

        // cv::Mat elementKernel2 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4),cv::Point(-1,-1));

        // show_image(binaryImage);

        // cv::Mat bwn;

        // vector<vector<cv::Point>> contours;
        // cv::Mat contourOutput = binaryImage.clone();
        // cv::findContours(contourOutput, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );

        

        // bitwise_not(binaryImage, bwn);

        // cv::Mat contourImage(binaryImage.size(), CV_8UC3, cv::Scalar(0,0,0));

        // cv::Scalar colors[3];
        // colors[0] = cv::Scalar(255, 255, 255);
        // // colors[1] = cv::Scalar(0, 255, 0);
        // // colors[2] = cv::Scalar(0, 0, 255);
        // for (size_t idx = 0; idx < contours.size(); idx++) {
        //     cv::drawContours(contourImage, contours, idx, colors[0]);
        // }
        
        // // cv::dilate(binaryImage,binaryImage,elementKernel1,cv::Point(-1,-1),1);

        // // cv::erode(binaryImage,binaryImage,elementKernel2,cv::Point(-1,-1),1);

        // bitwise_not(binaryImage, bwn);

        // show_image(bwn);
        // show_image(contourImage);
        // show_image(binaryImage);

        // show_image(binaryImage);

        vector<pair<vector<pair<int,int>>,vector<int>>> return_vec = cover_gen(binaryImage);

        //////////////// Graphic Analysis Starts ////////////////

        int n = 0;
        for(auto &[cover_point_list, direction_vector]:return_vec){
            n+=line_detection(back_to_rgb_copy, cover_point_list, direction_vector);
        }

        // show_image(back_to_rgb_copy);

        double A = 0;
        for(auto &[cover_points, dir_vec]: return_vec){
            A+=polygon_area(cover_points); // Resolution dependent calculation
        }

        double P = 0;
        for(auto &[cover_points, dir_vec]: return_vec){
            P+=find_perimeter(cover_points); // Resolution dependent calculation
        }

        avg += (P/n);

        cout<<A/n<<" "<<P/n<<endl;
        continue;

        //////////////// Graphic Analysis Ends ////////////////

        for(auto &ele1 : return_vec){
            ele1.first.push_back(ele1.first[0]);
            // cout<<"**********##########********"<<endl;
            // for(auto &vals:ele1.first)
            //     cout<<"====>>> ["<<vals.first<<" "<<vals.second<<"]"<<endl;
            // cout<<"**********##########********"<<endl;
            int min_x = INT_MAX;
            int max_x = INT_MIN;
            int min_y = INT_MAX;
            int max_y = INT_MIN;
            for(int i=1;i<ele1.first.size();i++){


                int x1 = ele1.first[i].first;
                int y1 = ele1.first[i].second;

                int x2 = ele1.first[i-1].first;
                int y2 = ele1.first[i-1].second;


                min_x = min({min_x, y1, y2});
                max_x = max({max_x, y1, y2});

                min_y = min({min_y, x1, x2});
                max_y = max({max_y, x1, x2});

                // cv::line(temp,cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(0, 0, 255),2);
                // show_image(graphic_img_copy);
            }

            int thickness = 2;

            int choice_b = 255;
            int choice_g = 0;
            int choice_r = 0;

            cv::line(back_to_rgb_copy,cv::Point(min_x, min_y), cv::Point(min_x, max_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);
            cv::line(back_to_rgb_copy,cv::Point(min_x, min_y), cv::Point(max_x, min_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);
            cv::line(back_to_rgb_copy,cv::Point(max_x, max_y), cv::Point(min_x, max_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);
            cv::line(back_to_rgb_copy,cv::Point(max_x, max_y), cv::Point(max_x, min_y), cv::Scalar(choice_b, choice_g, choice_r), thickness);



        }
        // show_image(back)
        // show_image(back_to_rgb_copy);
        // exit(-1);

        for(auto &[cover_point_list, direction_vector]: return_vec){

            line_detection(back_to_rgb_copy, cover_point_list, direction_vector);

        }

        // show_image(back_to_rgb_copy);

        string oow = pic_name + "_line_detection_output"+"."+ext;
        cv::imwrite(oow,back_to_rgb_copy);

        continue;

        
        
        /*

            Develop a function that takes in parameter cover1 and returns output as [cover1[i]]->{cover2's that belong to cover1[i]}

        */
        
        auto mapping = cover_mapping(return_vec, binaryImage_copy_copy);

            
        int graphic_area_cnt = 0;
        int possible_textual_area_cnt = 0;

        


        // show_image(back_to_rgb);
        string out_name = "./" + pic_name + "_output"+"."+ext;
        bool is_saved = cv::imwrite(out_name,back_to_rgb_copy);

        if(!is_saved){
            cout<<"Save Unsuccessful."<<endl;
            exit(0);
        }
    }  

    

    /////////////////////////////////////////////

    cout<<"******** AVG ********* "<<avg/(pic_names.size())<<endl;

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

