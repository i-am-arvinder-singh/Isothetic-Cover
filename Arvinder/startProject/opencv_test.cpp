
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

using namespace std;

// #define INNER 0
#define IMPROVED 1
#define LINE_GEN_STRICTNESS 1
#define CNT_LINE_AFTER 1
#define ERROR_TOLERANCE 110// in percent
#define THRESHOLD 0.6
// #define PRE_PROCESS 1

int GRID_SIZE = 23;
int PRE_PROCESS = 0;
int INNER = 0;

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
    cout<<"RETT errr : "<<ret_err<<endl;
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
    cv::Mat binaryImage){



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
    // cout<<"hreeeeeee"<<endl;
    // show_image(binaryImage_copy);

    std::vector<std::vector<bool>> is_pixel_present(BLOCK_HEIGHT,std::vector<bool>(BLOCK_WIDTH));

    std::cout<<"******************************"<<std::endl;

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
            std::cout<<is_pixel_present[i][j];
        }
        std::cout<<std::endl;
    }

    std::cout<<"******************************"<<std::endl;
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

            cout<<"-------------------Traversal Direction Below Cover No."<<number_of_covers<<"------------------"<<endl;

            for(auto ele:direction_vector){
                cout<<ele;
            }
            cout<<endl;
            cout<<direction_vector.size()<<endl;
            // Analyze shape
            cout<<"******===>> "<<"here"<<endl;
            // bool is_rectillinear = analyse_shape_regex(direction_vector);
            cout<<"******===>> "<<"here"<<endl;
            // cout<<direction_vector[35]<<" "<<direction_vector[108]<<endl;
            cout<<endl<<"-------------------------------------"<<endl;

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

            cout<<"here"<<endl;

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

    cv::Mat temp = binaryImage_copy_copy;

    // show_image(temp);

    // for(auto &ele1 : return_value){
    //     ele1.first.push_back(ele1.first[0]);
    //     // cout<<"**********##########********"<<endl;
    //     // for(auto &vals:ele1.first)
    //     //     cout<<"====>>> ["<<vals.first<<" "<<vals.second<<"]"<<endl;
    //     // cout<<"**********##########********"<<endl;
    //     for(int i=1;i<ele1.first.size();i++){
    //         int x1 = ele1.first[i].first;
    //         int y1 = ele1.first[i].second;

    //         int x2 = ele1.first[i-1].first;
    //         int y2 = ele1.first[i-1].second;
    //         cv::line(temp,cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 0, 0));
    //         // show_image(graphic_img_copy);
    //     }
    // }

    // show_image(temp);


    return return_value;


    // ####################################### Actual Algo Implementation Ends Here ######################################### //

}

int find_perimeter(vector<pair<int,int>> &point_list){
    int n = point_list.size();
    assert(n>1);
    return n;
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

    vector<string> pic_names ={"0002"};
    vector<string> exts = {"jpg"};    

    assert(pic_names.size()==exts.size());

    for(int i=0;i<pic_names.size();i++){

        cout<<"NOW PROCESSING=====> "<<i<<endl;

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

        // show_image(image);

        // show_image(grayImage);
        string o_name = "./" + pic_name + "_output_gray_image"+"."+ext;
        cv::imwrite(o_name,grayImage);

        // show_image(image);

        cv::Mat channels[3];

        cv::split(image,channels);


        // show_image(grayImage);
        cv::threshold(grayImage,binaryImage,200,255,cv::THRESH_BINARY);//40
        // show_image(binaryImage);

        cv::Mat binaryImage_copy_copy = binaryImage.clone();

        o_name = "./" + pic_name + "_output_binary_image"+"."+ext;
        cv::imwrite(o_name,binaryImage_copy_copy);

        // show_image(binaryImage_copy);

        cv::Mat back_to_rgb_copy;
        cv::cvtColor(binaryImage_copy_copy,back_to_rgb_copy,cv::COLOR_GRAY2BGR);
        // cv::dilate(binaryImage_copy,dst,elementKernel2,cv::Point(-1,-1),1);
        // cv::erode(dst,dst,elementKernel2,cv::Point(-1,-1),1);
        // show_image(dst);

        // cv::imwrite("./test5.jpeg",binaryImage_copy);



        // 5, 10, 15, 20

        // 10 test images => binary

        


        //////// COVER FINDING AND ANALYSIS /////////

        vector<pair<vector<pair<int,int>>,vector<int>>> return_vec = cover_gen(binaryImage);


            
        int graphic_area_cnt = 0;
        int possible_textual_area_cnt = 0;

        for(auto &ele :return_vec){

            bool is_rectillinear = analyse_shape_improved(ele.first);

            if(!is_rectillinear){

                cv::Mat graphic_img_ = binaryImage_copy_copy.clone();
                for(int i=0;i<graphic_img_.rows;i++){
                    for(int j=0;j<graphic_img_.cols;j++){
                        cv::Vec3b pixel = graphic_img_.at<cv::Vec3b>(cv::Point(j,i));
                        pixel.val[0] = 255;
                        pixel.val[1] = 255;
                        pixel.val[2] = 255;
                        graphic_img_.at<cv::Vec3b>(cv::Point(j,i)) = pixel;
                    }
                }
                // show_image(graphic_img_);
                
                int m = ele.first.size();

                cv::Point points[1][m];

                for(int i=0;i<m;i++){
                    points[0][i] = cv::Point(ele.first[i].second,ele.first[i].first);
                }

                const cv::Point* ppt[1] = {points[0]};
                int npt[] = {m};

                cv::fillPoly(graphic_img_,ppt,npt,1,cv::Scalar( 1, 1, 1 ),cv::LINE_8 );

                int cnt_pixs = 0;

                for(int i=0;i<graphic_img_.rows;i++){
                    for(int j=0;j<graphic_img_.cols;j++){
                        cv::Vec3b pixel = graphic_img_.at<cv::Vec3b>(cv::Point(j,i));
                        cv::Vec3b pixel_2 = binaryImage_copy_copy.at<cv::Vec3b>(cv::Point(j,i));
                        pixel.val[0] = (pixel.val[0]==255?255:pixel_2.val[0]);
                        pixel.val[1] = (pixel.val[1]==255?255:pixel_2.val[1]);
                        pixel.val[2] = (pixel.val[2]==255?255:pixel_2.val[2]);
                        if(pixel.val[0]!=255 || pixel.val[1]!=255 || pixel.val[2]!=255)
                            cnt_pixs++;
                        graphic_img_.at<cv::Vec3b>(cv::Point(j,i)) = pixel;
                    }
                }

                cv::Mat second_stage_img = graphic_img_;
                cv::cvtColor(second_stage_img,second_stage_img,cv::COLOR_GRAY2BGR);

                if(cnt_pixs>50){

                    int prev_gs = GRID_SIZE;
                    int prev_process = PRE_PROCESS;
                    int smaller_gs = 3;
                    int prev_inner = INNER;
                    GRID_SIZE = smaller_gs;
                    PRE_PROCESS = 0;
                    INNER = 0;

                    vector<pair<vector<pair<int,int>>,vector<int>>> cover_second_stage = cover_gen(graphic_img_);

                    GRID_SIZE = prev_gs;
                    PRE_PROCESS = prev_process;
                    INNER = prev_inner;

                    map<int,int> track_cover_sizes;

                    double sum_perimeter = 0;
                    double sum_area = 0;
                    int n_count = 0;

                    for(auto &ele1 : cover_second_stage){
                        
                        sum_perimeter+=(find_perimeter(ele1.first));
                        sum_area+=(polygon_area(ele1.first));
                        int ele1_size = ele1.first.size();
                        for(int i=0;i<ele1_size;i++){
                            auto prev = ele1.first[(i-1+ele1_size)%ele1_size];
                            auto cur = ele1.first[i];
                            auto next = ele1.first[(i+1)%ele1_size];
                            if(
                                (prev.first==cur.first && cur.first==next.first) ||
                                (prev.second==cur.second && cur.second==next.second)
                            ){
                                // n_count++;
                            }
                            else{
                                n_count++;
                            }
                        }

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
                            cv::line(second_stage_img,cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 0, 0));
                            // show_image(graphic_img_copy);
                        }
                        ele1.first.pop_back();
                        line_detection(second_stage_img, ele1.first, ele1.second);
                    }
                    
                    int max_cover_size = INT_MIN;

                    cout<<"**********##########********"<<endl;
                    int outer_bigger_cover_perimeter = find_perimeter(ele.first)*prev_gs;
                    cout<<"Outer bigger perimeter: "<<outer_bigger_cover_perimeter<<endl; 
                    for(auto &[x,y]:track_cover_sizes){
                        max_cover_size = max(max_cover_size,(x*smaller_gs));
                        cout<<"["<<(x*smaller_gs)<<"=>"<<y<<"]"<<endl;
                    }
                    cout<<"**********##########********"<<endl;
                    string out_name;
                    
                    graphic_area_cnt++;
                    double first_ratio = ((double)sum_area)/n_count;
                    double second_ratio = ((double)sum_perimeter)/n_count;
                    cout<<"^^^^^^^^^^^^^^^^^^^"<<endl;
                    cout<<"sum_area: "<<sum_area<<endl;
                    cout<<"sum_perimeter: "<<sum_area<<endl;
                    cout<<"first_ratio: "<<first_ratio<<endl;
                    cout<<"second_ratio: "<<second_ratio<<endl;
                    cout<<"^^^^^^^^^^^^^^^^^^^"<<endl;

                    if(first_ratio>35){
                        out_name = "./" + pic_name + "_output_graphic_element_EO_"+to_string(graphic_area_cnt)+"."+ext;

                    }
                    else if(second_ratio>7){
                        out_name = "./" + pic_name + "_output_graphic_element_ED_AP_"+to_string(graphic_area_cnt)+"."+ext;
                    }
                    else{
                        out_name = "./" + pic_name + "_output_graphic_element_HDS_"+to_string(graphic_area_cnt)+"."+ext;
                    }
                    

                    bool is_saved = cv::imwrite(out_name,second_stage_img);

                    show_image(second_stage_img);

                    if(!is_saved){
                        cout<<"Save Unsuccessful for text extraction."<<endl;
                        exit(0);
                    }




                }


            }
            else{ // Not rectillinear

                cv::Mat graphic_img_ = binaryImage_copy_copy.clone();
                for(int i=0;i<graphic_img_.rows;i++){
                    for(int j=0;j<graphic_img_.cols;j++){
                        cv::Vec3b pixel = graphic_img_.at<cv::Vec3b>(cv::Point(j,i));
                        pixel.val[0] = 255;
                        pixel.val[1] = 255;
                        pixel.val[2] = 255;
                        graphic_img_.at<cv::Vec3b>(cv::Point(j,i)) = pixel;
                    }
                }
                // show_image(graphic_img_);
                
                int m = ele.first.size();

                cv::Point points[1][m];

                for(int i=0;i<m;i++){
                    points[0][i] = cv::Point(ele.first[i].second,ele.first[i].first);
                }

                const cv::Point* ppt[1] = {points[0]};
                int npt[] = {m};

                cv::fillPoly(graphic_img_,ppt,npt,1,cv::Scalar( 1, 1, 1 ),cv::LINE_8 );

                int cnt_pixs = 0;

                for(int i=0;i<graphic_img_.rows;i++){
                    for(int j=0;j<graphic_img_.cols;j++){
                        cv::Vec3b pixel = graphic_img_.at<cv::Vec3b>(cv::Point(j,i));
                        cv::Vec3b pixel_2 = binaryImage_copy_copy.at<cv::Vec3b>(cv::Point(j,i));
                        pixel.val[0] = (pixel.val[0]==255?255:pixel_2.val[0]);
                        pixel.val[1] = (pixel.val[1]==255?255:pixel_2.val[1]);
                        pixel.val[2] = (pixel.val[2]==255?255:pixel_2.val[2]);
                        if(pixel.val[0]!=255 || pixel.val[1]!=255 || pixel.val[2]!=255)
                            cnt_pixs++;
                        graphic_img_.at<cv::Vec3b>(cv::Point(j,i)) = pixel;
                    }
                }

                cv::Mat second_stage_img = graphic_img_;
                cv::cvtColor(second_stage_img,second_stage_img,cv::COLOR_GRAY2BGR);

                if(cnt_pixs>50){
                    
                    // show_image(graphic_img_);

                    int prev_gs = GRID_SIZE;
                    int prev_process = PRE_PROCESS;
                    int smaller_gs = 3;
                    int prev_inner = INNER;
                    GRID_SIZE = smaller_gs;
                    PRE_PROCESS = 0;
                    INNER = 0;

                    vector<pair<vector<pair<int,int>>,vector<int>>> cover_second_stage = cover_gen(graphic_img_);

                    GRID_SIZE = prev_gs;
                    PRE_PROCESS = prev_process;
                    INNER = prev_inner;

                    
                    
                    map<int,int> track_cover_sizes;

                    double sum_perimeter = 0;
                    double sum_area = 0;
                    int n_count = 0;


                    for(auto &ele1 : cover_second_stage){
                        track_cover_sizes[find_perimeter(ele1.first)]++;

                        sum_perimeter+=(find_perimeter(ele1.first));
                        sum_area+=(polygon_area(ele1.first));
                        
                        int ele1_size = ele1.first.size();
                        for(int i=0;i<ele1_size;i++){
                            auto prev = ele1.first[(i-1+ele1_size)%ele1_size];
                            auto cur = ele1.first[i];
                            auto next = ele1.first[(i+1)%ele1_size];
                            if(
                                (prev.first==cur.first && cur.first==next.first) ||
                                (prev.second==cur.second && cur.second==next.second)
                            ){
                                // n_count++;
                            }
                            else{
                                n_count++;
                            }
                        }

                        // assert(n_count==ele1.first.size());


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
                            cv::line(second_stage_img,cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 0, 0));
                            // show_image(graphic_img_copy);
                        }
                        ele1.first.pop_back();
                        line_detection(second_stage_img, ele1.first, ele1.second);
                    }
                    
                    int max_cover_size = INT_MIN;

                    cout<<"**********##########********"<<endl;
                    int outer_bigger_cover_perimeter = find_perimeter(ele.first)*prev_gs;
                    cout<<"Outer bigger perimeter: "<<outer_bigger_cover_perimeter<<endl; 
                    for(auto &[x,y]:track_cover_sizes){
                        max_cover_size = max(max_cover_size,(x*smaller_gs));
                        cout<<"["<<(x*smaller_gs)<<"=>"<<y<<"]"<<endl;
                    }
                    cout<<"**********##########********"<<endl;
                    string out_name;
                    if(max_cover_size>1100){
                        graphic_area_cnt++;
                        double first_ratio = ((double)sum_area)/n_count;
                        double second_ratio = ((double)sum_perimeter)/n_count;

                        cout<<"^^^^^^^^^^^^^^^^^^^"<<endl;
                        cout<<"sum_area: "<<sum_area<<endl;
                        cout<<"sum_perimeter: "<<sum_area<<endl;
                        cout<<"first_ratio: "<<first_ratio<<endl;
                        cout<<"second_ratio: "<<second_ratio<<endl;
                        cout<<"^^^^^^^^^^^^^^^^^^^"<<endl;

                        if(first_ratio>35){
                            out_name = "./" + pic_name + "_output_graphic_element_EO_"+to_string(graphic_area_cnt)+"."+ext;

                        }
                        else if(second_ratio>7){
                            out_name = "./" + pic_name + "_output_graphic_element_ED_AP_"+to_string(graphic_area_cnt)+"."+ext;
                        }
                        else{
                            out_name = "./" + pic_name + "_output_graphic_element_HDS_"+to_string(graphic_area_cnt)+"."+ext;
                        }
                    }
                    else{

                        possible_textual_area_cnt++;
                        out_name = "./" + pic_name + "_output_textual_element_"+to_string(possible_textual_area_cnt)+"."+ext;
                    }

                    bool is_saved = cv::imwrite(out_name,second_stage_img);

                    show_image(second_stage_img);

                    if(!is_saved){
                        cout<<"Save Unsuccessful for text extraction."<<endl;
                        exit(0);
                    }

                }

            }

            ele.first.push_back(ele.first[0]);

            for(int i=1;i<ele.first.size();i++){
                int x1 = ele.first[i].first;
                int y1 = ele.first[i].second;

                int x2 = ele.first[i-1].first;
                int y2 = ele.first[i-1].second;
                // if(once){
                //     cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(0, 0, 255));    
                //     once = false;
                // }
                // else
                // cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 255, 0));
                if(is_rectillinear)
                    cv::line(back_to_rgb_copy, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 255, 0));
                else
                    cv::line(back_to_rgb_copy, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(255, 0, 255));
            }

            // cv::line(back_to_rgb_copy,cv::Point(0, 0), cv::Point(200, 200), cv::Scalar(255, 0, 0));


            

        }
        // show_image(back_to_rgb);
        string out_name = "./" + pic_name + "_output"+"."+ext;
        bool is_saved = cv::imwrite(out_name,back_to_rgb_copy);

        if(!is_saved){
            cout<<"Save Unsuccessful."<<endl;
            exit(0);
        }
    }    

    

    /////////////////////////////////////////////


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
    cout<<"Upto here?"<<endl;
    
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
            if(cur_right-cur_left==1)
                continue;
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

                // cout<<"&&&&&& "<<dir_val_in_range<<" "<<no_of_dir_in_range<<endl;

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
                        cout<<"&&&&&&====> "<<j<<" || "<<see<<" "<<w<<" "<<no_of_dir_in_range<<endl;
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

    cout<<"@@@@@@@ ST LINES::::: "<<new_line_list.size()<<endl;

    int cnt_color_lines = 0;

    for(auto &[l,r]:new_line_list){
        cout<<"####### ==> "<<l<<" "<<r<<endl;
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
            cv::line(back_to_rgb, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar(b_, g_, r_)); 
        }
    }


    return cnt_color_lines;

}