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

