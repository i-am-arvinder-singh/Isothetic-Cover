#include <bits/stdc++.h>

#include <opencv2/core.hpp>

#define endl "\n"

using namespace std;

struct pixel{
    int r,g,b;
};

void print(ofstream &image, vector<vector<pixel>> &mat){
    for(auto &vec:mat){
        for(auto &pix:vec){
            image<<pix.r<<" "<<pix.g<<" "<<pix.b<<endl;
        }
    }
}

int main(){

    ofstream image;
    image.open("./image.ppm");

    if(image.is_open()){
        image<<"P3"<<endl;
        int h = 2560;
        int w = 1600;
        image<<h<<" "<<w<<endl;
        image<<"255"<<endl;

        vector<vector<pixel>> mat(w,vector<pixel>(h));

        srand(time(NULL));

        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                int i_ = int(floor(((double)i/w)*255));
                int j_ = int(floor(((double)j/h)*255));
                pixel p;
                p.r = i_;
                p.g = j_;
                p.b = j_;
                mat[i][j] = p;
            }
        }

        print(image,mat);
    }

    image.close();

}