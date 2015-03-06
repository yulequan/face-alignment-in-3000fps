/*
Author: Bi Sai 
Date: 2014/06/18
This program is a reimplementation of algorithms in "Face Alignment by Explicit 
Shape Regression" by Cao et al.
If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com

Copyright (c) 2014 Bi Sai 
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "LBFRegressor.h"
using namespace std;
using namespace cv;

void TestDemo (){
    vector<Mat_<uchar> > test_images;
    vector<BoundingBox> test_bounding_boxs;
    int test_img_num = 507;
    int initial_number = 20;
    int landmark_num = 29;
    ifstream fin;

    for(int i = 0;i < test_img_num;i++){
        string image_name = "/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/testImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        test_images.push_back(temp);
    }
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/boundingbox_test.txt");
    for(int i = 0;i < test_img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0; 
        test_bounding_boxs.push_back(temp);
    }
    fin.close(); 
    
    LBFRegressor regressor;
    regressor.Load("/Users/lequan/workspace/xcode/myopencv/data/model.txt");
    while(true){
        int index = 1;
        cout<<"Input index:"<<endl;
        cin>>index;

        Mat_<double> current_shape = regressor.Predict(test_images[index],test_bounding_boxs[index],initial_number);
        cout << "Predict end"<<endl;
        Mat test_image_1;
        cvtColor(test_images[index],test_image_1, COLOR_GRAY2BGR);
       
        // draw bounding box
        circle(test_image_1,Point2d(test_bounding_boxs[index].start_x,test_bounding_boxs[index].start_y),3,Scalar(0,255,0),-1,8,0);
        circle(test_image_1,Point2d(test_bounding_boxs[index].start_x+test_bounding_boxs[index].width,test_bounding_boxs[index].start_y),3,Scalar(0,255,0),-1,8,0);
        circle(test_image_1,Point2d(test_bounding_boxs[index].start_x,test_bounding_boxs[index].start_y+test_bounding_boxs[index].height),3,Scalar(0,255,0),-1,8,0);
        circle(test_image_1,Point2d(test_bounding_boxs[index].start_x+test_bounding_boxs[index].width,test_bounding_boxs[index].start_y+test_bounding_boxs[index].height),3,Scalar(0,255,0),-1,8,0);
        
        // draw initialize shape ::blue
        Mat_<double>initializeshape = ReProjectShape(regressor.mean_shape_, test_bounding_boxs[index]);
        for(int i = 0;i < landmark_num;i++){
            circle(test_image_1,Point2d(initializeshape(i,0),initializeshape(i,1)),1,Scalar(255,0,0),-1,8,0);
        }
        cout <<"Initialize shape"<<endl;
        cout <<initializeshape <<endl;
       
        // draw result :: red
        for(int i = 0;i < landmark_num;i++){
            circle(test_image_1,Point2d(current_shape(i,0),current_shape(i,1)),1,Scalar(0,0,255),-1,8,0);
        }
//        cout << "Result shape"<<endl;
//        cout << current_shape<<endl;
        imshow("result",test_image_1);
        waitKey(0);
    }
    return ;
}


