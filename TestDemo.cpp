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
void LoadCofwTestData(vector<Mat_<uchar> >& images,
                      vector<Mat_<double> >& ground_truth_shapes,
                      vector<BoundingBox>& bounding_boxs);
void TestDemo (){
    vector<Mat_<uchar> > test_images;
    vector<BoundingBox> test_bounding_boxs;
    vector<Mat_<double> >test_ground_truth_shapes;
    string testdatapath = "/Users/lequan/Desktop/study/face/face-alignment-3000fps/Datasets/lfpw_testset/Path_Images.txt";
    int initial_number = 20;
//    LoadCofwTestData(test_images, test_ground_truth_shapes, test_bounding_boxs);
 
    LoadOpencvBbxData(testdatapath, test_images, test_ground_truth_shapes, test_bounding_boxs);
    LBFRegressor regressor;
    regressor.Load(modelPath+"model.txt");
    
    int index = 0;
    namedWindow("result",WINDOW_AUTOSIZE);
    while(true){
        index++;
        if (index >500){
            break;
        }
        cout << "Predict "<< index << endl;
        Mat_<double> current_shape = regressor.Predict(test_images[index],test_bounding_boxs[index],initial_number);
        Mat test_image_1;
        cvtColor(test_images[index],test_image_1, COLOR_GRAY2BGR);
       
        // draw bounding box
        rectangle(test_image_1, cvPoint(test_bounding_boxs[index].start_x,test_bounding_boxs[index].start_y),
                  cvPoint(test_bounding_boxs[index].start_x+test_bounding_boxs[index].width,test_bounding_boxs[index].start_y+test_bounding_boxs[index].height),Scalar(0,255,0), 1, 8, 0);
        // draw initialize shape ::blue
        Mat_<double>initializeshape = ReProjectShape(regressor.mean_shape_, test_bounding_boxs[index]);
        for(int i = 0;i < global_params.landmark_num;i++){
            circle(test_image_1,Point2d(initializeshape(i,0),initializeshape(i,1)),1,Scalar(255,0,0),-1,8,0);
        }
        
//       // draw ground truth ::yellow
//       for(int i = 0;i < global_params.landmark_num;i++){
//           circle(test_image_1,Point2d(test_ground_truth_shapes[index](i,0),test_ground_truth_shapes[index](i,1)),1,Scalar(0,255,255),-1,8,0);
//       }
         // draw result :: red
         for(int i = 0;i < global_params.landmark_num;i++){
             circle(test_image_1,Point2d(current_shape(i,0),current_shape(i,1)),1,Scalar(0,0,255),2,8,0);
         }
        imshow("result",test_image_1);
        int c = waitKey();
        if (c =='q'){
            destroyWindow("result");
            return;
        }
    }
    destroyWindow("result");
    return ;
}


void LoadCofwTestData(
                  vector<Mat_<uchar> >& images,
                  vector<Mat_<double> >& ground_truth_shapes,
                  vector<BoundingBox>& bounding_boxs){
    int img_num = 507;
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = "/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/testImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        images.push_back(temp);
    }
    
    ifstream fin;
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/boundingbox_test.txt");
    for(int i = 0;i < img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0;
        bounding_boxs.push_back(temp);
    }
    fin.close();
    
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/keypoints_test.txt");
    for(int i = 0;i < img_num;i++){
        Mat_<double> temp(global_params.landmark_num,2);
        for(int j = 0;j < global_params.landmark_num;j++){
            fin>>temp(j,0);
        }
        for(int j = 0;j < global_params.landmark_num;j++){
            fin>>temp(j,1);
        }
        ground_truth_shapes.push_back(temp);
    }
    fin.close();
}

