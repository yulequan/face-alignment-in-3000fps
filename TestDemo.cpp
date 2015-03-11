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
double TestDemo (){
    vector<Mat_<uchar> > test_images;
    vector<BoundingBox> test_bounding_boxs;
    vector<Mat_<double> >test_ground_truth_shapes;
    string testdatapath = "/Users/lequan/Desktop/study/face/face-alignment-3000fps/Datasets/lfpw_testset/Path_Images.txt";
    int initial_number = 20;
//    LoadCofwTestData(test_images, test_ground_truth_shapes, test_bounding_boxs);
//    LoadData(testdatapath, test_images, test_ground_truth_shapes, test_bounding_boxs);
    LoadOpencvBbxData(testdatapath, test_images, test_ground_truth_shapes, test_bounding_boxs);
    LBFRegressor regressor;
    regressor.Load(modelPath+"LBF.model");
    vector<Mat_<double> > current_shape = regressor.Predict(test_images,test_bounding_boxs,initial_number);
    double MRSE_sum = 0;
    for (int i =0; i<current_shape.size();i++){
        MRSE_sum += CalculateError(test_ground_truth_shapes[i], current_shape[i]);
//        // draw bounding box
//        rectangle(test_images[i], cvPoint(test_bounding_boxs[i].start_x,test_bounding_boxs[i].start_y),
//                  cvPoint(test_bounding_boxs[i].start_x+test_bounding_boxs[i].width,test_bounding_boxs[i].start_y+test_bounding_boxs[i].height),Scalar(0,255,0), 1, 8, 0);
//        // draw result :: red
//        for(int j = 0;j < global_params.landmark_num;j++){
//            circle(test_images[i],Point2d(current_shape[i](j,0),current_shape[i](j,1)),1,Scalar(0,0,255),3,8,0);
//        }
//        imshow("result", test_images[i]);
//        waitKey(0);
        
    }
    cout << test_ground_truth_shapes.size()<<endl;
    return MRSE_sum/test_ground_truth_shapes.size();
}
