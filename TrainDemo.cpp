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
void LoadCofwTrainData(vector<Mat_<uchar> >& images,
                       vector<Mat_<double> >& ground_truth_shapes,
                       vector<BoundingBox>& bounding_boxs);
void TrainDemo(){
    vector<Mat_<uchar> > images;
    vector<Mat_<double> > ground_truth_shapes;
    vector<BoundingBox> bounding_boxs;
    string traindatapath1 = "/Users/lequan/Desktop/study/face/face-alignment-3000fps/Datasets/afw/Path_Images.txt";
    string traindatapath2 = "/Users/lequan/Desktop/study/face/face-alignment-3000fps/Datasets/lfpw_trainset/Path_Images.txt";
    LBFRegressor regressor;

//    LoadCofwTrainData(images, ground_truth_shapes, bounding_boxs);
    LoadOpencvBbxData(traindatapath1, images, ground_truth_shapes, bounding_boxs);
    LoadOpencvBbxData(traindatapath2, images, ground_truth_shapes, bounding_boxs);
    regressor.Train(images,ground_truth_shapes,bounding_boxs);
    regressor.Save(modelPath+"LBF.model");
    return;
}
