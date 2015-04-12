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
#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "cv.h"
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>   
#include <utility> 

struct Params{
    
    double bagging_overlap;
    int max_numtrees;
    int max_depth;
    int landmark_num;// to be decided
    int initial_num;
    
    int max_numstage;
    double max_radio_radius[10];
    int max_numfeats[10]; // number of pixel pairs
    int max_numthreshs;
};
extern Params global_params;
extern cv::string modelPath;
extern cv::string dataPath;
class BoundingBox{
    public:
        double start_x;
        double start_y;
        double width;
        double height;
        double centroid_x;
        double centroid_y;
        BoundingBox(){
            start_x = 0;
            start_y = 0;
            width = 0;
            height = 0;
            centroid_x = 0;
            centroid_y = 0;
        }; 
};
cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& shapes,
                              const std::vector<BoundingBox>& bounding_box);

void GetShapeResidual(const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                      const std::vector<cv::Mat_<double> >& current_shapes,
                      const std::vector<BoundingBox>& bounding_boxs,
                      const cv::Mat_<double>& mean_shape,
                      std::vector<cv::Mat_<double> >& shape_residuals);

cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const cv::Mat_<double>& shape1, const cv::Mat_<double>& shape2,
                         cv::Mat_<double>& rotation,double& scale);
double calculate_covariance(const std::vector<double>& v_1,
                            const std::vector<double>& v_2);
void LoadData(std::string filepath,
              std::vector<cv::Mat_<uchar> >& images,
              std::vector<cv::Mat_<double> >& ground_truth_shapes,
              std::vector<BoundingBox> & bounding_box);
void LoadOpencvBbxData(std::string filepath,
                       std::vector<cv::Mat_<uchar> >& images,
                       std::vector<cv::Mat_<double> >& ground_truth_shapes,
                       std::vector<BoundingBox> & bounding_boxs
                       );

void TrainDemo();
double TestDemo();
int FaceDetectionAndAlignment(const char* inputname);
void ReadGlobalParamFromFile(cv::string path);
double CalculateError(cv::Mat_<double>& ground_truth_shape, cv::Mat_<double>& predicted_shape);
#endif
