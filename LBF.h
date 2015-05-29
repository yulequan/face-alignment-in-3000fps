//
//  LBF.h
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

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
void LoadDataAdjust(std::string filepath,
              std::vector<cv::Mat_<uchar> >& images,
              std::vector<cv::Mat_<double> >& ground_truth_shapes,
              std::vector<BoundingBox> & bounding_box);
void LoadOpencvBbxData(std::string filepath,
                       std::vector<cv::Mat_<uchar> >& images,
                       std::vector<cv::Mat_<double> >& ground_truth_shapes,
                       std::vector<BoundingBox> & bounding_boxs
                       );
void LoadCofwTrainData(std::vector<cv::Mat_<uchar> >& images,
                       std::vector<cv::Mat_<double> >& ground_truth_shapes,
                       std::vector<BoundingBox>& bounding_boxs);
void LoadCofwTestData(std::vector<cv::Mat_<uchar> >& images,
                       std::vector<cv::Mat_<double> >& ground_truth_shapes,
                       std::vector<BoundingBox>& bounding_boxs);

BoundingBox CalculateBoundingBox(cv::Mat_<double>& shape);
cv::Mat_<double> LoadGroundTruthShape(std::string& filename);
void adjustImage(cv::Mat_<uchar>& img,
                 cv::Mat_<double>& ground_truth_shape,
                 BoundingBox& bounding_box);

void  TrainModel(std::vector<std::string> trainDataName);
double TestModel(std::vector<std::string> testDataName);
int FaceDetectionAndAlignment(const char* inputname);
void ReadGlobalParamFromFile(cv::string path);
double CalculateError(const cv::Mat_<double>& ground_truth_shape, const cv::Mat_<double>& predicted_shape);
#endif
