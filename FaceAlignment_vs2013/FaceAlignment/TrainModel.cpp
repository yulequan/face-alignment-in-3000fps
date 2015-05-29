//
//  TrainDemo.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//
#include "LBFRegressor.h"
using namespace std;
using namespace cv;
void LoadCofwTrainData(vector<Mat_<uchar> >& images,
                       vector<Mat_<double> >& ground_truth_shapes,
                       vector<BoundingBox>& bounding_boxs);
void TrainModel(vector<string> trainDataName){
    vector<Mat_<uchar> > images;
    vector<Mat_<double> > ground_truth_shapes;
    vector<BoundingBox> bounding_boxs;

    for(int i=0;i<trainDataName.size();i++){
        string path;
        if(trainDataName[i]=="helen"||trainDataName[i]=="lfpw")
            path = dataPath + trainDataName[i] + "/trainset/Path_Images.txt";
        else
            path = dataPath + trainDataName[i] + "/Path_Images.txt";

       // LoadData(path, images, ground_truth_shapes, bounding_boxs);
          LoadOpencvBbxData(path, images, ground_truth_shapes, bounding_boxs);
    }

    LBFRegressor regressor;
    regressor.Train(images,ground_truth_shapes,bounding_boxs);
    regressor.Save(modelPath+"LBF.model");
    return;
}


