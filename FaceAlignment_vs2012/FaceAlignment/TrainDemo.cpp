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
void TrainDemo(){
    vector<Mat_<uchar> > images;
    vector<Mat_<double> > ground_truth_shapes;
    vector<BoundingBox> bounding_boxs;
    
    // you need to modify this section according to your training dataset
    string traindatapath1 = dataPath+"helen_trainset/Path_Images.txt";
    string traindatapath2 = dataPath+"afw/Path_Images.txt";
    string traindatapath3 = dataPath+"lfpw_trainset/Path_Images.txt";
    
    LBFRegressor regressor;
    LoadOpencvBbxData(traindatapath1, images, ground_truth_shapes, bounding_boxs);
    LoadOpencvBbxData(traindatapath2, images, ground_truth_shapes, bounding_boxs);
    LoadOpencvBbxData(traindatapath3, images, ground_truth_shapes, bounding_boxs);
    regressor.Train(images,ground_truth_shapes,bounding_boxs);
    regressor.Save(modelPath+"LBF.model");
    return;
}


void LoadCofwTrainData(vector<Mat_<uchar> >& images,
                       vector<Mat_<double> >& ground_truth_shapes,
                       vector<BoundingBox>& bounding_boxs){
    int img_num = 1345;
    cout<<"Read images..."<<endl;
    for(int i = 0;i < img_num;i++){
        string image_name = "/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/trainingImages/";
        image_name = image_name + to_string(i+1) + ".jpg";
        Mat_<uchar> temp = imread(image_name,0);
        images.push_back(temp);
    }
    
    ifstream fin;
    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/boundingbox.txt");
    for(int i = 0;i < img_num;i++){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width/2.0;
        temp.centroid_y = temp.start_y + temp.height/2.0; 
        bounding_boxs.push_back(temp);
    }
    fin.close(); 

    fin.open("/Users/lequan/workspace/xcode/myopencv/COFW_Dataset/keypoints.txt");
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

