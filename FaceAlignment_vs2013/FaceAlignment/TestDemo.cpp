//
//  TestDemo.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//
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
    int initial_number = 1;

    
    // you need to modify this section according to your training dataset
    string testdatapath = dataPath+"helen_testset/Path_Images.txt";
    
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
    cout << "test data size: "<<test_ground_truth_shapes.size()<<endl;
    return MRSE_sum/test_ground_truth_shapes.size();
}


void LoadCofwTestData(vector<Mat_<uchar> >& images,
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
