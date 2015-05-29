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

double TestModel (vector<string> testDataName){
    vector<Mat_<uchar> > test_images;
    vector<BoundingBox> test_bounding_boxs;
    vector<Mat_<double> >test_ground_truth_shapes;
    int initial_number = 1;
    for(int i=0;i<testDataName.size();i++){
        string path;
        if(testDataName[i]=="helen"||testDataName[i]=="lfpw")
            path = dataPath + testDataName[i] + "/testset/Path_Images.txt";
        else
            path = dataPath + testDataName[i] + "/Path_Images.txt";
        //LoadData(path, test_images, test_ground_truth_shapes, test_bounding_boxs);
        LoadOpencvBbxData(path, test_images, test_ground_truth_shapes, test_bounding_boxs);
    }
//    LoadCofwTestData(test_images, test_ground_truth_shapes, test_bounding_boxs);
    
    
    LBFRegressor regressor;
    regressor.Load(modelPath+"LBF.model");
    double t1 =(double)cvGetTickCount();
    vector<Mat_<double> > current_shapes = regressor.Predict(test_images,test_bounding_boxs,
                                                             test_ground_truth_shapes,initial_number);
    //vector<Mat_<double> >current_shapes = test_ground_truth_shapes;
    double t2 =(double)cvGetTickCount();
    
    cout << "test data size: "<<current_shapes.size()<<endl;
    cout << " average predict time is "<<(t2-t1)/((double)cvGetTickFrequency()*1000*current_shapes.size())<<" ms"<<endl;
    
    double MRSE_sum = 0;
    for (int i =0; i<current_shapes.size();i++){
        MRSE_sum += CalculateError(test_ground_truth_shapes[i], current_shapes[i]);
//        // draw bounding box
//        rectangle(test_images[i], cvPoint(test_bounding_boxs[i].start_x,test_bounding_boxs[i].start_y),
//                  cvPoint(test_bounding_boxs[i].start_x+test_bounding_boxs[i].width,test_bounding_boxs[i].start_y+test_bounding_boxs[i].height),Scalar(0,255,0), 1, 8, 0);
        // draw result :: red
//        for(int j = 0;j < global_params.landmark_num;j++){
//            circle(test_images[i],Point2d(current_shapes[i](j,0),current_shapes[i](j,1)),1,Scalar(255,255,255),-1,8,0);
//        }
//        imshow("result", test_images[i]);
//        char a = waitKey(0);
//        if(a=='s'){
//            imwrite(to_string(i)+".jpg",test_images[i]);
//        }
        
        
    }
    cout << "Mean Root Square Error is "<< MRSE_sum/current_shapes.size()*100 <<"%"<<endl;
    return MRSE_sum/current_shapes.size();
}


