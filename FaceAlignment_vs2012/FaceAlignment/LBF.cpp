//
//  LBF.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "LBF.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;

// parameters
Params global_params;

// you need to modify these paths
string modelPath ="../model/";
string dataPath = "/Users/lequan/Desktop/study/face/face-alignment-3000fps/Datasets/";
string cascadeName = "../haarcascade_frontalface_alt.xml";

void InitializeGlobalParam();
void PrintHelp();

int main( int argc, const char** argv ){
    
    //initialize parameters
    if (argc > 1 && strcmp(argv[1],"TrainModel")==0){
        InitializeGlobalParam();
    }
    else {
        ReadGlobalParamFromFile(modelPath+"LBF.model");
    }
    
    // main process
    if (argc==1){
        PrintHelp();
    }
    else if(strcmp(argv[1],"TrainModel")==0){
        TrainDemo();
    }
    else if (strcmp(argv[1], "TestModel")==0){
        double MRSE = TestDemo();
        cout << "Mean Root Square Error is "<< MRSE*100 <<"%"<<endl;
    }
    else if (strcmp(argv[1], "Demo")==0){
        if (argc == 2){
            return FaceDetectionAndAlignment("");
        }
        else if(argc ==3){
            return FaceDetectionAndAlignment(argv[2]);
        }
    }
    else {
        PrintHelp();
    }
    return 0;
}

// set the parameters when training models.
void InitializeGlobalParam(){
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 10;
    global_params.max_depth = 5;
    global_params.landmark_num = 68;
    global_params.initial_num = 5;
    
    global_params.max_numstage = 7;
    double m_max_radio_radius[10] = {0.4,0.3,0.2,0.15, 0.12, 0.10, 0.08, 0.06, 0.06,0.05};
    double m_max_numfeats[8] = {200,200, 200, 100, 100, 100, 80, 80};
    for (int i=0;i<10;i++){
        global_params.max_radio_radius[i] = m_max_radio_radius[i];
    }
    for (int i=0;i<8;i++){
        global_params.max_numfeats[i] = m_max_numfeats[i];
    }
    global_params.max_numthreshs = 500;
}

void ReadGlobalParamFromFile(string path){
    cout << "Loading GlobalParam from " <<path << endl;
    ifstream fin;
    fin.open(path);
    fin >> global_params.bagging_overlap;
    fin >> global_params.max_numtrees;
    fin >> global_params.max_depth;
    fin >> global_params.max_numthreshs;
    fin >> global_params.landmark_num;
    fin >> global_params.initial_num;
    fin >> global_params.max_numstage;
    
    for (int i = 0; i< global_params.max_numstage; i++){
        fin >> global_params.max_radio_radius[i];
    }
    
    for (int i = 0; i < global_params.max_numstage; i++){
        fin >> global_params.max_numfeats[i];
    }
    cout << "Loading GlobalParam end"<<endl;
    fin.close();
}
void PrintHelp(){
    cout << "Useage:"<<endl;
    cout << "1. train your own model:    LBF.out  TrainModel "<<endl;
    cout << "2. test model on dataset:   LBF.out  TestModel"<<endl;
    cout << "3. test model via a camera: LBF.out  Demo "<<endl;
    cout << "4. test model on a pic:     LBF.out  Demo xx.jpg"<<endl;
    cout << "5. test model on pic set:   LBF.out  Demo Img_Path.txt"<<endl;
    cout << endl;
}