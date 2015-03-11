/**
 * @author 
 * @version 2014/06/18
 */

#include "LBF.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;

// parameters
Params global_params;
string modelPath ="/Users/lequan/workspace/xcode/myopencv/model/";
string cascadeName = "haarcascade_frontalface_alt.xml";
double scale = 1.3;
void InitializeGlobalParam();
struct A {
    double a;
    int b[3];
};

int main( int argc, const char** argv ){
    if (argc > 1 && strcmp(argv[1],"TrainDemo")==0){
        InitializeGlobalParam();
    }
    else {
        ReadGlobalParamFromFile(modelPath+"LBF.model");
    }
    LBFRegressor regressor;
    regressor.Load(modelPath+"LBF.model");
    regressor.Save("/Users/lequan/workspace/xcode/myopencv/model/1.model");
//    // main process
//    if (argc==1){
//        return FaceDetectionAndAlignment("");
//    }
//    else if(strcmp(argv[1],"TrainDemo")==0){
//        TrainDemo();
//    }
//    else if (strcmp(argv[1], "TestDemo")==0){
//        double MRSE = TestDemo();
//        cout << "Mean Root Square Error is "<< MRSE*100 <<"%"<<endl;
//    }
//    else{
//        return FaceDetectionAndAlignment(argv[1]);
//    }
    return 0;
}

void InitializeGlobalParam(){
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 10;
    global_params.max_depth = 5;
    global_params.landmark_num = 68;
    global_params.initial_num = 10;
    
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
    cout << "Loading GlobalParam..." << endl;
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
    cout << "End"<<endl;
    fin.close();
}