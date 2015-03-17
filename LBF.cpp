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

int main( int argc, const char** argv ){
    if (argc > 1 && strcmp(argv[1],"TrainDemo")==0){
        InitializeGlobalParam();
    }
    else {
        ReadGlobalParamFromFile(modelPath+"1.model");
    }
//    LBFRegressor regressor;
//    regressor.Load(modelPath+"1.model");
//    regressor.Save("/Users/lequan/workspace/xcode/myopencv/model/1.model");
    // main process
    if (argc==1){
        return FaceDetectionAndAlignment("");
    }
    else if(strcmp(argv[1],"TrainDemo")==0){
        TrainDemo();
    }
    else if (strcmp(argv[1], "TestDemo")==0){
        double MRSE = TestDemo();
        cout << "Mean Root Square Error is "<< MRSE*100 <<"%"<<endl;
    }
    else{
        return FaceDetectionAndAlignment(argv[1]);
    }
    return 0;
}

void InitializeGlobalParam(){
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 10;
    global_params.max_depth = 4;
    global_params.landmark_num = 68;
    global_params.initial_num = 5;
    
    global_params.max_numstage = 5;
    double m_max_radio_radius[10] = {0.4,0.3,0.2,0.15, 0.12, 0.10, 0.08, 0.06, 0.06,0.05};
    double m_max_numfeats[8] = {200,200, 200, 100, 100, 100, 100, 100};
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
    fin.open(path,ios::binary);
    fin.read((char*)&global_params.bagging_overlap, sizeof(double));
    fin.read((char*)&global_params.max_numtrees, sizeof(int));
    fin.read((char*)&global_params.max_depth, sizeof(int));
    fin.read((char*)&global_params.max_numthreshs, sizeof(int));
    fin.read((char*)&global_params.landmark_num, sizeof(int));
    fin.read((char*)&global_params.initial_num, sizeof(int));
    fin.read((char*)&global_params.max_numstage, sizeof(int));
    fin.read((char*)global_params.max_radio_radius, sizeof(double)*global_params.max_numstage);
    fin.read((char*)global_params.max_numfeats, sizeof(int)*global_params.max_numstage);
    cout << "End"<<endl;
    fin.close();
}