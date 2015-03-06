/**
 * @author 
 * @version 2014/06/18
 */

#include "FaceAlignment.h"
using namespace std;
using namespace cv;
void InitializeGlobalParam(){
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 5;
    global_params.max_depth = 2;
    global_params.max_numthreshs = 100;
    global_params.landmark_num = 29;
    global_params.initial_num = 5;

    global_params.max_numstage = 3;
    double m_max_radio_radius[10] = {0.4,0.3,0.2,0.15, 0.1, 0.1, 0.1, 0.1, 0.08,0.08};
    double m_max_numfeats[8] = {5,5,5,5,5,5,4,4};
    for (int i=0;i<10;i++){
        global_params.max_radio_radius[i] = m_max_radio_radius[i];
    }
    for (int i=0;i<8;i++){
        global_params.max_numfeats[i] = m_max_numfeats[i];
    }
}

int main( int argc, char** argv){
//    InitializeGlobalParam();
//    if (argc < 2) {
//        cout << "Error" << endl;
//    }
//    else if (strcmp(argv[1], "Train") == 0){
//        TrainDemo();
//    }
//    else if (strcmp(argv[2],"Test") == 0){
//        TestDemo();
//    }
     TrainDemo();
}

