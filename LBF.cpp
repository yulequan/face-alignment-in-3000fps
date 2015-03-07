/**
 * @author 
 * @version 2014/06/18
 */

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

int main( int argc, const char** argv ){
    if (argc==1){
        return FaceDetectionAndAlignment("");
    }
    else if(strcmp(argv[1],"TrainDemo")==0){
        TrainDemo();
    }
    else if (strcmp(argv[1], "TestDemo")==0){
        TestDemo();
    }
    else{
        return FaceDetectionAndAlignment(argv[1]);
    }
    return 0;
}

