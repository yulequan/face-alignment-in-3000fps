//
//  RandomForest.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "RandomForest.h"
#include "RandomForest.h"
using namespace std;
using namespace cv;

void RandomForest::Train(
                         const vector<Mat_<uchar> >& images,
                         const vector<Mat_<double> >& ground_truth_shapes,
                         const vector<Mat_<double> >& current_shapes,
                         const vector<BoundingBox> & bounding_boxs,
                         const Mat_<double>& mean_shape,
                         const vector<Mat_<double> >& shapes_residual,
                         int stages
                         ){
    stages_ = stages;
    #pragma omp parallel for
    for (int i=0;i<num_landmark_;i++){
        clock_t tt = clock();
        int dbsize = (int)images.size();
        int Q = floor(dbsize*1.0/((1-overlap_ratio_)*max_numtrees_));
        int is,ie;
        vector<int> index;
        index.reserve(Q+1);
        for (int j =0;j <max_numtrees_; j++){
            index.clear();
            is = max( floor(j*Q - j*Q*overlap_ratio_ ), 0.0);
            ie = min(is + Q, dbsize-1);
            for (int k = is; k<=ie;k++){
                index.push_back(k);
            }
            rfs_[i][j].Train(images, ground_truth_shapes, current_shapes, bounding_boxs, mean_shape,shapes_residual,index,stages_,i);
        }
        double time = double(clock() - tt) / CLOCKS_PER_SEC;
        cout << "the train rf of "<< i <<"th landmark cost "<< time<<"s"<<endl;
    }

}
void RandomForest::Write(std::ofstream& fout){
    fout << stages_ <<endl;
    fout << max_numtrees_<<endl;
    fout << num_landmark_<<endl;
    fout << max_depth_ <<endl;
    fout << overlap_ratio_ <<endl;
    for (int i=0; i< num_landmark_;i++){
        for (int j = 0; j < max_numtrees_; j++){
            rfs_[i][j].Write(fout);
        }
    }
}
void RandomForest::Read(std::ifstream& fin){
    fin >> stages_;
    fin >> max_numtrees_;
    fin >> num_landmark_;
    fin >> max_depth_;
    fin >> overlap_ratio_;
    for (int i=0; i< num_landmark_;i++){
        for (int j = 0; j < max_numtrees_; j++){
            rfs_[i][j].Read(fin);
        }
    }
}
