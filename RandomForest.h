//
//  RandomForest.h
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#ifndef __myopencv__RandomForest__
#define __myopencv__RandomForest__

#include "Tree.h"

class RandomForest{
public:
    std::vector<std::vector<Tree> > rfs_;
    int max_numtrees_;
    int num_landmark_;
    int max_depth_;
    int stages_;
    double overlap_ratio_;
    
    
    RandomForest(){
        max_numtrees_ = global_params.max_numtrees;
        num_landmark_ = global_params.landmark_num;
        max_depth_    = global_params.max_depth;
        overlap_ratio_ = global_params.bagging_overlap;
        
        // resize the trees
        rfs_.resize(num_landmark_);
        for (int i=0;i<num_landmark_;i++){
            rfs_[i].resize(max_numtrees_);
        }
    }
    void Train(const std::vector<cv::Mat_<uchar> >& images,
               const std::vector<cv::Mat_<double> >& ground_truth_shapes,
               const std::vector<cv::Mat_<double> >& current_shapes,
               const std::vector<BoundingBox> & bounding_boxs,
               const cv::Mat_<double>& mean_shape,
               const std::vector<cv::Mat_<double> >& shapes_residual,
               int stages
               );
    void Read(std::ifstream& fin);
    void Write(std::ofstream& fout);
};



#endif /* defined(__myopencv__RandomForest__) */
