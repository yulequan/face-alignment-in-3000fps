//
//  Tree.h
//  myopencv
//
//  Created by lequan on 1/23/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#ifndef __myopencv__Tree__
#define __myopencv__Tree__

#include "LBF.h"

class Node {
public:
    //data
    bool issplit;
    int pnode;
    int depth;
    int cnodes[2];
    bool isleafnode;
    double thresh;
    double feat[4];
    std::vector<int> ind_samples;
   
    //Constructors
    Node(){
        ind_samples.clear();
        issplit = 0;
        pnode = 0;
        depth = 0;
        cnodes[0] = 0;
        cnodes[1] = 0;
        isleafnode = 0;
        thresh = 0;
        feat[0] = 0;
        feat[1] = 0;
        feat[2] = 0;
        feat[3] = 0;
    }
    void Write(std::ofstream& fout){
        fout << issplit<<" "<< pnode <<" "<<depth<<" " << cnodes[0]<<" "<<cnodes[1]<<" "<<isleafnode<<" "
        << thresh<<" "<<feat[0]<<" "<<feat[1]<<" "<<feat[2]<<" "<<feat[3]<<std::endl;
    }
    void Read(std::ifstream& fin){
        fin >> issplit >> pnode >> depth >> cnodes[0] >> cnodes[1] >> isleafnode
        >> thresh >> feat[0] >> feat[1] >> feat[2] >> feat[3];
    }
};

class Tree{
public:
    
    // id of the landmark
    int landmarkID_;
    // depth of the tree:
    int max_depth_;
    // number of maximum nodes:
    int max_numnodes_;
    //number of leaf nodes and nodes
    int num_leafnodes_;
    int num_nodes_;
    
    // sample pixel featurs' number, use when training RF
    int max_numfeats_;
    double max_radio_radius_;
    double overlap_ration_;
   
    // leafnodes id
    std::vector<int> id_leafnodes_;
    // tree nodes
    std::vector<Node> nodes_;
    
    
    Tree(){
        overlap_ration_ = global_params.bagging_overlap;
        max_depth_ = global_params.max_depth;
        max_numnodes_ = pow(2, max_depth_)-1;
        nodes_.resize(max_numnodes_);
    }
    void Train(const std::vector<cv::Mat_<uchar> >& images,
               const std::vector<cv::Mat_<double> >& ground_truth_shapes,
               const std::vector<cv::Mat_<double> >& current_shapes,
               const std::vector<BoundingBox> & bounding_boxs,
               const cv::Mat_<double>& mean_shape,
               const std::vector<cv::Mat_<double> >& regression_targets,
               const std::vector<int> index,
               int stages,
               int landmarkID
               );
    
    //Splite the node
    void Splitnode(const std::vector<cv::Mat_<uchar> >& images,
                   const std::vector<cv::Mat_<double> >& ground_truth_shapes,
                   const std::vector<cv::Mat_<double> >& current_shapes,
                   const std::vector<BoundingBox> & bounding_box,
                   const cv::Mat_<double>& mean_shape,
                   const cv::Mat_<double>& shapes_residual,
                   const std::vector<int> &ind_samples,
                   // output
                   double& thresh,
                   double* feat,
                   bool& isvaild,
                   std::vector<int>& lcind,
                   std::vector<int>& rcind
                   );
    
    //Predict
    void Predict();
    
    // Read/ write
    void Read(std::ifstream& fin);
    void Write(std:: ofstream& fout);
    
};





#endif /* defined(__myopencv__Tree__) */
