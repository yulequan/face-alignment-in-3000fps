//
//  Tree.cpp
//  myopencv
//
//  Created by lequan on 1/23/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "Tree.h"
using namespace std;
using namespace cv;

inline double calculate_var(const vector<double>& v_1 ){
    if (v_1.size() == 0)
        return 0;
    Mat_<double> v1(v_1);
    double mean_1 = mean(v1)[0];
    double mean_2 = mean(v1.mul(v1))[0];
    return mean_2 - mean_1*mean_1;
    
}
inline double calculate_var(const Mat_<double>& v1){
    double mean_1 = mean(v1)[0];
    double mean_2 = mean(v1.mul(v1))[0];
    return mean_2 - mean_1*mean_1;
    
}

void Tree::Train(const vector<Mat_<uchar> >& images,
                 const vector<Mat_<double> >& ground_truth_shapes,
                 const vector<Mat_<double> >& current_shapes,
                 const vector<BoundingBox> & bounding_boxs,
                 const Mat_<double>& mean_shape,
                 const vector<Mat_<double> >& regression_targets,
                 const vector<int> index,
                 int stages,
                 int landmarkID
                 ){
    // set the parameter
    landmarkID_ = landmarkID; // start from 0
    max_numfeats_ = global_params.max_numfeats[stages];
    max_radio_radius_ = global_params.max_radio_radius[stages];
    num_nodes_ = 1;
    num_leafnodes_ = 1;
    
    // index: indicate the training samples id in images
    int num_nodes_iter;
    int num_split;
    Mat_<double> shapes_residual((int)index.size(),2);
    // calculate regression targets: the difference between ground truth shapes and current shapes
    for(int i = 0;i < index.size();i++){
        shapes_residual(i,0) = regression_targets[index[i]](landmarkID_,0);
        shapes_residual(i,1) = regression_targets[index[i]](landmarkID_,1);
    }
    // initialize the root
    nodes_[0].issplit = false;
    nodes_[0].pnode = 0;
    nodes_[0].depth = 1;
    nodes_[0].cnodes[0] = 0;
    nodes_[0].cnodes[1] = 0;
    nodes_[0].isleafnode = 1;
    nodes_[0].thresh = 0;
    for (int i=0; i < 4;i++){
        nodes_[0].feat[i] = 1;
    }
    nodes_[0].ind_samples = index;
    

    bool stop = 0;
    int num_nodes = 1;
    int num_leafnodes = 1;
    double thresh;
    double feat[4];
    bool isvaild;
    vector<int> lcind,rcind;
    lcind.reserve(index.size());
    rcind.reserve(index.size());
    while(!stop){
        num_nodes_iter = num_nodes_;
        num_split = 0;
        for (int n = 0; n < num_nodes_iter; n++ ){
            if (!nodes_[n].issplit){
                if (nodes_[n].depth == max_depth_) {
                    if (nodes_[n].depth == 1){
                        nodes_[n].depth = 1;
                    }
                    nodes_[n].issplit = true;
                }
                else {
                    // separate the samples into left and right path
                    Splitnode(images,ground_truth_shapes,current_shapes,bounding_boxs,mean_shape,shapes_residual,
                              nodes_[n].ind_samples,thresh, feat, isvaild,lcind,rcind);
                    // set the threshold and featture for current node
                    nodes_[n].feat[0] = feat[0];
                    nodes_[n].feat[1] = feat[1];
                    nodes_[n].feat[2] = feat[2];
                    nodes_[n].feat[3] = feat[3];
                    nodes_[n].thresh  = thresh;
                    nodes_[n].issplit = true;
                    nodes_[n].isleafnode = false;
                    nodes_[n].cnodes[0] = num_nodes ;
                    nodes_[n].cnodes[1] = num_nodes +1;
                    
                    //add left and right child nodes into the random tree
                    nodes_[num_nodes].ind_samples = lcind;
                    nodes_[num_nodes].issplit = false;
                    nodes_[num_nodes].pnode = n;
                    nodes_[num_nodes].depth = nodes_[n].depth + 1;
                    nodes_[num_nodes].cnodes[0] = 0;
                    nodes_[num_nodes].cnodes[1] = 0;
                    nodes_[num_nodes].isleafnode = true;

                    nodes_[num_nodes +1].ind_samples = rcind;
                    nodes_[num_nodes +1].issplit = false;
                    nodes_[num_nodes +1].pnode = n;
                    nodes_[num_nodes +1].depth = nodes_[n].depth + 1;
                    nodes_[num_nodes +1].cnodes[0] = 0;
                    nodes_[num_nodes +1].cnodes[1] = 0;
                    nodes_[num_nodes +1].isleafnode = true;
                    
                    num_split++;
                    num_leafnodes++;
                    num_nodes +=2;
                }
            }
            
            
        }
        if (num_split == 0){
            stop = 1;
        }
        else{
            num_nodes_ = num_nodes;
            num_leafnodes_ = num_leafnodes;
        }
    }
    
    id_leafnodes_.clear();
    for (int i=0;i < num_nodes_;i++){
        if (nodes_[i].isleafnode == 1){
            id_leafnodes_.push_back(i);
        }
    }
}
void Tree::Splitnode(const vector<Mat_<uchar> >& images,
                     const vector<Mat_<double> >& ground_truth_shapes,
                     const vector<Mat_<double> >& current_shapes,
                     const vector<BoundingBox> & bounding_box,
                     const Mat_<double>& mean_shape,
                     const Mat_<double>& shapes_residual,
                     const vector<int> &ind_samples,
                     // output
                     double& thresh,
                     double* feat,
                     bool& isvaild,
                     vector<int>& lcind,
                     vector<int>& rcind
                     ){
    if (ind_samples.size() == 0){
        thresh = 0;
        feat = new double[4];
        feat[0] = 0;
        feat[1] = 0;
        feat[2] = 0;
        feat[3] = 0;
        lcind.clear();
        rcind.clear();
        isvaild = 1;
        return;
    }
    // get candidate pixel locations
    RNG random_generator(getTickCount());
    Mat_<double> candidate_pixel_locations(max_numfeats_,4);
    for(unsigned int i = 0;i < max_numfeats_;i++){
        double x1 = random_generator.uniform(-1.0,1.0);
        double y1 = random_generator.uniform(-1.0,1.0);
        double x2 = random_generator.uniform(-1.0,1.0);
        double y2 = random_generator.uniform(-1.0,1.0);
        if((x1*x1 + y1*y1 > 1.0)||(x2*x2 + y2*y2 > 1.0)){
            i--;
            continue;
        }
       // cout << x1 << " "<<y1 <<" "<< x2<<" "<< y2<<endl;
        candidate_pixel_locations(i,0) = x1 * max_radio_radius_;
        candidate_pixel_locations(i,1) = y1 * max_radio_radius_;
        candidate_pixel_locations(i,2) = x2 * max_radio_radius_;
        candidate_pixel_locations(i,3) = y2 * max_radio_radius_;
    }
    // get pixel difference feature
    Mat_<int> densities(max_numfeats_,(int)ind_samples.size());
    for (int i = 0;i < ind_samples.size();i++){
        Mat_<double> rotation;
        double scale;
        Mat_<double> temp = ProjectShape(current_shapes[ind_samples[i]],bounding_box[ind_samples[i]]);
        SimilarityTransform(temp,mean_shape,rotation,scale);
        // whether transpose or not ????
        for(int j = 0;j < max_numfeats_;j++){
            double project_x1 = rotation(0,0) * candidate_pixel_locations(j,0) + rotation(0,1) * candidate_pixel_locations(j,1);
            double project_y1 = rotation(1,0) * candidate_pixel_locations(j,0) + rotation(1,1) * candidate_pixel_locations(j,1);
            project_x1 = scale * project_x1 * bounding_box[ind_samples[i]].width / 2.0;
            project_y1 = scale * project_y1 * bounding_box[ind_samples[i]].height / 2.0;
            int real_x1 = project_x1 + current_shapes[ind_samples[i]](landmarkID_,0);
            int real_y1 = project_y1 + current_shapes[ind_samples[i]](landmarkID_,1);
            real_x1 = max(0.0,min((double)real_x1,images[ind_samples[i]].cols-1.0));
            real_y1 = max(0.0,min((double)real_y1,images[ind_samples[i]].rows-1.0));
            
            double project_x2 = rotation(0,0) * candidate_pixel_locations(j,2) + rotation(0,1) * candidate_pixel_locations(j,3);
            double project_y2 = rotation(1,0) * candidate_pixel_locations(j,2) + rotation(1,1) * candidate_pixel_locations(j,3);
            project_x2 = scale * project_x2 * bounding_box[ind_samples[i]].width / 2.0;
            project_y2 = scale * project_y2 * bounding_box[ind_samples[i]].height / 2.0;
            int real_x2 = project_x2 + current_shapes[ind_samples[i]](landmarkID_,0);
            int real_y2 = project_y2 + current_shapes[ind_samples[i]](landmarkID_,1);
            real_x2 = max(0.0,min((double)real_x2,images[ind_samples[i]].cols-1.0));
            real_y2 = max(0.0,min((double)real_y2,images[ind_samples[i]].rows-1.0));
            
            densities(j,i) = ((int)(images[ind_samples[i]](real_y1,real_x1))-(int)(images[ind_samples[i]](real_y2,real_x2)));
        }
    }
    // pick the feature
    Mat_<int> densities_sorted = densities.clone();
    cv::sort(densities, densities_sorted, CV_SORT_ASCENDING);
    vector<double> lc1,lc2;
    vector<double> rc1,rc2;
    lc1.reserve(ind_samples.size());
    rc1.reserve(ind_samples.size());
    lc2.reserve(ind_samples.size());
    rc2.reserve(ind_samples.size());
//    double E_x_2 = mean(shapes_residual.col(0).mul(shapes_residual.col(0)))[0];
//    double E_x = mean(shapes_residual.col(0))[0];
//    double E_y_2 = mean(shapes_residual.col(1).mul(shapes_residual.col(1)))[0];
//    double E_y = mean(shapes_residual.col(1))[0];
//    double var_overall = ind_samples.size()*((E_x_2 - E_x*E_x) + (E_y_2 - E_y*E_y));
    double var_overall =(calculate_var(shapes_residual.col(0))+calculate_var(shapes_residual.col(1))) * ind_samples.size();
    double max_var_reductions = 0;
    double threshold = 0;
    double var_lc = 0;
    double var_rc = 0;
    double var_reduce = 0;
    double max_id = 0;
    for (int i = 0;i <max_numfeats_;i++){
        lc1.clear();
        lc2.clear();
        rc1.clear();
        rc2.clear();
        int ind =(int)(ind_samples.size() * random_generator.uniform(0.05,0.95));
        threshold = densities_sorted(i,ind);
        for (int j=0;j < ind_samples.size();j++){
            if (densities(i,j) < threshold){
                lc1.push_back(shapes_residual(j,0));
                lc2.push_back(shapes_residual(j,1));
            }
            else{
                rc1.push_back(shapes_residual(j,0));
                rc2.push_back(shapes_residual(j,1));            }
        }
        var_lc = (calculate_var(lc1)+calculate_var(lc2)) * lc1.size();
        var_rc = (calculate_var(rc1)+calculate_var(rc2)) * rc1.size();
        var_reduce = var_overall - var_lc - var_rc;
//       cout << var_reduce<<endl;
        if (var_reduce > max_var_reductions){
            max_var_reductions = var_reduce;
            thresh = threshold;
            max_id = i;
        }
    }
    
    isvaild = 1;
    feat[0] =candidate_pixel_locations(max_id,0)/max_radio_radius_;
    feat[1] =candidate_pixel_locations(max_id,1)/max_radio_radius_;
    feat[2] =candidate_pixel_locations(max_id,2)/max_radio_radius_;
    feat[3] =candidate_pixel_locations(max_id,3)/max_radio_radius_;
//    cout << max_id<< " "<<max_var_reductions <<endl;
//    cout << feat[0] << " "<<feat[1] <<" "<< feat[2]<<" "<< feat[3]<<endl;
    lcind.clear();
    rcind.clear();
    for (int j=0;j < ind_samples.size();j++){
        if (densities(max_id,j) < thresh){
            lcind.push_back(ind_samples[j]);
        }
        else{
            rcind.push_back(ind_samples[j]);
        }
    }
}

void Tree::Write(std:: ofstream& fout){
    fout << landmarkID_<<endl;
    fout << max_depth_<<endl;
    fout << max_numnodes_<<endl;
    fout << num_leafnodes_<<endl;
    fout << num_nodes_<<endl;
    fout << max_numfeats_<<endl;
    fout << max_radio_radius_<<endl;
   // fout << overlap_ration_ << endl;
    fout << 0.4 << endl;
    
    fout << id_leafnodes_.size()<<endl;
    for (int i=0;i<id_leafnodes_.size();i++){
        fout << id_leafnodes_[i]<< " ";
    }
    fout <<endl;
    
    for (int i=0; i <max_numnodes_;i++){
        nodes_[i].Write(fout);
    }
}
void Tree::Read(std::ifstream& fin){
    fin >> landmarkID_;
    fin >> max_depth_;
    fin >> max_numnodes_;
    fin >> num_leafnodes_;
    fin >> num_nodes_;
    fin >> max_numfeats_;
    fin >> max_radio_radius_;
    fin >> overlap_ration_;
    int num ;
    fin >> num;
    id_leafnodes_.resize(num);
    for (int i=0;i<num;i++){
        fin >> id_leafnodes_[i];
    }
    
    for (int i=0; i <max_numnodes_;i++){
        nodes_[i].Read(fin);
    }
}


