//
//  LBFRegressor.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "LBFRegressor.h"
using namespace std;
using namespace cv;
struct feature_node ** LBFRegressor::DeriveBinaryFeat(
                                    const RandomForest& randf,
                                    const vector<Mat_<uchar> >& images,
                                    const vector<Mat_<double> >& current_shapes,
                                    const vector<BoundingBox> & bounding_boxs,
                                    const Mat_<double>& mean_shape){
    // calculate the overall dimension of binary feature, concatenate the
    // features for all landmarks, and the feature of one landmark is sum
    // leafnodes of all random trees;
    int dims_binfeat = 0;
    Mat_<int> ind_bincode(randf.num_landmark_,randf.max_numtrees_);
    for ( int i =0;i < randf.num_landmark_;i++){
        for (int j =0; j< randf.max_numtrees_; j++){
            ind_bincode(i,j)= randf.rfs_[i][j].num_leafnodes_;
            dims_binfeat = dims_binfeat + randf.rfs_[i][j].num_leafnodes_;
        }
    }
    
    // initilaize the memory for binfeatures
    struct feature_node **binfeatures;
    binfeatures = new struct feature_node* [images.size()];
    for (int i=0;i<images.size();i++){
         binfeatures[i] = new struct feature_node[randf.max_numtrees_*randf.num_landmark_+1];
        // binfeatures[i] = new struct feature_node[dims_binfeat+1];
    }
    
    int bincode;
    int ind;
    int leafnode_per_tree = pow(2,(randf.max_depth_-1));
    Mat_<double> rotation;
    double scale;

    // extract feature for each samples
    for (int i=0;i < images.size();i++){
        SimilarityTransform(ProjectShape(current_shapes[i],bounding_boxs[i]),mean_shape,rotation,scale);
        for (int j =0; j <randf.num_landmark_; j++){
            for(int k = 0; k< randf.max_numtrees_;k++){

                bincode = GetCodefromTree(randf.rfs_[j][k],images[i],current_shapes[i],bounding_boxs[i],rotation,scale);
                ind = j * randf.max_numtrees_ + k;
                binfeatures[i][ind].index = leafnode_per_tree * ind + bincode;
                binfeatures[i][ind].value = 1;
            }
        }
        binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].index = -1;
        binfeatures[i][randf.num_landmark_ * randf.max_numtrees_].value = -1;
    }
    return binfeatures;
}
int  LBFRegressor::GetCodefromTree(const Tree& tree,
                                   const Mat_<uchar>& image,
                                   const Mat_<double>& shape,
                                   const BoundingBox& bounding_box,
                                   const Mat_<double>& rotation,
                                   const double scale){
    int currnode = 0;
    int bincode = 0;
    while(1){
        double x1 = tree.nodes_[currnode].feat[0];
        double y1 = tree.nodes_[currnode].feat[1];
        double x2 = tree.nodes_[currnode].feat[2];
        double y2 = tree.nodes_[currnode].feat[3];
        
        double project_x1 = rotation(0,0) * x1 + rotation(0,1) * y1;
        double project_y1 = rotation(1,0) * x1 + rotation(1,1) * y1;
        project_x1 = scale * project_x1 * bounding_box.width / 2.0;
        project_y1 = scale * project_y1 * bounding_box.height / 2.0;
        int real_x1 = project_x1 + shape(tree.landmarkID_,0);
        int real_y1 = project_y1 + shape(tree.landmarkID_,1);
        real_x1 = max(0.0,min((double)real_x1,image.cols-1.0));
        real_y1 = max(0.0,min((double)real_y1,image.rows-1.0));
        
        double project_x2 = rotation(0,0) * x2 + rotation(0,1) * y2;
        double project_y2 = rotation(1,0) * x2 + rotation(1,1) * y2;
        project_x2 = scale * project_x2 * bounding_box.width / 2.0;
        project_y2 = scale * project_y2 * bounding_box.height / 2.0;
        int real_x2 = project_x2 + shape(tree.landmarkID_,0);
        int real_y2 = project_y2 + shape(tree.landmarkID_,1);
        real_x2 = max(0.0,min((double)real_x2,image.cols-1.0));
        real_y2 = max(0.0,min((double)real_y2,image.rows-1.0));
        double pdf = ((int)(image(real_y1,real_x1))-(int)(image(real_y2,real_x2)));
        if (pdf < tree.nodes_[currnode].thresh){
            currnode =tree.nodes_[currnode].cnodes[0];
        }
        else{
            currnode =tree.nodes_[currnode].cnodes[1];
        }
        if (tree.nodes_[currnode].isleafnode == 1){
            bincode = 1;
            for (vector<int>::const_iterator citer=tree.id_leafnodes_.begin();citer!=tree.id_leafnodes_.end();citer++){
                if (*citer == currnode){
                    return bincode;
                }
                bincode++;
            }
            return bincode;
        }
    }
    return bincode;
};

void LBFRegressor::GlobalRegression(struct feature_node **binfeatures,
                                    const vector<Mat_<double> >& shapes_residual,
                                    vector<Mat_<double> >& current_shapes,
                                    const vector<BoundingBox> & bounding_boxs,
                                    const Mat_<double>& mean_shape,
                                    //Mat_<double>& W,
                                    vector<struct model*>& models,
                                    int num_feature,
                                    int num_train_sample,
                                    int stages
                                    ){
    // shapes_residual: n*(l*2)
    // construct the problem(expect y)
    struct problem* prob = new struct problem;
    prob -> l = num_train_sample;
    prob -> n = num_feature;
    prob -> x = binfeatures;
    prob -> bias = -1;
    
    // construct the parameter
    struct parameter* param = new struct parameter;
    param-> solver_type = L2R_L2LOSS_SVR_DUAL;
    param->C = 1.0/num_train_sample;
    param->p = 0;
    
    // initialize the y
    int num_residual = shapes_residual[0].rows*2;
    double** yy = new double*[num_residual];
    
    for (int i=0;i<num_residual;i++){
        yy[i] = new double[num_train_sample];
    }
    for (int i=0;i < num_train_sample;i++){
        for (int j=0;j<num_residual;j++){
            if (j < num_residual/2){
                yy[j][i] = shapes_residual[i](j,0);
            }
            else{
                yy[j][i] = shapes_residual[i](j-num_residual/2,1);
            }
        }
    }
    
    //train
    models.clear();
    models.resize(num_residual);
    for (int i=0;i < num_residual;i++){
        clock_t t1 = clock();
        cout << "Train "<< i <<"th landmark"<<endl;
        prob->y = yy[i];
        check_parameter(prob, param);
        struct model* lbfmodel  = train(prob, param);
        models[i] = lbfmodel;
        double time =double(clock() - t1) / CLOCKS_PER_SEC;
        cout << "linear regression of one landmark cost "<< time <<"s"<<endl;
    }
    // update the current shape and shapes_residual
    double tmp;
    double scale;
    Mat_<double>rotation;
    Mat_<double> deltashape_bar(num_residual/2,2);
    Mat_<double> deltashape_bar1(num_residual/2,2);
    for (int i=0;i<num_train_sample;i++){
        for (int j=0;j<num_residual;j++){
            tmp = predict(models[j],binfeatures[i]);
            if (j < num_residual/2){
                deltashape_bar(j,0) = tmp;
            }
            else{
                deltashape_bar(j-num_residual/2,1) = tmp;
            }
        }
        // transfer or not to be decided
        // now transfer
        SimilarityTransform(ProjectShape(current_shapes[i],bounding_boxs[i]),mean_shape,rotation,scale);
        transpose(rotation,rotation);
        deltashape_bar1 = scale * deltashape_bar * rotation;
        current_shapes[i] = ReProjectShape((ProjectShape(current_shapes[i],bounding_boxs[i])+deltashape_bar1),bounding_boxs[i]);
        
        //updata shapes_residual
       // shapes_residual[i] = shapes_residual[i] - deltashape_bar;
    }
}

void LBFRegressor::GlobalPrediction(struct feature_node** binfeatures,
                      const vector<struct model*>& models,
                      vector<Mat_<double> >& current_shapes,
                      const vector<BoundingBox> & bounding_boxs,
                      int num_train_sample,
                      int stages){
    int num_residual = current_shapes[0].rows*2;
    double tmp;
    double scale;
    Mat_<double>rotation;
    Mat_<double> deltashape_bar(num_residual/2,2);
    Mat_<double> deltashape_bar1(num_residual/2,2);
    for (int i=0;i<num_train_sample;i++){
        for (int j=0;j<num_residual;j++){
            tmp = predict(models[j],binfeatures[i]);
            if (j < num_residual/2){
                deltashape_bar(j,0) = tmp;
            }
            else{
                deltashape_bar(j-num_residual/2,1) = tmp;
            }
        }
        // transfer or not to be decided
        // now transfer
        SimilarityTransform(ProjectShape(current_shapes[i],bounding_boxs[i]),mean_shape_,rotation,scale);
        transpose(rotation,rotation);
        deltashape_bar1 = scale * deltashape_bar * rotation;
        current_shapes[i] = ReProjectShape((ProjectShape(current_shapes[i],bounding_boxs[i])+deltashape_bar1),bounding_boxs[i]);
    }
}

void LBFRegressor::Train(const vector<Mat_<uchar> >& images,
                         const vector<Mat_<double> >& ground_truth_shapes,
                         const vector<BoundingBox> & bounding_boxs){
    
    // data augmentation and multiple initialization
    vector<Mat_<uchar> > augmented_images;
    vector<BoundingBox> augmented_bounding_boxs;
    vector<Mat_<double> > augmented_ground_truth_shapes;
    vector<Mat_<double> > current_shapes;
    
    RNG random_generator(getTickCount());
    for(int i = 0;i < images.size();i++){
        for(int j = 0;j < global_params.initial_num;j++){
            int index = 0;
            do{
                // index = (i+j+1) % (images.size());
                index = random_generator.uniform(0, (int)images.size());
            }while(index == i);
            
            // 1. Select ground truth shapes of other images as initial shapes
            augmented_images.push_back(images[i]);
            augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);
            augmented_bounding_boxs.push_back(bounding_boxs[i]);
            
            // 2. Project current shape to bounding box of ground truth shapes
            Mat_<double> temp = ProjectShape(ground_truth_shapes[index], bounding_boxs[index]);
            temp = ReProjectShape(temp, bounding_boxs[i]);
            current_shapes.push_back(temp);
        }
    }
    
    // get mean shape from training shapes(only origin train images)
    mean_shape_ = GetMeanShape(ground_truth_shapes,bounding_boxs);
    // train random forest
    int num_feature = global_params.landmark_num * global_params.max_numtrees * pow(2,(global_params.max_depth-1));
    int num_train_sample = (int)augmented_images.size();
    for (int stage = 0; stage < global_params.max_numstage; stage++){
        clock_t t = clock();
        GetShapeResidual(augmented_ground_truth_shapes,current_shapes,augmented_bounding_boxs,
                         mean_shape_,shapes_residual_);
        cout << "learn random forest for "<< stage <<"th stage" <<endl;
        RandomForest_[stage].Train(augmented_images,augmented_ground_truth_shapes, current_shapes, augmented_bounding_boxs, mean_shape_, shapes_residual_, stage);
        
        cout << "derive binary codes given learned random forest in current stage"<< endl;
        struct feature_node ** binfeatures ;
        binfeatures = DeriveBinaryFeat(RandomForest_[stage], augmented_images, current_shapes, augmented_bounding_boxs, mean_shape_);
        
        cout << "learn global linear regression given binary feature" << endl;
        GlobalRegression(binfeatures, shapes_residual_, current_shapes, augmented_bounding_boxs, mean_shape_, Models_[stage], num_feature, num_train_sample, stage);
        ReleaseFeatureSpace(binfeatures,(int)augmented_images.size());
        double time = double(clock() - t) / CLOCKS_PER_SEC;
        cout << "the rf of "<< stage<<" stage has been trained, cost "<<time <<" s"<<endl<<endl;
    }
}
void LBFRegressor::ReleaseFeatureSpace(struct feature_node ** binfeatures,
                         int num_train_sample){
    for (int i = 0;i < num_train_sample;i++){
            delete[] binfeatures[i];
    }
    delete[] binfeatures;
}

Mat_<double>  LBFRegressor::Predict(const cv::Mat_<uchar>& image,
                                    const BoundingBox& bounding_box,
                                    int initial_num){
    
    Mat_<double> result = Mat::zeros(global_params.landmark_num,2, CV_64FC1);

    vector<Mat_<uchar> > images;
    vector<Mat_<double> > current_shapes;
    vector<BoundingBox>  bounding_boxs;
    
    images.push_back(image);
    bounding_boxs.push_back(bounding_box);
    Mat_<double> current_shape = ReProjectShape(mean_shape_, bounding_box);
    current_shapes.push_back(current_shape);
    int num_train_sample = (int)images.size();
    for ( int stage = 0; stage < global_params.max_numstage; stage++){
        struct feature_node ** binfeatures ;
        binfeatures = DeriveBinaryFeat(RandomForest_[stage],images,current_shapes,bounding_boxs, mean_shape_);
        GlobalPrediction(binfeatures, Models_[stage], current_shapes,bounding_boxs,num_train_sample,stage);
    }
    return current_shapes[0];
}



void LBFRegressor::Save(string path){
    cout << endl<<"Saving model..." << endl;
    ofstream fout;
    fout.open(path);
    // write the Regressor to file
    WriteGlobalParam(fout);
    WriteRegressor(fout);
    fout.close();
    cout << "End" << endl;

    
}

void LBFRegressor::Load(string path){
    cout << "Loading model..." << endl;
    ifstream fin;
    fin.open(path);
    ReadGlobalParam(fin);
    ReadRegressor(fin);
    fin.close();
    cout << "End"<<endl;
}
void  LBFRegressor::WriteGlobalParam(ofstream& fout){
    fout << global_params.bagging_overlap << endl;
    fout << global_params.max_numtrees << endl;
    fout << global_params.max_depth << endl;
    fout << global_params.max_numthreshs << endl;
    fout << global_params.landmark_num << endl;
    fout << global_params.initial_num << endl;
    fout << global_params.max_numstage << endl;
    
    for (int i = 0; i< global_params.max_numstage; i++){
        fout << global_params.max_radio_radius[i] << " ";
    }
    fout << endl;
    
    for (int i = 0; i < global_params.max_numstage; i++){
        fout << global_params.max_numfeats[i] << " ";
    }
    fout << endl;
}
void  LBFRegressor::WriteRegressor(ofstream& fout){
    for(int i = 0;i < global_params.landmark_num;i++){
        fout << mean_shape_(i,0)<<" "<< mean_shape_(i,1)<<" ";
    }
    fout<<endl;
    
    for (int i=0; i < global_params.max_numstage; i++ ){
        RandomForest_[i].Write(fout);
        fout << Models_[i].size()<< endl;
        for (int j=0; j<Models_[i].size();j++){
            stringstream name;
            name <<"/Users/lequan/workspace/xcode/myopencv/data/"<< i << "_" <<j<<".model";
            save_model(name.str().c_str(), Models_[i][j]);
        }
    }
}
void  LBFRegressor::ReadGlobalParam(ifstream& fin){
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
}

void LBFRegressor::ReadRegressor(ifstream& fin){
    mean_shape_ = Mat::zeros(global_params.landmark_num,2,CV_64FC1);
    for(int i = 0;i < global_params.landmark_num;i++){
        fin >> mean_shape_(i,0) >> mean_shape_(i,1);
    }
    for (int i=0; i < global_params.max_numstage; i++ ){
        RandomForest_[i].Read(fin);
        int num =0;
        fin >> num;
        Models_[i].resize(num);
        for (int j=0;j<num;j++){
            stringstream name;
            name <<"/Users/lequan/workspace/xcode/myopencv/data/"<< i << "_" <<j<<".model";
            Models_[i][j] = load_model(name.str().c_str());
        }
    }
}

