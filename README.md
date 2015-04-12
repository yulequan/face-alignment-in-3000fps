# Face-alignment-via-3000fps

This project is a C++ reimplementation of face alignment in 3000fps in the CVPR 2014 paper:
[ Face Alignment at 3000 FPS via Regressing Local Binary Features. ]().

### Usage

1. Download datasets and get Path_Images.txt as [jwyang/face-alignment](https://github.com/jwyang/face-alignment). 

2. Download [liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) and compiler it to get blas lib (liblinear/blas/blas.a). You need to use your own blas.a to replace `blas.a` in the folder `build`.

3. To compiler the program: go to folder `build` and 
   
   cmake .
  
   make

4. To train a new model: set global parameters in `LBF.cpp` and dataset in `TrainDemo.cpp`. Use `"LBF.out TrainModel"`.
5. To test a model on dataset: set test dataset in `TestDemo.cpp`. Use `""LBF.out TestModel"`.

###Model
I have trained a model on AFW, HELEN,LFPW dataset. You can download it from [here](http://pan.baidu.com/s/1326PS). 


### FAQ
* How to get the bounding box of image ?

	I use the face detector in OpenCV to get the bounding box.You can use any detector to get the bounding box but you you must provide a bounding box of similar measure with the training data. 

### Contact 
If you have any question, you can create an `issue` on GitHub.

### Reference Project
* [soundsilence/FaceAlignment](https://github.com/soundsilence/FaceAlignment)

* [jwyang/face-alignment](https://github.com/jwyang/face-alignment)  




 
  
