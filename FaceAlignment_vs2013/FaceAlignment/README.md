# Face-alignment-in-3000fps

This project is a C++ reimplementation of face alignment in 3000fps in the CVPR 2014 paper:
[ Face Alignment at 3000 FPS via Regressing Local Binary Features. ]().
### VS project 
 I have added a VS 2013 project. You can try it!
### Usage

1. Download datasets and get Path_Images.txt as [jwyang/face-alignment](https://github.com/jwyang/face-alignment). 

2. To compiler the program: go to folder `build` and 
   
   cmake .
  
   make

3. To train a new model: set global parameters in `LBF.cpp` and dataset in `TrainDemo.cpp`. Use `"LBF.out TrainModel"`.
4. To test a model on dataset: set test dataset in `TestDemo.cpp`. Use `"LBF.out TestModel"`.

###Model
I have trained a model on AFW, HELEN,LFPW dataset. You can download it from [here](http://pan.baidu.com/s/1326PS) or [google drive](https://drive.google.com/folderview?id=0ByeDfKY7bL0_fmg2RWN2V0xtQ19veW1wdFVJRjBaRHBuUmJNNERHc0YyQ2lLVXJodDZTbk0&usp=sharing). 


### FAQ
* How to get the bounding box of image ?

	I use the face detector in OpenCV to get the bounding box.You can use any detector to get the bounding box but you must provide a bounding box of similar measure with the training data. 

### Contact 
If you have any question, you can create an `issue` on GitHub.
Or you can email yulequan@zju.edu.cn

### Reference Project
* [soundsilence/FaceAlignment](https://github.com/soundsilence/FaceAlignment)

* [jwyang/face-alignment](https://github.com/jwyang/face-alignment)  




 
  
