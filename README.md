# Face-alignment-in-3000fps

This project is a C++ reimplementation of face alignment in 3000fps in the CVPR 2014 paper:
[ Face Alignment at 3000 FPS via Regressing Local Binary Features. ]().

### Update openMP support !!!
 I modify my code to support openMP. You can use it in GCC(Linux) or in VS (Windows).
 
 If you use it in Linux, you should comment or uncomment `FIND_PACKAGE( OpenMP REQUIRED)`  in CmakeLists.txt.
 
 If you use it in Windows, you can directly use it. 
 
### VS project 
 I add a VS project. 
 
### Usage

1. Download datasets and get Path_Images.txt as [jwyang/face-alignment](https://github.com/jwyang/face-alignment). 

2. To compiler the program: go to folder `build` and 
   
   cmake .
  
   make

3. To train a new model: set global parameters, model path, train database name in `LBF.cpp`. Use `"LBF.out TrainModel"`.


4. To test a model on dataset: set model path, test dataset name in `LBF.cpp`. Use `"LBF.out TestModel"`.

###Model
I have trained a model on AFW, HELEN,LFPW dataset. You can download it from [here](https://pan.baidu.com/s/1jHNXa8A
) or [google drive](https://drive.google.com/folderview?id=0ByeDfKY7bL0_fmg2RWN2V0xtQ19veW1wdFVJRjBaRHBuUmJNNERHc0YyQ2lLVXJodDZTbk0&usp=sharing). 


### FAQ
* How to get the bounding box of image ?

	I use the face detector in OpenCV to get the bounding box.You can use any detector to get the bounding box but you must provide a bounding box of similar measure with the training data. 

* How about the liblinear?

	I add the liblinear source code as the project code. So you can directly compiler this project and don't need to consider to compiler this library.

### Contact 
If you have any question, you can create an `issue` on GitHub.
Or you can email ylqzd2011@gmail.com

### Reference Project
* [soundsilence/FaceAlignment](https://github.com/soundsilence/FaceAlignment)

* [jwyang/face-alignment](https://github.com/jwyang/face-alignment)  




 
  
