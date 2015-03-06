cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

set(source ./FaceAlignment.h ./liblinear/tron.cpp ./liblinear/tron.h ./liblinear/linear.h ./liblinear/linear.cpp ./Demo.cpp ./TrainDemo.cpp ./TestDemo.cpp
./Utils.cpp ./Tree.h ./Tree.cpp ./RandomForest.cpp ./RandomForest.h ./LBFRegressor.h ./LBFRegressor.cpp)
add_executable(Demo.out ${source})
target_link_libraries(Demo.out ${OpenCV_LIBS} blas.a) 



