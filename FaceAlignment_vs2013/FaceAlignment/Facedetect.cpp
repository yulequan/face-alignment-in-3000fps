//
//  Facedetect.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//
#include "LBFRegressor.h"
using namespace std;
using namespace cv;
int save_count=0;
void detectAndDraw(Mat& img,
                   CascadeClassifier& nestedCascade, LBFRegressor& regressor,
                   double scale, bool tryflip );

int FaceDetectionAndAlignment(const char* inputname){
       extern string cascadeName;
    string inputName;
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    bool tryflip = false;
    double scale  = 1.3;
    CascadeClassifier cascade;
    
    if (inputname!=NULL){
        inputName.assign(inputname);
    }
    
    // name is empty or a number
    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') ){
        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture){
            cout << "Capture from CAM " <<  c << " didn't work" << endl;
            return -1;
        }
    }
    // name is not empty
    else if( inputName.size() ){
        if (inputName.find(".jpg")!=string::npos||inputName.find(".png")!=string::npos
            ||inputName.find(".bmp")!=string::npos){
            image = imread( inputName, 1 );
            if (image.empty()){
                cout << "Read Image fail" << endl;
                return -1;
            }
        }
        else if(inputName.find(".mp4")!=string::npos||inputName.find(".avi")!=string::npos
                ||inputName.find(".wmv")!=string::npos){
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture) cout << "Capture from AVI didn't work" << endl;
            return -1;
        }
    }
    // -- 0. Load LBF model
    LBFRegressor regressor;
    regressor.Load(modelPath+"LBF.model");
    
    // -- 1. Load the cascades
    if( !cascade.load( cascadeName ) ){
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    // cvNamedWindow( "result", 1 );
    // -- 2. Read the video stream
    if( capture ){
        cout << "In capture ..." << endl;
        for(;;){
            IplImage* iplImg = cvQueryFrame( capture );
            frame = cvarrToMat(iplImg);
            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );

            detectAndDraw( frameCopy, cascade,regressor, scale, tryflip );

            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }

        waitKey(0);

_cleanup_:
        cvReleaseCapture( &capture );
    }
    else{
       
        if( !image.empty() ){
            cout << "In image read" << endl;
            detectAndDraw( image, cascade,regressor,  scale, tryflip );
            waitKey(0);
        }
        else if( !inputName.empty() ){
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            cout << "In image set model" << endl;
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f ){
                char buf[1000+1];
                while( fgets( buf, 1000, f ) ){
                    int len = (int)strlen(buf), c;
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );    
                    if( !image.empty() ){
                        detectAndDraw(image, cascade,regressor,scale, tryflip );
                        c = waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;
                    }
                    else{
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    cvDestroyWindow("result");

    return 0;
}


void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    LBFRegressor& regressor,
                    double scale, bool tryflip ){
    int i = 0;
    double t = 0;
    vector<Rect> faces,faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    
    // --Detection
    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );
    if( tryflip ){
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    
    // --Alignment
    t =(double)cvGetTickCount();
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ ){
        Point center;
        Scalar color = colors[i%8];
        BoundingBox boundingbox;
        
        boundingbox.start_x = r->x*scale;
        boundingbox.start_y = r->y*scale;
        boundingbox.width   = (r->width-1)*scale;
        boundingbox.height  = (r->height-1)*scale;
        boundingbox.centroid_x = boundingbox.start_x + boundingbox.width/2.0;
        boundingbox.centroid_y = boundingbox.start_y + boundingbox.height/2.0;
        
        t =(double)cvGetTickCount();
        Mat_<double> current_shape = regressor.Predict(gray,boundingbox,1);
        t = (double)cvGetTickCount() - t;
        printf( "alignment time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
//        // draw bounding box
//        rectangle(img, cvPoint(boundingbox.start_x,boundingbox.start_y),
//                  cvPoint(boundingbox.start_x+boundingbox.width,boundingbox.start_y+boundingbox.height),Scalar(0,255,0), 1, 8, 0);
        // draw result :: red
        for(int i = 0;i < global_params.landmark_num;i++){
             circle(img,Point2d(current_shape(i,0),current_shape(i,1)),3,Scalar(255,255,255),-1,8,0);
        }
    }
    cv::imshow( "result", img );
    char a = waitKey(0);
    if(a=='s'){
        save_count++;
        imwrite(to_string(save_count)+".jpg", img);
    }
}
