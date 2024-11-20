#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0/127.5;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(127.5, 127.5, 127.5);

const std::string configFile = "../models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
const std::string modelFile = "../models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
std::vector<std::string> classes;

const std::string classFile = "../models/coco_class_labels.txt";

cv::Mat detect_objects(cv::dnn::Net net, cv::Mat frame){
    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight),
                                               meanVal, true, false);
    net.setInput(inputBlob);
    cv::Mat detection = net.forward("detection_out");
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    return detectionMat;
}

void display_text(cv::Mat& img, std::string text, int x, int y){
    // Get text size
    int baseLine;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseLine);
    // Use text size to create a black rectangle
    rectangle(img, Point(x,y-textSize.height-baseLine), Point(x+textSize.width,y+baseLine),
             Scalar(0,0,0),-1);
    // Display text inside the rectangle
    putText(img, text, Point(x,y-5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 1, LINE_AA);
}

void display_objects(Mat& img, Mat objects, float threshold = 0.25){
    // For every detected object
    for (int i = 0; i < objects.rows; i++){
        int classId = objects.at<float>(i, 1);
        float score = objects.at<float>(i, 2);

        // Recover original cordinates from normalized coordinates
        int x = static_cast<int>(objects.at<float>(i, 3) * img.cols);
        int y = static_cast<int>(objects.at<float>(i, 4) * img.rows);
        int w = static_cast<int>(objects.at<float>(i, 5) * img.cols - x);
        int h = static_cast<int>(objects.at<float>(i, 6) * img.rows - y);

        // Check if the detection is of good quality
        if (score > threshold){
            display_text(img, classes[classId].c_str(), x, y);
            rectangle(img, Point(x,y), Point(x+w, y+h), Scalar(255,255,255), 2);
        }
    }
}

int main(){
cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelFile, configFile);
ifstream ifs(classFile.c_str());
string line;
while (getline(ifs, line))
{
    classes.push_back(line);
}

Mat img, objects;

img = imread(DATA_PATH + "images/street.jpg");
objects = detect_objects(net, img);
display_objects(img, objects);
imshow("Image",img);
waitKey(0);
img = imread(DATA_PATH + "images/baseball.jpg");
objects = detect_objects(net, img);
display_objects(img, objects);
imshow("Image",img);
waitKey(0);
img = imread(DATA_PATH + "images/soccer.jpg");
objects = detect_objects(net, img);
display_objects(img, objects);
imshow("Image",img);
waitKey(0);
destroyAllWindows();
return 0;
}
