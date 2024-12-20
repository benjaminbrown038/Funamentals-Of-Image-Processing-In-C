#include <iostream>
#include "dataPath.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(){
string filename = DATA_PATH + "images/gaussian-noise.png";
Mat image = imread(filename);

Mat dst1, dst2;

// Box filter- kernel size 3
blur( image, dst1, Size( 3, 3 ), Point(-1,-1) );

//Box filter kernel size 7
blur(image,dst2,Size(7,7),Point(-1,-1));

imshow("Original Image",image);
waitKey(0);
imshow("Box Blur Result 1 : KernelSize = 3",dst1);
waitKey(0);
imshow("Box Blur Result 2 : KernelSize = 7",dst2);
waitKey(0);

return 0;
}
