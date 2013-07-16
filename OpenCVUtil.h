#ifndef __OPENCV_UTIL_H__
#define __OPENCV_UTIL_H__

#include <opencv2\opencv.hpp>
#include <string>

// Returns the depth of a matrix as string (e.g. "CV_8U", "CV_16U" etc.)
std::string getDepth(cv::Mat);

// Returns if a point is within an ellipse. The first three
// arguments describe the ellipse (center, axes and angle 
// respectively). The last argument is the point to be tested
bool isInside(cv::Point2f, cv::Size, float,cv::Point2f);

// MATLAB style histogram plotter
// First argument represents the matrix, second argument
// represents name of the window
void hist(cv::Mat,std::string);

// Multiply an N channel matrix with a one channel matrix.
// If TargetDepth is Negative then depth of inputNch is used.
// Otherwise, both the inputs are converted to targetDepth
// and the output depth also becomes targetDepth.
cv::Mat multiplyWith1ch(cv::Mat inputNch, cv::Mat input1ch,
	int targetDepth = -1);

// MATLAB style FFT calculator
cv::Mat fft(cv::Mat);

// TODO: Check and Debug
// Fits an image within a specified screenwidth and screenheight
// Args: inImg, ScreenWidth, ScreenHeight
cv::Mat fitIn(cv::Mat, int, int);

// TODO: Check and Debug
// MATLAB style subplot. The arguments are wheretodraw,
//whattodraw, subplot rows, subplot columns and subplot current cell
void subplot(cv::Mat &, cv::Mat,
		int, int, int);

//circular shift n rows from up to down if n > 0, -n rows from down to up if n < 0
void shiftRows(cv::Mat& mat,int n) ;

//circular shift n columns from left to right if n > 0, -n columns from right to left if n < 0
void shiftCols(cv::Mat& mat, int n);
		
//circular shift one row from up to down
void shiftRows(cv::Mat& mat);		

// TODO: Debug it
// MATLAB style matrix min
cv::Mat mMin(cv::Mat inMatrix, int dim, cv::Mat &outMinIdx);
#endif