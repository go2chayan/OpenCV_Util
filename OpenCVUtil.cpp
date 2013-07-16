#include "OpenCVUtil.h"

std::string getDepth(cv::Mat inMat){
	int dep = inMat.depth();
	std::string retVal;
	switch(dep){
		case CV_8U:
			retVal = "CV_8U";
			break;
		case CV_8S:
			retVal = "CV_8S";
			break;
		case CV_16U:
			retVal = "CV_16U";
			break;
		case CV_16S:
			retVal = "CV_16S";
			break;
		case CV_32S:
			retVal = "CV_32S";
			break;
		case CV_32F:
			retVal = "CV_32F";
			break;
		case CV_64F:
			retVal = "CV_64F";
			break;
	}
	return retVal;
}
bool isInside(cv::Point2f centerofEllipse, cv::Size Axes, float angle,
			cv::Point2f pointToCheck){

		// Distance of foci from center
		float d = std::sqrt(std::pow(Axes.width,2.)-std::pow(Axes.height,2.));
		cv::Point2f foc1 = cv::Point2f(centerofEllipse.x + d*cos(angle*3.141592654/180),
				centerofEllipse.y + d*sin(angle*3.141592654/180));
		cv::Point2f foc2 = cv::Point2f(centerofEllipse.x - d*cos(angle*3.141592654/180),
						centerofEllipse.y - d*sin(angle*3.141592654/180));
		cv::Point2f diff1 = (pointToCheck - foc1);
	    cv::Point2f diff2 = (pointToCheck - foc2);
		float argInside = std::sqrt(diff1.dot(diff1))+std::sqrt(diff2.dot(diff2));
		return (argInside<2*Axes.width);
}
void hist(cv::Mat inImg,std::string inStr = "hist"){
  cv::Mat hist;

  // Establish the number of bins
  int histSize = 256;

  // Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  // Calculate Histogram
  cv::calcHist(&inImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
  
  // Draw the histograms
  int hist_w = 512; int hist_h = 150;
  int bin_w = cvRound( (double) hist_w/histSize );

  cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  
  /// Draw
  for( int i = 1; i < histSize; i++ )
      cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       cv::Scalar( 255, 255, 255), 2, 8, 0);
  /// Display
  imshow(inStr.c_str(), histImage );
}
cv::Mat fft(cv::Mat inImg){
	// Optimal fft size and complex matrix for fft
	cv::Mat padded,complexI;
	int m = cv::getOptimalDFTSize(inImg.rows);int n = cv::getOptimalDFTSize(inImg.cols);
	cv::copyMakeBorder(inImg,padded,0,m-inImg.rows,0,n-inImg.cols,cv::BORDER_CONSTANT,0);
	padded.convertTo(padded,CV_32FC1);
	std::vector<cv::Mat > matArr;matArr.push_back(padded);
	matArr.push_back(cv::Mat::zeros(cv::Size(padded.cols,padded.rows),CV_32F));
	cv::merge(matArr,complexI);

	// perform the FFT and split
	cv::dft(complexI,complexI);

	return complexI;
}
cv::Mat fitIn(cv::Mat inImg, int ScreenWidth, int ScreenHeight){
	int n=0,m=0;cv::Mat outImg;
	float aspRatioIm = float(inImg.cols)/float(inImg.rows);
	float aspRatioScr = float(ScreenWidth)/float(ScreenHeight);
	n = aspRatioScr > aspRatioIm ? aspRatioIm*ScreenHeight : ScreenWidth;
	m = aspRatioScr > aspRatioIm ? ScreenHeight : ScreenWidth/aspRatioIm;
	cv::resize(inImg,outImg,cv::Size(n,m));
	return outImg;
}
cv::Mat multiplyWith1ch(cv::Mat inputNch, cv::Mat input1ch,
	int targetDepth){
	CV_Assert((inputNch.rows==input1ch.rows) &&
		(inputNch.cols==input1ch.cols));
	cv::Mat retVal;std::vector<cv::Mat> temp;
	if(targetDepth>=0)
		input1ch.convertTo(input1ch,CV_MAKETYPE(inputNch.depth(),1));
	else{
		input1ch.convertTo(input1ch,CV_MAKETYPE(targetDepth,1));
		inputNch.convertTo(inputNch,CV_MAKETYPE(targetDepth,
			inputNch.channels()));
	}
	for(int i=0;i<inputNch.channels();i++)
		temp.push_back(input1ch);
	cv::merge(temp,retVal);
	retVal = retVal.mul(inputNch);
	return retVal;
}
void subplot(cv::Mat &whereToDraw, cv::Mat whatToDraw,
		int subPlotRows, int subPlotCols, int subPlotCurrCell){
			subPlotCurrCell--;
	int m = whereToDraw.rows/subPlotRows;
	int n = whereToDraw.cols/subPlotCols;
	int x = n*(subPlotCurrCell % subPlotCols), y = m*((subPlotCurrCell+1) / subPlotCols);
	cv::Mat ResizedIm = fitIn(whatToDraw,m,n);
	cv::Mat ROI = whereToDraw(cv::Range(y,y+ResizedIm.rows),
		cv::Range(x,x+ResizedIm.cols));
	ResizedIm.copyTo(ROI);
}
cv::Mat mMin(cv::Mat inMatrix, int dim, cv::Mat &outMinIdx){
	int m = inMatrix.rows, n = inMatrix.cols;
	assert(!inMatrix.empty()); // input matrix must not be empty

	double minVal=0,maxVal=0;cv::Point minLoc,maxLoc;
	std::vector<double> minVec, maxVec; 
	std::vector<double> minLocVec, maxLocVec;

	cv::Mat retVal;
	if(dim==0){
		for(int i=0;i<n;i++){
			cv::minMaxLoc(inMatrix.col(i),&minVal,&maxVal,&minLoc,&maxLoc);
			minVec.push_back(minVal);
			minLocVec.push_back(minLoc.y);
		}
		retVal = cv::Mat(minVec).t();
		outMinIdx = cv::Mat(minLocVec).t();
	}else if(dim==1){
		for(int i=0;i<m;i++){
			cv::minMaxLoc(inMatrix.row(i),&minVal,&maxVal,&minLoc,&maxLoc);
			minVec.push_back(minVal);
			minLocVec.push_back(minLoc.x);
		}
		retVal = cv::Mat(minVec).clone(); // Interesting ... the function will not
		outMinIdx = cv::Mat(minLocVec).clone(); // return correct value if not cloned
	}
	return retVal;
}
//circular shift one row from up to down
void shiftRows(cv::Mat& mat) {

    cv::Mat temp;
    cv::Mat m;
    int k = (mat.rows-1);
    mat.row(k).copyTo(temp);
    for(; k > 0 ; k-- ) {
        m = mat.row(k);
        mat.row(k-1).copyTo(m);
    }
    m = mat.row(0);
    temp.copyTo(m);
}
//circular shift n rows from up to down if n > 0, -n rows from down to up if n < 0
void shiftRows(cv::Mat& mat,int n) {
    if( n < 0 ) {
        n = -n;
        flip(mat,mat,0);
        for(int k=0; k < n;k++) {
            shiftRows(mat);
        }
        flip(mat,mat,0);
    } else {
        for(int k=0; k < n;k++) {
            shiftRows(mat);
        }
    }
}
//circular shift n columns from left to right if n > 0, -n columns from right to left if n < 0
void shiftCols(cv::Mat& mat, int n) {
    if(n < 0){
        n = -n;
        flip(mat,mat,1);
        transpose(mat,mat);
        shiftRows(mat,n);
        transpose(mat,mat);
        flip(mat,mat,1);
    } else {
        transpose(mat,mat);
        shiftRows(mat,n);
        transpose(mat,mat);
    }
}