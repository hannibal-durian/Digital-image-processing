//============================================================================
// Name        : Dip1.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip1.h"

#include <stdexcept>
namespace dip1 {


/**
 * @brief function that performs some kind of (simple) image processing
 * @param img input image
 * @returns output image
 */
cv::Mat doSomethingThatMyTutorIsGonnaLike(const cv::Mat& img) {
	int r = img.rows;
	int c = img.cols;
	cv::Mat img2 = img.clone();
	
	for (int i = 1; i < r - 1; i++)
	{
		for (int j = 1; j < c - 1; j++)
		{
			img2.at<cv::Vec3b>(i, j)[0] = (img.at<cv::Vec3b>(i, j)[0], img.at<cv::Vec3b>(i + 1, j + 1)[0],
				img.at<cv::Vec3b>(i + 1, j)[0], img.at<cv::Vec3b>(i, j + 1)[0], img.at<cv::Vec3b>(i + 1, j - 1)[0],
				img.at<cv::Vec3b>(i - 1, j + 1)[0], img.at<cv::Vec3b>(i - 1, j)[0], img.at<cv::Vec3b>(i, j - 1)[0],
				img.at<cv::Vec3b>(i - 1, j - 1)[0]) / 9;
			img2.at<cv::Vec3b>(i, j)[1] = (img.at<cv::Vec3b>(i, j)[1], img.at<cv::Vec3b>(i + 1, j + 1)[1],
				img.at<cv::Vec3b>(i + 1, j)[1], img.at<cv::Vec3b>(i, j + 1)[1], img.at<cv::Vec3b>(i + 1, j - 1)[1],
				img.at<cv::Vec3b>(i - 1, j + 1)[1], img.at<cv::Vec3b>(i - 1, j)[1], img.at<cv::Vec3b>(i, j - 1)[1],
				img.at<cv::Vec3b>(i - 1, j - 1)[1]) / 9;
			img2.at<cv::Vec3b>(i, j)[2] = (img.at<cv::Vec3b>(i, j)[2], img.at<cv::Vec3b>(i + 1, j + 1)[2],
				img.at<cv::Vec3b>(i + 1, j)[2], img.at<cv::Vec3b>(i, j + 1)[2], img.at<cv::Vec3b>(i + 1, j - 1)[2],
				img.at<cv::Vec3b>(i - 1, j + 1)[2], img.at<cv::Vec3b>(i - 1, j)[2], img.at<cv::Vec3b>(i, j - 1)[2],
				img.at<cv::Vec3b>(i - 1, j - 1)[2]) / 9;
		}

	}
	cv::Mat zero_blue = img.clone();
	for (int r = 0; r < zero_blue.rows; r++)
	{
		for (int c = 0; c < zero_blue.cols; c++)
		{
			zero_blue.at<cv::Vec3b>(r, c)[0] = zero_blue.at<cv::Vec3b>(r, c)[0] * 0;//zero blue channel
        }
	}
	cv::namedWindow("zero_blue");
	cv::imshow("zero_blue", zero_blue);
	cv::imwrite("zero_blue.jpg", zero_blue);

	cv::Mat img3 = img.clone();
	cv::Mat splitChannels[3];
	cv::split(img3, splitChannels);
	splitChannels[2] = cv::Mat::zeros(splitChannels[2].size(), CV_8U);
	cv::Mat kill_red;
	cv::merge(splitChannels, 3, kill_red);
	cv::namedWindow("kill_red");
	cv::imshow("kill_red", kill_red);
	cv::imwrite("kill_red.jpg", kill_red);
	
	cv::Mat outputGray(img2.size(), CV_8U);
	unsigned char grayValue, maxValue = 1;
	for (int y = 0; y < img2.rows; y++)
	{
		for (int x = 0; x < img2.cols; x++)
		{
			grayValue = img2.at<uchar>(y, x);
			maxValue = cv::max(maxValue, grayValue);
		}
	}
	float scale = 255.0 / maxValue;
	for (int y = 0; y < img2.rows; y++)
	{
		for (int x = 0; x < img2.cols; x++)
		{
			outputGray.at<uchar>(y, x) = static_cast<unsigned char>(img2.at<uchar>(y, x) * scale + 0.5);
		}
	}
	cv::namedWindow("gray");
	cv::imshow("gray", outputGray);
	cv::imwrite("result1.jpg", outputGray);
	return img2;
}





/******************************
      GIVEN FUNCTIONS
 ******************************/

/**
 * @brief function loads input image, calls processing function, and saves result
 * @param fname path to input image
 */
void run(const std::string &filename) {

    // window names
    std::string win1 = "Original image";
    std::string win2 = "Result";

    // some images
    cv::Mat inputImage, outputImage;

    // load image
    std::cout << "loading image" << std::endl;
    inputImage = cv::imread("lena.jpg");
    std::cout << "done" << std::endl;
    
    // check if image can be loaded
    if (!inputImage.data)
        throw std::runtime_error(std::string("ERROR: Cannot read file ") + "lena.jpg");

    // show input image
    cv::namedWindow(win1.c_str());
    cv::imshow(win1.c_str(), inputImage);
    
    // do something (reasonable!)
    outputImage = doSomethingThatMyTutorIsGonnaLike(inputImage);
    
    // show result
    cv::namedWindow(win2.c_str());
    cv::imshow(win2.c_str(), outputImage);
    
    // save result
    cv::imwrite("result.jpg", outputImage);
    
    // wait a bit
    cv::waitKey(0);
}


}
