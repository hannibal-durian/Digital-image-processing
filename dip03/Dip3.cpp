//============================================================================
// Name    : Dip3.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip3.h"

#include <stdexcept>

namespace dip3 {

const char * const filterModeNames[NUM_FILTER_MODES] = {
    "FM_SPATIAL_CONVOLUTION",
    "FM_FREQUENCY_CONVOLUTION",
    "FM_SEPERABLE_FILTER",
    "FM_INTEGRAL_IMAGE",
};



/**
 * @brief Generates 1D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel1D(int kSize){

    // TO DO !!!
	//const static double pi = 3.1415926;
	cv::Mat GS_filter = cv::Mat_<float>::zeros(1, kSize);
	int half = kSize / 2;	//以（half,half）为中心建立坐标系进行计算
	float sigma = kSize / 5;
	float sum = 0;
	for (int i = 0; i < kSize; i++)
	{
		float g = exp(-(i - half) * (i - half) / (2 * sigma * sigma));// 只需计算指数部分，常数会在归一化的过程中消去
		sum += g;
		GS_filter.at<float>(i) = g;
	}
	for (int i = 0; i < kSize; i++)// 归一化
		GS_filter.at<float>(i) /= sum;
	return GS_filter;
    //return cv::Mat_<float>::zeros(1, kSize);
}

/**
 * @brief Generates 2D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel2D(int kSize){

    // TO DO !!!
	cv::Mat templateMatrix = cv::Mat_<float>::zeros(kSize, kSize);
	const static double pi = 3.1415926;
	float sigma = kSize / 5;
	int origin = kSize / 2;
	double x2, y2;
	double sum = 0;
	for (int i = 0; i < kSize; i++)
	{
		x2 = pow(i - origin, 2);
		for (int j = 0; j < kSize; j++)
		{
			y2 = pow(j - origin, 2);
			double g = exp(-(x2 + y2) / (2 * sigma * sigma));
			sum += g;
			templateMatrix.at<float>(i,j) = g;
		}
	}
	for (int i = 0; i < kSize; i++)
	{
		for (int j = 0; j < kSize; j++)
		{
			templateMatrix.at<float>(i, j) /= sum;
		}
	}
	return templateMatrix;
    //return cv::Mat_<float>::zeros(kSize, kSize);
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @returns Circular shifted matrix
 */
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy){

   // TO DO !!!
	cv::Mat tmp = cv::Mat::zeros(in.rows, in.cols, in.type());
	int new_x = 0;
	int new_y = 0;
	for (int y = 0; y < in.rows; y++)
	{
		// calulate new y-coordinate
		new_y = y + dy;
		if (new_y < 0)
			new_y = new_y + in.rows;
		if (new_y >= in.rows)
			new_y = new_y - in.rows;
		for (int x = 0; x < in.cols; x++) {
			// calculate new x-coordinate
			new_x = x + dx;
			if (new_x < 0)
				new_x = new_x + in.cols;
			if (new_x >= in.cols)
				new_x = new_x - in.cols;
			tmp.at<float>(new_y, new_x) = in.at<float>(y, x);
		}
	}
   return tmp;
}


/**
 * @brief Performes convolution by multiplication in frequency domain
 * @param in Input image
 * @param kernel Filter kernel
 * @returns Output image
 */
cv::Mat_<float> frequencyConvolution(const cv::Mat_<float>& in, const cv::Mat_<float>& kernel){

   // TO DO !!!
	//cv::Mat f_in = in.clone();
	cv::Mat OutputImage = in.clone();
	//cv::dft(in, f_in, 0);
	//cv::Mat f_kernel = kernel.clone();
	//cv::dft(kernel, f_kernel, 0);
	//cv::mulSpectrums(f_in, f_kernel, f_in, 0);
	//cv::dft(f_in, OutputImage, cv::DFT_INVERSE + cv::DFT_SCALE);

	//OutputImage.create(abs(in.rows - kernel.rows) + 1, abs(in.cols - kernel.cols) + 1, in.type());
	//cv::Size dftSize;

	//dftSize.width = cv::getOptimalDFTSize(in.cols + kernel.cols - 1);
	//dftSize.height = cv::getOptimalDFTSize(in.rows + kernel.rows - 1);

	//cv::Mat tempA(dftSize, in.type(), cv::Scalar::all(0));
	//cv::Mat tempB(dftSize, kernel.type(), cv::Scalar::all(0));

	//cv::Mat roiA(tempA, cv::Rect(0, 0, in.cols, in.rows));
	//in.copyTo(roiA);
	//cv::Mat roiB(tempB, cv::Rect(0, 0, kernel.cols, kernel.rows));
	//kernel.copyTo(roiB);

	//cv::dft(tempA, tempA, 0, in.rows);
	//cv::dft(tempB, tempB, 0, kernel.rows);

	//cv::mulSpectrums(tempA, tempB, tempA, 0);

	//cv::dft(tempA, tempA, cv::DFT_INVERSE + cv::DFT_SCALE, OutputImage.rows);

	//tempA(cv::Rect(0, 0, OutputImage.cols, OutputImage.rows)).copyTo(OutputImage);

	cv::Mat largermat(in.rows,in.cols,kernel.type(),cv::Scalar::all(0));
	float radius;
	radius = (kernel.cols-1)/2;
	for (int y=0; y<kernel.rows;y++)
	{
        for (int x=0; x<kernel.cols;x++)
        {
            largermat.at<float>(y,x)=kernel.at<float>(y,x);
        }
	}
	cv::Mat newlargemat = circShift(largermat,-radius,-radius);
	cv::Mat newlargemat_com;
	cv::dft(newlargemat,newlargemat_com,0);
	cv::Mat f_in = in.clone();
	cv::dft(in,f_in,0);
	cv::mulSpectrums(f_in, newlargemat_com, f_in, 0);
	cv::dft(f_in, OutputImage, cv::DFT_INVERSE + cv::DFT_SCALE);
	return OutputImage;
    //return in;
}


/**
 * @brief  Performs UnSharp Masking to enhance fine image structures
 * @param in The input image
 * @param filterMode How convolution for smoothing operation is done
 * @param size Size of used smoothing kernel
 * @param thresh Minimal intensity difference to perform operation
 * @param scale Scaling of edge enhancement
 * @returns Enhanced image
 */
cv::Mat_<float> usm(const cv::Mat_<float>& in, FilterMode filterMode, int size, float thresh, float scale)
{
   // TO DO !!!

   // use smoothImage(...) for smoothing
	int imageheight = in.rows;
	int imagewidth = in.cols;
	cv::Mat Enhanced_1 = in.clone();
	cv::Mat Enhanced_2 = in.clone();
	cv::Mat Enhanced_3 = in.clone();
	cv::Mat Enhanced_4 = in.clone();
	Enhanced_1 = smoothImage(in, size, filterMode);
	for (int y = 0; y < imageheight; y++)
	{
		for (int x = 0; x < imagewidth; x++)
		{
			Enhanced_2.at<float>(y, x) = abs(in.at<float>(y, x) - Enhanced_1.at<float>(y, x));
		}
	}
	for (int y = 0; y < imageheight; y++)
	{
		for (int x = 0; x < imagewidth; x++)
		{
			float value = Enhanced_2.at<float>(y, x);
			if (value > thresh)
			{
				Enhanced_3.at<float>(y, x) = Enhanced_2.at<float>(y, x) * scale;
			}
			else
			{
				Enhanced_3.at<float>(y, x) = Enhanced_2.at<float>(y, x);
			}
		}
	}
	for (int y = 0; y < imageheight; y++)
	{
		for (int x = 0; x < imagewidth; x++)
		{
			Enhanced_4.at<float>(y, x) = in.at<float>(y, x) + Enhanced_3.at<float>(y, x);
		}
	}
	return Enhanced_4;
   //return in;
}


/**
 * @brief Convolution in spatial domain
 * @param src Input image
 * @param kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{

   // Hopefully already DONE, copy from last homework
	cv::Mat_<float> kern;
	cv::flip(kernel, kern, -1);
	//compute the radius of kernel
	cv::Mat OutputImage = src.clone();
	int sub_x = (kern.rows - 1) / 2;
	int sub_y = (kern.cols - 1) / 2;
	// image borders
	cv::Mat new_src;
	cv::copyMakeBorder(src, new_src, sub_y, sub_y, sub_x, sub_x, cv::BORDER_CONSTANT, 0);
	for (int image_x = sub_x; image_x < new_src.rows - sub_x; image_x++)
	{
		for (int image_y = sub_y; image_y < new_src.cols - sub_y; image_y++)
		{
			float pix_value = 0.;
			for (int kernel_x = 0; kernel_x < kern.rows; kernel_x++)
			{
				for (int kernel_y = 0; kernel_y < kern.cols; kernel_y++)
				{
					float weihgt = kern.at<float>(kernel_x, kernel_y);
					float value = new_src.at<float>(image_x + kernel_x - sub_x, image_y + kernel_y - sub_y);
					pix_value += weihgt * value;
				}
			}
			OutputImage.at<float>(image_x - sub_x, image_y - sub_y) = pix_value;
		}
	}
	return OutputImage;
   //return src;
}


/**
 * @brief Convolution in spatial domain by seperable filters
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel){

   // TO DO !!!
	cv::Mat OutputImage = src.clone();
	//int sub_x = (kernel.rows - 1) / 2;
	int border = (kernel.cols - 1) / 2;
	cv::Mat_<float> kern;
	cv::flip(kernel, kern, -1);
	// image borders
	cv::Mat new_src;
	cv::copyMakeBorder(src, new_src, border, border, border, border, cv::BORDER_CONSTANT, 0);
	int rows = new_src.rows - border;
	int cols = new_src.cols - border;
	for (int i = border; i < rows; i++)
	{
		for (int j = border; j < cols; j++)
		{
			float sum = 0;
			for (int k = -border; k <= border; k++)
			{
     			sum += kern.at<float>(border + k) * new_src.at<float>(i, j + k); // 先做水平方向的卷积
			}
			OutputImage.at<float>(i - border, j - border) = sum;
		}
	}
	cv::Mat new_src1 = OutputImage.clone();
	cv::copyMakeBorder(new_src1, new_src1, border, border, border, border, cv::BORDER_CONSTANT, 0);
	// 竖直方向
	for (int i = border; i < rows; i++)
	{
		for (int j = border; j < cols; j++)
		{
			float sum = 0;
			for (int k = -border; k <= border; k++)
			{
				sum += kern.at<float>(border + k) * new_src1.at<float>(i + k, j); // 列不变，行变化；竖直方向的卷积
			}
			OutputImage.at<float>(i - border, j - border) = sum;
		}
	}
	return OutputImage;
   //return src;

}


/**
 * @brief Convolution in spatial domain by integral images
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> satFilter(const cv::Mat_<float>& src, int size){

   // optional
   //void integral(InputArray src,OutputArray sum,OutputArray sqsum,int sdepth = -1,int sqdepth = -1);
	cv::Mat sum, sqrsum;
	cv::integral(src, sum, sqrsum);
	int w = src.cols;
	int h = src.rows;
	cv::Mat result = cv::Mat::zeros(src.size(), src.type());
	int x2 = 0, y2 = 0;
	int x1 = 0, y1 = 0;
	int ksize = 5;
	int radius = ksize / 2;

	int cx = 0, cy = 0;
	for (int row = 0; row < h + radius; row++)
	{
		y2 = (row + 1) > h ? h : (row + 1);
		y1 = (row - ksize) < 0 ? 0 : (row - ksize);
		for (int col = 0; col < w + radius; col++)
		{
			x2 = (col + 1) > w ? w : (col + 1);
			x1 = (col - ksize) < 0 ? 0 : (col - ksize);
			cx = (col - radius) < 0 ? 0 : col - radius;
			cy = (row - radius) < 0 ? 0 : row - radius;
			int num = (x2 - x1) * (y2 - y1);
			int tl = sum.at<int>(y1, x1);
			int tr = sum.at<int>(y2, x1);
			int bl = sum.at<int>(y1, x2);
			int br = sum.at<int>(y2, x2);
			int s = (br - bl - tr + tl);
			result.at<float>(cy, cx) = cv::saturate_cast<float>(s / num);

		}
	}
	return result;
   //return src;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * @brief Performs a smoothing operation but allows the algorithm to be chosen
 * @param in Input image
 * @param size Size of filter kernel
 * @param type How is smoothing performed?
 * @returns Smoothed image
 */
cv::Mat_<float> smoothImage(const cv::Mat_<float>& in, int size, FilterMode filterMode)
{
    switch(filterMode) {
        case FM_SPATIAL_CONVOLUTION: return spatialConvolution(in, createGaussianKernel2D(size));	// 2D spatial convolution
        case FM_FREQUENCY_CONVOLUTION: return frequencyConvolution(in, createGaussianKernel2D(size));	// 2D convolution via multiplication in frequency domain
        case FM_SEPERABLE_FILTER: return separableFilter(in, createGaussianKernel1D(size));	// seperable filter
        case FM_INTEGRAL_IMAGE: return satFilter(in, size);		// integral image
        default:
            throw std::runtime_error("Unhandled filter type!");
    }
}



}

