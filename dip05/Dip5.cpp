//============================================================================
// Name        : Dip5.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip5.h"


namespace dip5 {


/**
* @brief Generates gaussian filter kernel of given size
* @param kSize Kernel size (used to calculate standard deviation)
* @returns The generated filter kernel
*/
cv::Mat_<float> createGaussianKernel1D(float sigma)
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);
    // Hopefully already DONE, copy from last homework, just make sure you compute the kernel size from the given sigma (and not the other way around)
	cv::Mat GS_filter = cv::Mat_<float>::zeros(1, kSize);
	int half = kSize / 2;	//以（half,half）为中心建立坐标系进行计算
	//float sigma = kSize / 5;
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
* @brief Convolution in spatial domain by seperable filters
* @param src Input image
* @param size Size of filter kernel
* @returns Convolution result
*/
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernelX, const cv::Mat_<float>& kernelY)
{
    // Hopefully already DONE, copy from last homework
    // But do mind that this one gets two different kernels for horizontal and vertical convolutions.
	cv::Mat OutputImage = src.clone();
	int sub_x = (kernelX.cols - 1) / 2;
	int sub_y = (kernelY.cols - 1) / 2;
	//int border = (kernel.cols - 1) / 2;
	cv::Mat_<float> kern_x;
	cv::Mat_<float> kern_y;
	cv::flip(kernelX, kern_x, -1);//1
	cv::flip(kernelY, kern_y, -1);//0
	// image borders
	cv::Mat new_src;
	cv::copyMakeBorder(src, new_src, sub_y, sub_y, sub_x, sub_x, cv::BORDER_CONSTANT, 0);
	int rows = new_src.rows - sub_y;
	int cols = new_src.cols - sub_x;
	for (int i = sub_y; i < rows; i++)
	{
		for (int j = sub_x; j < cols; j++)
		{
			float sum = 0;
			for (int k = -sub_x; k <= sub_x; k++)
			{
				sum += kern_x.at<float>(sub_x + k) * new_src.at<float>(i, j + k); // 先做水平方向的卷积
			}
			OutputImage.at<float>(i - sub_y, j - sub_x) = sum;
		}
	}
	cv::Mat new_src1 = OutputImage.clone();
	cv::copyMakeBorder(new_src1, new_src1, sub_y, sub_y, sub_x, sub_x, cv::BORDER_CONSTANT, 0);
	// 竖直方向
	for (int i = sub_y; i < rows; i++)
	{
		for (int j = sub_x; j < cols; j++)
		{
			float sum = 0;
			for (int k = -sub_y; k <= sub_y; k++)
			{
				sum += kern_y.at<float>(sub_y + k) * new_src1.at<float>(i + k, j); // 列不变，行变化；竖直方向的卷积
			}
			OutputImage.at<float>(i - sub_y, j - sub_x) = sum;
		}
	}
	return OutputImage;
    //return src;
}


/**
 * @brief Creates kernel representing fst derivative of a Gaussian kernel (1-dimensional)
 * @param sigma standard deviation of the Gaussian kernel
 * @returns the calculated kernel
 */
cv::Mat_<float> createFstDevKernel1D(float sigma)
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);
    // TO DO !!!
	cv::Mat FD_filter = cv::Mat_<float>::zeros(1, kSize);
	int half = kSize / 2;
	float pi = 3.141592653;
	for (int i = 0; i < kSize; i++)
	{
		float g = exp((-(i-half) * (i-half)) / (2 * sigma * sigma));
		float derivative = (-(i-half) / (2 * pi * sigma * sigma * sigma * sigma)) * g;
		FD_filter.at<float>(i) = derivative;
	}
	return FD_filter;
    //return cv::Mat_<float>::zeros(1, kSize);
}


/**
 * @brief Calculates the directional gradients through convolution
 * @param img The input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param gradX Matrix through which to return the x component of the directional gradients
 * @param gradY Matrix through which to return the y component of the directional gradients
 */
void calculateDirectionalGradients(const cv::Mat_<float>& img, float sigmaGrad,
                            cv::Mat_<float>& gradX, cv::Mat_<float>& gradY)
{
    // TO DO !!!

    gradX.create(img.rows, img.cols);
    gradY.create(img.rows, img.cols);
	cv::Mat FstDevKernel = createFstDevKernel1D(sigmaGrad);
	cv::Mat Gaussiankernel = createGaussianKernel1D(sigmaGrad);
	gradX = separableFilter(img, FstDevKernel, Gaussiankernel);
	gradY = separableFilter(img, Gaussiankernel, FstDevKernel);
}

/**
 * @brief Calculates the structure tensors (per pixel)
 * @param gradX The x component of the directional gradients
 * @param gradY The y component of the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation".
 * @param A00 Matrix through which to return the A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 Matrix through which to return the A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 Matrix through which to return the A_{1,1} elements of the structure tensor of each pixel.
 */
void calculateStructureTensor(const cv::Mat_<float>& gradX, const cv::Mat_<float>& gradY, float sigmaNeighborhood,
                            cv::Mat_<float>& A00, cv::Mat_<float>& A01, cv::Mat_<float>& A11)
{
    A00.create(gradX.rows, gradX.cols);
    A01.create(gradX.rows, gradX.cols);
    A11.create(gradX.rows, gradX.cols);

    // TO DO !!!
	cv::Mat GS_filter = createGaussianKernel1D(sigmaNeighborhood);
	int x = gradX.cols;
	int y = gradX.rows;
	cv::Mat a=A00.clone();
	cv::Mat b=A01.clone();
	cv::Mat c=A11.clone();
	//a=gradX*gradX;
	//b=gradX*gradY;
	//c=gradY*gradY;
	a=gradX.mul(gradX);
	b=gradX.mul(gradY);
	c=gradY.mul(gradY);

	A00 = separableFilter(a, GS_filter, GS_filter);
	//A00 = a;
	A01 = separableFilter(b, GS_filter, GS_filter);
	A11 = separableFilter(c, GS_filter, GS_filter);
	//A11 = c;
}

/**
 * @brief Calculates the feature point weight and isotropy from the structure tensors.
 * @param A00 The A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 The A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 The A_{1,1} elements of the structure tensor of each pixel.
 * @param weight Matrix through which to return the weights of each pixel.
 * @param isotropy Matrix through which to return the isotropy of each pixel.
 */
void calculateFoerstnerWeightIsotropy(const cv::Mat_<float>& A00, const cv::Mat_<float>& A01, const cv::Mat_<float>& A11,
                                    cv::Mat_<float>& weight, cv::Mat_<float>& isotropy)
{
    weight.create(A00.rows, A00.cols);
    isotropy.create(A00.rows, A00.cols);

    // TO DO !!!
	int x = A00.cols;
	int y = A00.rows;
	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			cv::Mat_<float> A =cv::Mat_<float>::zeros(2, 2);
			A.at<float>(0, 0) = A00.at<float>(i, j);
			A.at<float>(0, 1) = A01.at<float>(i, j);
			A.at<float>(1, 0) = A01.at<float>(i, j);
			A.at<float>(1, 1) = A11.at<float>(i, j);
			float tra = (cv::trace(A).val[0]);
			weight.at<float>(i, j) = cv::determinant(A) / std::max(tra, 1e-8f);
			isotropy.at<float>(i, j) = (4 * cv::determinant(A)) / std::max(tra*tra, 1e-8f);
		}

	}
}


/**
 * @brief Finds Foerstner interest points in an image and returns their location.
 * @param img The greyscale input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation" of the structure tensor.
 * @param fractionalMinWeight Threshold on the weight as a fraction of the mean of all locally maximal weights.
 * @param minIsotropy Threshold on the isotropy of interest points.
 * @returns List of interest point locations.
 */
std::vector<cv::Vec2i> getFoerstnerInterestPoints(const cv::Mat_<float>& img, float sigmaGrad, float sigmaNeighborhood, float fractionalMinWeight, float minIsotropy)
{
    // TO DO !!!
	cv::Mat_<float> gradX, gradY, A00, A01, A11, weight, isotropy;

	calculateDirectionalGradients(img, sigmaGrad, gradX, gradY);
	calculateStructureTensor(gradX, gradY, sigmaNeighborhood, A00, A01, A11);
	calculateFoerstnerWeightIsotropy(A00, A01, A11, weight, isotropy);
	//cv::mean(weight);
	float w = (cv::mean(weight).val[0]);
	std::vector<cv::Vec2i> InterestPoints;
	int x = img.cols;
	int y = img.rows;

	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			//float w = weight.at<float>(i, j);
			if (weight.at<float>(i, j) > w*fractionalMinWeight)
			{
				//float isot = isotropy.at<float>(i, j);
				if (isotropy.at<float>(i, j) > minIsotropy)
				{
					if (isLocalMaximum(weight, j, i))
					{
						cv::Vec2i p;
						p[0] = j;
						p[1] = i;
						InterestPoints.push_back(p);
					}
				}
			}
		}
	}
    return InterestPoints;
}



/* *****************************
  GIVEN FUNCTIONS
***************************** */


// Use this to compute kernel sizes so that the unit tests can simply hard checks for correctness.
unsigned getOddKernelSizeForSigma(float sigma)
{
    unsigned kSize = (unsigned) std::ceil(5.0f * sigma) | 1;
    if (kSize < 3) kSize = 3;
    return kSize;
}

bool isLocalMaximum(const cv::Mat_<float>& weight, int x, int y)
{
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            int x_ = std::min(std::max(x+j, 0), weight.cols-1);
            int y_ = std::min(std::max(y+i, 0), weight.rows-1);
            if (weight(y_, x_) > weight(y, x))
                return false;
        }
    return true;
}

}
