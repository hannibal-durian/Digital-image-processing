//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip2.h"

namespace dip2 {


/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @params src Input image
 * @params kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
   // TO DO !!

	//kernel flip
	cv::Mat_<float> kern;
	cv::flip(kernel,kern,-1);
	 //计算卷积核的半径 compute the radius of kernel
	cv::Mat OutputImage = src.clone();
	int sub_x = (kern.rows -1) / 2;
	int sub_y = (kern.cols -1) / 2;
	//边界问题  image borders
	cv::Mat new_src;
	//new_src = cv::Mat::zeros(2 * sub_x + src.rows, 2 * sub_y + src.cols, src.type());//初始化一个补全图像的大小。
	//cv::Rect real_roi_of_image = cv::Rect(sub_x, sub_y, src.rows, src.cols);
	//cv::Mat real_mat_of_image = new_src(real_roi_of_image);
	//src.copyTo(new_src(real_roi_of_image));
	cv::copyMakeBorder(src, new_src, sub_y, sub_y, sub_x, sub_x, cv::BORDER_CONSTANT, 0);
	//遍历图片  iterate over the image
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
					float value = new_src.at<float>(image_x + kernel_x -sub_x, image_y + kernel_y-sub_y);
					pix_value += weihgt * value;
				}
			}
			OutputImage.at<float>(image_x - sub_x, image_y - sub_y) = pix_value;
		}
	}
	return OutputImage;
   //return src.clone();
}

/**
 * @brief Moving average filter (aka box filter)
 * @note: you might want to use Dip2::spatialConvolution(...) within this function
 * @param src Input image
 * @param kSize Window size used by local average
 * @returns Filtered image
 */
cv::Mat_<float> averageFilter(const cv::Mat_<float>& src, int kSize)
{
   // TO DO !!

	//cv::Mat_<float>  average_kernel;
	//average_kernel = cv::Mat_<float>::ones(cv::Size(kSize,kSize));
	float num = 1./(kSize * kSize);
	cv::Mat average_kernel = cv::Mat(kSize, kSize,CV_32FC1,num);
	//float num = 1/(kSize * kSize);
	//average_kernel = average_kernel * (num);
	//cv::Mat new_src = src.clone();
	cv::Mat OutputImage = src.clone();
	OutputImage = spatialConvolution(src, average_kernel);
	return OutputImage;

/*	cv::Mat dst = src.clone();
	int start = (kSize-1) / 2;
	cv::Mat new_src;
	cv::copyMakeBorder(src, new_src, start, start, start, start, cv::BorderTypes::BORDER_CONSTANT, 0);
	int rows = new_src.rows;
	int cols = new_src.cols;
	for (int m = start; m < rows - start; m++)
	{
		for (int n = start; n < cols - start; n++)
		{
			int sum = 0;
			for (int i = 0; i < kSize; i++)
			{
				for (int j = 0; j < kSize; j++)
				{
					sum += new_src.at<float>(m + i - start, m + j-start);
				}
			}
			dst.at<float>(m, n) = float(sum / kSize / kSize);
		}
	}
	return dst;
*/
    //return src.clone();
}

/**
 * @brief Median filter
 * @param src Input image
 * @param kSize Window size used by median operation
 * @returns Filtered image
 */
cv::Mat_<float> medianFilter(const cv::Mat_<float>& src, int kSize)
{
   // TO DO !!
	cv::Mat dst = src.clone();
	//获取图片的宽，高和像素信息，
	cv::Mat media_kernel;
	media_kernel = cv::Mat::ones(kSize, kSize,CV_32FC1);
	int sub_x = (media_kernel.rows - 1) / 2;
	int sub_y = (media_kernel.cols - 1) / 2;
	int  num = kSize * kSize;
	//边界问题  image borders
	cv::Mat new_src = src.clone();
	//new_src = cv::Mat::zeros(2 * sub_x + src.rows, 2 * sub_y + src.cols, src.type());//初始化一个补全图像的大小。
	//cv::Rect real_roi_of_image = cv::Rect(sub_x, sub_y, src.rows, src.cols);
	//cv::Mat real_mat_of_image = new_src(real_roi_of_image);
	//src.copyTo(new_src(real_roi_of_image));
	cv::copyMakeBorder(src, new_src, sub_y, sub_y, sub_x, sub_x, cv::BorderTypes::BORDER_CONSTANT, 0);
	//中值滤波
	for (int i = sub_x; i < new_src.rows - sub_x; ++i)
	{
		for (int j = sub_y; j < new_src.cols - sub_y; ++j)
		{
			std::vector<float> pixel;
			for (int kernel_x = 0; kernel_x < media_kernel.rows; kernel_x++)
			{
				for (int kernel_y = 0; kernel_y < media_kernel.cols; kernel_y++)
				{
					float value = new_src.at<float>(i + kernel_x -sub_x, j + kernel_y-sub_y);
					pixel.push_back(value);
				}
			}
			//排序
			std::sort(pixel.begin(), pixel.end());
			//获取该中心点的值
			dst.at<float>(i-sub_x, j-sub_y) = pixel[(num+1) / 2];
		}
	}
	return dst;
    //return src.clone();
}

/**
 * @brief Bilateral filer
 * @param src Input image
 * @param kSize Size of the kernel
 * @param sigma_spatial Standard-deviation of the spatial kernel
 * @param sigma_radiometric Standard-deviation of the radiometric kernel
 * @returns Filtered image
 */
cv::Mat_<float> bilateralFilter(const cv::Mat_<float>& src, int kSize, float sigma_spatial, float sigma_radiometric)
{
    // TO DO !!
	//cv::Mat dst = src.clone();
	//cv::bilateralFilter(src, dst, kSize, sigma_spatial, sigma_radiometric);
	//return dst;

	double space_coeff = -0.5 / (sigma_spatial * sigma_spatial);
	double color_coeff = -0.5 / (sigma_radiometric * sigma_radiometric);
	int radius = (kSize-1) / 2;
	cv::Mat temp = src.clone();
	cv::Mat dst = src.clone();
	float pi = 3.141592653;
    float x1 = 1./(2*pi*sigma_spatial*sigma_spatial);
    float x2 = 1./(2*pi*sigma_radiometric*sigma_radiometric);
	cv::copyMakeBorder(src, temp, radius, radius, radius, radius, cv::BorderTypes::BORDER_CONSTANT, 0);
	for (int k = radius; k < temp.rows - radius; ++k)
	{
		for (int l = radius; l < temp.cols - radius; ++l)
		{
			float zaehler = 0.;
			float nenner = 0.;
			for (int i = -radius; i <= radius; ++i)
			{
				for (int j = -radius; j <= radius; ++j)
				{
					//空间域模板权值 wd 和 灰度域 模板权值 wr
					double wd = x1*exp((pow(i,2) + pow(j,2)) * space_coeff);
					double wr = x2*exp(pow((temp.at<float>(i+k, j+l)) - (temp.at<float>(k, l)),2) * color_coeff);
					double w = wd * wr;
					zaehler += temp.at<float>(i+k,j+l) * w;
					nenner += w;
				}
			}
			dst.at<float>(k - radius, l - radius) = zaehler / nenner;
		}
	}
    return dst;

	//return src.clone();
}

/**
 * @brief Non-local means filter
 * @note: This one is optional!
 * @param src Input image
 * @param searchSize Size of search region
 * @param sigma Optional parameter for weighting function
 * @returns Filtered image
 */
cv::Mat_<float> nlmFilter(const cv::Mat_<float>& src, int searchSize, double sigma)
{

    return src.clone();
}



/**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
 */
NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType)
{
    // TO DO !!
	//NOISE_TYPE_1  ----Shot Noise
	//NOISE_TYPE_2  ----Gaussian Noise
	if (noiseType == NOISE_TYPE_1)
	{
		return NR_MEDIAN_FILTER;
	}
	if (noiseType == NOISE_TYPE_2)
	{
		return NR_MOVING_AVERAGE_FILTER;
	}
    //return (NoiseReductionAlgorithm) -1;
}



cv::Mat_<float> denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType, dip2::NoiseReductionAlgorithm noiseReductionAlgorithm)
{
    // TO DO !!

    // for each combination find reasonable filter parameters

    switch (noiseReductionAlgorithm) {
        case dip2::NR_MOVING_AVERAGE_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::averageFilter(src, 5);
                case NOISE_TYPE_2:
                    return dip2::averageFilter(src, 5);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_MEDIAN_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::medianFilter(src, 7);
                case NOISE_TYPE_2:
                    return dip2::medianFilter(src, 7);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_BILATERAL_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::bilateralFilter(src, 7, 150.0f, 50.0f);
                case NOISE_TYPE_2:
                    return dip2::bilateralFilter(src, 5, 100.0f, 30.0f);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        default:
            throw std::runtime_error("Unhandled filter type!");
    }
}





// Helpers, don't mind these

const char *noiseTypeNames[NUM_NOISE_TYPES] = {
    "NOISE_TYPE_1",
    "NOISE_TYPE_2",
};

const char *noiseReductionAlgorithmNames[NUM_FILTERS] = {
    "NR_MOVING_AVERAGE_FILTER",
    "NR_MEDIAN_FILTER",
    "NR_BILATERAL_FILTER",
};


}
