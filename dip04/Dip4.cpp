//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip4.h"

namespace dip4 {

using namespace std::complex_literals;

/*

===== std::complex cheat sheet =====

Initialization:

std::complex<float> a(1.0f, 2.0f);
std::complex<float> a = 1.0f + 2.0if;

Common Operations:

std::complex<float> a, b, c;

a = b + c;
a = b - c;
a = b * c;
a = b / c;

std::sin, std::cos, std::tan, std::sqrt, std::pow, std::exp, .... all work as expected

Access & Specific Operations:

std::complex<float> a = ...;

float real = a.real();
float imag = a.imag();
float phase = std::arg(a);
float magnitude = std::abs(a);
float squared_magnitude = std::norm(a);

std::complex<float> complex_conjugate_a = std::conj(a);

*/



/**
 * @brief Computes the complex valued forward DFT of a real valued input
 * @param input real valued input
 * @return Complex valued output, each pixel storing real and imaginary parts
 */
cv::Mat_<std::complex<float>> DFTReal2Complex(const cv::Mat_<float>& input)
{
    // TO DO !!!
	cv::Mat_<std::complex<float>> COMP = cv::Mat_<std::complex<float>>(input.rows, input.cols);
	cv::dft(input, COMP, cv::DFT_COMPLEX_OUTPUT);
	return COMP;
    //return cv::Mat_<std::complex<float>>(input.rows, input.cols);
}


/**
 * @brief Computes the real valued inverse DFT of a complex valued input
 * @param input Complex valued input, each pixel storing real and imaginary parts
 * @return Real valued output
 */
cv::Mat_<float> IDFTComplex2Real(const cv::Mat_<std::complex<float>>& input)
{
    // TO DO !!!
	cv::Mat_<float> Re = cv::Mat_<float>(input.rows, input.cols);
	cv::dft(input, Re, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
	//cv::idft(input, Re, cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
	return Re;
    //return cv::Mat_<float>(input.rows, input.cols);
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @return Circular shifted matrix
*/
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy)
{
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
    //return in;
}


/**
 * @brief Computes the thresholded inverse filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return The inverse filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeInverseFilter(const cv::Mat_<std::complex<float>>& input, const float eps)
{
    // TO DO !!!
	int r = input.rows;
	int c = input.cols;
	cv::Mat_<std::complex<float>> Q = input.clone();
	float max = 0;
	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < c; j++)
		{
            float magnitude = std::abs(input.at<std::complex<float>>(i, j));
			if (max <= magnitude)
			{
				max = magnitude;
			}
        }

	}
	float T = max * eps;
	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < c; j++)
		{
			float magni= std::abs(input.at<std::complex<float>>(i, j));
			if (magni < T)
			{
                std::complex<float> B{T,0};
				Q.at<std::complex<float>>(i, j) = 1.0f / B;
			}
			else
			{
				Q.at<std::complex<float>>(i, j) = 1.0f / input.at<std::complex<float>>(i, j);
			}

		}

	}
    return Q;
}


/**
 * @brief Applies a filter (in frequency domain)
 * @param input Image in frequency domain (complex valued)
 * @param filter Filter in frequency domain (complex valued), same size as input
 * @return The filtered image, complex valued, in frequency domain
 */
cv::Mat_<std::complex<float>> applyFilter(const cv::Mat_<std::complex<float>>& input, const cv::Mat_<std::complex<float>>& filter)
{
    // TO DO !!!
    cv::Mat_<std::complex<float>> output = cv::Mat_<std::complex<float>>(input.rows, input.cols);
	cv::mulSpectrums(input, filter, output, 0);
    return output;
}


/**
 * @brief Function applies the inverse filter to restorate a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return Restorated output image
 */
cv::Mat_<float> inverseFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, const float eps)
{
    // TO DO !!!
	cv::Mat padded(degraded.rows, degraded.cols, filter.type(), cv::Scalar::all(0));

	float radius_r = (filter.rows - 1) / 2;
	float radius_c = (filter.cols - 1) / 2;

	for (int y = 0; y < filter.rows; y++)
	{
		for (int x = 0; x < filter.cols; x++)
		{
			padded.at<float>(y, x) = filter.at<float>(y, x);
		}
	}

	cv::Mat fullfiter = circShift(padded, -radius_r, -radius_c);
	cv::Mat_<std::complex<float>> Complex_degraded = DFTReal2Complex(degraded);
	cv::Mat_<std::complex<float>> Complex_filter = DFTReal2Complex(fullfiter);
	cv::Mat_<std::complex<float>> Q = computeInverseFilter(Complex_filter, eps);
	cv::Mat_<std::complex<float>> restorted = applyFilter(Complex_degraded, Q);
	cv::Mat_<std::complex<float>> output = IDFTComplex2Real(restorted);
	return output;

    //return degraded;
}


/**
 * @brief Computes the Wiener filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param snr Signal to noise ratio
 * @return The wiener filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeWienerFilter(const cv::Mat_<std::complex<float>>& input, const float snr)
{
    // TO DO !!!
	int r = input.rows;
	int c = input.cols;
	cv::Mat_<std::complex<float>> Q = input.clone();
	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < c; j++)
		{
			std::complex<float> complex_conjugate_p = std::conj(input.at<std::complex<float>>(i, j));
			float squared_magnitude_p = std::norm(input.at<std::complex<float>>(i, j));
			std::complex<float> B{squared_magnitude_p + (1.0f / pow(snr, 2)),0};
			//std::complex<float> a = (squared_magnitude_p + (1.0f / pow(snr, 2)))*1.0f + 0.0if;
			Q.at<std::complex<float>>(i, j) = complex_conjugate_p / B;
		}
	}
	return Q;
    //return input;
}

/**
 * @brief Function applies the wiener filter to restore a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param snr Signal to noise ratio of the input image
 * @return Restored output image
 */
cv::Mat_<float> wienerFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, float snr)
{
    // TO DO !!!
	int r = (degraded.rows - filter.rows) / 2;
	int c = (degraded.cols - filter.cols) / 2;
	//cv::Mat padded;
	//cv::copyMakeBorder(filter, padded, r, r, c, c, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	cv::Mat padded(degraded.rows, degraded.cols, filter.type(), cv::Scalar::all(0));
	for (int y = 0; y < filter.rows; y++)
	{
		for (int x = 0; x < filter.cols; x++)
		{
			padded.at<float>(y, x) = filter.at<float>(y, x);
		}
	}

	float radius_r;
	float radius_c;
	radius_r = (filter.rows - 1) / 2;
	radius_c = (filter.cols - 1) / 2;

	cv::Mat fullfiter = circShift(padded, -radius_r, -radius_c);
	cv::Mat_<std::complex<float>> Complex_degraded = DFTReal2Complex(degraded);
	cv::Mat_<std::complex<float>> Complex_filter = DFTReal2Complex(fullfiter);
	cv::Mat_<std::complex<float>> Q = computeWienerFilter(Complex_filter, snr);
	cv::Mat_<std::complex<float>> restorted = applyFilter(Complex_degraded, Q);
	cv::Mat_<std::complex<float>> output = IDFTComplex2Real(restorted);
	return output;
    //return degraded;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * function degrades the given image with gaussian blur and additive gaussian noise
 * @param img Input image
 * @param degradedImg Degraded output image
 * @param filterDev Standard deviation of kernel for gaussian blur
 * @param snr Signal to noise ratio for additive gaussian noise
 * @return The used gaussian kernel
 */
cv::Mat_<float> degradeImage(const cv::Mat_<float>& img, cv::Mat_<float>& degradedImg, float filterDev, float snr)
{

    int kSize = round(filterDev*3)*2 - 1;

    cv::Mat gaussKernel = cv::getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();

    cv::Mat imgs = img.clone();
    cv::dft( imgs, imgs, img.rows);
    cv::Mat kernels = cv::Mat::zeros( img.rows, img.cols, CV_32FC1);
    int dx, dy; dx = dy = (kSize-1)/2.;
    for(int i=0; i<kSize; i++)
        for(int j=0; j<kSize; j++)
            kernels.at<float>((i - dy + img.rows) % img.rows,(j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i,j);
	cv::dft( kernels, kernels );
	cv::mulSpectrums( imgs, kernels, imgs, 0 );
	cv::dft( imgs, degradedImg,  cv::DFT_INVERSE + cv::DFT_SCALE, img.rows );

    cv::Mat mean, stddev;
    cv::meanStdDev(img, mean, stddev);

    cv::Mat noise = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
    cv::randn(noise, 0, stddev.at<double>(0)/snr);
    degradedImg = degradedImg + noise;
    cv::threshold(degradedImg, degradedImg, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(degradedImg, degradedImg, 0, 0, cv::THRESH_TOZERO);

    return gaussKernel;
}


}
