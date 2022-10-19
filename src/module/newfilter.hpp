

/***************************************************************/
/*
*   Distribution code Version 1.1 -- 09/21/2014 by Qi Zhang Copyright 2014, The Chinese University of Hong Kong.
*
*   The Code is created based on the method described in the following paper 
*   [1] "100+ Times Faster Weighted Median Filter", Qi Zhang, Li Xu, Jiaya Jia, IEEE Conference on 
*		Computer Vision and Pattern Recognition (CVPR), 2014
*   
*   Due to the adaption for supporting mask and different types of input, this code is
*   slightly slower than the one claimed in the original paper. Please use
*   our executable on our website for performance comparison.
*
*   The code and the algorithm are for non-comercial use only.
*
/***************************************************************/


#ifndef JOINT_WMF_H
#define JOINT_WMF_H

/***************************************************************/
/* 
 * Standard IO library is required.
 * STL String library is required.
 *
/***************************************************************/
#include <cstdio>
#include <string>

/***************************************************************/
/* 
 * OpenCV 2.4 is required. 
 * The following code is already built on OpenCV 2.4.2.
 *
/***************************************************************/
#include "opencv2/core/core.hpp"
#include <time.h>

//Use the namespace of CV and STD
using namespace std;
using namespace cv;

class JointWMF{

public:

	/***************************************************************/
	/* Function: filter 
	 *
	 * Description: filter implementation of joint-histogram weighted median framework
	 *				including clustering of feature image, adaptive quantization of input image.
	 * 
	 * Input arguments:
	 *			I: input image (any # of channels). Accept only CV_32F and CV_8U type.
	 *	  feature: the feature image ("F" in the paper). Accept only CV_8UC1 and CV_8UC3 type (the # of channels should be 1 or 3).  
	 *          r: radius of filtering kernel, should be a positive integer.
	 *      sigma: filter range standard deviation for the feature image.
	 *         nI: # of quantization level of input image. (only when the input image is CV_32F type)
	 *         nF: # of clusters of feature value. (only when the feature image is 3-channel)
	 *       iter: # of filtering times/iterations. (without changing the feature map)
	 * weightType: the type of weight definition, including:
	 *					exp: exp(-|I1-I2|^2/(2*sigma^2))
	 *					iv1: (|I1-I2|+sigma)^-1
	 *					iv2: (|I1-I2|^2+sigma^2)^-1
	 *					cos: dot(I1,I2)/(|I1|*|I2|)
	 *					jac: (min(r1,r2)+min(g1,g2)+min(b1,b2))/(max(r1,r2)+max(g1,g2)+max(b1,b2))
	 *					off: unweighted
	 *		 mask: a 0-1 mask that has the same size with I. This mask is used to ignore the effect of some pixels. If the pixel value on mask is 0, 
	 *			   the pixel will be ignored when maintaining the joint-histogram. This is useful for applications like optical flow occlusion handling.
	 *
	 * Note:
	 *		1. When feature image clustering (when F is 3-channel) OR adaptive quantization (when I is floating point image) is 
	 *         performed, the result is an approximation. To increase the accuracy, using a larger "nI" or "nF" will help. 
	 *
	 */
	/***************************************************************/

	static Mat filter(Mat &I, Mat &feature, int r, float sigma=25.5, int nI=256, int nF=256, int iter=1, string weightType="exp", Mat mask=Mat());

	/***************************************************************/
	/* Function: filterCore
	 * 
	 * Description: filter core implementation only containing joint-histogram weighted median framework
	 * 
	 * input arguments:
	 *			I: input image. Only accept CV_32S type.
	 *          F: feature image. Only accept CV_32S type.
	 *       wMap: a 2D array that defines the distance between each pair of feature values. wMap[i][j] is the weight between feature value "i" and "j".
	 *          r: radius of filtering kernel, should be a positive integer.
	 *         nI: # of possible values in I, i.e., all values of I should in range [0, nI)
	 *         nF: # of possible values in F, i.e., all values of F should in range [0, nF)
	 *		 mask: a 0-1 mask that has the same size with I, for ignoring the effect of some pixels, as introduced in function "filter"
	 */
	/***************************************************************/

	static Mat filterCore(Mat &I, Mat &F, float **wMap, int r=20, int nF=256, int nI=256, Mat mask=Mat());

private:

	/***************************************************************/
	/* Function: updateBCB
	 * Description: maintain the necklace table of BCB
	/***************************************************************/
	static inline void updateBCB(int &num,int *f,int *b,int i,int v);

	/***************************************************************/
	/* Function: float2D
	 * Description: allocate a 2D float array with dimension "dim1 x dim2"
	/***************************************************************/
	static float** float2D(int dim1, int dim2);

	/***************************************************************/
	/* Function: float2D_release
	 * Description: deallocate the 2D array created by float2D()
	/***************************************************************/
	static void float2D_release(float** p);

	/***************************************************************/
	/* Function: int2D
	 * Description: allocate a 2D integer array with dimension "dim1 x dim2"
	/***************************************************************/
	static int** int2D(int dim1, int dim2);

	/***************************************************************/
	/* Function: int2D_release
	 * Description: deallocate the 2D array created by int2D()
	/***************************************************************/
	static void int2D_release(int** p);

	/***************************************************************/
	/* Function: featureIndexing
	 * Description: convert uchar feature image "F" to CV_32SC1 type. 
	 *				If F is 3-channel, perform k-means clustering
	 *				If F is 1-channel, only perform type-casting
	/***************************************************************/
	static void featureIndexing(Mat &F, float **&wMap, int &nF, float sigmaI, string weightType);

	/***************************************************************/
	/* Function: from32FTo32S
	 * Description: adaptive quantization for changing a floating-point 1D image to integer image.
	 *				The adaptive quantization strategy is based on binary search, which searches an 
	 *				upper bound of quantization error.
	 *				The function also return a mapping between quantized value (32F) and quantized index (32S).
	 *				The mapping is used to convert integer image back to floating-point image after filtering.
	/***************************************************************/
	static void from32FTo32S(Mat &img, Mat &outImg, int nI, float *mapping);

	/***************************************************************/
	/* Function: from32STo32F
	 * Description: convert the quantization index image back to the floating-point image accroding to the mapping
	/***************************************************************/
	static void from32STo32F(Mat &img, Mat &outImg, float *mapping);
};

#endif