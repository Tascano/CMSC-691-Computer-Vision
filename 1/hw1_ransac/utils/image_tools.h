#ifndef canny_edge_h
#define canny_edge_h

#include "math.h"
#include "float.h"
#include "string.h"

#include "stb_image.h"
#include "stb_image_write.h"
#include "data_types.h"

Mat8U* load_image(const char *fname);
void save_rgb_image(Mat8U *inputImg, const char *fname);
void save_gray_image(Mat8U *inputImg, const char *fname);
Mat8U *draw_line(Mat8U *inputImg, pt2i fstPoint, pt2i sndPoint, uchar *color);

Mat8U* clone_mat_8u(const Mat8U *inputMat);
Mat16S* clone_mat_16s(const Mat16S *inputMat);
Mat8U* copy_mat_8u(const Mat8U *inputMat);
Mat16S* init_mat_zeros_16S(const int width, const int height, const int depth);
Mat8U* init_mat_zeros_8U(const int width, const int height, const int depth);
Mat16S* init_mat_ones_16S(const int width, const int height, const int depth);
Mat8U* init_mat_ones_8U(const int width, const int height, const int depth);
Mat32F *init_mat_zeros_32F(const int width, const int height, const int depth);

KernelData* compute_gaussian_kernel(const float sigma, const int kernel_size);
Mat8U *convolution_8u(const Mat8U *src_img, const KernelData *kernel);
Mat16S *convolution_16s(const Mat8U *inputMat, const KernelData *kernel);

Mat32F* edge_detection(Mat8U* inputMat, float dge_direction[]);
Mat32F* non_max_suppress(Mat32F *edge_magnitude, float *edge_direction);
Mat8U *double_threshold(const Mat32F *inputMat, const unsigned int lowThreshold, const unsigned int highThreshold);

Mat32F* hysteresis_recursion(Mat32F* inputMat, long x, long y, int lowThreshold);
Mat8U *edges_tracing(Mat32F *src_img, const int lowThreshold, const int highThreshold);
Mat8U *simple_edge_tracing(const Mat32F *inputMat, const unsigned int lowThreshold, const unsigned int highThreshold);


Mat8U* rgb2gray(const Mat8U* src_img);
Mat8U *gaussian_blur(const Mat8U *gray_img);

Mat8U* canny(const char* image);
TPoint* get_points(Mat8U *inputImg, int* num_points);

void debug_gaussian_kernel(KernelData* kernel);

#endif