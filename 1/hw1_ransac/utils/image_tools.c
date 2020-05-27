#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "image_tools.h"

KernelData *compute_gaussian_kernel(const float sigma, const int kernel_size)
{
	if (kernel_size % 2 != 1)
	{
		printf("[ERROR] compute_gaussian_kernel- Kernel is 2k + 1\n");
		return NULL;
	}
	float sum = 0.0;
	KernelData *kernel = (KernelData *)malloc(sizeof(KernelData));
	kernel->sigma = sigma;
	kernel->kernel_size = kernel_size;
	kernel->data = (float *)malloc(sizeof(float) * kernel_size * kernel_size);
	int ksize = kernel_size >> 1;
	for (int i = 1; i <= kernel_size; i++)
	{
		for (int j = 1; j <= kernel_size; j++)
		{
			float H = exp(-((pow(i - (ksize + 1), 2) + pow(j - (ksize + 1), 2)) / (2 * sigma * sigma))) / (2 * M_PI * pow(sigma, 2));
			kernel->data[(i - 1) * kernel_size + (j - 1)] = H;
			sum += H;
		}
	}
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			kernel->data[i * kernel_size + j] /= sum;
		}
	}
	return kernel;
}
Mat8U *convolution_8u(const Mat8U *src_img, const KernelData *kernel)
{
	if (!src_img)
	{
		printf("[ERROR] convolution2D - Please check the input image data");
		exit(-1);
	}
	if (!kernel)
	{
		printf("[ERROR] convolution2D - Input Gaussian kernel is invalid\n");
		exit(-1);
	}
	Mat8U *dest_img = clone_mat_8u(src_img);
	float sum = 0.0;
	int kCenter = kernel->kernel_size / 2;

	for (int i = 0; i < dest_img->height; ++i)
	{
		for (int j = 0; j < dest_img->width; ++j)
		{
			sum = 0.0;
			for (int m = 0; m < kernel->kernel_size; ++m)
			{
				int mm = kernel->kernel_size - 1 - m;

				for (int n = 0; n < kernel->kernel_size; ++n)
				{
					int nn = kernel->kernel_size - 1 - n;
					int rowIndex = i + (kCenter - mm);
					int colIndex = j + (kCenter - nn);
					if (rowIndex >= 0 && rowIndex < dest_img->height && colIndex >= 0 && colIndex < dest_img->width)
					sum += src_img->data[dest_img->width * rowIndex + colIndex] * kernel->data[kernel->kernel_size * mm + nn];
				}
			}
			dest_img->data[dest_img->width * i + j] = (uchar)((float)fabs(sum) + 0.5f);
		}
	}
	return dest_img;
}

Mat16S *convolution_16s(const Mat8U *inputMat, const KernelData *kernel)
{
	if (!inputMat)
	{
		printf("[ERROR] convolution2D - Please check the input image data");
		exit(-1);
	}
	if (!kernel)
	{
		printf("[ERROR] convolution2D - Input Gaussian kernel is invalid\n");
		exit(-1);
	}

	Mat16S *outputMat = init_mat_zeros_16S(inputMat->width, inputMat->height, 1);

	float sum = 0.0;
	int kCenter = kernel->kernel_size / 2;
    char16 *inPtr = (char16 *)&inputMat->data[inputMat->width * kCenter + kCenter];
	char16 *inPtr2 = (char16 *)&inputMat->data[inputMat->width * kCenter + kCenter];
	char16 *outPtr = outputMat->data;
	float *kPtr = kernel->data;
    for (int i = 0; i < inputMat->height; ++i)
	{
		int rowMax = i + kCenter;
		int rowMin = i - inputMat->height + kCenter;
        for (int j = 0; j < inputMat->width; ++j)
		{
			int colMax = j + kCenter;
			int colMin = j - inputMat->width + kCenter;
            sum = 0.0;
			for (int m = 0; m < kernel->kernel_size; ++m)
			{
			if (m <= rowMax && m > rowMin)
				{
					for (int n = 0; n < kernel->kernel_size; ++n)
					{
					    if (n <= colMax && n > colMin)
							sum += *(inPtr - n) * *kPtr;
                        ++kPtr;
					}
				}
				else
					kPtr += kernel->kernel_size;
                inPtr -= inputMat->width;
			}

			if (sum >= 0)
				*outPtr = (char16)(sum + 0.5f);
			else
				*outPtr = (char16)(sum - 0.5f);

			kPtr = kernel->data;
			inPtr = ++inPtr2;
			++outPtr;
		}
	}

	return outputMat;
}

void debug_gaussian_kernel(KernelData *kernel)
{
	for (int i = 0; i < kernel->kernel_size; i++)
	{
		printf("[ ");
		for (int j = 0; j < kernel->kernel_size; j++)
		{
			printf("%f ", kernel->data[i * kernel->kernel_size + j]);
		}
		printf("]\n");
	}
}
Mat8U *gaussian_blur(const Mat8U *gray_img)
{

	if (!gray_img)
	{
		printf("[ERROR] gaussian_blur func - Invalid input image\n");
		exit(1);
	}

	if (gray_img->depth > 1)
	{
		printf("[ERROR] gaussian_blur func - Input image need a gray image\n");
		exit(1);
	}

	int KERNEL_SIZE = 5;
	float SIGMA = 1.0;
	KernelData *kernel = compute_gaussian_kernel(SIGMA, KERNEL_SIZE);
	Mat8U *blurImg = convolution_8u(gray_img, kernel);

	free(kernel);
	return blurImg;
}

Mat32F *edge_detection(Mat8U *inputMat, float edge_direction[])
{
	float Gx[9] = {1.0, 0.0, -1.0,
				   2.0, 0.0, -2.0,
				   1.0, 0.0, -1.0};
	float Gy[9] = {1.0, 2.0, 1.0,
				   0.0, 0.0, 0.0,
				   -1.0, -2.0, -1.0};

	float value_gx = 0.0;
	float value_gy = 0.0;
	float value_max = 0.0;

	int width = inputMat->width;
	int height = inputMat->height;

	Mat32F *outputMat = init_mat_zeros_32F(width, height, 1);

	float angle = 0.0;
	int pad_offset = 2;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			value_gx = 0.0;
			value_gy = 0.0;
			if (i > pad_offset && i < height - pad_offset && j > pad_offset && j < width - pad_offset)
			{
				for (int k = 0; k < 3; k++)
				{
					for (int l = 0; l < 3; l++)
					{
						int xn = j + k - 1;
						int yn = i + l - 1;
						int index = xn + yn * width;
						value_gx += Gx[3 * k + l] * inputMat->data[index];
						value_gy += Gy[3 * k + l] * inputMat->data[index];
					}
				}
			}

			outputMat->data[i * width + j] = (float)hypot(value_gx, value_gy);
			value_max = outputMat->data[i * width + j] > value_max ? outputMat->data[i * width + j] : value_max;

			if ((value_gy != 0.0) || (value_gx != 0.0))
			{
				angle = atan2(value_gy, value_gx) * 180.0 / M_PI;
			}
			else
			{
				angle = 0.0;
			}
			if (((angle > -22.5) && (angle <= 22.5)) || ((angle > 157.5) && (angle <= -157.5)))
			{
				edge_direction[i * width + j] = 0;
			}
			else if (((angle > 22.5) && (angle <= 67.5)) || ((angle > -157.5) && (angle <= -112.5)))
			{
				edge_direction[i * width + j] = 45;
			}
			else if (((angle > 67.5) && (angle <= 112.5)) || ((angle > -112.5) && (angle <= -67.5)))
			{
				edge_direction[i * width + j] = 90;
			}
			else if (((angle > 112.5) && (angle <= 157.5)) || ((angle > -67.5) && (angle <= -22.5)))
			{
				edge_direction[i * width + j] = 135;
			}
		}
	}
	return outputMat;
}

Mat32F *non_max_suppress(Mat32F *edge_magnitude, float *edge_direction)
{
	if (!edge_magnitude)
	{
		printf("[ERROR] Non_max_suppress func - Invalid magnitude data\n");
		exit(1);
	}

	if (!edge_direction)
	{
		printf("[ERROR] Non_max_suppress func - Invalid direction data\n");
		exit(1);
	}

	int width = edge_magnitude->width;
	int height = edge_magnitude->height;

	Mat32F *outputMat = init_mat_zeros_32F(width, height, 1);

	float pixel1 = 0.0;
	float pixel2 = 0.0;
	// Non-maximum suppression, straightforward implementation.
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (edge_direction[i * width + j] == 0)
			{
				pixel1 = edge_magnitude->data[(i + 1) * width + j];
				pixel2 = edge_magnitude->data[(i - 1) * width + j];
			}
			else if (edge_direction[i * width + j] == 45)
			{
				pixel1 = edge_magnitude->data[(i + 1) * width + j - 1];
				pixel2 = edge_magnitude->data[(i - 1) * width + j + 1];
			}
			else if (edge_direction[i * width + j] == 90)
			{
				pixel1 = edge_magnitude->data[i * width + j - 1];
				pixel2 = edge_magnitude->data[i * width + j + 1];
			}
			else if (edge_direction[i * width + j] == 135)
			{
				pixel1 = edge_magnitude->data[(i + 1) * width + j + 1];
				pixel2 = edge_magnitude->data[(i - 1) * width + j - 1];
			}
			float pixel = edge_magnitude->data[i * width + j];
			if ((pixel >= pixel1) && (pixel > pixel2))
			{
				outputMat->data[i * width + j] = pixel;
			}
			else
			{
				outputMat->data[i * width + j] = 0.0;
			}
		}
	}

	return outputMat;
}

Mat8U *double_threshold(const Mat32F *inputMat, const unsigned int lowThreshold, const unsigned int highThreshold)
{
	if (!inputMat)
	{
		printf("[ERROR] canny func - Could not load the image\n");
		exit(1);
	}

	int width = inputMat->width;
	int height = inputMat->height;

	Mat8U *outputMat = init_mat_zeros_8U(width, height, 1);
	const unsigned int weak = 55;
	const unsigned int strong = 255;

	float in_value = 0.0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			in_value = inputMat->data[i * width + j];
			if (in_value >= highThreshold)
			{
				outputMat->data[i * width + j] = strong;
			}
			else if (in_value >= lowThreshold && in_value < highThreshold)
			{
				outputMat->data[i * width + j] = weak;
			}
			else if (in_value < lowThreshold)
			{
				outputMat->data[i * width + j] = 0;
			}
		}
	}
	return outputMat;
}

Mat8U *simple_edge_tracing(const Mat32F *inputMat, const unsigned int lowThreshold, const unsigned int highThreshold)
{
	if (!inputMat)
	{
		printf("[ERROR] simple_edge_tracing func, Invalid input data\n");
		exit(1);
	}
	Mat8U *outputImg = double_threshold(inputMat, lowThreshold, highThreshold);
	int width = inputMat->width;
	int height = inputMat->height;
	const int weak = 55;
	const int strong = 255;

	for (int i = 1; i < height - 1; i++)
	{
		uchar *src_ptr = &outputImg->data[i * width + 1];

		for (int j = 1; j < width - 1; j++)
		{
			uchar *center_ptr = src_ptr;
			uchar *north_ptr = center_ptr - width;
			uchar *south_ptr = center_ptr + width;
			uchar *west_ptr = center_ptr + 1;
			uchar *east_ptr = center_ptr - 1;
			uchar *nw_ptr = north_ptr + 1;
			uchar *ne_ptr = north_ptr - 1;
			uchar *sw_ptr = south_ptr + 1;
			uchar *se_ptr = south_ptr - 1;
			if (*src_ptr == weak)
			{
				if (*center_ptr == strong || *north_ptr == strong || *south_ptr == strong || *west_ptr == strong || *east_ptr == strong || *nw_ptr == strong || *ne_ptr == strong || *sw_ptr == strong || *sw_ptr == strong || *se_ptr == strong)
				{
					*src_ptr = strong;
				}
				else
				{
					*src_ptr = 0;
				}
			}
			src_ptr++;
		}
	}
	return outputImg;
}

Mat32F *hysteresis_recursion(Mat32F *inputMat, long x, long y, int lowThreshold)
{
	int value = 0;
	int width = inputMat->width;
	int height = inputMat->height;

	for (long x1 = x - 1; x1 <= x + 1; x1++)
	{
		for (long y1 = y - 1; y1 <= y + 1; y1++)
		{
			if ((x1 < height) & (y1 < width) & (x1 >= 0) & (y1 >= 0) & (x1 != x) & (y1 != y))
			{

				value = inputMat->data[x1 * width + y1];
				if (value != 255)
				{
					if (value >= lowThreshold)
					{
						inputMat->data[x1 * width + y1] = MAX_BRIGHTNESS;
						hysteresis_recursion(inputMat, x1, y1, lowThreshold);
					}
					else
					{
						inputMat->data[x1 * width + y1] = 0;
					}
				}
			}
		}
	}
	return inputMat;
}

Mat8U *edges_tracing(Mat32F *src_img, const int lowThreshold, const int highThreshold)
{
	if (!src_img)
	{
		printf("[ERROR] tracing_edge func, Invalid input data\n");
		exit(1);
	}

	int width = src_img->width;
	int height = src_img->height;
	Mat8U *outputImg = init_mat_zeros_8U(width, height, 1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src_img->data[i * width + j] >= highThreshold)
			{
				outputImg->data[i * width + j] = MAX_BRIGHTNESS;
				src_img = hysteresis_recursion(src_img, i, j, lowThreshold);
			}
		}
	}

	return outputImg;
}

Mat8U *canny(const char *image_path)
{
	Mat8U *in_img = load_image(image_path);
	if (!in_img)
	{
		printf("[ERROR] canny func - Could not load the image\n");
		return NULL;
	}
	int width = in_img->width;
	int height = in_img->height;

	Mat8U *gray_img = rgb2gray(in_img);

	Mat8U *blur_img = gaussian_blur(gray_img);


	float *edge_direction = (float *)malloc(sizeof(float) * width * height);
	Mat32F *edge_magnitude = edge_detection(blur_img, edge_direction);

	Mat8U *edges = simple_edge_tracing(edge_magnitude, 50, 100);


	free(in_img);
	free(gray_img);
	free(blur_img);
	free(edge_direction);
	free(edge_magnitude);
	return edges;
}

TPoint *get_points(Mat8U *inputImg, int *num_points)
{
	if (!inputImg){
		printf("[ERROR] get_points - Invalid input \n");
		return NULL;
	}
	*num_points = 0;
	for (int i = 0; i < inputImg->height; i++)
	{
		for (int j = 0; j < inputImg->width; j++)
		{
			if (inputImg->data[j + i * inputImg->width] > 0)
			{
				(*num_points)++;
			}
		}
	}

	TPoint *point_list = (TPoint *)malloc(sizeof(*point_list) * (*num_points));

	int count = 0;
	for (int i = 0; i < inputImg->height; i++)
	{
		for (int j = 0; j < inputImg->width; j++)
		{
			if (inputImg->data[j + i * inputImg->width] > 0)
			{
				point_list[count].x = (double)j;
				point_list[count].y = (double)i;
				count++;
			}
		}
	}
	if (count == 0)
		return NULL;
	return point_list;
}
Mat8U *load_image(const char *fname)
{
	int img_w = 0;
	int img_h = 0;
	int img_chan = 0;

	uchar *data = stbi_load(fname, &img_w, &img_h, &img_chan, 3);
	if (!data)
	{
		printf("[ERROR] LoadImage func - Could not load the image\n");
		exit(1);
	}

	Mat8U *img = (Mat8U *)malloc(sizeof(Mat8U));
	if (img_chan > 3 || img_chan < 2)
	{
		printf("[ERROR] LoadImage func : expected 3 channels (red green blue)\n");
		exit(1);
	}
	else if (img_chan == 3)
	{
		img->data = (uchar *)malloc(img_w * img_h * img_chan);
		memcpy(img->data, data, img_w * img_h * img_chan);
	}
	else if (img_chan == 1)
	{
		img->data = (uchar *)malloc(img_w * img_h);
		memcpy(img->data, data, img_w * img_h);
	}

	img->width = img_w;
	img->height = img_h;
	img->depth = img_chan;
	free(data);
	return img;
}

Mat8U *clone_mat_8u(const Mat8U *inputMat)
{
	if (!inputMat)
	{
		printf("[ERROR] CloneMat8U func - Invalid input mat data\n");
		exit(-1);
	}
	Mat8U *outpuMat = (Mat8U *)malloc(sizeof(Mat8U));
	outpuMat->data = (uchar *)malloc(inputMat->width * inputMat->height * inputMat->depth);
	memset(outpuMat->data, 0, inputMat->width * inputMat->height * inputMat->depth);
	outpuMat->width = inputMat->width;
	outpuMat->height = inputMat->height;
	outpuMat->depth = inputMat->depth;
	return outpuMat;
}

Mat16S *clone_mat_16s(const Mat16S *inputMat)
{
	if (!inputMat)
	{
		printf("[ERROR] clone_mat_16s func - Invalid input mat data\n");
		exit(-1);
	}
	Mat16S *outpuMat = (Mat16S *)malloc(sizeof(Mat16S));
	outpuMat->data = (char16 *)malloc(inputMat->width * inputMat->height * inputMat->depth);
	memset(outpuMat->data, 0, inputMat->width * inputMat->height * inputMat->depth);
	outpuMat->width = inputMat->width;
	outpuMat->height = inputMat->height;
	outpuMat->depth = inputMat->depth;
	return outpuMat;
}

Mat8U *copy_mat_8u(const Mat8U *inputMat)
{
	if (!inputMat)
	{
		printf("[ERROR] copy_mat_8u func - Invalid input image\n");
		exit(-1);
	}

	Mat8U *outputMat = (Mat8U *)malloc(sizeof(Mat8U));
	outputMat->data = (uchar *)malloc(inputMat->width * inputMat->height * inputMat->depth);
	outputMat->width = inputMat->width;
	outputMat->height = inputMat->height;
	outputMat->depth = inputMat->depth;

	if (inputMat->depth == 1)
	{
		for (int i = 0; i < inputMat->height; i++)
		{
			uchar *ptr = &outputMat->data[i * inputMat->width];
			uchar *src_ptr = &inputMat->data[i * inputMat->width];
			for (int j = 0; j < inputMat->width; j++)
			{
				*ptr = *src_ptr;
				ptr++;
				src_ptr++;
			}
		}
	}
	if (inputMat->depth == 3)
	{
		for (int i = 0; i < inputMat->height; i++)
		{
			for (int j = 0; j < inputMat->width; j++)
			{
				outputMat->data[3 * (i * inputMat->width + j) + 2] = inputMat->data[3 * (i * inputMat->width + j) + 2];
				outputMat->data[3 * (i * inputMat->width + j) + 1] = inputMat->data[3 * (i * inputMat->width + j) + 1];
				outputMat->data[3 * (i * inputMat->width + j) + 0] = inputMat->data[3 * (i * inputMat->width + j) + 0];
			}
		}
	}

	return outputMat;
}

Mat16S *copy_mat_16s(const Mat16S *inputMat)
{
	if (!inputMat)
	{
		printf("[ERROR] copy_mat_16s func - Invalid input image\n");
		exit(-1);
	}

	Mat16S *outputMat = (Mat16S *)malloc(sizeof(Mat16S));
	outputMat->data = (char16 *)malloc(inputMat->width * inputMat->height * inputMat->depth);
	outputMat->width = inputMat->width;
	outputMat->height = inputMat->height;
	outputMat->depth = inputMat->depth;

	if (inputMat->depth == 1)
	{
		for (int i = 0; i < inputMat->height; i++)
		{
			char16 *ptr = &outputMat->data[i * inputMat->width];
			char16 *src_ptr = &inputMat->data[i * inputMat->width];
			for (int j = 0; j < inputMat->width; j++)
			{
				*ptr = *src_ptr;
				ptr++;
				src_ptr++;
			}
		}
	}
	if (inputMat->depth == 3)
	{
		for (int i = 0; i < inputMat->height; i++)
		{
			for (int j = 0; j < inputMat->width; j++)
			{
				outputMat->data[3 * (i * inputMat->width + j) + 2] = inputMat->data[3 * (i * inputMat->width + j) + 2];
				outputMat->data[3 * (i * inputMat->width + j) + 1] = inputMat->data[3 * (i * inputMat->width + j) + 1];
				outputMat->data[3 * (i * inputMat->width + j) + 0] = inputMat->data[3 * (i * inputMat->width + j) + 0];
			}
		}
	}

	return outputMat;
}

Mat16S *init_mat_zeros_16S(const int width, const int height, const int depth)
{
	Mat16S *outputMat = (Mat16S *)malloc(sizeof(Mat16S));
	outputMat->data = (char16 *)malloc(sizeof(char16) * width * height * depth);
	memset(outputMat->data, 0, sizeof(char16) * width * height * depth);
	outputMat->width = width;
	outputMat->height = height;
	outputMat->depth = depth;
	return outputMat;
}

Mat32F *init_mat_zeros_32F(const int width, const int height, const int depth)
{
	Mat32F *outputMat = (Mat32F *)malloc(sizeof(Mat32F));
	outputMat->data = (float *)malloc(sizeof(float) * width * height * depth);
	memset(outputMat->data, 0, sizeof(float) * width * height * depth);
	outputMat->width = width;
	outputMat->height = height;
	outputMat->depth = depth;
	return outputMat;
}
Mat8U *init_mat_zeros_8U(const int width, const int height, const int depth)
{
	Mat8U *outputMat = (Mat8U *)malloc(sizeof(Mat8U));
	outputMat->data = (uchar *)malloc(sizeof(uchar) * width * height * depth);
	memset(outputMat->data, 0, sizeof(uchar) * width * height * depth);
	outputMat->width = width;
	outputMat->height = height;
	outputMat->depth = depth;
	return outputMat;
}
Mat16S *init_mat_ones_16S(const int width, const int height, const int depth)
{
	Mat16S *outputMat = (Mat16S *)malloc(sizeof(Mat16S));
	outputMat->data = (char16 *)malloc(sizeof(char16) * width * height * depth);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			*(outputMat->data) = 1;
			outputMat->data++;
		}
	}
	outputMat->width = width;
	outputMat->height = height;
	outputMat->depth = depth;
	return outputMat;
}
Mat8U *init_mat_ones_8U(const int width, const int height, const int depth)
{
	Mat8U *outputMat = (Mat8U *)malloc(sizeof(Mat8U));
	outputMat->data = (uchar *)malloc(sizeof(uchar) * width * height * depth);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			*(outputMat->data) = 1;
			outputMat->data++;
		}
	}
	outputMat->width = width;
	outputMat->height = height;
	outputMat->depth = depth;
	return outputMat;
}

// https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
Mat8U *rgb2gray(const Mat8U *src_img)
{
	if (!src_img || !src_img->data)
	{
		printf("[ERROR] rgb2gray func - Invalid input image\n");
		exit(1);
	}

	Mat8U *dest_img = (Mat8U *)malloc(sizeof(Mat8U));
	dest_img->data = (uchar *)malloc(src_img->width * src_img->height);
	dest_img->width = src_img->width;
	dest_img->height = src_img->height;
	dest_img->depth = 1;

	for (int y = 0; y < dest_img->height; y++)
	{
		for (int x = 0; x < dest_img->width; x++)
		{
			uchar r = src_img->data[3 * (y * src_img->width + x) + 2];
			uchar g = src_img->data[3 * (y * src_img->width + x) + 1];
			uchar b = src_img->data[3 * (y * src_img->width + x) + 0];
			dest_img->data[x + y * src_img->width] = (uchar)(0.299 * r + 0.587 * g + 0.114 * b);
		}
	}
	return dest_img;
}

void save_rgb_image(Mat8U *inputImg, const char *fname)
{
	if (!inputImg)
	{
		printf("[ERROR] save_rgb_image func - Invalid input image\n");
		exit(1);
	}
	if (inputImg->depth != 3)
	{
		printf("[ERROR] save_gray_image func - Input image is not the color image\n");
		exit(1);
	}
	int width = inputImg->width;
	int height = inputImg->height;

	stbi_write_png(fname, width, height, 3, inputImg->data, width * 3);
	printf("Saved rgb image: %s %d x %d (w x h)\n", fname, width, height);
}

void save_gray_image(Mat8U *inputImg, const char *fname)
{
	if (!inputImg)
	{
		printf("[ERROR] save_gray_image func - Invalid input image\n");
		exit(1);
	}

	if (inputImg->depth > 1)
	{
		printf("[ERROR] save_gray_image func - Input image is not gray\n");
		exit(1);
	}

	int width = inputImg->width;
	int height = inputImg->height;
	int y, x, c, i = 0;
	uchar *data = malloc(width * height * 3);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			data[i++] = inputImg->data[y * width + x];
			data[i++] = inputImg->data[y * width + x];
			data[i++] = inputImg->data[y * width + x];
		}
	}
	stbi_write_png(fname, width, height, 3, data, width * 3);
	free(data);
	printf("Saved gray image: %s %d x %d (w x h)\n", fname, width, height);
}

Mat8U *draw_line(Mat8U *inputImg, pt2i fstPoint, pt2i sndPoint, uchar *color)
{
	if (!inputImg)
	{
		printf("[ERROR] save_gray_image func - Input image is not gray\n");
		exit(1);
	}

	if (fstPoint.x < 0)
		fstPoint.x = 0;
	if (fstPoint.y < 0)
		fstPoint.y = 0;
	if (sndPoint.x < 0)
		sndPoint.x = 0;
	if (sndPoint.y < 0)
		sndPoint.y = 0;
	if (fstPoint.x >= inputImg->width)
	{
		fstPoint.x = inputImg->width - 1;
	}
	if (fstPoint.y >= inputImg->height)
	{
		fstPoint.y = inputImg->height - 1;
	}
	if (sndPoint.x >= inputImg->width)
	{
		sndPoint.x = inputImg->width - 1;
	}
	if (sndPoint.y >= inputImg->height)
	{
		sndPoint.y = inputImg->height - 1;
	}

	Mat8U *outputImg = copy_mat_8u(inputImg);
	int dx = abs(sndPoint.x - fstPoint.x);
	int sx = fstPoint.x < sndPoint.x ? 1 : -1;
	int dy = abs(sndPoint.y - fstPoint.y);
	int sy = fstPoint.y < sndPoint.y ? 1 : -1;
	int err = (dx > dy ? dx : -dy) / 2;
	int e2 = 0;
	while (true)
	{
		if (inputImg->depth == 3)
		{
			outputImg->data[3 * (fstPoint.y * inputImg->width + fstPoint.x) + 2] = color[0]; //red
			outputImg->data[3 * (fstPoint.y * inputImg->width + fstPoint.x) + 1] = color[1]; //green
			outputImg->data[3 * (fstPoint.y * inputImg->width + fstPoint.x) + 0] = color[2]; //blue
		}
		else if (inputImg->depth == 1)
		{
			outputImg->data[fstPoint.y * inputImg->width + fstPoint.x] = 255;
		}
		else
		{
			printf("[ERROR] draw_line func - Invalid input image\n");
			break;
		}
		if (fstPoint.x == sndPoint.x && fstPoint.y == sndPoint.y)
			break;
		e2 = err;
		if (e2 > -dx)
		{
			err -= dy;
			fstPoint.x += sx;
		}
		if (e2 < dy)
		{
			err += dx;
			fstPoint.y += sy;
		}
	}
	return outputImg;
}