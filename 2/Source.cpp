/*
 Reference: Source code of SIFT is taken from OpenCV Contribution on Github:
 https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/sift.cpp

 A lot of my refernce regarding coding and using OpenCV was from this really good resource :
 https://www.opencv-srf.com/p/introduction.html

 Function References:
 https://thispointer.com/c-check-if-given-path-is-a-file-or-directory-using-boost-c17-filesystem-library/

 Please note:
 I use <filesystem> (#include <filesystem>; namespace fs = std::filesystem) to have to make Vistual Studio understands c++17

 Alternate use for Ubuntu is using <dirent.h> library function
 (https://stackoverflow.com/questions/2089514/how-to-know-and-load-all-images-in-a-specific-folder)
 */

#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/utils/tls.hpp>
#include <opencv2/calib3d.hpp>
#include <stdarg.h>

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;

//FeatureManipulation
class FeatureManipulator
{
public:
	FeatureManipulator() {
		cv::namedWindow(this->namedTemplate, cv::WINDOW_FREERATIO);
		cv::namedWindow(this->namedOutputWindows, cv::WINDOW_FREERATIO);
	}

	FeatureManipulator(string imagePath) {
		this->imgPathFolder = imagePath;
		cv::namedWindow(this->namedTemplate, cv::WINDOW_FREERATIO);
		cv::namedWindow(this->namedOutputWindows, cv::WINDOW_FREERATIO);
	}
	~FeatureManipulator() {
		cv::destroyAllWindows();
	}

	void setImageFolderPath(string folderPath) {
		this->imgPathFolder = folderPath;
	}
	bool run();

private:
	string namedOutputWindows = "Output";

	string namedTemplate = "Template";
	string imgPathFolder = "";
	int frameRate = 30;
	cv::Mat frame;
	bool checkIfDirectory(const string filePath);
	bool checkIfFile(const string fileName);
	bool listingFileInFolder(const string folderPath, vector<string>& fileList);
	bool detectFeaturePts(const cv::Mat image, vector<cv::Point2f>& corners);
	bool detectFeatureDesciptorPts(const cv::Mat image, vector<cv::Point2f>& corners, vector<cv::KeyPoint>& keyPoints, cv::Mat& desc);
	bool mactchingFeaturePts(vector<cv::KeyPoint> keyPt1, vector<cv::KeyPoint> keyPt2, cv::Mat desc1, cv::Mat desc2, vector<cv::DMatch>& goodMatches);
	bool transformObjects(vector<cv::KeyPoint> keyPts1, vector<cv::KeyPoint> keyPts2, vector<cv::DMatch> goodMatches, vector<cv::Point2f> box_corners, vector<cv::Point2f>& new_box_corners);
	static void MouseCallbackEvent(int event, int x, int y, int flags, void* userdata);
	bool drawBox(cv::Mat& image, cv::Point pt1, cv::Point pt3, cv::Scalar color);
	bool drawBox(cv::Mat& image, vector<cv::Point2f>pts, cv::Scalar color);
};
// FeatureManipulator class End

int main()
{
	FeatureManipulator featureMatching("pictureframes");
	featureMatching.run();

	return 0;
}

typedef struct _MouseData {
	bool isMouseDown = false;
	cv::Point pt1 = cv::Point(-1, -1);
	cv::Point pt2 = cv::Point(-1, -1);
	cv::Point pt3 = cv::Point(-1, -1);
	cv::Point pt4 = cv::Point(-1, -1);
	bool bDraw = false;
}MouseData;

#define GBLUR_BLSIZE_X 5 
#define GBLUR_BLSIZE_Y 5
#define SOBEL_KSIZE 3
#define HARRIS_K_FREE 0.04 
#define LOW_CANNY_THRESHOLD 10
#define HIGH_CANNY_THRESHOLD 200
#define POS_OF_TEMPLATE_FRAME 74
#define RADIO_THRESH 0.7f 

//  START SIFT
// taken from https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/sift.cpp

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

#define DoG_TYPE_SHORT 0
#if DoG_TYPE_SHORT
// intermediate type used for DoG pyramids
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif

class SIFT : public Feature2D
{
public:
	explicit SIFT(int nfeatures = 0, int nOctaveLayers = 3,
		double contrastThreshold = 0.04, double edgeThreshold = 10,
		double sigma = 1.6);

	//Ptr<SIFT> create(int _nfeatures, int _nOctaveLayers,
	//	double _contrastThreshold, double _edgeThreshold, double _sigma);

	//! returns the descriptor size in floats (128)
	int descriptorSize() const CV_OVERRIDE;

	//! returns the descriptor type
	int descriptorType() const CV_OVERRIDE;

	//! returns the default norm type
	int defaultNorm() const CV_OVERRIDE;

	//! finds the keypoints and computes descriptors for them using SIFT algorithm.
	//! Optionally it can compute descriptors for the user-provided keypoints
	void detectAndCompute(InputArray img, InputArray mask,
		std::vector<KeyPoint>& keypoints,
		OutputArray descriptors,
		bool useProvidedKeypoints = false) CV_OVERRIDE;

	void buildGaussianPyramid(const Mat& base, std::vector<Mat>& pyr, int nOctaves) const;
	void buildDoGPyramid(const std::vector<Mat>& pyr, std::vector<Mat>& dogpyr) const;
	void findScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
		std::vector<KeyPoint>& keypoints) const;

protected:
	CV_PROP_RW int nfeatures;
	CV_PROP_RW int nOctaveLayers;
	CV_PROP_RW double contrastThreshold;
	CV_PROP_RW double edgeThreshold;
	CV_PROP_RW double sigma;
};


static inline void
unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
	octave = kpt.octave & 255;
	layer = (kpt.octave >> 8) & 255;
	octave = octave < 128 ? octave : (-128 | octave);
	scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
}

static Mat createInitialImage(const Mat& img, bool doubleImageSize, float sigma)
{
	Mat gray, gray_fpt;
	if (img.channels() == 3 || img.channels() == 4)
	{
		cvtColor(img, gray, COLOR_BGR2GRAY);
		gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
	}
	else
		img.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

	float sig_diff;

	if (doubleImageSize)
	{
		sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));
		Mat dbl;
#if DoG_TYPE_SHORT
		resize(gray_fpt, dbl, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR_EXACT);
#else
		resize(gray_fpt, dbl, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR);
#endif
		GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
		return dbl;
	}
	else
	{
		sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
		GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
		return gray_fpt;
	}
}


void SIFT::buildGaussianPyramid(const Mat& base, std::vector<Mat>& pyr, int nOctaves) const
{
	std::vector<double> sig(nOctaveLayers + 3);
	pyr.resize(nOctaves * (nOctaveLayers + 3));

	// precompute Gaussian sigmas using the following formula:
	//  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
	sig[0] = sigma;
	double k = std::pow(2., 1. / nOctaveLayers);
	for (int i = 1; i < nOctaveLayers + 3; i++)
	{
		double sig_prev = std::pow(k, (double)(i - 1)) * sigma;
		double sig_total = sig_prev * k;
		sig[i] = std::sqrt(sig_total * sig_total - sig_prev * sig_prev);
	}

	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < nOctaveLayers + 3; i++)
		{
			Mat& dst = pyr[o * (nOctaveLayers + 3) + i];
			if (o == 0 && i == 0)
				dst = base;
			// base of new octave is halved image from end of previous octave
			else if (i == 0)
			{
				const Mat& src = pyr[(o - 1) * (nOctaveLayers + 3) + nOctaveLayers];
				resize(src, dst, Size(src.cols / 2, src.rows / 2),
					0, 0, INTER_NEAREST);
			}
			else
			{
				const Mat& src = pyr[o * (nOctaveLayers + 3) + i - 1];
				GaussianBlur(src, dst, Size(), sig[i], sig[i]);
			}
		}
	}
}


class buildDoGPyramidComputer : public ParallelLoopBody
{
public:
	buildDoGPyramidComputer(
		int _nOctaveLayers,
		const std::vector<Mat>& _gpyr,
		std::vector<Mat>& _dogpyr)
		: nOctaveLayers(_nOctaveLayers),
		gpyr(_gpyr),
		dogpyr(_dogpyr) { }

	void operator()(const cv::Range& range) const CV_OVERRIDE
	{
		const int begin = range.start;
		const int end = range.end;

		for (int a = begin; a < end; a++)
		{
			const int o = a / (nOctaveLayers + 2);
			const int i = a % (nOctaveLayers + 2);

			const Mat& src1 = gpyr[o * (nOctaveLayers + 3) + i];
			const Mat& src2 = gpyr[o * (nOctaveLayers + 3) + i + 1];
			Mat& dst = dogpyr[o * (nOctaveLayers + 2) + i];
			subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
		}
	}

private:
	int nOctaveLayers;
	const std::vector<Mat>& gpyr;
	std::vector<Mat>& dogpyr;
};

void SIFT::buildDoGPyramid(const std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr) const
{
	int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
	dogpyr.resize(nOctaves * (nOctaveLayers + 2));

	parallel_for_(Range(0, nOctaves * (nOctaveLayers + 2)), buildDoGPyramidComputer(nOctaveLayers, gpyr, dogpyr));
}

// Computes a gradient orientation histogram at a specified pixel
static float calcOrientationHist(const Mat& img, Point pt, int radius,
	float sigma, float* hist, int n)
{
	int i, j, k, len = (radius * 2 + 1) * (radius * 2 + 1);

	float expf_scale = -1.f / (2.f * sigma * sigma);
	AutoBuffer<float> buf(len * 4 + n + 4);
	float* X = buf.data(), * Y = X + len, * Mag = X, * Ori = Y + len, * W = Ori + len;
	float* temphist = W + len + 2;

	for (i = 0; i < n; i++)
		temphist[i] = 0.f;

	for (i = -radius, k = 0; i <= radius; i++)
	{
		int y = pt.y + i;
		if (y <= 0 || y >= img.rows - 1)
			continue;
		for (j = -radius; j <= radius; j++)
		{
			int x = pt.x + j;
			if (x <= 0 || x >= img.cols - 1)
				continue;

			float dx = (float)(img.at<sift_wt>(y, x + 1) - img.at<sift_wt>(y, x - 1));
			float dy = (float)(img.at<sift_wt>(y - 1, x) - img.at<sift_wt>(y + 1, x));

			X[k] = dx; Y[k] = dy; W[k] = (i * i + j * j) * expf_scale;
			k++;
		}
	}

	len = k;

	// compute gradient values, orientations and the weights over the pixel neighborhood
	cv::hal::exp32f(W, W, len);
	cv::hal::fastAtan2(Y, X, Ori, len, true);
	cv::hal::magnitude32f(X, Y, Mag, len);

	k = 0;

	for (; k < len; k++)
	{
		int bin = cvRound((n / 360.f) * Ori[k]);
		if (bin >= n)
			bin -= n;
		if (bin < 0)
			bin += n;
		temphist[bin] += W[k] * Mag[k];
	}

	// smooth the histogram
	temphist[-1] = temphist[n - 1];
	temphist[-2] = temphist[n - 2];
	temphist[n] = temphist[0];
	temphist[n + 1] = temphist[1];

	i = 0;

	for (; i < n; i++)
	{
		hist[i] = (temphist[i - 2] + temphist[i + 2]) * (1.f / 16.f) +
			(temphist[i - 1] + temphist[i + 1]) * (4.f / 16.f) +
			temphist[i] * (6.f / 16.f);
	}

	float maxval = hist[0];
	for (i = 1; i < n; i++)
		maxval = std::max(maxval, hist[i]);

	return maxval;
}

// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
static bool adjustLocalExtrema(const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
	int& layer, int& r, int& c, int nOctaveLayers,
	float contrastThreshold, float edgeThreshold, float sigma)
{
	const float img_scale = 1.f / (255 * SIFT_FIXPT_SCALE);
	const float deriv_scale = img_scale * 0.5f;
	const float second_deriv_scale = img_scale;
	const float cross_deriv_scale = img_scale * 0.25f;

	float xi = 0, xr = 0, xc = 0, contr = 0;
	int i = 0;

	for (; i < SIFT_MAX_INTERP_STEPS; i++)
	{
		int idx = octv * (nOctaveLayers + 2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];

		Vec3f dD((img.at<sift_wt>(r, c + 1) - img.at<sift_wt>(r, c - 1)) * deriv_scale,
			(img.at<sift_wt>(r + 1, c) - img.at<sift_wt>(r - 1, c)) * deriv_scale,
			(next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c)) * deriv_scale);

		float v2 = (float)img.at<sift_wt>(r, c) * 2;
		float dxx = (img.at<sift_wt>(r, c + 1) + img.at<sift_wt>(r, c - 1) - v2) * second_deriv_scale;
		float dyy = (img.at<sift_wt>(r + 1, c) + img.at<sift_wt>(r - 1, c) - v2) * second_deriv_scale;
		float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2) * second_deriv_scale;
		float dxy = (img.at<sift_wt>(r + 1, c + 1) - img.at<sift_wt>(r + 1, c - 1) -
			img.at<sift_wt>(r - 1, c + 1) + img.at<sift_wt>(r - 1, c - 1)) * cross_deriv_scale;
		float dxs = (next.at<sift_wt>(r, c + 1) - next.at<sift_wt>(r, c - 1) -
			prev.at<sift_wt>(r, c + 1) + prev.at<sift_wt>(r, c - 1)) * cross_deriv_scale;
		float dys = (next.at<sift_wt>(r + 1, c) - next.at<sift_wt>(r - 1, c) -
			prev.at<sift_wt>(r + 1, c) + prev.at<sift_wt>(r - 1, c)) * cross_deriv_scale;

		Matx33f H(dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);

		Vec3f X = H.solve(dD, DECOMP_LU);

		xi = -X[2];
		xr = -X[1];
		xc = -X[0];

		if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
			break;

		if (std::abs(xi) > (float)(INT_MAX / 3) ||
			std::abs(xr) > (float)(INT_MAX / 3) ||
			std::abs(xc) > (float)(INT_MAX / 3))
			return false;

		c += cvRound(xc);
		r += cvRound(xr);
		layer += cvRound(xi);

		if (layer < 1 || layer > nOctaveLayers ||
			c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER ||
			r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER)
			return false;
	}

	// ensure convergence of interpolation
	if (i >= SIFT_MAX_INTERP_STEPS)
		return false;

	{
		int idx = octv * (nOctaveLayers + 2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];
		Matx31f dD((img.at<sift_wt>(r, c + 1) - img.at<sift_wt>(r, c - 1)) * deriv_scale,
			(img.at<sift_wt>(r + 1, c) - img.at<sift_wt>(r - 1, c)) * deriv_scale,
			(next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c)) * deriv_scale);
		float t = dD.dot(Matx31f(xc, xr, xi));

		contr = img.at<sift_wt>(r, c) * img_scale + t * 0.5f;
		if (std::abs(contr) * nOctaveLayers < contrastThreshold)
			return false;

		// principal curvatures are computed using the trace and det of Hessian
		float v2 = img.at<sift_wt>(r, c) * 2.f;
		float dxx = (img.at<sift_wt>(r, c + 1) + img.at<sift_wt>(r, c - 1) - v2) * second_deriv_scale;
		float dyy = (img.at<sift_wt>(r + 1, c) + img.at<sift_wt>(r - 1, c) - v2) * second_deriv_scale;
		float dxy = (img.at<sift_wt>(r + 1, c + 1) - img.at<sift_wt>(r + 1, c - 1) -
			img.at<sift_wt>(r - 1, c + 1) + img.at<sift_wt>(r - 1, c - 1)) * cross_deriv_scale;
		float tr = dxx + dyy;
		float det = dxx * dyy - dxy * dxy;

		if (det <= 0 || tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det)
			return false;
	}

	kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
	kpt.size = sigma * powf(2.f, (layer + xi) / nOctaveLayers) * (1 << octv) * 2;
	kpt.response = std::abs(contr);

	return true;
}


class findScaleSpaceExtremaComputer : public ParallelLoopBody
{
public:
	findScaleSpaceExtremaComputer(
		int _o,
		int _i,
		int _threshold,
		int _idx,
		int _step,
		int _cols,
		int _nOctaveLayers,
		double _contrastThreshold,
		double _edgeThreshold,
		double _sigma,
		const std::vector<Mat>& _gauss_pyr,
		const std::vector<Mat>& _dog_pyr,
		TLSData<std::vector<KeyPoint> >& _tls_kpts_struct)

		: o(_o),
		i(_i),
		threshold(_threshold),
		idx(_idx),
		step(_step),
		cols(_cols),
		nOctaveLayers(_nOctaveLayers),
		contrastThreshold(_contrastThreshold),
		edgeThreshold(_edgeThreshold),
		sigma(_sigma),
		gauss_pyr(_gauss_pyr),
		dog_pyr(_dog_pyr),
		tls_kpts_struct(_tls_kpts_struct) { }
	void operator()(const cv::Range& range) const CV_OVERRIDE
	{
		const int begin = range.start;
		const int end = range.end;

		static const int n = SIFT_ORI_HIST_BINS;
		float hist[n];

		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx - 1];
		const Mat& next = dog_pyr[idx + 1];

		std::vector<KeyPoint>* tls_kpts = tls_kpts_struct.get();

		KeyPoint kpt;
		for (int r = begin; r < end; r++)
		{
			const sift_wt* currptr = img.ptr<sift_wt>(r);
			const sift_wt* prevptr = prev.ptr<sift_wt>(r);
			const sift_wt* nextptr = next.ptr<sift_wt>(r);

			for (int c = SIFT_IMG_BORDER; c < cols - SIFT_IMG_BORDER; c++)
			{
				sift_wt val = currptr[c];

				// find local extrema with pixel accuracy
				if (std::abs(val) > threshold &&
					((val > 0 && val >= currptr[c - 1] && val >= currptr[c + 1] &&
						val >= currptr[c - step - 1] && val >= currptr[c - step] && val >= currptr[c - step + 1] &&
						val >= currptr[c + step - 1] && val >= currptr[c + step] && val >= currptr[c + step + 1] &&
						val >= nextptr[c] && val >= nextptr[c - 1] && val >= nextptr[c + 1] &&
						val >= nextptr[c - step - 1] && val >= nextptr[c - step] && val >= nextptr[c - step + 1] &&
						val >= nextptr[c + step - 1] && val >= nextptr[c + step] && val >= nextptr[c + step + 1] &&
						val >= prevptr[c] && val >= prevptr[c - 1] && val >= prevptr[c + 1] &&
						val >= prevptr[c - step - 1] && val >= prevptr[c - step] && val >= prevptr[c - step + 1] &&
						val >= prevptr[c + step - 1] && val >= prevptr[c + step] && val >= prevptr[c + step + 1]) ||
						(val < 0 && val <= currptr[c - 1] && val <= currptr[c + 1] &&
							val <= currptr[c - step - 1] && val <= currptr[c - step] && val <= currptr[c - step + 1] &&
							val <= currptr[c + step - 1] && val <= currptr[c + step] && val <= currptr[c + step + 1] &&
							val <= nextptr[c] && val <= nextptr[c - 1] && val <= nextptr[c + 1] &&
							val <= nextptr[c - step - 1] && val <= nextptr[c - step] && val <= nextptr[c - step + 1] &&
							val <= nextptr[c + step - 1] && val <= nextptr[c + step] && val <= nextptr[c + step + 1] &&
							val <= prevptr[c] && val <= prevptr[c - 1] && val <= prevptr[c + 1] &&
							val <= prevptr[c - step - 1] && val <= prevptr[c - step] && val <= prevptr[c - step + 1] &&
							val <= prevptr[c + step - 1] && val <= prevptr[c + step] && val <= prevptr[c + step + 1])))
				{
					int r1 = r, c1 = c, layer = i;
					if (!adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
						nOctaveLayers, (float)contrastThreshold,
						(float)edgeThreshold, (float)sigma))
						continue;
					float scl_octv = kpt.size * 0.5f / (1 << o);
					float omax = calcOrientationHist(gauss_pyr[o * (nOctaveLayers + 3) + layer],
						Point(c1, r1),
						cvRound(SIFT_ORI_RADIUS * scl_octv),
						SIFT_ORI_SIG_FCTR * scl_octv,
						hist, n);
					float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
					for (int j = 0; j < n; j++)
					{
						int l = j > 0 ? j - 1 : n - 1;
						int r2 = j < n - 1 ? j + 1 : 0;

						if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr)
						{
							float bin = j + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2]);
							bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
							kpt.angle = 360.f - (float)((360.f / n) * bin);
							if (std::abs(kpt.angle - 360.f) < FLT_EPSILON)
								kpt.angle = 0.f;
							{
								tls_kpts->push_back(kpt);
							}
						}
					}
				}
			}
		}
	}
private:
	int o, i;
	int threshold;
	int idx, step, cols;
	int nOctaveLayers;
	double contrastThreshold;
	double edgeThreshold;
	double sigma;
	const std::vector<Mat>& gauss_pyr;
	const std::vector<Mat>& dog_pyr;
	TLSData<std::vector<KeyPoint> >& tls_kpts_struct;
};

// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
void SIFT::findScaleSpaceExtrema(const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
	std::vector<KeyPoint>& keypoints) const
{
	const int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);
	const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);

	keypoints.clear();
	TLSDataAccumulator<std::vector<KeyPoint> > tls_kpts_struct;

	for (int o = 0; o < nOctaves; o++)
		for (int i = 1; i <= nOctaveLayers; i++)
		{
			const int idx = o * (nOctaveLayers + 2) + i;
			const Mat& img = dog_pyr[idx];
			const int step = (int)img.step1();
			const int rows = img.rows, cols = img.cols;

			parallel_for_(Range(SIFT_IMG_BORDER, rows - SIFT_IMG_BORDER),
				findScaleSpaceExtremaComputer(
					o, i, threshold, idx, step, cols,
					nOctaveLayers,
					contrastThreshold,
					edgeThreshold,
					sigma,
					gauss_pyr, dog_pyr, tls_kpts_struct));
		}

	std::vector<std::vector<KeyPoint>*> kpt_vecs;
	tls_kpts_struct.gather(kpt_vecs);
	for (size_t i = 0; i < kpt_vecs.size(); ++i) {
		keypoints.insert(keypoints.end(), kpt_vecs[i]->begin(), kpt_vecs[i]->end());
	}
}


static void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
	int d, int n, float* dst)
{
	Point pt(cvRound(ptf.x), cvRound(ptf.y));
	float cos_t = cosf(ori * (float)(CV_PI / 180));
	float sin_t = sinf(ori * (float)(CV_PI / 180));
	float bins_per_rad = n / 360.f;
	float exp_scale = -1.f / (d * d * 0.5f);
	float hist_width = SIFT_DESCR_SCL_FCTR * scl;
	int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
	// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
	radius = std::min(radius, (int)sqrt(((double)img.cols) * img.cols + ((double)img.rows) * img.rows));
	cos_t /= hist_width;
	sin_t /= hist_width;

	int i, j, k, len = (radius * 2 + 1) * (radius * 2 + 1), histlen = (d + 2) * (d + 2) * (n + 2);
	int rows = img.rows, cols = img.cols;

	AutoBuffer<float> buf(len * 6 + histlen);
	float* X = buf.data(), * Y = X + len, * Mag = Y, * Ori = Mag + len, * W = Ori + len;
	float* RBin = W + len, * CBin = RBin + len, * hist = CBin + len;

	for (i = 0; i < d + 2; i++)
	{
		for (j = 0; j < d + 2; j++)
			for (k = 0; k < n + 2; k++)
				hist[(i * (d + 2) + j) * (n + 2) + k] = 0.;
	}

	for (i = -radius, k = 0; i <= radius; i++)
		for (j = -radius; j <= radius; j++)
		{
			// Calculate sample's histogram array coords rotated relative to ori.
			// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			// r_rot = 1.5) have full weight placed in row 1 after interpolation.
			float c_rot = j * cos_t - i * sin_t;
			float r_rot = j * sin_t + i * cos_t;
			float rbin = r_rot + d / 2 - 0.5f;
			float cbin = c_rot + d / 2 - 0.5f;
			int r = pt.y + i, c = pt.x + j;

			if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
				r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
			{
				float dx = (float)(img.at<sift_wt>(r, c + 1) - img.at<sift_wt>(r, c - 1));
				float dy = (float)(img.at<sift_wt>(r - 1, c) - img.at<sift_wt>(r + 1, c));
				X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
				W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;
				k++;
			}
		}

	len = k;
	cv::hal::fastAtan2(Y, X, Ori, len, true);
	cv::hal::magnitude32f(X, Y, Mag, len);
	cv::hal::exp32f(W, W, len);

	k = 0;
	for (; k < len; k++)
	{
		float rbin = RBin[k], cbin = CBin[k];
		float obin = (Ori[k] - ori) * bins_per_rad;
		float mag = Mag[k] * W[k];

		int r0 = cvFloor(rbin);
		int c0 = cvFloor(cbin);
		int o0 = cvFloor(obin);
		rbin -= r0;
		cbin -= c0;
		obin -= o0;

		if (o0 < 0)
			o0 += n;
		if (o0 >= n)
			o0 -= n;

		// histogram update using tri-linear interpolation
		float v_r1 = mag * rbin, v_r0 = mag - v_r1;
		float v_rc11 = v_r1 * cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0 * cbin, v_rc00 = v_r0 - v_rc01;
		float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
		float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
		float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
		float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;

		int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
		hist[idx] += v_rco000;
		hist[idx + 1] += v_rco001;
		hist[idx + (n + 2)] += v_rco010;
		hist[idx + (n + 3)] += v_rco011;
		hist[idx + (d + 2) * (n + 2)] += v_rco100;
		hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;
		hist[idx + (d + 3) * (n + 2)] += v_rco110;
		hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;
	}

	// finalize histogram, since the orientation histograms are circular
	for (i = 0; i < d; i++)
		for (j = 0; j < d; j++)
		{
			int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
			hist[idx] += hist[idx + n];
			hist[idx + 1] += hist[idx + n + 1];
			for (k = 0; k < n; k++)
				dst[(i * d + j) * n + k] = hist[idx + k];
		}
	// copy histogram to the descriptor,
	// apply hysteresis thresholding
	// and scale the result, so that it can be easily converted
	// to byte array
	float nrm2 = 0;
	len = d * d * n;
	k = 0;

	for (; k < len; k++)
		nrm2 += dst[k] * dst[k];

	float thr = std::sqrt(nrm2) * SIFT_DESCR_MAG_THR;

	i = 0, nrm2 = 0;
	for (; i < len; i++)
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val * val;
	}
	nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

	k = 0;
	for (; k < len; k++)
	{
		dst[k] = saturate_cast<uchar>(dst[k] * nrm2);
	}
}

class calcDescriptorsComputer //: public ParallelLoopBody
{
public:
	calcDescriptorsComputer(const std::vector<Mat>& _gpyr,
		const std::vector<KeyPoint>& _keypoints,
		Mat& _descriptors,
		int _nOctaveLayers,
		int _firstOctave)
		: gpyr(_gpyr),
		keypoints(_keypoints),
		descriptors(_descriptors),
		nOctaveLayers(_nOctaveLayers),
		firstOctave(_firstOctave) { }

	void operator()(const cv::Range& range) const //CV_OVERRIDE
	{
		const int begin = range.start;
		const int end = range.end;

		static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

		for (int i = begin; i < end; i++)
		{
			KeyPoint kpt = keypoints[i];
			int octave, layer;
			float scale;
			unpackOctave(kpt, octave, layer, scale);
			CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
			float size = kpt.size * scale;
			Point2f ptf(kpt.pt.x * scale, kpt.pt.y * scale);
			const Mat& img = gpyr[(octave - firstOctave) * (nOctaveLayers + 3) + layer];

			float angle = 360.f - kpt.angle;
			if (std::abs(angle - 360.f) < FLT_EPSILON)
				angle = 0.f;
			calcSIFTDescriptor(img, ptf, angle, size * 0.5f, d, n, descriptors.ptr<float>((int)i));
		}
	}
private:
	const std::vector<Mat>& gpyr;
	const std::vector<KeyPoint>& keypoints;
	Mat& descriptors;
	int nOctaveLayers;
	int firstOctave;
};

static void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
	Mat& descriptors, int nOctaveLayers, int firstOctave)
{
	parallel_for_(Range(0, static_cast<int>(keypoints.size())), calcDescriptorsComputer(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave));
}


SIFT::SIFT(int _nfeatures, int _nOctaveLayers,
	double _contrastThreshold, double _edgeThreshold, double _sigma)
	: nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
	contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma)
{
}

int SIFT::descriptorSize() const
{
	return SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
}

int SIFT::descriptorType() const
{
	return CV_32F;
}

int SIFT::defaultNorm() const
{
	return NORM_L2;
}


void SIFT::detectAndCompute(InputArray _image, InputArray _mask,
	std::vector<KeyPoint>& keypoints,
	OutputArray _descriptors,
	bool useProvidedKeypoints)
{
	int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
	Mat image = _image.getMat(), mask = _mask.getMat();

	if (image.empty() || image.depth() != CV_8U)
		CV_Error(Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

	if (!mask.empty() && mask.type() != CV_8UC1)
		CV_Error(Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)");

	if (useProvidedKeypoints)
	{
		firstOctave = 0;
		int maxOctave = INT_MIN;
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			int octave, layer;
			float scale;
			unpackOctave(keypoints[i], octave, layer, scale);
			firstOctave = std::min(firstOctave, octave);
			maxOctave = std::max(maxOctave, octave);
			actualNLayers = std::max(actualNLayers, layer - 2);
		}

		firstOctave = std::min(firstOctave, 0);
		CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
		actualNOctaves = maxOctave - firstOctave + 1;
	}

	Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
	std::vector<Mat> gpyr;
	int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(std::log((double)std::min(base.cols, base.rows)) / std::log(2.) - 2) - firstOctave;

	//double t, tf = getTickFrequency();
	//t = (double)getTickCount();
	buildGaussianPyramid(base, gpyr, nOctaves);

	//t = (double)getTickCount() - t;
	//printf("pyramid construction time: %g\n", t*1000./tf);

	if (!useProvidedKeypoints)
	{
		std::vector<Mat> dogpyr;
		buildDoGPyramid(gpyr, dogpyr);
		//t = (double)getTickCount();
		findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
		KeyPointsFilter::removeDuplicatedSorted(keypoints);

		if (nfeatures > 0)
			KeyPointsFilter::retainBest(keypoints, nfeatures);
		//t = (double)getTickCount() - t;
		//printf("keypoint detection time: %g\n", t*1000./tf);

		if (firstOctave < 0)
			for (size_t i = 0; i < keypoints.size(); i++)
			{
				KeyPoint& kpt = keypoints[i];
				float scale = 1.f / (float)(1 << -firstOctave);
				kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
				kpt.pt *= scale;
				kpt.size *= scale;
			}

		if (!mask.empty())
			KeyPointsFilter::runByPixelsMask(keypoints, mask);
	}
	else
	{
		// filter keypoints by mask
		//KeyPointsFilter::runByPixelsMask( keypoints, mask );
	}

	if (_descriptors.needed())
	{
		//t = (double)getTickCount();
		int dsize = descriptorSize();
		_descriptors.create((int)keypoints.size(), dsize, CV_32F);
		Mat descriptors = _descriptors.getMat();

		calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
		//t = (double)getTickCount() - t;
		//printf("descriptor extraction time: %g\n", t*1000./tf);
	}
}
// END OF SIFT CLASS

// FeatureManipulator class - Description 
bool FeatureManipulator::run() {
	if (!checkIfDirectory(this->imgPathFolder)) {
		cerr << "[ERROR] Folder: \"" << this->imgPathFolder << "\" does not exist\n";
		return false;
	}

	// Set mouse callback handler
	try {
		MouseData mouseData;
		cv::setMouseCallback(this->namedOutputWindows, this->MouseCallbackEvent, &mouseData);

		vector<string> fileList;
		listingFileInFolder(this->imgPathFolder, fileList);

		// Get the template frame
		vector<cv::Point2f> templateCorners;
		vector<cv::KeyPoint> keyPointsTemplate;
		cv::Mat descTemplate;
		cv::Mat templateImg;

		vector<cv::Point2f> working_pts{ mouseData.pt1, mouseData.pt2, mouseData.pt3, mouseData.pt4 };
		vector<cv::Point2f> layer_pts{ mouseData.pt1, mouseData.pt2, mouseData.pt3, mouseData.pt4 };
		// loop through the images
		for (const auto it : fileList) {

			frame = cv::imread(it, cv::IMREAD_COLOR);
			cv::Mat origFrame;
			cv::copyTo(frame, origFrame, cv::noArray());
			cv::Mat image = frame.clone();
			vector<cv::Point2f> corners;
			cv::Mat desc;
			cv::Rect roi;

			vector<cv::KeyPoint> keyPoints;
			detectFeatureDesciptorPts(image, corners, keyPoints, desc);
			for (const auto itr : corners) {
				cv::circle(frame, itr, 3, cv::Scalar(0, 255, 255), 1);
			}
			// always drawbox
			drawBox(this->frame, working_pts, cv::Scalar(0, 255, 0));

			if (mouseData.bDraw && mouseData.isMouseDown) {
				drawBox(this->frame, mouseData.pt1, mouseData.pt3, cv::Scalar(0, 0, 255));
			}
			else if (mouseData.bDraw && !mouseData.isMouseDown) {
				templateCorners.clear();
				keyPointsTemplate.clear();
				if (mouseData.pt1.x > mouseData.pt3.x) {
					roi.x = mouseData.pt3.x;
				}
				else {
					roi.x = mouseData.pt1.x;
				}
				if (mouseData.pt1.y > mouseData.pt3.y) {
					roi.y = mouseData.pt3.y;
				}
				else {
					roi.y = mouseData.pt1.y;
				}

				roi.width = abs(mouseData.pt1.x - mouseData.pt3.x);
				roi.height = abs(mouseData.pt1.y - mouseData.pt3.y);
				templateImg = origFrame(roi);
				detectFeatureDesciptorPts(templateImg, templateCorners, keyPointsTemplate, descTemplate);
				for (const auto itr : templateCorners) {
					cv::circle(templateImg, itr, 3, cv::Scalar(0, 255, 255), 1);
				}
				printf("Update new template image\n");
				cv::imshow(namedTemplate, templateImg);
				mouseData.bDraw = false;

				// Update new points
				working_pts[0] = mouseData.pt1;
				working_pts[1] = mouseData.pt2;
				working_pts[2] = mouseData.pt3;
				working_pts[3] = mouseData.pt4;
				layer_pts[0] = cv::Point2f(0, 0);
				layer_pts[1] = cv::Point2f(roi.width, 0);
				layer_pts[2] = cv::Point2f(roi.width, roi.height);
				layer_pts[3] = cv::Point2f(0, roi.height);
			}

			if (!keyPointsTemplate.empty()) {
				vector<cv::DMatch> goodMatches;
				mactchingFeaturePts(keyPointsTemplate, keyPoints, descTemplate, desc, goodMatches);
				vector<cv::Point2f> new_box_corners;
				if (transformObjects(keyPointsTemplate, keyPoints, goodMatches, layer_pts, new_box_corners))
				{
					for (int i = 0; i < layer_pts.size(); i++) {
						working_pts[i].x = new_box_corners[i].x;
						working_pts[i].y = new_box_corners[i].y;
					}
				}

				cv::Mat res;
				drawMatches(templateImg, keyPointsTemplate, origFrame, keyPoints, goodMatches, res,
					cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				cv::imshow("matching", res);
			}

			cv::imshow(this->namedOutputWindows, frame);
			if (cv::waitKey(30) == 'q') {
				break;
			}

		}
		return true;
	}
	catch (cv::Exception & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}

	return false;
}

bool FeatureManipulator::detectFeaturePts(const cv::Mat image, vector<cv::Point2f>& corners) {
	try {
		cv::Mat grayImg;
		if (image.channels() > 1) {
			cv::cvtColor(image, grayImg, cv::COLOR_RGB2GRAY);
		}
		else {
			grayImg = image.clone();
		}
		cv::goodFeaturesToTrack(grayImg, corners, 200, 0.3, 7, cv::Mat(), 7, false, 0.04);
		return true;
	}
	catch (cv::Exception & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;
}


bool FeatureManipulator::detectFeatureDesciptorPts(const cv::Mat image, vector<cv::Point2f>& corners, vector<cv::KeyPoint>& keyPoints, cv::Mat& desc) {
	cv::KeyPoint test;
	try {
		cv::Mat grayImg;
		if (image.channels() > 1) {
			cv::cvtColor(image, grayImg, cv::COLOR_RGB2GRAY);
		}
		else {
			grayImg = image.clone();
		}

		SIFT detector;
		detector.detectAndCompute(grayImg, noArray(), keyPoints, desc);

		for (const auto it : keyPoints) {
			corners.push_back(it.pt);
		}

		return true;
	}
	catch (cv::Exception & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;
}

// Using FLANN features matching
bool FeatureManipulator::mactchingFeaturePts(vector<cv::KeyPoint> keyPt1, vector<cv::KeyPoint> keyPt2,
	cv::Mat desc1, cv::Mat desc2, vector<cv::DMatch>& goodMatches) {
	try {
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		std::vector< std::vector<DMatch> > knn_matches;
		matcher->knnMatch(desc1, desc2, knn_matches, 2);
		// Filter matches using the Lowe's ratio test

		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < RADIO_THRESH * knn_matches[i][1].distance)
			{
				goodMatches.push_back(knn_matches[i][0]);
			}
		}
		return true;
	}
	catch (cv::Exception & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;
}

bool FeatureManipulator::transformObjects(vector<cv::KeyPoint> keyPts1, vector<cv::KeyPoint> keyPts2, vector<cv::DMatch> goodMatches,
	vector<cv::Point2f> box_corners, vector<cv::Point2f>& new_box_corners) {
	try {
		//Localize the object
		vector<cv::Point2f> obj;
		vector<cv::Point2f> scene;


		for (size_t i = 0; i < goodMatches.size(); i++)
		{
			//Get the keypoints from the good matches
			obj.push_back(keyPts1[goodMatches[i].queryIdx].pt);
			scene.push_back(keyPts2[goodMatches[i].trainIdx].pt);
		}
		cv::Mat H_matrix = cv::findHomography(obj, scene, cv::RANSAC, 5.0);

		cv::perspectiveTransform(box_corners, new_box_corners, H_matrix);
		cout << H_matrix << endl;
		cout << "Box corners\n";
		cout << box_corners << endl;
		cout << new_box_corners << endl;
		return true;
	}
	catch (cv::Exception & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;

}

void FeatureManipulator::MouseCallbackEvent(int event, int x, int y, int flags, void* userdata)
{
	try {
		MouseData* mouseData = (MouseData*)userdata;
		if (event == cv::EVENT_LBUTTONDOWN)
		{
			mouseData->isMouseDown = true;
			mouseData->pt1.x = x;
			mouseData->pt1.y = y;
			mouseData->pt3.x = -1;
			mouseData->pt3.y = -1;
			mouseData->pt2.x = -1;
			mouseData->pt2.y = -1;
			mouseData->pt4.x = -1;
			mouseData->pt4.y = -1;
			mouseData->bDraw = true;
		}
		else if (event == cv::EVENT_MOUSEMOVE) {
			if (!mouseData->bDraw) return;
			mouseData->pt3.x = x;
			mouseData->pt3.y = y;
			mouseData->pt2.x = mouseData->pt3.x;
			mouseData->pt2.y = mouseData->pt1.y;
			mouseData->pt4.x = mouseData->pt1.x;
			mouseData->pt4.y = mouseData->pt3.y;
		}
		else if (event == cv::EVENT_LBUTTONUP)
		{
			mouseData->isMouseDown = false;
			mouseData->pt3.x = x;
			mouseData->pt3.y = y;
			mouseData->pt2.x = mouseData->pt3.x;
			mouseData->pt2.y = mouseData->pt1.y;
			mouseData->pt4.x = mouseData->pt1.x;
			mouseData->pt4.y = mouseData->pt3.y;
		}
	}
	catch (cv::Exception & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
}


bool FeatureManipulator::drawBox(cv::Mat& image, cv::Point pt1, cv::Point pt3, cv::Scalar color) {
	try {
		if (pt1.x < 0 || pt1.y < 0 || pt3.x < 0 || pt3.y < 0)
			return false;
		cv::Point pt2 = cv::Point(pt3.x, pt1.y);
		cv::Point pt4 = cv::Point(pt1.x, pt3.y);
		cv::line(image, pt1, pt2, color, 2);
		cv::line(image, pt2, pt3, color, 2);
		cv::line(image, pt3, pt4, color, 2);
		cv::line(image, pt4, pt1, color, 2);
		return true;
	}
	catch (cv::Exception & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;
}

bool FeatureManipulator::drawBox(cv::Mat& image, vector<cv::Point2f>pts, cv::Scalar color) {
	try {
		if (pts[0].x < 0 || pts[0].y < 0 || pts[1].x < 0 || pts[1].y < 0 ||
			pts[2].x < 0 || pts[2].y < 0 || pts[3].x < 0 || pts[3].y < 0)
			return false;
		cv::line(image, pts[0], pts[1], color, 2);
		cv::line(image, pts[1], pts[2], color, 2);
		cv::line(image, pts[2], pts[3], color, 2);
		cv::line(image, pts[3], pts[0], color, 2);
		return true;
	}
	catch (cv::Exception & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;
}

bool FeatureManipulator::checkIfDirectory(string filePath) {
	try {
		fs::path pathObj(filePath);
		if (fs::exists(pathObj) && fs::is_directory(pathObj)) {
			return true;
		}
	}
	catch (fs::filesystem_error & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;
}

bool FeatureManipulator::checkIfFile(string fileName) {
	try {
		fs::path pathObj(fileName);
		if (fs::exists(pathObj) && fs::is_regular_file(pathObj)) {
			return true;
		}
	}
	catch (fs::filesystem_error & err) {
		cerr << "[ERROR] " << err.what() << endl;
	}
	return false;
}

bool FeatureManipulator::listingFileInFolder(const string folderPath, vector<string>& fileList) {
	if (!checkIfDirectory(folderPath)) {
		cerr << "[ERROR] Folder: \"" << folderPath << "\" does not exist\n";
		return false;
	}
	for (const auto& entry : fs::directory_iterator(folderPath)) {
		fileList.push_back(entry.path().string());
	}
	return true;
}
//FeatureManipulator class - Description - End