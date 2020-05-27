#include "ransac.h"

// form of line: y = mx + b  m: slope b: output intercept

double get_distance(pt2f pt, double m, double b)
{
	double dist = fabs(b + m * pt.x - pt.y) / sqrt(1 + m * m);
}

bool fit_line(ArrPoint2f pt, double *m, double *b, double *r)
{
	double sumx = 0.0;
	double sumx2 = 0.0;
	double sumxy = 0.0;
	double sumy = 0.0;
	double sumy2 = 0.0;
	int num_points = vec_size(&pt.x);

	for (int i = 0; i < num_points; i++)
	{
		float x = vec_get(&pt.x, i);
		float y = vec_get(&pt.y, i);
		sumx += x;
		sumx2 += pow(x, 2);
		sumxy += x * y;
		sumy += y;
		sumy2 += pow(y, 2);
	}

	double denom = (num_points * sumx2 - pow(sumx, 2));
	if (denom == 0)
	{
		*m = 0;
		*b = 0;
		if (r)
			*r = 0;
		return false;
	}

	*m = (num_points * sumxy - sumx * sumy) / denom;
	*b = (sumy * sumx2 - sumx * sumxy) / denom;
	if (r != NULL)
	{
		*r = (sumxy - sumx * sumy / num_points) /
			 sqrt((sumx2 - pow(sumx, 2) / num_points) *
				  (sumy2 - pow(sumy, 2) / num_points));
	}
	return true;
}

double uniformRandom(void)
{
	return (double)rand() / (double)RAND_MAX;
}

double gaussianRandom(void)
{
	/*Gaussian routine from Numerical Recipes*/
	static int next_gaussian = 0;
	static double saved_gaussian_value;

	double fac, rsq, v1, v2;

	if (next_gaussian == 0)
	{
		do
		{
			v1 = 2 * uniformRandom() - 1;
			v2 = 2 * uniformRandom() - 1;
			rsq = v1 * v1 + v2 * v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2 * log(rsq) / rsq);
		saved_gaussian_value = v1 * fac;
		next_gaussian = 1;
		return v2 * fac;
	}
	else
	{
		next_gaussian = 0;
		return saved_gaussian_value;
	}
}

// Get a straight line fitting sample, that is, randomly select 2 points on the line sampling point set
bool getSample(int *set, Vector *sset, int size)
{
	int i[2];
	if (size > 2)
	{
		do
		{
			for (int n = 0; n < 2; n++)
				i[n] = (int)(uniformRandom() * (size - 1));
		} while (!(i[1] != i[0]));
		for (int n = 0; n < 2; n++)
		{
			vec_append(sset, i[n]);
		}
	}
	else
	{
		return false;
	}
	return true;
}

bool verifyComposition(const pt2f pt1, const pt2f pt2)
{
	if (fabs(pt1.x - pt2.x) < 5 && fabs(pt1.y - pt2.y) < 5)
		return false;
	return true;
}

//RANSAC line fitting
void fitLineRANSAC(ArrPoint2f ptSet, double *m, double *b, double *r, bool *inlierFlag)
{
	double residual_error = 2.99;
	double min_dist = 3;
	int sample_count = 0;
	int N = 500;
	double res = 0;

	bool stop_loop = false;
	int maximum = 0;

	int size = vec_size(&ptSet.x);
    for (int i = 0; i < size; i++)
		inlierFlag[i] = false;

	double resids_[size];
	for (int i = 0; i < size; i++)
		resids_[i] = 3;
	srand((unsigned int)time(NULL));
	int ptsID[size];
	for (unsigned int i = 0; i < size; i++)
		ptsID[i] = i;

	while (N > sample_count && !stop_loop)
	{
		bool inlierstemp[size];
		double residualstemp[size];
		Vector ptss;
		vec_init(&ptss);
		int inlier_count = 0;
		if (!getSample(ptsID, &ptss, size))
		{
			stop_loop = true;
			continue;
		}

		ArrPoint2f pt_sam;
		vec_init(&pt_sam.x);
		vec_init(&pt_sam.y);

		int idx0 = vec_get(&ptss, 0);
		int idx1 = vec_get(&ptss, 1);
		vec_free_memory(&ptss);

		pt2f pt1 = {vec_get(&ptSet.x, idx0), vec_get(&ptSet.y, idx0)};
		pt2f pt2 = {vec_get(&ptSet.x, idx1), vec_get(&ptSet.y, idx1)};

		vec_append(&pt_sam.x, vec_get(&ptSet.x, idx0));
		vec_append(&pt_sam.y, vec_get(&ptSet.y, idx0));
		vec_append(&pt_sam.x, vec_get(&ptSet.x, idx1));
		vec_append(&pt_sam.x, vec_get(&ptSet.x, idx1));
        if (!verifyComposition(pt1, pt2))
		{
			++sample_count;
			continue;
		}
		fit_line(pt_sam, m, b, r);
    	vec_free_memory(&pt_sam.x);
		vec_free_memory(&pt_sam.y);
        for (unsigned int i = 0; i < size; i++)
		{
			pt2f pt = {vec_get(&ptSet.x, i), vec_get(&ptSet.y, i)};
			double dist = get_distance(pt, *m, *b);
			inlierstemp[i] = false;
			if (dist <= min_dist)
			{
				++inlier_count;
				inlierstemp[i] = true;
			}
		}
		if (inlier_count >= maximum)
		{
			maximum = inlier_count;
			for (int ii = 0; ii < size; ii++)
			{
				inlierFlag[ii] = inlierstemp[ii];
			}
		}
		if (inlier_count == 0)
		{
			N = 500;
		}
		else
		{
			double epsilon = 1.0 - (double)inlier_count / (double)size;
			double p = 0.99;
			double s = 2.0;
			N = (int)((log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s))));
		}
		++sample_count;
	}
	ArrPoint2f pset;
	vec_init(&pset.x);
	vec_init(&pset.y);
	for (unsigned int i = 0; i < size; i++)
	{
		if (inlierFlag[i])
		{
			vec_append(&pset.x, vec_get(&ptSet.x, i));
			vec_append(&pset.y, vec_get(&ptSet.y, i));
		}
	}

	fit_line(pset, m, b, r);
	vec_free_memory(&pset.x);
	vec_free_memory(&pset.y);
    }

double GetDistance(TPoint LP0, TPoint LP1, TPoint Point)
{
	double distance = abs(((LP1.y - LP0.y) * Point.x - (LP1.x - LP0.x) * Point.y + LP1.x * LP0.y - LP1.y * LP0.x) / sqrt(pow((LP1.y - LP0.y), 2) + pow(LP1.x - LP0.x, 2)));
	return distance;
};

int ransac_fitting(TPoint *Data, int numData, double Threshold, double Confidence,
				   double ApproximateInlierFraction, TPoint *Point0, TPoint *Point1, int *Inliers)
{

	srand(time(NULL));

	TPoint LPoint0;
	TPoint LPoint1;
	TPoint TP;
	int TInliers[numData];
	int numOfIn = 0;
	int maxNumOfIn = 0;
	double w = 0.00;
	double p = 0.00;
	int k = log(1 - Confidence) / log(1 - pow(ApproximateInlierFraction, 2));
	for (int i = 0; i <= k; i++)
	{
		int testP0 = (rand() % numData);
		LPoint0.x = Data[testP0].x;
		LPoint0.y = Data[testP0].y;

		int testP1 = (rand() % numData);
		LPoint1.x = Data[testP1].x;
		LPoint1.y = Data[testP1].y;

		numOfIn = 0;

		int numTInliers = 0;
		for (size_t Index = 0; Index != numData; ++Index)
		{
			TP.x = Data[Index].x;
			TP.y = Data[Index].y;
			double dist = GetDistance(LPoint0, LPoint1, TP);
			if (GetDistance(LPoint0, LPoint1, TP) <= Threshold)
			{
				TInliers[numTInliers++] = 1;
				numOfIn++;
			}
			else
			{
				TInliers[numTInliers++] = 0;
			}
		}

		if (maxNumOfIn < numOfIn)
		{
			maxNumOfIn = numOfIn;
			Point0->x = LPoint0.x;
			Point0->y = LPoint0.y;
			Point1->x = LPoint1.x;
			Point1->y = LPoint1.y;
			for (int ind = 0; ind < numData; ind++)
			{
				Inliers[ind] = TInliers[ind];
			}
		}
	}
}
