#ifndef ransac_h
#define ransac_h

#include "data_types.h"
#include "time.h"

double uniformRandom();

double gaussianRandom();


void ransac_fit_line();


bool fit_line (ArrPoint2f pt, double* m, double* b, double* r);
void fitLineRANSAC(ArrPoint2f ptSet, double *m, double *b, double *r, bool *inlierFlag);


double GetDistance(TPoint LP0, TPoint LP1, TPoint Point);

int ransac_fitting(TPoint *Data, int numData, double Threshold, double Confidence,
				   double ApproximateInlierFraction, TPoint *Point0, TPoint *Point1, int *Inliers);

#endif