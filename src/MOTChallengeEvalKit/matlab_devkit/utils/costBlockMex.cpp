#include <matrix.h>
#include "mex.h"
#include <cmath>
#include <omp.h>
#include <algorithm>
using namespace std;

inline int index(int i, int j, int numRows) // 1-indexed to C
{
	return (j-1)*numRows + i-1;
}

void correspondingFrames(double* frames1, int N, double* frames2, int M, int* loc)
{
	int pos = 0;
	int i = 0;
	while (i < N && pos < M) {
		while (pos < M) {
			if (frames1[i] == frames2[pos]) {
				loc[i] = pos; pos++; i++;
				break;
			}
			else if (frames1[i] < frames2[pos]) {
				loc[i] = -1; i++;
				if (i == N) break;
			}
			else pos++;
		}
	}
}

void computeDistances(double* tr1, double* tr2, int numPoints1, int numPoints2, int* position, bool world, double* distance)
{
	if (world)
	{
		double* wx1 = &tr1[index(1, 7, numPoints1)];
		double* wy1 = &tr1[index(1, 8, numPoints1)];
		double* wx2 = &tr2[index(1, 7, numPoints2)];
		double* wy2 = &tr2[index(1, 8, numPoints2)];

		for (int i = 0; i < numPoints1; i++)
		{
			if (position[i] == -1)
				distance[i] = 1e9;
			else
			{
				double dx = wx1[i] - wx2[position[i]];
				double dy = wy1[i] - wy2[position[i]];
				distance[i] = sqrt(dx*dx + dy*dy);
			}
		}
	}
	else
	{
		double* l1 = &tr1[index(1, 3, numPoints1)];
		double* t1 = &tr1[index(1, 4, numPoints1)];
		double* w1 = &tr1[index(1, 5, numPoints1)];
		double* h1 = &tr1[index(1, 6, numPoints1)];
		double* l2 = &tr2[index(1, 3, numPoints2)];
		double* t2 = &tr2[index(1, 4, numPoints2)];
		double* w2 = &tr2[index(1, 5, numPoints2)];
		double* h2 = &tr2[index(1, 6, numPoints2)];

		for (int i = 0; i < numPoints1; i++)
		{
			if (position[i] == -1)
				distance[i] = 0;
			else
			{
				double area1 = w1[i] * h1[i];
				double area2 = w2[position[i]] * h2[position[i]];

				double x_overlap = max(0.0, min(l1[i] + w1[i], l2[position[i]] + w2[position[i]]) - max(l1[i], l2[position[i]]));
				double y_overlap = max(0.0, min(t1[i] + h1[i], t2[position[i]] + h2[position[i]]) - max(t1[i], t2[position[i]]));
				double intersectionArea = x_overlap*y_overlap;
				double unionArea = area1 + area2 - intersectionArea;
				double iou = intersectionArea / unionArea;
				distance[i] = iou;
			}
		}
	}
}

void compute(double* tr1, double* tr2, const int* dim1, const int* dim2, double threshold, bool world, double& cost, double& fp, double& fn)
{
	int numPoints1 = dim1[0];
	int numPoints2 = dim2[0];
	int numCols1 = dim1[1];
	int numCols2 = dim2[1];
	int tr1start, tr1end, tr2start, tr2end;
	tr1start = tr1[index(1,1, numPoints1)];
	tr1end = tr1[index(numPoints1,1, numPoints1)];
	tr2start = tr2[index(1,1, numPoints2)];
	tr2end = tr2[index(numPoints2,1, numPoints2)];


	bool overlapTest = ((tr1start >= tr2start && tr1start <= tr2end) ||
		(tr1end >= tr2start && tr1end <= tr2end) ||
		(tr2start >= tr1start && tr2start <= tr1end) ||
		(tr2end >= tr1start && tr2end <= tr1end));
		
	if (!overlapTest)
	{
		fp = numPoints2;
		fn = numPoints1;
		cost = numPoints1 + numPoints2;
		return;
	}

	int* positionGT = new int[numPoints1];
	int* positionPred = new int[numPoints2];
	for (int i = 0; i < numPoints1; i++) positionGT[i] = -1;
	for (int i = 0; i < numPoints2; i++) positionPred[i] = -1;
	double* distanceGT = new double[numPoints1];
	double* distancePred = new double[numPoints2];
	double* frames1 = &tr1[index(1, 1, numPoints1)];
	double* frames2 = &tr2[index(1, 1, numPoints2)];
	
	correspondingFrames(frames1, numPoints1, frames2, numPoints2, positionGT);
	correspondingFrames(frames2, numPoints2, frames1, numPoints1, positionPred);
	computeDistances(tr1, tr2, numPoints1, numPoints2, positionGT, world, distanceGT);
	computeDistances(tr2, tr1, numPoints2, numPoints1, positionPred, world, distancePred);

	fp = 0; fn = 0;
	if (world) {
		for (int i = 0; i < numPoints1; i++) if (distanceGT[i] > threshold) fn++;
		for (int i = 0; i < numPoints2; i++) if (distancePred[i] > threshold) fp++;
	}
	else {
		for (int i = 0; i < numPoints1; i++) if (distanceGT[i] < threshold) fn++;
		for (int i = 0; i < numPoints2; i++) if (distancePred[i] < threshold) fp++;
	}
	cost = fp + fn;

	delete[] positionGT;
	delete[] positionPred;
	delete[] distanceGT;
	delete[] distancePred;

}





void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{

	int numGT = mxGetNumberOfElements(prhs[0]);
	int numPred = mxGetNumberOfElements(prhs[1]);
	int numEl = numGT + numPred;

	double threshold = (double)mxGetScalar(prhs[2]);
	bool world = (bool)mxGetScalar(prhs[3]);

	double *cost, *fp, *fn;
	plhs[0] = mxCreateDoubleMatrix(numGT, numPred, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(numGT, numPred, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(numGT, numPred, mxREAL);
	cost = mxGetPr(plhs[0]);
	fp = mxGetPr(plhs[1]);
	fn = mxGetPr(plhs[2]);

#pragma omp parallel for 
	for (int i = 0; i < numGT; i++) {
#pragma omp parallel for 
		for (int j = 0; j < numPred; j++) {


			const int  *dim1, *dim2;
			dim1 = (int *)mxGetDimensions(mxGetCell(prhs[0], i));
			dim2 = (int *)mxGetDimensions(mxGetCell(prhs[1], j));
			double* tr1 = mxGetPr(mxGetCell(prhs[0],i));
			double* tr2 = mxGetPr(mxGetCell(prhs[1],j));
			int ind = j*numGT+ i;
			compute(tr1, tr2, dim1, dim2, threshold, world, cost[ind], fp[ind], fn[ind]);
		}
	}
	
}