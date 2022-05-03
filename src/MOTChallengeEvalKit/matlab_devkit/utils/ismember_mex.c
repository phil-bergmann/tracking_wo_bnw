/*
 *Expects two sorted arrays !!!
 *
 *Example:
 * >>ismember_mex( [1 3 5], [1 2 3 4 6 7 8] )
 *ans =

     1
     1
     0
 */

#include <matrix.h>
#include "mex.h"

#include <matrix.h>

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	double *in1, *in2, *locb;
	mxLogical *out;
	int N,M;

	in1 = mxGetPr(prhs[0]);
	in2 = mxGetPr(prhs[1]);

	N = (int)mxGetNumberOfElements(prhs[0]);
	M = (int)mxGetNumberOfElements(prhs[1]);

	plhs[0] = mxCreateLogicalMatrix(N, 1);
	out = mxGetLogicals(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(N, 1, mxREAL);
	locb = mxGetPr(plhs[1]);
	int pos = 0;
	int i = 0;
	while (i < N && pos < M)
	{
		while (pos < M)
		{
			if (in1[i] == in2[pos])
			{
				out[i] = true;
				locb[i] = pos + 1;
				pos++;
				i++;
				break;
			}
			else if (in1[i] < in2[pos])
			{
				out[i] = false;
				locb[i] = 0;
				i++;
				if (i == N) break;
			}
			else
			{
				pos++;
			}
		}
		
	}

	
}
