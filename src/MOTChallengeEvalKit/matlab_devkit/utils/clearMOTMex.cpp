#include <matrix.h>
#include "mex.h"
#include <cmath>
#include <omp.h>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>
#include <unordered_map>
using namespace std;


inline int index(int i, int j, int numRows) // 2D 0-indexed to C
{
	return j*numRows + i;
}

inline double euclidean(double x1, double y1, double x2, double y2)
{
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

double boxiou(double l1, double t1, double w1, double h1, double l2, double t2, double w2, double h2)
{
	double area1 = w1 * h1;
	double area2 = w2 * h2;

	double x_overlap = max(0.0, min(l1 + w1, l2 + w2) - max(l1, l2));
	double y_overlap = max(0.0, min(t1 + h1, t2 + h2) - max(t1, t2));
	double intersectionArea = x_overlap*y_overlap;
	double unionArea = area1 + area2 - intersectionArea;
	double iou = intersectionArea / unionArea;
	return iou;
}

// Min cost bipartite matching via shortest augmenting paths
//
// Code from https://github.com/jaehyunp/
//
// This is an O(n^3) implementation of a shortest augmenting path
// algorithm for finding min cost perfect matchings in dense
// graphs.  In practice, it solves 1000x1000 problems in around 1
// second.
//
//   cost[i][j] = cost for pairing left node i with right node j
//   Lmate[i] = index of right node that left node i pairs with
//   Rmate[j] = index of left node that right node j pairs with
//
// The values in cost[i][j] may be positive or negative.  To perform
// maximization, simply negate the cost[][] matrix.



typedef vector<double> VD;
typedef vector<VD> VVD;
typedef vector<int> VI;

double MinCostMatching(const VVD &cost, VI &Lmate, VI &Rmate) {
	int n = int(cost.size());

	// construct dual feasible solution
	VD u(n);
	VD v(n);
	for (int i = 0; i < n; i++) {
		u[i] = cost[i][0];
		for (int j = 1; j < n; j++) u[i] = min(u[i], cost[i][j]);
	}
	for (int j = 0; j < n; j++) {
		v[j] = cost[0][j] - u[0];
		for (int i = 1; i < n; i++) v[j] = min(v[j], cost[i][j] - u[i]);
	}

	// construct primal solution satisfying complementary slackness
	Lmate = VI(n, -1);
	Rmate = VI(n, -1);
	int mated = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (Rmate[j] != -1) continue;
			if (fabs(cost[i][j] - u[i] - v[j]) < 1e-10) {
				Lmate[i] = j;
				Rmate[j] = i;
				mated++;
				break;
			}
		}
	}

	VD dist(n);
	VI dad(n);
	VI seen(n);

	// repeat until primal solution is feasible
	while (mated < n) {

		// find an unmatched left node
		int s = 0;
		while (Lmate[s] != -1) s++;

		// initialize Dijkstra
		fill(dad.begin(), dad.end(), -1);
		fill(seen.begin(), seen.end(), 0);
		for (int k = 0; k < n; k++)
			dist[k] = cost[s][k] - u[s] - v[k];

		int j = 0;
		while (true) {

			// find closest
			j = -1;
			for (int k = 0; k < n; k++) {
				if (seen[k]) continue;
				if (j == -1 || dist[k] < dist[j]) j = k;
			}
			seen[j] = 1;

			// termination condition
			if (Rmate[j] == -1) break;

			// relax neighbors
			const int i = Rmate[j];
			for (int k = 0; k < n; k++) {
				if (seen[k]) continue;
				const double new_dist = dist[j] + cost[i][k] - u[i] - v[k];
				if (dist[k] > new_dist) {
					dist[k] = new_dist;
					dad[k] = j;
				}
			}
		}

		// update dual variables
		for (int k = 0; k < n; k++) {
			if (k == j || !seen[k]) continue;
			const int i = Rmate[k];
			v[k] += dist[k] - dist[j];
			u[i] -= dist[k] - dist[j];
		}
		u[s] += dist[j];

		// augment along path
		while (dad[j] >= 0) {
			const int d = dad[j];
			Rmate[j] = Rmate[d];
			Lmate[Rmate[j]] = j;
			j = d;
		}
		Rmate[j] = s;
		Lmate[s] = j;

		mated++;
	}

	double value = 0;
	for (int i = 0; i < n; i++)
		value += cost[i][Lmate[i]];

	return value;
}

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
    // gtMat and resMat should contain IDs in range [1,...,numIDs]
    // and frames in range [1, ..., numFrames]
	// data format: frame, ID, left, top, width, height, worldX, worldY
    double *gtMat = mxGetPr(prhs[0]);
    double *resMat = mxGetPr(prhs[1]);
	double threshold = (double)mxGetScalar(prhs[2]);
	bool world = (bool)mxGetScalar(prhs[3]);
	bool VERBOSE = (bool)mxGetScalar(prhs[4]);

	int *dimGT, *dimST;
	dimGT = (int *)mxGetDimensions(prhs[0]);
	dimST = (int *)mxGetDimensions(prhs[1]);
    int rowsGT = dimGT[0], colsGT = dimGT[1], rowsST = dimST[0], colsST = dimST[1];
    int Fgt=0, Ngt = 0, Nst = 0;
    
    
    for (int i = 0; i < rowsGT; i++) {
        int frame = gtMat[index(i,0,rowsGT)];
        int ID = gtMat[index(i,1,rowsGT)];
        if (frame > Fgt) Fgt = frame;
        if (ID > Ngt) Ngt = ID;
    }
    for (int i = 0; i < rowsST; i++) {
		int frame = resMat[index(i, 0, rowsST)];
		if (frame > Fgt) Fgt = frame;
		int ID = resMat[index(i, 1, rowsST)];
        if (ID > Nst) Nst = ID;
    } 
    
	plhs[0] = mxCreateDoubleMatrix(1, Fgt, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1, Fgt, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1, Fgt, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(1, Fgt, mxREAL);
	plhs[4] = mxCreateDoubleMatrix(1, Fgt, mxREAL);
	plhs[5] = mxCreateDoubleMatrix(Fgt, Ngt, mxREAL);
	plhs[6] = mxCreateDoubleMatrix(Fgt, Ngt, mxREAL);
	plhs[7] = mxCreateDoubleMatrix(Fgt, Nst, mxREAL);


	double* mmeOut = mxGetPr(plhs[0]);
	double* cOut = mxGetPr(plhs[1]);
	double* fpOut = mxGetPr(plhs[2]);
	double* mOut = mxGetPr(plhs[3]);
	double* gOut = mxGetPr(plhs[4]);
	double* dOut = mxGetPr(plhs[5]);
	double* MOut = mxGetPr(plhs[6]);
	double* allfalseposOut = mxGetPr(plhs[7]);
	//double* MOut = mxGetPr(plhs[8]);

	double INF = 1e9;

	vector<unordered_map<int, int>> gtInd(Fgt);
	vector<unordered_map<int, int>> stInd(Fgt);
	vector<unordered_map<int,int>> M(Fgt);
	vector<int> mme(Fgt, 0); // ID Switchtes(mismatches)
	vector<int> c(Fgt, 0); // matches found
	vector<int> fp(Fgt, 0); // false positives
	vector<int> m(Fgt, 0); // misses = false negatives
	vector<int> g(Fgt, 0); // gt count for each frame
	vector<vector<double>> d(Fgt, vector<double>(Ngt, 0)); // all distances mapped to [0..1]
	vector<vector<int>> allfalsepos(Fgt, vector<int>(Nst, 0));

	for (int i = 0; i < rowsGT; i++) {
		int frame = gtMat[index(i, 0, rowsGT)]-1;
		int ID = gtMat[index(i, 1, rowsGT)]-1;
		gtInd[frame][ID] = i;
	}
	for (int i = 0; i < rowsST; i++) {
		int frame = resMat[index(i, 0, rowsST)]-1;
		int ID = resMat[index(i, 1, rowsST)]-1;
		stInd[frame][ID] = i;
	}

	for (int i = 0; i < Fgt; i++) {
		for (unordered_map<int, int>::iterator it = gtInd[i].begin(); it != gtInd[i].end(); it++) g[i]++;
	}

	
	for (int t = 0; t < Fgt; t++)
	{
		//if ((t + 1) % 1000 == 0) mexEvalString("fprintf('.');");  // print every 1000th frame

		if (t > 0)
		{
			vector<int> mappings;
			for (unordered_map<int,int>::iterator it = M[t - 1].begin(); it != M[t - 1].end(); it++) mappings.push_back(it->first);
			sort(mappings.begin(), mappings.end());
			for (int k = 0; k < mappings.size(); k++)
			{
				unordered_map<int, int>::const_iterator foundGtind = gtInd[t].find(mappings[k]);
				unordered_map<int, int>::const_iterator foundStind = stInd[t].find(M[t - 1][mappings[k]]);

				if (foundGtind != gtInd[t].end() && foundStind != stInd[t].end())
				{
					bool matched = false;
					if (world)
					{
						double gtx, gty, stx, sty;
						int rowgt = gtInd[t][mappings[k]];
						int rowres = stInd[t][M[t - 1][mappings[k]]];
						gtx = gtMat[index(rowgt, 6, rowsGT)];
						gty = gtMat[index(rowgt, 7, rowsGT)];
						stx = resMat[index(rowres, 6, rowsST)];
						sty = resMat[index(rowres, 7, rowsST)];
						double dist = euclidean(gtx, gty, stx, sty);
						matched = (dist <= threshold);
					}
					else
					{
						double gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight;
						int rowgt = gtInd[t][mappings[k]];
						int rowres = stInd[t][M[t - 1][mappings[k]]];
						gtleft = gtMat[index(rowgt, 2, rowsGT)];
						gttop = gtMat[index(rowgt, 3, rowsGT)];
						gtwidth = gtMat[index(rowgt, 4, rowsGT)];
						gtheight = gtMat[index(rowgt, 5, rowsGT)];
						stleft = resMat[index(rowres, 2, rowsST)];
						sttop = resMat[index(rowres, 3, rowsST)];
						stwidth = resMat[index(rowres, 4, rowsST)];
						stheight = resMat[index(rowres, 5, rowsST)];
						double dist = 1 - boxiou(gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight);
						matched = (dist <= threshold);
					}

					if (matched) {
						M[t][mappings[k]] = M[t - 1][mappings[k]];
						if (VERBOSE) printf("%d: preserve %d\n", t+1, mappings[k]+1);
					}
				}
			}
		}

		vector<int> unmappedGt, unmappedEs, stindt, findm;
		for (unordered_map<int,int>::iterator it = gtInd[t].begin(); it != gtInd[t].end(); it++) {
			unordered_map<int, int>::const_iterator found = M[t].find(it->first);
			if (found==M[t].end()) unmappedGt.push_back(it->first);
		}
		for (unordered_map<int,int>::iterator it = M[t].begin(); it != M[t].end(); it++) findm.push_back(it->second);
		for (unordered_map<int,int>::iterator it = stInd[t].begin(); it != stInd[t].end(); it++) stindt.push_back(it->first);

		sort(stindt.begin(), stindt.end());
		sort(findm.begin(), findm.end());
		set_difference(stindt.begin(), stindt.end(), findm.begin(), findm.end(), inserter(unmappedEs, unmappedEs.end()));

        sort(unmappedGt.begin(), unmappedGt.end());
        
		int squareSize = max(unmappedGt.size(), unmappedEs.size());
		vector<vector<double>> alldist(squareSize, vector<double>(squareSize, INF));

		if (VERBOSE)
		{
			printf("%d: UnmappedGTs: ", t+1);
			for (int i = 0; i < unmappedGt.size(); i++) printf("%d, ", unmappedGt[i]+1);
			printf("\n%d: UnmappedEs: ", t+1);
			for (int i = 0; i < unmappedEs.size(); i++) printf("%d, ", unmappedEs[i]+1);
			printf("\n");
		}

        int uid = 0; // Unique identifier
		for (int i = 0; i < unmappedGt.size(); i++)
		{
			for (int j = 0; j < unmappedEs.size(); j++)
			{
				int o = unmappedGt[i];
				int e = unmappedEs[j];
				if (world)
				{
					double gtx, gty, stx, sty;
					int rowgt = gtInd[t][o];
					int rowres = stInd[t][e];
					gtx = gtMat[index(rowgt, 6, rowsGT)];
					gty = gtMat[index(rowgt, 7, rowsGT)];
					stx = resMat[index(rowres, 6, rowsST)];
					sty = resMat[index(rowres, 7, rowsST)];
					double dist = euclidean(gtx, gty, stx, sty);
					if (dist <= threshold) 
                    {
                        alldist[i][j] = dist;
                        // Add unique identifier to break ties
                        alldist[i][j] += 1e-9 * uid;
                        uid++;
                    }
				}
				else
				{
					double gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight;
					int rowgt = gtInd[t][o];
					int rowres = stInd[t][e];
					gtleft = gtMat[index(rowgt, 2, rowsGT)];
					gttop = gtMat[index(rowgt, 3, rowsGT)];
					gtwidth = gtMat[index(rowgt, 4, rowsGT)];
					gtheight = gtMat[index(rowgt, 5, rowsGT)];
					stleft = resMat[index(rowres, 2, rowsST)];
					sttop = resMat[index(rowres, 3, rowsST)];
					stwidth = resMat[index(rowres, 4, rowsST)];
					stheight = resMat[index(rowres, 5, rowsST)];
					double dist = 1 - boxiou(gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight);
                    if (dist <= threshold) 
                    {
                        alldist[i][j] = dist;
                        // Add unique identifier to break ties
                        alldist[i][j] += 1e-9 * uid;
                        uid++;
                    }
				}
			}
		}

		vector<int> Lmate, Rmate;
		MinCostMatching(alldist, Lmate, Rmate);

		for (int k = 0; k < Lmate.size(); k++) {
			if (alldist[k][Lmate[k]] == INF) continue;
			M[t][unmappedGt[k]] = unmappedEs[Lmate[k]];
			if (VERBOSE) printf("%d: map %d with %d\n", t+1, unmappedGt[k]+1, unmappedEs[Lmate[k]]+1);
		}

		vector<int> curtracked, alltrackers, mappedtrackers, falsepositives, set1;

		for (unordered_map<int,int>::iterator it = M[t].begin(); it != M[t].end(); it++) {
			curtracked.push_back(it->first);
			set1.push_back(it->second);
		}
		for (unordered_map<int,int>::iterator it = stInd[t].begin(); it != stInd[t].end(); it++) alltrackers.push_back(it->first);

		sort(set1.begin(), set1.end());
		sort(alltrackers.begin(), alltrackers.end());
		set_intersection(set1.begin(), set1.end(), alltrackers.begin(), alltrackers.end(), inserter(mappedtrackers, mappedtrackers.begin()));
		set_difference(alltrackers.begin(), alltrackers.end(), mappedtrackers.begin(), mappedtrackers.end(), inserter(falsepositives, falsepositives.end()));

		for (int k = 0; k < falsepositives.size(); k++) allfalsepos[t][falsepositives[k]] = falsepositives[k];

		//  mismatch errors
		if (t > 0)
		{
			for (int k = 0; k < curtracked.size(); k++)
			{
				int ct = curtracked[k];
				int lastnonempty = -1;
				for (int j = t - 1; j >= 0; j--) {
					if (M[j].find(ct) != M[j].end()) {
						lastnonempty = j; break;
					}
				}
				if (gtInd[t-1].find(ct)!=gtInd[t-1].end() && lastnonempty != -1)
				{
					int mtct = -1, mlastnonemptyct = -1;
					if (M[t].find(ct) != M[t].end()) mtct = M[t][ct];
					if (M[lastnonempty].find(ct) != M[lastnonempty].end()) mlastnonemptyct = M[lastnonempty][ct];

					if (mtct != mlastnonemptyct)
						mme[t]++;
				}
			}
		}

		c[t] = curtracked.size();
		for (int k = 0; k < curtracked.size(); k++)
		{
			int ct = curtracked[k];
			int eid = M[t][ct];
			if (world)
			{
				double gtx, gty, stx, sty;
				int rowgt = gtInd[t][ct];
				int rowres = stInd[t][eid];
				gtx = gtMat[index(rowgt, 6, rowsGT)];
				gty = gtMat[index(rowgt, 7, rowsGT)];
				stx = resMat[index(rowres, 6, rowsST)];
				sty = resMat[index(rowres, 7, rowsST)];
				d[t][ct] = euclidean(gtx, gty, stx, sty);
			}
			else
			{
				double gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight;
				int rowgt = gtInd[t][ct];
				int rowres = stInd[t][eid];
				gtleft = gtMat[index(rowgt, 2, rowsGT)];
				gttop = gtMat[index(rowgt, 3, rowsGT)];
				gtwidth = gtMat[index(rowgt, 4, rowsGT)];
				gtheight = gtMat[index(rowgt, 5, rowsGT)];
				stleft = resMat[index(rowres, 2, rowsST)];
				sttop = resMat[index(rowres, 3, rowsST)];
				stwidth = resMat[index(rowres, 4, rowsST)];
				stheight = resMat[index(rowres, 5, rowsST)];
				d[t][ct] = 1 - boxiou(gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight);
			}
		}

		for (unordered_map<int,int>::iterator it = stInd[t].begin(); it != stInd[t].end(); it++) fp[t]++;
		fp[t] -= c[t];
		m[t] = g[t] - c[t];
	}
	
	// Copy back to matlab
	for (int k = 0; k < Fgt; k++) {
		mmeOut[k] = mme[k];
		cOut[k] = c[k];
		fpOut[k] = fp[k];
		gOut[k] = g[k];
		mOut[k] = m[k];
	}

	for (int i = 0; i < Fgt; i++) {
		for (int j = 0; j < Ngt; j++) {
			dOut[index(i, j, Fgt)] = d[i][j];
			dOut[index(i, j, Fgt)] = (d[i][j] == INF ? mxGetInf() : d[i][j]);
		}

		for (unordered_map<int, int>::iterator it = M[i].begin(); it != M[i].end(); it++) {
			int j = it->first;
			MOut[index(i, j, Fgt)] = M[i][j] + 1; // matlab indexed
		}

		for (int j = 0; j < Nst; j++) {
			allfalseposOut[index(i, j, Fgt)] = allfalsepos[i][j];
		}
	}
	

}