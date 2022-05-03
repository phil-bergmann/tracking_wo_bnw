mex utils/MinCostMatching.cpp -outdir utils CXXFLAGS="$CXXFLAGS --std=c++11"
mex utils/clearMOTMex.cpp -outdir utils CXXFLAGS="$CXXFLAGS --std=c++11"
mex utils/costBlockMex.cpp -outdir utils COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="$CXXFLAGS --std=c++11"
