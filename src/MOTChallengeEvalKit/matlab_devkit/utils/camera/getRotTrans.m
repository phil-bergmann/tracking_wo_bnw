function [mR, mT]=getRotTrans(camPar)
% 
% 
% (C) Anton Andriyenko, 2012
%
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without written permission from the authors.

if camPar.ortho
    mR=camPar.mR; mT=camPar.mT;
else
    %%% Rotation Translation %%%
    mT=[camPar.mExt.mTx;camPar.mExt.mTy;camPar.mExt.mTz];
    sa = sin(camPar.mExt.mRx);
    ca = cos(camPar.mExt.mRx);
    sb = sin(camPar.mExt.mRy);
    cb = cos(camPar.mExt.mRy);
    sg = sin(camPar.mExt.mRz);
    cg = cos(camPar.mExt.mRz);
    
    mR11 = cb * cg;
    mR12 = cg * sa * sb - ca * sg;
    mR13 = sa * sg + ca * cg * sb;
    mR21 = cb * sg;
    mR22 = sa * sb * sg + ca * cg;
    mR23 = ca * sb * sg - cg * sa;
    mR31 = -sb;
    mR32 = cb * sa;
    mR33 = ca * cb;
    
    mR=[mR11 mR12 mR13;
        mR21 mR22 mR23;
        mR31 mR32 mR33];
end
end