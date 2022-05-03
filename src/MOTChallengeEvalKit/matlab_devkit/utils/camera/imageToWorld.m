function [Xw, Yw, Zw]=imageToWorld(Xi, Yi, camPar)

% if scalar, just up/downscale = orthographic
if camPar.ortho
    Xw=Xi*camPar.scale;
    Yw=Yi*camPar.scale;
    Zw=0;
else
    
    mGeo=camPar.mGeo;
    mExt=camPar.mExt;
    mInt=camPar.mInt;
    
    mTx=mExt.mTx;
    mTy=mExt.mTy;
    mTz=mExt.mTz;
    
    mT=[mExt.mTx;mExt.mTy;mExt.mTz];
    
    %% internal init
    sa = sin(mExt.mRx);
    ca = cos(mExt.mRx);
    sb = sin(mExt.mRy);
    cb = cos(mExt.mRy);
    sg = sin(mExt.mRz);
    cg = cos(mExt.mRz);
    
    mR11 = cb * cg;
    mR12 = cg * sa * sb - ca * sg;
    mR13 = sa * sg + ca * cg * sb;
    mR21 = cb * sg;
    mR22 = sa * sb * sg + ca * cg;
    mR23 = ca * sb * sg - cg * sa;
    mR31 = -sb;
    mR32 = cb * sa;
    mR33 = ca * cb;
    
    
    
    % 		/* convert from image to distorted sensor coordinates */
    Xd = mGeo.mDpx * (Xi - mInt.mCx) / mInt.mSx;
    Yd = mGeo.mDpy * (Yi - mInt.mCy);
    
    % 		/* convert from distorted sensor to undistorted sensor plane coordinates */
    [Xu Yu]=distortedToUndistortedSensorCoord (Xd, Yd, mInt.mKappa1);
    
    % 		/* calculate the corresponding xw and yw world coordinates	 */
    % 		/* (these equations were derived by simply inverting	 */
    % 		/* the perspective projection equations using Macsyma)	 */
    Zw=0;
    common_denominator = ((mR11 * mR32 - mR12 * mR31) * Yu + ...
        (mR22 * mR31 - mR21 * mR32) * Xu - ...
        mInt.mFocal * mR11 * mR22 + mInt.mFocal * mR12 * mR21);
    
    Xw = (((mR12 * mR33 - mR13 * mR32) * Yu + ...
        (mR23 * mR32 - mR22 * mR33) * Xu - ...
        mInt.mFocal * mR12 * mR23 + mInt.mFocal * mR13 * mR22) * Zw + ...
        (mR12 * mTz - mR32 * mTx) * Yu + ...
        (mR32 * mTy - mR22 * mTz) * Xu - ...
        mInt.mFocal * mR12 * mTy + mInt.mFocal * mR22 * mTx) / common_denominator;
    
    Yw = -(((mR11 * mR33 - mR13 * mR31) * Yu + ...
        (mR23 * mR31 - mR21 * mR33) * Xu - ...
        mInt.mFocal * mR11 * mR23 + mInt.mFocal * mR13 * mR21) * Zw + ...
        (mR11 * mTz - mR31 * mTx) * Yu + ...
        (mR31 * mTy - mR21 * mTz) * Xu - ...
        mInt.mFocal * mR11 * mTy + mInt.mFocal * mR21 * mTx) / common_denominator;
% else
%     error('imageToWorld: camera parameters format unknown');
end
end