function [Xi Yi]=worldToImage(Xw,Yw,Zw,mR,mT,mInt,mGeo)

% if scalar, just up/downscale = orthographic
if length(mR)==1
     Xi=Xw/mR;
     Yi=Yw/mR;
 else
    
    %		/* convert from world coordinates to camera coordinates */
    x=[mR mT]*[Xw;Yw;Zw;1];
    xc = x(1);
    yc = x(2);
    zc = x(3);
    
    %		/* convert from camera coordinates to undistorted sensor plane coordinates */
    Xu = mInt.mFocal * xc / zc;
    Yu = mInt.mFocal * yc / zc;
    
    %		/* convert from undistorted to distorted sensor plane coordinates */
    [Xd Yd]=undistortedToDistortedSensorCoord (Xu, Yu, mInt.mKappa1);
    %         Xd=Xu;
    %         Yd=Yu;
    
    %         Rusq=Xu*Xu+Yu*Yu;
    %         Ru=sqrt(Xu*Xu+Yu*Yu);
    %         Xd=Xu*(1+mInt.mKappa1*Rusq);
    %         Yd=Yu*(1+mInt.mKappa1*Rusq);
    
    
    %		/* convert from distorted sensor plane coordinates to image coordinates */
    Xi = Xd * mInt.mSx / mGeo.mDpx + mInt.mCx;
    Yi = Yd / mGeo.mDpy + mInt.mCy;
 end
end