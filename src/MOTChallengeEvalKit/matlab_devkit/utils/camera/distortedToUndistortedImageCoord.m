function [Xfu Yfu]=distortedToUndistortedImageCoord (Xfd, Yfd, mDpx, mDpy, mCx, mCy, mSx, mKappa1)


% 		/* convert from image to sensor coordinates */
		Xd = mDpx * (Xfd - mCx) / mSx;
		Yd = mDpy * (Yfd - mCy);
		
% 		/* convert from distorted sensor to undistorted sensor plane coordinates */
		[Xu Yu]=distortedToUndistortedSensorCoord(Xd, Yd, mKappa1);
		
% 		/* convert from sensor to image coordinates */
		Xfu = Xu * mSx / mDpx + mCx;
		Yfu = Yu / mDpy + mCy;
	
end