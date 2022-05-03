function [Xfd Yfd]=undistortedToDistortedImageCoord (Xfu, Yfu, mDpx, mDpy, mCx, mCy, mSx, mKappa1)

	
% 		/* convert from image to sensor coordinates */
		Xu = mDpx * (Xfu - mCx) / mSx;
		Yu = mDpy * (Yfu - mCy);
		
% 		/* convert from undistorted sensor to distorted sensor plane coordinates */
		[Xd Yd]=undistortedToDistortedSensorCoord(Xu, Yu, mKappa1);
		
% 		/* convert from sensor to image coordinates */
		Xfd = Xd * mSx / mDpx + mCx;
		Yfd = Yd / mDpy + mCy;
end