function [Xu Yu]=distortedToUndistortedSensorCoord (Xd, Yd, mKappa1)
% 	/* convert from distorted to undistorted sensor plane coordinates */
	distortion_factor = 1 + mKappa1 * (Xd*Xd + Yd*Yd);
	Xu = Xd * distortion_factor;
	Yu = Yd * distortion_factor;
end