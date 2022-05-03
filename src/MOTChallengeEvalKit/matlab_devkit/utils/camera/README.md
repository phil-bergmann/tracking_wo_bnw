# Camera Utility Files

This folder contains scripts for computing image-to-world and world-to-image projections.

`calib_towncentre.m` is to be used with the AVG-TownCentre sequence.

The rest is for the PETS sequence, which follows the Tsai camera model. parseCameraParameters can be used to read the camera xml file and the two scripts `imageToWorld.m` and `worldToImage.m` contain the projection routines.