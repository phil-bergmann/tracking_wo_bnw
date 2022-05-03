import cv2
import glob
import json
import re
import math
import numpy as np
### Module imports ###
import sys
#from modules.reconstruction.Triangulate import Triangulate

class Camera:
    """
    Class implementation for representing a camera
    """
    
    def __init__(self, intrinsicPath=None, extrinsicPath=None):
        """
        Initialize object
        """
        
        self.dist = None # Lens distortion coefficients
        self.K = None    # Intrinsic camera parameters
        self.R = None    # Extrinsic camera rotation
        self.t = None    # Extrinsic camera translation
        self.plane = None # Water interface
        self.roi = None

        if(intrinsicPath is not None):
            self.K, self.dist = self.loadIntrinsic(intrinsicPath)

        if(extrinsicPath is not None):
            self.calcExtrinsicFromJson(extrinsicPath)

    def loadIntrinsic(self, path):
        """
        Checks whether a 2D point is within the camera ROI
        The camera ROI is defined by the corners of the water
        
        Input:
            path: path to json file with the intrinsic parameters
            
        Output:
            K: camera matrix
            dist: distortion coefficients
        """    
        # Load json file
        with open(path) as f:
            data = f.read()

        # Remove comments
        pattern = re.compile('/\*.*?\*/', re.DOTALL | re.MULTILINE)
        data = re.sub(pattern, ' ', data)
        
        # Parse json
        data = json.loads(data)

        # Load camera matrix K and distortion coefficients
        K = np.array(data["K"])
        dist = np.array(data["Distortion"]).flatten()
        return K, dist
        

    def withinRoi(self, x, y):
        """
        Checks whether a 2D point is within the camera ROI
        The camera ROI is defined by the corners of the water
        
        Input:
            x: x coordinate of the 2D point
            y: y coordinate of the 2D point
            
        Output:
            Boolean on whether the 2D point is within the region of interest
        """
        
        if(self.roi is None or self.K is None):
            return True
        
        p1 = cv2.undistortPoints(np.array([[[x,y]]]), self.K, self.dist).flatten()
        if(p1[0] < self.roi['x'][0] or p1[0] > self.roi['x'][1]):
            return False
        if(p1[1] < self.roi['y'][0] or p1[1] > self.roi['y'][1]):
            return False
        return True
        

    def backprojectPoint(self, x, y):
        """
        Backproject 2D point into a 3D ray i.e. finds R = R^-1 K^-1 [x y 1]^T
        
        Input:
            x: x coordinate of the 2D point
            y: y coordinate of the 2D point
        
        Output:
            ray: 
            ray0:
        """
        
        if(self.R is None or self.t is None):
            print("Camera: Error - Extrinsic parameters is needed to back-project a point")
            return
        if(self.K is None or self.dist is None):
            print("Camera: Error - Intrinsic parameters is needed to back-project a point")
            return
        
        # Calculate R = K^-1 [x y 1]^T and account for distortion
        ray = cv2.undistortPoints(np.array([[[x,y]]]), self.K, self.dist)
        ray = ray[0][0] # Unwrap point from array of array
        ray = np.array([ray[0], ray[1], 1.0])
        
        # Calculate R^-1 R
        ray = np.dot(np.linalg.inv(self.rot), ray)
        ray /= np.linalg.norm(ray)

        # Calculate camera center, i.e. -R^-1 t
        ray0 = self.pos
        return ray, ray0


    def forwardprojectPoint(self, x, y, z, correctRefraction=True, verbose=False):
        """
        Forwards project a 3D point onto the camera plane
        
        Input:
            x: x coordinate of the 3D point
            y: y coordinate of the 3D point
            z: z coordinate of the 3D point
            correctRefraction: Whether to correct for refraction when projecting
            verbose: Whether to write information when correcting for refraction
            
        Output:
            point: 2D point on the camera plane
        """
        
        if(correctRefraction is False):
            p3 = cv2.projectPoints(np.array([[[x,y,z]]]), self.R, self.t, self.K, self.dist)[0]
            return p3.flatten()
        
        p1 = np.array([x,y,z])
        c1 = self.pos.flatten()
        w = self.plane.normal
        
        # 1) Plane between p1 and c1, perpendicular to w
        n = np.cross((p1-c1), w)
        if(verbose):
            print("Plane normal: {0}".format(n))

        # 2) Find plane origin and x/y directions
        #    i.e. project camera position onto refraction plane
        p0 = self.plane.intersectionWithRay(-w, c1)        
        if(verbose):
            print("Plane origin: {0}".format(p0))

        pX = c1-p0
        pX = pX / np.linalg.norm(pX)
        pY = np.cross(n, pX)
        pY = pY / np.linalg.norm(pY)
        if(verbose):
            print("Plane x direction: {0}".format(pX))
            print("Plane y direction: {0}".format(pY))
            print("Direction dot check: \n{0}\n{1}\n{2}".format(np.dot(pX,pY),
                                                                np.dot(n,pX),
                                                                np.dot(n,pY)))

        # 3) Project 3d position and camera position onto 2D plane
        p1_proj = np.array([np.dot(pX, p1-p0),
                            np.dot(pY, p1-p0)])
        c1_proj = np.array([np.dot(pX, c1-p0),
                            np.dot(pY, c1-p0)])
        if(verbose):
            print("P1 projection: {0}".format(p1_proj)) 
            print("C1 projection: {0}".format(c1_proj))

        # 4) Construct 4'th order polynomial
        sx = p1_proj[0]
        sy = p1_proj[1]
        e = c1_proj[0]
        r = 1.33
        N = (1/r**2) - 1

        y4 = N
        y3 = -2*N*sy
        y2 = (N * sy**2+(sx**2/r**2)-e**2)
        y1 = 2 * e**2 * sy
        y0 = -e**2 * sy**2

        coeffs = [y4, y3, y2, y1, y0]
        res = np.roots(coeffs)
        
        real = np.real(res)
        resRange = (min(1e-6,sy),max(1e-6,sy))

        finalRes = []
        for r in real:
            if(r > resRange[0] and r < resRange[1]):                
                finalRes.append(r)
        finalRes = finalRes[np.argmax([abs(x) for x in finalRes])]
        refPoint = (finalRes*pY)+p0
            
        if(verbose):
            print("\n")
            print("4th order poly details:")
            print(" - Range: {0}".format(resRange))
            print(" - Roots: {0}".format(real))
            print(" - finalRes: {0}".format(finalRes))
            print(" - pY: {0}".format(pY))
            print(" - p0: {0}".format(p0))
            print(" - Intersection point: {0}".format(refPoint))

        p3 = cv2.projectPoints(np.array([[[*refPoint]]]), self.R, self.t, self.K, self.dist)[0]
        return p3.flatten()
        

    def getExtrinsicMat(self):
        """
        Returns the extrinsic camera matrix i.e. [R | t]
        
        Ouput:
            Mat: Numpy array of sie 3x4
        """
        
        if(self.R is None or self.t is None):
            print("Camera: Error - Extrinsic parameters is needed for the extrinsic matrix")
            return        
        return np.concatenate((self.rot, self.t), axis=1)
        
    
    # Get object points for checkerboard calibration
    def getObjectPoints(self, checkerboardSize, squareSize):
        """
        Get object points for checkerboard calibration
        
        Input:
            checkerboardSize: Tuple containing the number of inner points along each direction of the checkerboard
            squareSize: The size of the squares in centimeters
            
        Output:
            objP: Numpy matrix containing the detected object points
        """
        
        objP = np.zeros((checkerboardSize[0]*checkerboardSize[1],3), np.float32)
        objP[:,:2] = np.mgrid[0:checkerboardSize[0],0:checkerboardSize[1]].T.reshape(-1,2)*squareSize
        return objP


    def showCorners(self, img, checkerboardSize, corners, resize=0.8):
        """
        Show the checkerboard corners for 5 seconds
        
        Input:
            img: Input image which is drawn on
            checkerboardSize: Tuple containing the number of inner points along each direction of the checkerboard
            corner: Array of detected corners
            resize: Amount to resize the image by
        """
        
        img = cv2.drawChessboardCorners(img, checkerboardSize,corners, True)
        img = cv2.resize(img, (0,0), fx=resize, fy=resize)     
        cv2.imshow('detected corners',img)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()


    def undistortImage(self, img):
        """
        Undistort image using distortion coefficients and intrinsic paramters
        
        Input:
            img: Image which has to be undistorted
            
        Output:
            dst: The undistorted image        
        """
        
        h,w = img.shape[:2]
        newCam, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w,h), 1, (w,h))
        mapX, mapY = cv2.initUndistortRectifyMap(self.K, self.dist, None, newCam, (w,h), 5)
        dst = cv2.remap(img, mapX, mapY, cv2.INTER_LINEAR)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        print("Hejsan")
        return dst


    def calibrateFromFolder(self, imageFolder, checkerboardSize, squareSize, debug=False, verbose=False):
        """
        Find intrinsic parameters for the camera using a folder of images
        
        Input:
            imageFolder: String path to folder containing images for calibration
            checkerboardSize: Tuple containing the number of inner points along each direction of the checkerboard
            squareSize: The size of the squares in centimeters
            debug: Boolean indicating whether to write debug messages when calibrating
            verbose: Boolean indicating whether to explicitely write the image paths used
            
        Output:
            intri: Intrinsic camera parameters
            dist: Lens distortion coefficients
        """
        
        imageNames = glob.glob(imageFolder)
        images = []
        if(verbose):
            print("Calibration image names:")
        for imgPath in imageNames:
            if(verbose):
                print(imgPath)
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            images.append(img)

        return self.calibrate(images, checkerboardSize, squareSize, debug=debug, verbose=verbose)
        
    
    def calibrate(self, images, checkerboardSize, squareSize, debug=False, verbose=False):
        """
        Find intrinsic parameters for the camera
        
        Input:
            images: A list of numpy arrays, containing the images to be used
            checkerboardSize: Tuple containing the number of inner points along each direction of the checkerboard
            squareSize: The size of the squares in centimeters
            debug: Boolean indicating whether to write debug messages when calibrating
            verbose: Boolean indicating whether to explicitely write the image paths used
            
        Output:
            intri: Intrinsic camera parameters
            dist: Lens distortion coefficients                
        """
        
        if(len(images) < 1):
            print("Camera: Error - Too few images for calibration")
            return

        # Find checkerboard corners in each image
        objP = self.getObjectPoints(checkerboardSize, squareSize)
        objPoints = []
        imgPoints = []
        imgCounter = 0
        for img in images:
            ret, corners = cv2.findChessboardCorners(img, checkerboardSize, None)
            imgCounter += 1
            if(ret):
                objPoints.append(objP)
                imgPoints.append(corners)
                if(debug):
                    self.showCorners(img, checkerboardSize, corners)
            else:
                print("Camera: Info - Unable to find corners in an image during calibration")
            if(verbose):
                print("Camera calibration - progress: {0} / {1}".format(imgCounter,len(images)))

        # Calibrate the camera
        ret, intri, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, img.shape[::-1], None, None, flags=cv2.CALIB_RATIONAL_MODEL)
        if(ret):
            self.dist = dist
            self.K = intri
            return intri, dist
        else:
            print("Camera: Error - Calibration failed!")
        return

    def getPosition(self):
        """
        Calculate camera position i.e. -R^-1 t
        
        Output:
            camPos: Numpy array indicating a 3D position
        """
        
        if(self.R is None or self.t is None):
            print("Camera: Error - Extrinsic parameters is needed to find the camera postion")
            return
        rotMat = self.getRotationMat()
        camPos = -np.dot(rotMat.T, self.t)
        return camPos.T


    def getRotationMat(self):
        """
        Returns the rotation matrix of the camera
        
        Output:
            Mat: Numpy matrix containing the rotation matrix
        """
        
        if(self.R is None):
            print("Camera: Error - Extrinsic parameters is needed to return rotation matrix")
            return
        return cv2.Rodrigues(self.R)[0]


    def calcExtrinsicFromJson(self, jsonPath, method=None):
        """
        Find extrinsic parameters for the camera using
        image <--> world reference points from a JSON file
        
        Input:
            jsonPath: String path to the JSON file
            method: Indicating whether a specific method should be used when calculating extrinsic parameters. Default is None
        """
        
        # Load json file
        with open(jsonPath) as f:
            data = f.read()

        # Remove comments
        pattern = re.compile('/\*.*?\*/', re.DOTALL | re.MULTILINE)
        data = re.sub(pattern, ' ', data)
        
        # Parse json
        data = json.loads(data)

        # Convert to numpy arrays
        cameraPoints = np.zeros((4,1,2))
        worldPoints = np.zeros((4,3))

        for i,entry in enumerate(data):
            cameraPoints[i][0][0] = entry["camera"]["x"]
            cameraPoints[i][0][1] = entry["camera"]["y"]

            worldPoints[i][0] = entry["world"]["x"]
            worldPoints[i][1] = entry["world"]["y"]
            worldPoints[i][2] = entry["world"]["z"]

        # Calc extrinsic parameters
        if(method == None):
            self.calcExtrinsic(worldPoints.astype(float), cameraPoints.astype(float))
        else:
            self.calcExtrinsic(worldPoints.astype(float), cameraPoints.astype(float), method=method)

        self.rot = cv2.Rodrigues(self.R)[0]
        self.pos = self.getPosition()
    
    
    def calcExtrinsic(self, worldPoints, cameraPoints, method=cv2.SOLVEPNP_ITERATIVE):
        """
        Find extrinsic parameters for the camera
        Mainly two methods:
            cv2.SOLVEPNP_P3P and cv2.SOLVEPNP_ITERATIVE
        See: http://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#ggaf8729b87a4ca8e16b9b0e747de6af27da9f589872a7f7d687dc58294e01ea33a5
        
        Input:
            worldPoints: World coordinates (x,y,z) in centimeters. Is represented as a 4 x 3 matrix, one of each corner of the aquarium
            cameraPoints: Camera coordinates in pixels. Is represented as a 4 x 1 x 2 matrix, one of each corner of the aquarium 
            method: Method to use when calculating extrinsic parameters. Default is cv2.SOLVEPNP_ITERATIVE
            
        Output:
            rvec: Rotation vector that together with tvec can transform from world ot camera coordinates
            tvec: Translation vector that together with rvec can transform from world ot camera coordinates
        """
        
        if(self.K is None or self.dist is None):
            print("Camera: Error - Calibrate camera before finding extrinsic parameters!")
            return

        ret, rvec, tvec = cv2.solvePnP(worldPoints,cameraPoints,self.K,self.dist,flags=method)
        if(ret):
            self.R = rvec
            self.t = tvec
            self.plane = Plane(worldPoints)
            # Ensure that the plane normal points towards the camera
            if(np.dot(self.getPosition(), self.plane.normal) < 0):
                self.plane.normal = -self.plane.normal

            # Create roi
            roiPts = cv2.undistortPoints(cameraPoints, self.K, self.dist)
            roiPts = roiPts.reshape(4,2)
            self.roi = {}
            self.roi["x"] = (min(roiPts[:,0]), max(roiPts[:,0]))
            self.roi["y"] = (min(roiPts[:,1]), max(roiPts[:,1]))
            return rvec, tvec
        else:
            print("Camera: Error - Failed to find extrinsic parameters")
        return 

class Plane:
    """
    Class implementation for representing a plane
    """
    
    def __init__(self, points=None):
        """
        Initialize object
        """
        
        self.points = None
        self.normal = None
        self.x = None
        self.y = None
        if(points is not None):
            self.normal = self.calculateNormal(points)
            self.points = points


    def calculateNormal(self, points, verbose=False):
        """    
        Calculates the plane normal n = [a b c] and d for the plane: ax + by + cz + d = 0
        
        Input:
            points: List of 3D points used to calculate the plane
            verbose: Whether to write the resulting plane normal and plane
        
        Output:
            n: A numpy vector containing the 3D plane normal
        """
        
        if(len(points) < 4):
            print("Error calculating plane normal. 4 or more points needed")
        #Calculate plane normal
        self.x = points[1]-points[2]
        self.x = self.x/np.linalg.norm(self.x)
        self.y = points[3]-points[2]
        self.y = self.y/np.linalg.norm(self.y)
        n = np.cross(self.x,self.y)
        n /= np.linalg.norm(n)
        if(verbose):
            print("Plane normal: \n {0} \n plane d: {1}".format(n,d))
        return n


    def intersectionWithRay(self, r, r0, verbose=False):
        # TODO : What happens if the plane and ray are parallel
        
        """    
        Calcuates the intersection between a plane and a ray
        
        Input: 
            r: Numpy vector containing the ray direction
            ro: Numpy vector containing a point on the ray
            verbose: Whether to print information regarding the calculated itnersection
            
        Output:
            intersection: A 3D point indicating the intersection between a ray and plane
        """
        
        n0 = self.points[0]
        t = np.dot((n0 - r0), self.normal)
        t /= np.dot(r,self.normal)
        intersection = (t * r) + r0
        if(verbose):
            print("t: \n" + str(t))
            print("Intersection: \n" + str(intersection))
        return intersection.flatten()
