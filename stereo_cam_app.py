# import collections
import cv2
import numpy as np
import sys
from typing import NamedTuple
import os.path
import glob

# Basic data structure to keep both images from a stereo capture
class StereoImg(NamedTuple):
    left_img: np.ndarray
    right_img: np.ndarray


# Get the StereoImg using the path passed as parameters
def get_stereo_img(argv, img_type):
    # We start by checking if the paths are pointing to some valid files.
    if not os.path.isfile(argv[0]):
        print("The path : ", argv[0], " doesn't point to a valid file.")
        sys.exit()
    elif len(argv) == 2 and not os.path.isfile(argv[0]):
        print("The path : ", argv[1], " doesn't point to a valid file.")
        sys.exit()

    # If there's only one path, we assume both images are together and we split them in a left and right image.
    if len(argv) == 1:
        merged_img = cv2.imread(argv[0], img_type)
        # Checks if the image has been read correctly
        if merged_img is None:
            print("Something went wrong and the image at the path : ", argv[0], " couldn't be read.")
            sys.exit()
        merged_h, merged_w = merged_img.shape

        width_cutoff = merged_w // 2
        img_left = merged_img[:, :width_cutoff]
        img_right = merged_img[:, width_cutoff:]

    # Otherwise, we just have to read both images
    else:
        img_left = cv2.imread(argv[0], img_type)
        img_right = cv2.imread(argv[1], img_type)
        # Checks if both images have been read correctly
        if img_left is None:
            print("Something went wrong and the image at the path : ", argv[0], " couldn't be read.")
            sys.exit()
        elif img_right is None:
            print("Something went wrong and the image at the path : ", argv[1], " couldn't be read.")
            sys.exit()

    # Returns the combination the left and right image in a StereoImg
    images = StereoImg(img_left, img_right)
    return images


# Calibrates the images using the calibration data. The data can be found in the README
def calibrate_stereo_cam(argv, img_type):
    
    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
    chessboardSize = (9,7)
    frameSize = (640,480)


    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    objp = objp * 20
    #print(objp)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    imgpointsR = [] # 2d points in image plane.

    for i in range(1,4):
        path=['Img/Calibration/mire'+str(i)+'.jpg']
        image=get_stereo_img(path,0)
        cv2.imwrite('Img/Calibration/stereoLeft/mire'+str(i)+'.jpg', image[0])
        cv2.imwrite('Img/Calibration/stereoRight/mire'+str(i)+'.jpg', image[1])
    
    imagesRight = glob.glob('Img/Calibration/stereoRight/*.JPG')
    imagesLeft = glob.glob('Img/Calibration/stereoLeft/*.JPG')

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):

        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL and retR == True:

            objpoints.append(objp)

            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv2.imshow('img left', imgL)
            cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv2.imshow('img right', imgR)
            cv2.waitKey(1000)


    cv2.destroyAllWindows()




    ############## CALIBRATION #######################################################

    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))



    ########## Stereo Vision Calibration #############################################

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same 

    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)




    ########## Stereo Rectification #################################################

    rectifyScale= 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

    stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)
    
    img=get_stereo_img(argv, img_type)
    imgL=img[0]
    imgR=img[1]
    dstL = cv2.undistort(imgL, newCameraMatrixL, distL, None, None)
    dstR = cv2.undistort(imgL, newCameraMatrixL, distL, None, None)
    cv2.imwrite('Img/ImgRectL/undistL.jpg', dstL)
    cv2.imwrite('Img/ImgRectR/undistR.jpg', dstR) 

    print("Printing the new parameters!")
    print("Camera matrix : \n")
    print(newCameraMatrixL)
    print("dist : \n")
    print(distL)
    print("rvecs : \n")
    print(rectL)
    print("tvecs : \n")
    print(projMatrixL)
    
    dst = StereoImg(dstL, dstR)
    return dst
    # Returns an updated StereoImg


# Do the matching between the images
def image_matching(stereo_img):
    print("No yet implemented")
    #
    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
    # disparity = stereo.compute(stereo_img.left_img, stereo_img.right_img)
    #
    # cv2.imshow('disparity', disparity)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Returns a 2D matrix with the disparities for each pixel


# Fill a 2D matrix with the coordinates of each pixel. The method should return the error coefficient found between
# the two images and a way to see the data (either by creating an image using the coordinates or by printing some
# kind of graph)
def depth_rendering():
    print("No yet implemented")
    # Returns a 2D matrix with the coordinates of each pixel in the scene


# Debugging method used to show the content of a StereoImg
def debug_print_lr(stereo_img):
    # That's only there for debugging purpose, it shows the image read from the files
    cv2.imshow('left', stereo_img.left_img)
    cv2.imshow('right', stereo_img.right_img)
    # Stop the program until we press a key and then closes every window that has been opened
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main method, everything starts and ends here!
def main(argv):
    # If the number of parameters is > 2, exit the process
    if len(argv) > 2 or len(argv) == 0:
        print("The number of arguments should be either 1 or 2, please refer to the README.md for more information "
              "about the execution of this program.")
        sys.exit()

    # Get the images from the path passed as parameters Since we want to match the stereo capture, we are setting the
    # capture type to greyscale. It can be modified by changing the second parameter of the function
    stereo_img = get_stereo_img(argv, 0)

    # Debugging method used to print both left and right images
    debug_print_lr(stereo_img)

    # To implement...
    calibrate_stereo_cam(['Img/nico1.jpg'],0)

    # # Proceeds to the matching of the left and right images and returns a 2D matrix with the disparity for each pixels
    # image_matching(stereo_img)

    # To implement...
    depth_rendering()


if __name__ == "__main__":
    main(sys.argv[1:])
