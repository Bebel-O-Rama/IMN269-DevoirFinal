# import collections
import cv2
import numpy as np
import sys
from typing import NamedTuple
import os.path

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
        merged_img = cv2.imread(argv[0])
        grayed_merged_img = cv2.cvtColor(merged_img, cv2.COLOR_BGR2GRAY)
        # Checks if the image has been read correctly
        if merged_img is None:
            print("Something went wrong and the image at the path : ", argv[0], " couldn't be read.")
            sys.exit()
        merged_h, merged_w = grayed_merged_img.shape

        width_cutoff = merged_w // 2
        img_left = grayed_merged_img[:, :width_cutoff]
        img_right = grayed_merged_img[:, width_cutoff:]

    # Otherwise, we just have to read both images
    else:
        img_left = cv2.imread(argv[0])
        img_right = cv2.imread(argv[1])
        grayed_img_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        grayed_img_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
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


# def calibrate_stereo_cam(stereo_img, isPrintigCorner):
#     # termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     chessboardSize = (9, 7)
#     frameSize = (1280, 960)
#
#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
#
#     objp = objp * 20
#     print(objp)
#
#     # Arrays to store object points and image points from all the images.
#     objpoints = []  # 3d point in real world space
#     imgpointsL = []  # 2d points in image plane.
#     imgpointsR = []  # 2d points in image plane.
#
#     # Read and parse the mire into a left and right array
#     stereo_mire_1 = get_stereo_img(['Img/Calibration/mire1.jpg'], 0)
#     stereo_mire_2 = get_stereo_img(['Img/Calibration/mire2.jpg'], 0)
#     stereo_mire_3 = get_stereo_img(['Img/Calibration/mire3.jpg'], 0)
#
#     mire_left = [stereo_mire_1.left_img, stereo_mire_2.left_img, stereo_mire_3.left_img]
#     mire_right = [stereo_mire_1.right_img, stereo_mire_2.right_img, stereo_mire_3.right_img]
#
#     for imgLeft, imgRight in zip(mire_left, mire_right):
#         # Find the chess board corners
#         retL, cornersL = cv2.findChessboardCorners(imgLeft, chessboardSize, None)
#         retR, cornersR = cv2.findChessboardCorners(imgRight, chessboardSize, None)
#
#         # If found, add object points, image points (after refining them)
#         if retL and retR == True:
#
#             objpoints.append(objp)
#
#             cornersL = cv2.cornerSubPix(imgLeft, cornersL, (5,5), (-1,-1), criteria)
#             imgpointsL.append(cornersL)
#
#             cornersR = cv2.cornerSubPix(imgRight, cornersR, (5,5), (-1,-1), criteria)
#             imgpointsR.append(cornersR)
#
#             if isPrintigCorner:
#                 # Draw and display the corners
#                 cv2.drawChessboardCorners(imgLeft, chessboardSize, cornersL, retL)
#                 cv2.imshow('img left', imgLeft)
#                 cv2.drawChessboardCorners(imgRight, chessboardSize, cornersR, retR)
#                 cv2.imshow('img right', imgRight)
#                 cv2.waitKey(1000)
#     cv2.destroyAllWindows()
#
#     retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
#     heightL, widthL = imgLeft.shape
#     newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1,
#                                                            (widthL, heightL))
#
#     retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
#     heightR, widthR = imgRight.shape
#     newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1,
#                                                            (widthR, heightR))
#
#     flags = 0
#     flags |= cv2.CALIB_FIX_INTRINSIC
#     # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
#     # Hence intrinsic parameters are the same
#
#     criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
#     retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(
#         objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, imgLeft.shape[::-1],
#         criteria_stereo, flags)
#
#
#     rectifyScale = 1
#     rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv2.stereoRectify(newCameraMatrixL, distL,
#                                                                                newCameraMatrixR, distR,
#                                                                                imgLeft.shape[::-1], rot, trans,
#                                                                                rectifyScale, (0, 0))
#
#     stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, imgLeft.shape[::-1], cv2.CV_16SC2)
#     stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, imgRight.shape[::-1], cv2.CV_16SC2)
#
#     dstL = cv2.undistort(stereo_img.left_img, newCameraMatrixL, distL, None, None)
#     dstR = cv2.undistort(stereo_img.right_img, newCameraMatrixR, distR, None, None)
#     cv2.imshow('left', dstL)
#     cv2.imshow('right', dstR)
#
#     cv2.waitKey(0)
#
#     print("Printing the new parameters!")
#     print("Camera matrix (left): \n")
#     print(newCameraMatrixL)
#     print("Camera matrix (right): \n")
#     print(newCameraMatrixR)
#     print("dist : \n")
#     print(distL)
#     print("rvecs : \n")
#     print(rectL)
#     print("tvecs : \n")
#     print(projMatrixL)

    # print("Saving parameters!")
    # cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_WRITE)
    #
    # cv_file.write('stereoMapL_x', stereoMapL[0])
    # cv_file.write('stereoMapL_y', stereoMapL[1])
    # cv_file.write('stereoMapR_x', stereoMapR[0])
    # cv_file.write('stereoMapR_y', stereoMapR[1])
    #
    # cv_file.release()


def calibrate_stereo_cam(stereo_img, isPrintingCorner):
    # Puts the stereo_img to calibrate into this array
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size = [1280, 960]
    chessboard_sqr_size = 20
    chessboard_size = (9, 7)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_sqr_size

    # Arrays to store object points and image points from all the images.
    objpointsL = []  # 3d point in real world space for left camera
    objpointsR = []  # 3d point in real world space for left camera
    objpoints = []  # 3d point in real world space for left camera

    imgpointsL = []  # 2d points in the left image plane.
    imgpointsR = []  # 2d points in right image plane.

    # Read and parse the mire into a left and right array
    stereo_mire_1 = get_stereo_img(['Img/Calibration/mire1.jpg'], 0)
    stereo_mire_2 = get_stereo_img(['Img/Calibration/mire2.jpg'], 0)
    stereo_mire_3 = get_stereo_img(['Img/Calibration/mire3.jpg'], 0)

    mire_left = [stereo_mire_1.left_img, stereo_mire_2.left_img, stereo_mire_3.left_img]
    mire_right = [stereo_mire_1.right_img, stereo_mire_2.right_img, stereo_mire_3.right_img]

    # mire_left = [stereo_mire_1.left_img]
    # mire_right = [stereo_mire_1.right_img]


    for img_L, img_R in zip(mire_left, mire_right):
        ret_L, corners_left = cv2.findChessboardCorners(img_L, chessboard_size, None)
        ret_R, corners_right = cv2.findChessboardCorners(img_R, chessboard_size, None)

        if ret_L and ret_R:
            objpoints.append(objp)
            corners_left = cv2.cornerSubPix(img_L, corners_left, (11, 11), (-1, -1), criteria)
            imgpointsL.append(corners_left)
            corners_right = cv2.cornerSubPix(img_R, corners_right, (11, 11), (-1, -1), criteria)
            imgpointsR.append(corners_right)
            # Draw and display the corners if the option has be set to true
            if isPrintingCorner:
                cv2.drawChessboardCorners(img_L, chessboard_size, corners_left, ret_L)
                cv2.imshow('Chessboard left', img_L)
                cv2.drawChessboardCorners(img_R, chessboard_size, corners_right, ret_R)
                cv2.imshow('Chessboard right', img_R)
                cv2.waitKey(0)

    # # Get the information from the left camera chessboard pattern
    # for img_L in mire_left:
    #     ret, corners = cv2.findChessboardCorners(img_L, (9, 7), None)
    #     # If found, add object points, image points (after refining them)
    #     if ret == True:
    #         objpointsL.append(objp)
    #         corners2 = cv2.cornerSubPix(img_L, corners, (11, 11), (-1, -1), criteria)
    #         imgpointsL.append(corners2)
    #         # Draw and display the corners if the option has be set to true
    #         if isPrintingCorner:
    #             cv2.drawChessboardCorners(img_L, (9, 7), corners2, ret)
    #             cv2.imshow('path', img_L)
    #             cv2.waitKey(0)
    #
    # # Get the information from the right camera chessboard pattern
    # for img_R in mire_right:
    #     ret, corners = cv2.findChessboardCorners(img_R, (9, 7), None)
    #     # If found, add object points, image points (after refining them)
    #     if ret == True:
    #         objpointsR.append(objp)
    #         corners2 = cv2.cornerSubPix(img_R, corners, (11, 11), (-1, -1), criteria)
    #         imgpointsR.append(corners2)
    #         # Draw and display the corners if the option has be set to true
    #         if isPrintingCorner:
    #             cv2.drawChessboardCorners(img_R, (9, 7), corners2, ret)
    #             cv2.imshow('path', img_R)
    #             cv2.waitKey(0)

    # Get the calibration parameters for both cameras
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, img_L.shape[::-1], None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, img_R.shape[::-1], None, None)

    print("-------------------------------")
    print("calibrateCamera (left):")
    print('retL\n', retL)
    print('matrix\n', mtxL)
    print('disto Left\n', distL)
    print('rot vec Left\n', rvecsL)
    print('transl vec Left\n', tvecsL)


    print("-------------------------------")
    print("calibrateCamera (right):")
    print('retR\n', retR)
    print('matrix\n', mtxR)
    print('disto Right\n', distR)
    print('rot vec Right\n', rvecsR)
    print('transl vec Right\n', tvecsR)

    # Refine the results
    hL, wL = img_L.shape[:2]
    hR, wR = img_R.shape[:2]

    # Maybe mettre 0 pour dist
    new_left_matrix, roi_L = cv2.getOptimalNewCameraMatrix(mtxL, 0, img_size, 1, (wL, hL))
    new_right_matrix, roi_R = cv2.getOptimalNewCameraMatrix(mtxR, 0, img_size, 1, (wR, hR))

    # Perform the stereo calibration with the intrinsic parameters
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Maybe mettre 0 pour dist
    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR,
                                                                                        new_left_matrix, 0, new_right_matrix,
                                                                                        0, img_L.shape[::-1],
                                                                                        criteria_stereo, flags)

    print("-------------------------------")
    print("stereoCalibrate :")
    print('retS\n', retS)
    print('new left matrix\n', new_mtxL)
    print('new right matrix\n', new_mtxR)
    print('disto left\n', distL)
    print('disto right\n', distR)
    print('Rotation\n', Rot)
    print('Translation\n', Trns)
    print('E: \n', Emat)
    print('F: \n', Fmat)


    # rectify_scale = 1
    # rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
    #                                                                           img_L.shape[::-1], Rot, Trns,
    #                                                                           rectify_scale, (0, 0))
    #
    # # Undistort the images
    # cleaned_left = cv2.undistort(stereo_img.left_img, mtxL, distL, None, new_left_matrix)
    # cv2.imshow('first', cleaned_left)
    # cv2.waitKey(0)
    #
    # x, y, w, h = roi_L
    # cleaned_left = cleaned_left[y:y+h, x:x+w]
    # cv2.imshow('post_ROI', cleaned_left)
    # cv2.waitKey(0)
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #
    # # Params from camera calibration
    # camMats = [cameraMatrix1, cameraMatrix2]
    # distCoeffs = [distCoeffs1, distCoeffs2]
    #
    # camSources = [0, 1]
    # for src in camSources:
    #     distCoeffs[src][0][4] = 0.0  # use only the first 2 values in distCoeffs
    #
    # # The rectification process
    # newCams = [0, 0]
    # roi = [0, 0]
    # for src in camSources:
    #     newCams[src], roi[src] = cv2.getOptimalNewCameraMatrix(cameraMatrix=camMats[src],
    #                                                            distCoeffs=distCoeffs[src],
    #                                                            imageSize=(w, h),
    #                                                            alpha=0)
    #
    # rectFrames = [0, 0]
    # for src in camSources:
    #     rectFrames[src] = cv2.undistort(frames[src],
    #                                     camMats[src],
    #                                     distCoeffs[src])
    #
    # # See the results
    # view = np.hstack([frames[0], frames[1]])
    # rectView = np.hstack([rectFrames[0], rectFrames[1]])
    #
    # cv2.imshow('view', view)
    # cv2.imshow('rectView', rectView)
    #
    # # Wait indefinitely for any keypress
    # cv2.waitKey(0)


# # Calibrates the images using the calibration data. The data can be found in the README
# def calibrate_git(stereo_img):
#
#     ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
#     chessboardSize = (9,7)
#     frameSize = (640,480)
#     # frameSize = (1280, 960)
#
#     # termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#
#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
#     objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
#
#     objp = objp * 20
#     #print(objp)
#
#     # Arrays to store object points and image points from all the images.
#     objpoints = [] # 3d point in real world space
#     imgpointsL = [] # 2d points in image plane.
#     imgpointsR = [] # 2d points in image plane.
#
#     for i in range(1,4):
#         path=['Img/Calibration/mire'+str(i)+'.jpg']
#         image=get_stereo_img(path,0)
#         cv2.imwrite('Img/Calibration/stereoLeft/mire'+str(i)+'.jpg', image[0])
#         cv2.imwrite('Img/Calibration/stereoRight/mire'+str(i)+'.jpg', image[1])
#
#     imagesRight = glob.glob('Img/Calibration/stereoRight/*.JPG')
#     imagesLeft = glob.glob('Img/Calibration/stereoLeft/*.JPG')
#
#     for imgLeft, imgRight in zip(imagesLeft, imagesRight):
#
#         imgL = cv2.imread(imgLeft)
#         imgR = cv2.imread(imgRight)
#         grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#         grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
#
#         # Find the chess board corners
#         retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
#         retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)
#
#         # If found, add object points, image points (after refining them)
#         if retL and retR == True:
#
#             objpoints.append(objp)
#
#             cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
#             imgpointsL.append(cornersL)
#
#             cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
#             imgpointsR.append(cornersR)
#
#             # # Draw and display the corners
#             # cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
#             # cv2.imshow('img left', imgL)
#             # cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
#             # cv2.imshow('img right', imgR)
#             # cv2.waitKey(1000)
#
#
#     cv2.destroyAllWindows()
#
#
#
#
#     ############## CALIBRATION #######################################################
#
#     retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
#     heightL, widthL, channelsL = imgL.shape
#     newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
#
#     retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
#     heightR, widthR, channelsR = imgR.shape
#     newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))
#
#
#
#     ########## Stereo Vision Calibration #############################################
#
#     flags = 0
#     flags |= cv2.CALIB_FIX_INTRINSIC
#     # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
#     # Hence intrinsic parameters are the same
#
#     criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
#     retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)
#
#
#
#
#     ########## Stereo Rectification #################################################
#
#     # rectifyScale= 1
#     # rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))
#
#     # stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
#     # stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)
#
#     # imgL=img[0]
#     # imgR=img[1]
#     dstL = cv2.undistort(stereo_img.left_img, newCameraMatrixL, distL, None, None)
#     dstR = cv2.undistort(stereo_img.right_img, newCameraMatrixR, distR, None, None)
#     # cv2.imwrite('Img/ImgRectL/undistL.jpg', dstL)
#     # cv2.imwrite('Img/ImgRectR/undistR.jpg', dstR)
#
#     cv2.imshow('left', dstL)
#     cv2.imshow('right', dstR)
#
#     cv2.waitKey(0)
#
#
#     print("Printing the new parameters!")
#     print("Camera matrix (left): \n")
#     print(newCameraMatrixL)
#     print("Camera matrix (right): \n")
#     print(newCameraMatrixR)
#     print("dist : \n")
#     print(distL)
#     # print("rvecs : \n")
#     # print(rectL)
#     # print("tvecs : \n")
#     # print(projMatrixL)
#
#     dst = StereoImg(dstL, dstR)
#     return dst
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

    # Debugging method used to print both left and right images.
    # debug_print_lr(stereo_img)

    # calibrate_git(stereo_img)

    # Calibrate the images using the mire found in the folder Img/Calibration/.
    # The second parameter is a bool to determine if the process shows the corner of the chessboard pattern or not
    stereo_img_rect = calibrate_stereo_cam(stereo_img, True)
    
    # Debugging method used to print both left and right images rectified
    # debug_print_lr(stereo_img_rect)

    # # Proceeds to the matching of the left and right images and returns a 2D matrix with the disparity for each pixels
    # image_matching(stereo_img)

    # To implement...
    depth_rendering()


if __name__ == "__main__":
    main(sys.argv[1:])
