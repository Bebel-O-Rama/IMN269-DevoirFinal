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
def get_stereo_img(argv):
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
        grayed_img_l = grayed_merged_img[:, :width_cutoff]
        grayed_img_r = grayed_merged_img[:, width_cutoff:]

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
    images = StereoImg(grayed_img_l, grayed_img_r)
    return images


def calibrate_stereo_cam(stereo_img, is_debugging):
    # Constant used in the method
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size = [1280, 960]
    chessboard_sqr_size = 20
    chessboard_size = (9, 7)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_sqr_size

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space for left camera
    imgpointsL = []  # 2d points in the left image plane.
    imgpointsR = []  # 2d points in right image plane.

    # Gets the mire in the folder 'Img/Calibration
    mire_left = []
    mire_right = []

    # Read and parse the mire into a left and right array
    print("Fetch the chessboard patterns in 'Img/Calibration/'")
    for i in range(1, 22):
        mire_full = get_stereo_img(['Img/Calibration/mire' + str(i) + '.jpg'])
        mire_left.append(mire_full.left_img)
        mire_right.append(mire_full.right_img)

    print("Find the corners in the chessboards pattern")
    for img_L, img_R in zip(mire_left, mire_right):
        ret_l, corners_left = cv2.findChessboardCorners(img_L, chessboard_size, None)
        ret_r, corners_right = cv2.findChessboardCorners(img_R, chessboard_size, None)

        if ret_l and ret_r:
            obj_points.append(objp)
            corners_left = cv2.cornerSubPix(img_L, corners_left, (11, 11), (-1, -1), criteria)
            imgpointsL.append(corners_left)
            corners_right = cv2.cornerSubPix(img_R, corners_right, (11, 11), (-1, -1), criteria)
            imgpointsR.append(corners_right)
            # Draw and display the corners if the option has be set to true
            if is_debugging:
                cv2.drawChessboardCorners(img_L, chessboard_size, corners_left, ret_l)
                cv2.imshow('Chessboard left', img_L)
                cv2.drawChessboardCorners(img_R, chessboard_size, corners_right, ret_r)
                cv2.imshow('Chessboard right', img_R)
                cv2.waitKey(0)

    print("Get the calibration parameters for both cameras")
    # Get the calibration parameters for both cameras
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_points, imgpointsL, img_size, None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_points, imgpointsR, img_size, None, None)

    print("-------------------------------")
    print("calibrateCamera (left):")
    print('return value (left)\n', retL)
    print('matrix (left)\n', mtxL)
    print('distortion (left)\n', distL)
    print('rotation vector (left)\n', rvecsL)
    print('translation vector (left)\n', tvecsL)

    print("-------------------------------")
    print("calibrateCamera (right):")
    print('return value (right)\n', retR)
    print('matrix (right)\n', mtxR)
    print('distortion (right)\n', distR)
    print('rotation vector (right)\n', rvecsR)
    print('translation vector (right)\n', tvecsR)

    new_left_matrix, roi_L = cv2.getOptimalNewCameraMatrix(mtxL, 0, img_size, 1, img_size)
    new_right_matrix, roi_R = cv2.getOptimalNewCameraMatrix(mtxR, 0, img_size, 1, img_size)

    # Perform the stereo calibration with the intrinsic parameters
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retStereo, new_mtxL, distL, new_mtxR, distR, rot, trans, essentialMtx, fundamentalMtx = cv2.stereoCalibrate(
        obj_points, imgpointsL, imgpointsR,
        new_left_matrix, 0, new_right_matrix,
        0, img_size,
        criteria_stereo, flags)

    print("-------------------------------")
    print("stereoCalibrate :")
    print('return value\n', retStereo)
    print('new left matrix\n', new_mtxL)
    print('new right matrix\n', new_mtxR)
    print('distortion (left)\n', distL)
    print('distortion (right)\n', distR)
    print('Rotation\n', rot)
    print('Translation\n', trans)
    print('Essential matrix: \n', essentialMtx)
    print('Fundamental matrix: \n', fundamentalMtx)

    rectify_scale = 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, 0, new_mtxR, 0,
                                                                              img_size, rot, trans,
                                                                              rectify_scale, (0, 0))
    print("-------------------------------")
    print("stereoRectify :")
    print('rectification matrix (left)\n', rect_l)
    print('rectification matrix (right)\n', rect_r)
    print('projection matrix (left)\n', proj_mat_l)
    print('projection matrix (right)\n', proj_mat_r)
    print('Q\n', Q)
    print('ROI (left)\n', roiL)
    print('ROI (right)\n', roiR)

    stereoMapL = cv2.initUndistortRectifyMap(new_mtxL, 0, rect_l, proj_mat_l, img_size, cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(new_mtxR, 0, rect_r, proj_mat_r, img_size, cv2.CV_16SC2)

    # Calibrate the images using the all the calibration data
    fixed_left_img = cv2.remap(stereo_img.left_img, stereoMapL[0], stereoMapL[1], cv2.INTER_LANCZOS4)
    fixed_right_img = cv2.remap(stereo_img.right_img, stereoMapR[0], stereoMapR[1], cv2.INTER_LANCZOS4)

    # Resize the images to fit the new region of interest. Also put both images to the same size
    x = roiL[0] if roiL[0] > roiR[0] else roiR[0]
    y = roiL[1] if roiL[1] > roiR[1] else roiR[1]
    w = roiL[2] + roiL[0] if roiL[2] + roiL[0] < roiR[2] + roiR[0] else roiR[2] + roiR[0]
    h = roiL[3] + roiL[1] if roiL[3] + roiR[0] < roiR[3] + roiR[0] else roiR[3] + roiR[1]

    fixed_left_img = fixed_left_img[y:h, x:w]
    fixed_right_img = fixed_right_img[y:h, x:w]

    calibrated_img = StereoImg(fixed_left_img, fixed_right_img)

    if is_debugging:
        cv2.imshow('left img ', calibrated_img.left_img)
        cv2.imshow('right img ', calibrated_img.right_img)
        cv2.waitKey(0)

    return calibrated_img


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
    print("----------------------------------")
    print("Getting the stereo images to process")
    print("----------------------------------")
    stereo_img = get_stereo_img(argv)

    # Debugging method used to print both left and right images.
    debug_print_lr(stereo_img)

    # Calibrate the images using the mire found in the folder Img/Calibration/.
    # The second parameter is a bool to determine if the process shows the corner of the chessboard pattern or not
    print("----------------------------------")
    print("Calibrating the images")
    print("----------------------------------")
    stereo_img_rect = calibrate_stereo_cam(stereo_img, False)

    # Debugging method used to print both left and right images rectified
    debug_print_lr(stereo_img_rect)

    # Proceeds to the matching of the left and right images and returns a 2D matrix with the disparity for each pixels
    print("----------------------------------")
    print("Matching both images to get a disparity map")
    print("----------------------------------")
    image_matching(stereo_img)

    # To implement...
    print("----------------------------------")
    print("Creating a 3D render of the stereo capture")
    print("----------------------------------")
    depth_rendering()


if __name__ == "__main__":
    main(sys.argv[1:])
