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
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(obj_points, imgpointsL, img_size, None, None)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(obj_points, imgpointsR, img_size, None, None)

    print("-------------------------------")
    print("calibrateCamera (left):")
    print('return value (left)\n', ret_l)
    print('matrix (left)\n', mtx_l)
    print('distortion (left)\n', dist_l)
    print('rotation vector (left)\n', rvecs_l)
    print('translation vector (left)\n', tvecs_l)

    print("-------------------------------")
    print("calibrateCamera (right):")
    print('return value (right)\n', ret_r)
    print('matrix (right)\n', mtx_r)
    print('distortion (right)\n', dist_r)
    print('rotation vector (right)\n', rvecs_r)
    print('translation vector (right)\n', tvecs_r)

    new_left_matrix, roi_L = cv2.getOptimalNewCameraMatrix(mtx_l, 0, img_size, 1, img_size)
    new_right_matrix, roi_R = cv2.getOptimalNewCameraMatrix(mtx_r, 0, img_size, 1, img_size)

    # Perform the stereo calibration with the intrinsic parameters
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ret_stereo, new_mtx_l, dist_l, new_mtx_r, dist_r, rot, trans, essential_mtx, fundamental_mtx = cv2.stereoCalibrate(
        obj_points, imgpointsL, imgpointsR,
        new_left_matrix, 0, new_right_matrix,
        0, img_size,
        criteria_stereo, flags)

    print("-------------------------------")
    print("stereoCalibrate :")
    print('return value\n', ret_stereo)
    print('new left matrix\n', new_mtx_l)
    print('new right matrix\n', new_mtx_r)
    print('distortion (left)\n', dist_l)
    print('distortion (right)\n', dist_r)
    print('Rotation\n', rot)
    print('Translation\n', trans)
    print('Essential matrix: \n', essential_mtx)
    print('Fundamental matrix: \n', fundamental_mtx)

    rectify_scale = 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roi_l, roi_r = cv2.stereoRectify(new_mtx_l, 0, new_mtx_r, 0,
                                                                              img_size, rot, trans,
                                                                              rectify_scale, (0, 0))
    print("-------------------------------")
    print("stereoRectify :")
    print('rectification matrix (left)\n', rect_l)
    print('rectification matrix (right)\n', rect_r)
    print('projection matrix (left)\n', proj_mat_l)
    print('projection matrix (right)\n', proj_mat_r)
    print('Q\n', Q)
    print('ROI (left)\n', roi_l)
    print('ROI (right)\n', roi_r)

    stereo_map_l = cv2.initUndistortRectifyMap(new_mtx_l, 0, rect_l, proj_mat_l, img_size, cv2.CV_16SC2)
    stereo_map_r = cv2.initUndistortRectifyMap(new_mtx_r, 0, rect_r, proj_mat_r, img_size, cv2.CV_16SC2)

    # Calibrate the images using the all the calibration data
    fixed_left_img = cv2.remap(stereo_img.left_img, stereo_map_l[0], stereo_map_l[1], cv2.INTER_LANCZOS4)
    fixed_right_img = cv2.remap(stereo_img.right_img, stereo_map_r[0], stereo_map_r[1], cv2.INTER_LANCZOS4)

    # Resize the images to fit the new region of interest. Also put both images to the same size
    x = roi_l[0] if roi_l[0] > roi_r[0] else roi_r[0]
    y = roi_l[1] if roi_l[1] > roi_r[1] else roi_r[1]
    w = roi_l[2] + roi_l[0] if roi_l[2] + roi_l[0] < roi_r[2] + roi_r[0] else roi_r[2] + roi_r[0]
    h = roi_l[3] + roi_l[1] if roi_l[3] + roi_l[1] < roi_r[3] + roi_r[1] else roi_r[3] + roi_r[1]

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
