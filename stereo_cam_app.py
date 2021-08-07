#  Nicolas Auclair-Labbé - aucn2303
#  Ala Antabli - anta2801


# import collections
import cv2
import numpy as np
import sys
from typing import NamedTuple
import os.path
import open3d

# Basic data structure to keep both images from a stereo capture
class StereoImg(NamedTuple):
    left_img: np.ndarray
    right_img: np.ndarray


# Get the StereoImg using the path passed as parameters
def get_stereo_img(argv, isGray):
    # We check if we want to parse only with grayscale or with colors
    color_code = cv2.COLOR_BGR2GRAY if isGray else cv2.COLOR_BGR2RGB
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
        grayed_merged_img = cv2.cvtColor(merged_img, color_code)
        # Checks if the image has been read correctly
        if merged_img is None:
            print("Something went wrong and the image at the path : ", argv[0], " couldn't be read.")
            sys.exit()


        if isGray:
            merged_h, merged_w = grayed_merged_img.shape
            width_cutoff = merged_w // 2
            grayed_img_l = grayed_merged_img[:, :width_cutoff]
            grayed_img_r = grayed_merged_img[:, width_cutoff:]
        else:
            merged_h, merged_w, color_channel = grayed_merged_img.shape
            width_cutoff = merged_w // 2
            grayed_img_l = grayed_merged_img[:, :width_cutoff]
            grayed_img_r = grayed_merged_img[:, width_cutoff:]

    # Otherwise, we just have to read both images
    else:
        img_left = cv2.imread(argv[0])
        img_right = cv2.imread(argv[1])
        grayed_img_l = cv2.cvtColor(img_left, color_code)
        grayed_img_r = cv2.cvtColor(img_right, color_code)
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


def calibrate_stereo_cam(stereo_img, stereo_img_color, is_debugging):
    # Constant used in the method
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size = [1280, 960]
    chessboard_sqr_size = 20
    chessboard_size = (9, 7)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_sqr_size

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space for left camera
    img_points_l = []  # 2d points in the left image plane.
    img_points_r = []  # 2d points in right image plane.

    # Gets the mire in the folder 'Img/Calibration
    mire_left = []
    mire_right = []

    # Read and parse the mire into a left and right array
    print("Fetch the chessboard patterns in 'Img/Calibration/'")
    for i in range(1, 22):
        mire_full = get_stereo_img(['Img/Calibration/mire' + str(i) + '.jpg'], True)
        mire_left.append(mire_full.left_img)
        mire_right.append(mire_full.right_img)

    print("Find the corners in the chessboards pattern")
    for img_L, img_R in zip(mire_left, mire_right):
        ret_l, corners_left = cv2.findChessboardCorners(img_L, chessboard_size, None)
        ret_r, corners_right = cv2.findChessboardCorners(img_R, chessboard_size, None)

        if ret_l and ret_r:
            obj_points.append(objp)
            corners_left = cv2.cornerSubPix(img_L, corners_left, (11, 11), (-1, -1), criteria)
            img_points_l.append(corners_left)
            corners_right = cv2.cornerSubPix(img_R, corners_right, (11, 11), (-1, -1), criteria)
            img_points_r.append(corners_right)
            # Draw and display the corners if the option has be set to true
            if is_debugging:
                cv2.drawChessboardCorners(img_L, chessboard_size, corners_left, ret_l)
                cv2.imshow('Chessboard left', img_L)
                cv2.drawChessboardCorners(img_R, chessboard_size, corners_right, ret_r)
                cv2.imshow('Chessboard right', img_R)
                cv2.waitKey(0)

    print("Get the calibration parameters for both cameras")
    # Get the calibration parameters for both cameras
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(obj_points, img_points_l, img_size, None, None)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(obj_points, img_points_r, img_size, None, None)

    print("-------------------------------")
    print("calibrateCamera (left):")
    print('return value (left)\n', ret_l)
    print('matrix (left)\n', mtx_l)
    print('distortion (left)\n', dist_l)
    # print('rotation vector (left)\n', rvecs_l)
    # print('translation vector (left)\n', tvecs_l)

    print("-------------------------------")
    print("calibrateCamera (right):")
    print('return value (right)\n', ret_r)
    print('matrix (right)\n', mtx_r)
    print('distortion (right)\n', dist_r)


    new_left_matrix, roi_L = cv2.getOptimalNewCameraMatrix(mtx_l, 0, img_size, 1, img_size)
    new_right_matrix, roi_R = cv2.getOptimalNewCameraMatrix(mtx_r, 0, img_size, 1, img_size)

    # Perform the stereo calibration with the intrinsic parameters
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ret_stereo, new_mtx_l, dist_l, new_mtx_r, dist_r, rot, trans, essential_mtx, fundamental_mtx = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
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
    # We also want to resize the color images to use later for the 3D rendering
    x = roi_l[0] if roi_l[0] > roi_r[0] else roi_r[0]
    y = roi_l[1] if roi_l[1] > roi_r[1] else roi_r[1]
    w = roi_l[2] + roi_l[0] if roi_l[2] + roi_l[0] < roi_r[2] + roi_r[0] else roi_r[2] + roi_r[0]
    h = roi_l[3] + roi_l[1] if roi_l[3] + roi_l[1] < roi_r[3] + roi_r[1] else roi_r[3] + roi_r[1]

    roi_left_stereo_img = stereo_img_color.left_img[y:h, x:w]
    roi_right_stereo_img = stereo_img_color.right_img[y:h, x:w]
    fixed_left_img = fixed_left_img[y:h, x:w]
    fixed_right_img = fixed_right_img[y:h, x:w]

    calibrated_img = StereoImg(fixed_left_img, fixed_right_img)
    roi_stereo_col = StereoImg(roi_left_stereo_img, roi_right_stereo_img)

    mean_error_left = 0
    mean_error_right = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs_l[i], tvecs_l[i], mtx_l, dist_l)
        error = cv2.norm(img_points_l[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error_left += error
    print("total projection error (left): {}".format(mean_error_left / len(obj_points)))
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
        error = cv2.norm(img_points_r[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error_right += error
    print("total projection error (right): {}".format(mean_error_right / len(obj_points)))

    if is_debugging:
        cv2.imshow('left img ', calibrated_img.left_img)
        cv2.imshow('right img ', calibrated_img.right_img)
        cv2.waitKey(0)

    return calibrated_img, trans, roi_stereo_col


# Do the matching between the images
def image_matching(stereo_img_rect):
    # Variables used to do the matching on the image 'Img/nico2.jpg'
    block_size = 13
    min_disp = 0
    max_disp = 256
    num_disp = max_disp - min_disp
    uniquenessRatio = 1
    speckleWindowSize = 1000
    speckleRange = 50
    disp12MaxDiff = 10

    # Create the map used to evaluate the disparity of the stereo capture
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )

    # Apply the map on the stereo images and normalize them to show better results
    disparity_SGBM = stereo.compute(stereo_img_rect.left_img, stereo_img_rect.right_img)
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                  beta=0, norm_type=cv2.NORM_MINMAX)

    disparity_SGBM = np.uint8(disparity_SGBM)

    # Return the disparity map to be used in the 3D rendering step
    return disparity_SGBM

# Fill a 2D matrix with the coordinates of each pixel. The method should return the error coefficient found between
# the two images and a way to see the data (either by creating an image using the coordinates or by printing some
# kind of graph)
def depth_rendering(disparity_map, roi_stereo_color, translation):

    h, w = roi_stereo_color.left_img.shape[:2]
    focal_length = 12
    distance_stereo_cam = np.linalg.norm(translation)

    ratio_for_Z = distance_stereo_cam/focal_length



    Q = np.float32([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, distance_stereo_cam/focal_length, 0],
                     [0, 0, 0, 1]])

    # points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

    points_3D = np.zeros((h, w, 3), dtype=float)

    for i in range(0, h-1):
        for j in range(0, w-1):
            points_3D[i][j] = [j, -i, disparity_map[i][j] * ratio_for_Z]

    # points_3D = disparity_map

    # for i in disparity_map:
    #     for j in disparity_map[0]:
    #         points_3D[i][j] = disparity_map[i][j] * distance_stereo_cam/focal_length

    # It gets rid of any points without any disparities
    mask_map = disparity_map > disparity_map.min()

    print("Convert into a cloud file (.ply), then loads it")

    # Creates some variables to convert the 3D image into a cloud
    colors = roi_stereo_color.left_img
    output_file = 'reconstructed_nico2.ply'
    # Uses the mask on the points and the colored image
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]

    output_colors = output_colors.reshape(-1, 3)
    output_points = np.hstack([output_points.reshape(-1, 3), output_colors])

    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		'''
    # Save the cloud
    with open(output_file, 'w') as f:
        f.write(ply_header % dict(vert_num=len(output_points)))
        np.savetxt(f, output_points, '%f %f %f %d %d %d')
    # Load the cloud to show to everyone
    cloud = open3d.io.read_point_cloud("reconstructed_nico2.ply")  # Read the point cloud
    open3d.visualization.draw_geometries([cloud])  # Visualize the point cloud


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
    stereo_img = get_stereo_img(argv, True)
    stereo_img_color = get_stereo_img(argv, False)
    # Debugging method used to print both left and right images.
    # debug_print_lr(stereo_img)

    # Calibrate the images using the mire found in the folder Img/Calibration/.
    # The second parameter is a bool to determine if the process shows the corner of the chessboard pattern or not
    # Returns the calibrated images, the translation vector (we want the difference between both camera) a resized color stereo set for 3D rendering
    print("----------------------------------")
    print("Calibrating the images")
    print("----------------------------------")
    stereo_img_rect, translation, roi_stereo_color = calibrate_stereo_cam(stereo_img, stereo_img_color, False)

    # Debugging method used to print both left and right images rectified
    # debug_print_lr(roi_stereo_color)

    # Proceeds to the matching of the left and right images and returns a 2D matrix with the disparity for each pixels
    # Returns the disparity map
    print("----------------------------------")
    print("Matching both images to get a disparity map")
    print("----------------------------------")
    disparity_map = image_matching(stereo_img_rect)

    # # Shows the disparity map returned from the method 'image_matching'
    # cv2.imshow('disparity_map', disparity_map)
    # cv2.waitKey(0)

    # Creates a 3D render of the stereo capture using the disparity map of both images in the stereo capture
    print("----------------------------------")
    print("Creating a 3D render of the stereo capture")
    print("----------------------------------")
    depth_rendering(disparity_map, roi_stereo_color, translation)

    print("----------------------------------")
    print("Exiting the program")
    print("----------------------------------")

if __name__ == "__main__":
    main(sys.argv[1:])
