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
def calibrate_stereo_cam():
    print("No yet implemented")
    # Returns an updated StereoImg


# Do the matching between the images
def image_matching():
    print("No yet implemented")
    # Returns a 2D matrix with the depth for each pixel


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
    calibrate_stereo_cam()
    image_matching()
    depth_rendering()


if __name__ == "__main__":
    main(sys.argv[1:])