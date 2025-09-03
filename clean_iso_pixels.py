import numpy as np
import cv2


def clean_iso_pixels(frame):

    if frame.dtype == 'uint16':
        alpha1 = (1 / 65535)
    elif frame.dtype == 'uint8':
        alpha1 = (1 / 255)
    else:
        raise Exception("frame type should be uint8/16.")
    # Change to bit values:
    input_image = cv2.convertScaleAbs(frame, alpha=alpha1)
    input_image_comp = cv2.bitwise_not(input_image)

    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], np.uint8)

    hitormiss1 = cv2.morphologyEx(input_image, cv2.MORPH_ERODE, kernel1)
    hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
    hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)
    del_isolated = input_image - hitormiss

    return del_isolated
