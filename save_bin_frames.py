"""
Make binary take chosen threshold, patj folder and maximum images and make them binary.
save the binary imgs in threshold folder in the given path directory.
return the path to the new images
"""

import cv2
from nd2reader import ND2Reader
import glob
import numpy as np
import os
import imutils
from natsort import natsorted
import time
from ParticleDetection.clean_iso_pixels import clean_iso_pixels
from skimage.morphology import skeletonize



def save_bin_frames(filename,
                    threshold1,
                    threshold2,
                    init_frame=1,
                    clean_isolated_px=True,
                    delete_first_frame=True,
                    first_img=1,
                    skeleton=True,
                    max_frames=None,
                    path=None):
    # make dir for the new data:
    directory, nd_name = os.path.split(os.path.splitext(filename)[0])
    if path is None:
        dir_data = os.path.splitext(filename)[0]
    else:
        dir_data = path
    if not os.path.exists(dir_data):
        os.mkdir(dir_data)

    if os.path.exists(dir_data) or os.path.isdir(dir_data):
        print(f'The {dir_data} is a directory')

    # make directory for images:
    dir_data_frames = os.path.join(dir_data, f'frames_{init_frame}_{max_frames}')
    if not os.path.exists(dir_data_frames):
        os.mkdir(dir_data_frames)

    if os.path.exists(dir_data_frames) or os.path.isdir(dir_data_frames):
        print(f'The {dir_data_frames} is a directory')

    # clean the directory
    for file in os.scandir(dir_data_frames):
        os.remove(file.path)

    # open ND2 video, make binary, subtract first image, clean and save:
    with ND2Reader(filename) as frames:

        if max_frames is None:
            max_frames = frames.metadata['num_frames']

        _, frame_0 = cv2.threshold(frames[first_img], threshold1, 65535, cv2.THRESH_BINARY)
        st = time.time()
        frames_range = range(init_frame, max_frames)

        for i in frames_range:
            #print(i)

            if i < init_frame - 1:
                continue
            else:
                _, a = cv2.threshold(frames[i], threshold2, 65535, cv2.THRESH_BINARY)

                if delete_first_frame:
                    cur_frame = frame_0 - a
                else:
                    cur_frame = a

                if clean_isolated_px:
                    cur_frame = clean_iso_pixels(cur_frame)

                if skeleton:
                    # Dilation and erosion to "bond" 2 close contours that probably got split...
                    cur_frame = cv2.dilate(cur_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)), iterations=2)
                    # Skeletonize
                    cur_frame = skeletonize(cur_frame)
                    # Dilate again the skeleton to ~1um diameter fiber
                    cur_frame = cv2.dilate(cur_frame.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                                           iterations=1)

                img_path = os.path.join(dir_data_frames, 'file_' + str(i) + '.tif')
                cv2.imwrite(img_path, cur_frame*255, params=(int(cv2.IMWRITE_TIFF_COMPRESSION), 1))

                if i % 100 == 0:
                    end = time.time()
                    elapsed_time = end - st
                    print('Frame num.:', i)
                    print('Execution time of 100 frames:', elapsed_time, 'seconds')
                    st = time.time()

            if i == max_frames:
                print(f'Are there more than {max_frames} images?')
                break

    print(f'Images were saved in {dir_data_frames}')
    print(f'From {max_frames}, {i}  frames were saved.')
    return dir_data_frames
