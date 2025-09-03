import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import os
import imutils
import math
from natsort import natsorted
import csv
import time
from statistics import mean
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize


from ParticleDetection.clean_iso_pixels import clean_iso_pixels

'''
Finding contours in frame with minimum area of dia=1 um and length of 10 um (2D dimension),
Comparing contours with adjacent frame based on distance < 50 pixels (~32 um) and area difference< 50%.
Tracking object: velocity, distance, area, perimeter, minimum enclosing box (min length/width),
fiber straightness, fiber curliness, angle with x axis...
'''


def find_center_cnt(contour):
    m1 = cv2.moments(contour)
    #print(m1['m00'])
    cx = int(m1['m10'] / m1['m00'])  # center (x,y) of the contour
    cy = int(m1['m01'] / m1['m00'])
    return round(cx), round(cy)


def obj_features(contour, fiber_dia, micron_to_pixel):
    m1 = cv2.moments(contour)
    area = m1["m00"] * micron_to_pixel  # (px)*(um/px)
    cx = int(m1['m10'] / m1['m00'])  # center (x,y) of the contour
    cy = int(m1['m01'] / m1['m00'])

    min_rect = cv2.minAreaRect(contour)  # ( center (x,y), (width, height), angle of rotation ).
    straight_rect = cv2.boundingRect(contour)
    straight_rect = [round(elem) for elem in straight_rect]
    box = cv2.boxPoints(min_rect)  # box points-array of four vertices of rectangle.
    box = np.int0(box)
    perimeter = cv2.arcLength(contour, True) * micron_to_pixel  # um
    l_area = area / fiber_dia  # um
    l_perimeter = (perimeter - 2 * fiber_dia) / 2  # um
    (x, y), (w, h), angle = min_rect
    min_length = max(w, h) * micron_to_pixel  # um
    min_width = min(w, h) * micron_to_pixel  # um
    fiber_length = min(l_perimeter, l_area)
    df = pd.DataFrame(box, columns=['x', 'y'])

    ymax = df.iloc[df['y'].idxmax(), :]
    xmax = df.iloc[df['x'].idxmax(), :]
    xmin = df.iloc[df['x'].idxmin(), :]

    if math.dist(ymax, xmax) > math.dist(ymax, xmin):
        theta = math.atan(round(abs(xmax['y'] - ymax['y']) / abs(xmax['x'] - ymax['x']), 2))
    else:
        theta = math.atan(round(abs(xmin['y'] - ymax['y']) / abs(xmin['x'] - ymax['x']), 2))

    theta_deg = theta * 180 / math.pi

    return {"cx": round(cx), "cy": round(cy), "w": round(w), "h": round(h), "angle": round(angle),
            "area": round(area), "perimeter": round(perimeter), "min_length": round(min_length),
            "min_width": round(min_width,2), "fiber_length": round(fiber_length), "theta_deg": round(theta_deg),
            "fiber_dia": round(fiber_dia), "straight_rect": straight_rect}


def intersection_over_union(box1, box2):  # input box: x, y, w, h of straight_rect
    x, y, w, h = box1
    box1 = (x-w/2, y-h/2, x+w/2, y+h/2)  # top left pt, bottom right pt
    box1 = [round(num) for num in box1]
    print(f'box1:{box1}')
    area_box1 = w*h

    x2, y2, w2, h2 = box2
    box2  = (x2-w2/2, y2-h2/2, x2+w2/2, y2+h2/2)
    box2 = [round(num) for num in box2]
    print(f'box2:{box2}')
    area_box2 = w2*h2

    x_inter_left = max(box1[0], box2[0]) # box has its top left corner more to the right.
    y_inter_top = max(box1[1], box2[1]) # box has its top left corner lower than the other.
    x_inter_right = min(box1[2], box2[2])
    y_inter_bottom = min(box1[3], box2[3])

    print(f'x_inter_left:{x_inter_left}')
    print(f'y_inter_top:{y_inter_top}')
    print(f'x_inter_right:{x_inter_right}')
    print(f'y_inter_bottom:{y_inter_bottom}')

    area_inter = max(0, x_inter_right - x_inter_left) * max(0, y_inter_bottom - y_inter_top)

    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
    return iou


def object_tracking(dir_data_frames,  # directory where the (binary) frames where saved in this format:'file_0.tif',..
                    time_interval,  # time between frames
                    pixel_size,  # micron to pixel ratio
                    init_frame=0,  # first image to start, default is the first.
                    min_dst=150,  # distance between the object in the following frame
                    max_frames=None,  # last image to finish, default is the last.
                    save_csv=True,
                    fiber_min_dia=1,  # unit: micron
                    dir_data=None,  #  directory to save data, if not given, save where frames directory is.
                    ):
    obj_id = 0  # Number of objects
    tracking_objects = []  # track_id: list of object that are still in the frame
    tracking_position = {}  # (cx, cy, w, h, anglr)
    velocity_tracking = {}  # velocity between following frames (not necessarily consecutive frames)
    velocity_x_tracking = {}
    velocity_y_tracking = {}
    distance_tracking = {}  # the distance between the first frame and current position.
    frame_tracking = {}
    area_tracking = {}
    perimeter_tracking = {}
    fiber_length_tracking = {}
    min_length_tracking = {}
    min_width_tracking = {}
    fiber_straightness_tracking = {}
    fiber_curl_major_tracking = {}
    fiber_curl_minor_tracking = {}
    theta_deg_tracking = {}

    def update_dic(track_id,  # the original obj_id tracking
                   corr_cnt_id,  # contour index in contours vector corresponding with prev obj
                   frame_contours,  # contours of all objects in frame
                   curr_frame,  # frame num.
                   new_obj=False):
        # save to dictionaries
        features = obj_features(contour=frame_contours[corr_cnt_id], fiber_dia=fiber_min_dia, micron_to_pixel=pixel_size)

        tracking_position.setdefault(track_id, []).append([features["cx"], features["cy"],
                                                           features["w"], features["h"], features["angle"]])
        area_tracking.setdefault(track_id, []).append(features["area"])
        perimeter_tracking.setdefault(track_id, []).append(features["perimeter"])
        fiber_length_tracking.setdefault(track_id, []).append(features["fiber_length"])
        min_length_tracking.setdefault(track_id, []).append(features["min_length"])
        min_width_tracking.setdefault(track_id, []).append(features["min_width"])
        theta_deg_tracking.setdefault(track_id, []).append(features["theta_deg"])
        frame_tracking.setdefault(track_id, []).append(curr_frame)
        # calculate straightness/curliness:
        fiber_straightness = features["min_length"] / features[
            "fiber_length"]  # 1- if fiber is straight
        fiber_curl_major = features["fiber_length"] / features["min_length"]
        fiber_curl_minor = features["min_width"] / features["fiber_dia"]

        fiber_straightness_tracking.setdefault(track_id, []).append(round(fiber_straightness, 2))
        fiber_curl_major_tracking.setdefault(track_id, []).append(round(fiber_curl_major, 2))
        fiber_curl_minor_tracking.setdefault(track_id, []).append(round(fiber_curl_minor, 2))

        if not new_obj:
            curr_cx, curr_cy = tracking_position[track_id][-1][0], tracking_position[track_id][-1][1]
            prev_cx, prev_cy = tracking_position[track_id][-2][0], tracking_position[track_id][-2][1]

            # velocities tracking:
            time_bet_frames = time_interval * (frame_tracking[track_id][-1] - frame_tracking[track_id][-2])
            delta_distance = round(math.dist([curr_cx, curr_cy], [prev_cx, prev_cy]), 2)  # Euclidean distance

            velocity = delta_distance * pixel_size / time_bet_frames
            velocity_tracking.setdefault(track_id, []).append(round(velocity, 2))

            velocity_x = (curr_cx - prev_cx) * pixel_size / time_bet_frames
            velocity_x_tracking.setdefault(track_id, []).append(round(velocity_x, 2))

            velocity_y = (curr_cy - prev_cy) * pixel_size / time_bet_frames
            velocity_y_tracking.setdefault(track_id, []).append(round(velocity_y, 2))
            # accumulative distance:
            first_cx = tracking_position[track_id][0][0]
            first_cy = tracking_position[track_id][0][1]

            distance_tracking.setdefault(track_id, []).append(round(math.dist([features["cx"],
                                                                               features["cy"]], [first_cx, first_cy]),
                                                                    2))

        if new_obj:
            velocity_tracking[obj_id] = [0]
            velocity_x_tracking[obj_id] = [0]
            velocity_y_tracking[obj_id] = [0]
            distance_tracking[obj_id] = [0]

    saved_items = {}
    subtract_ids = []
    subst_contours = {} # dic of id with low velocities that their new_cnt need to be subtracted from following images.

    # iterable frames directory:
    dir_iter = os.path.join(dir_data_frames, "*.tif")
    # data saving directory:
    if dir_data is None:
        dir_data = os.path.split(dir_data_frames)[0]

    for i, img_path in enumerate(natsorted(glob.glob(dir_iter))):
        m = os.path.splitext(os.path.basename(img_path))[0]
        current_frame = int(m.split('_')[1])

        if i % 100 == 0:
            st = time.time()
        if i < init_frame:
            continue
        if i == init_frame:
            frame_0 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if max_frames is not None and i > max_frames:
            break
        else:
            # Finding the contours/objects in current frame:
            frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            frame= frame_0-frame

            # Dilation and erosion to "bond" 2 close contours that probably got split...
            #frame = cv2.dilate(frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8)), iterations=2)
            # Skeletonize
            #frame = skeletonize(frame)
            # Dilate again the skeleton to ~1um diameter fiber
            #frame = cv2.dilate(frame.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)

            # Before finding new contours, let's remove old one that are stuck in th frame:
            copy_frame = frame.copy().astype(np.uint8)
            # draw the relevant contours in black in the frame:
            for subst_cnt in subst_contours.values():
                copy_frame = cv2.drawContours(copy_frame, subst_cnt, 0, color=(0, 0, 0),
                                              thickness=-1)

            contours, _ = cv2.findContours(copy_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # make a new df that will contain the delts distance between old and new cnts:
            tracking_objects_copy = tracking_objects.copy()
            delta_dist_df = pd.DataFrame(columns=tracking_objects_copy)

            # Loop on each contour found in the frame, only if its area is greater than min area, continue:
            cnt_index = [] # list of cnt index in contours
            for j, new_cnt in enumerate(contours):
                M = cv2.moments(new_cnt)
                min_area = 17 / pixel_size  # unit: px min area fiber 17 um
                if M["m00"] > min_area:
                    #print(j)
                    # Only at the beginning of if object tracking is empty we save all objects,
                    # and we don't compare to previous.

                    if i == init_frame or not tracking_objects:

                        # save to dictionaries
                        tracking_objects += [obj_id]
                        update_dic(track_id=obj_id, corr_cnt_id=j, frame_contours=contours, curr_frame=current_frame, new_obj=True)

                        obj_id += 1
                    # Compare current object with previous frame objects
                    else:

                        # Get the contour cx, cy:
                        cx, cy = find_center_cnt(new_cnt)
                        # if object center x is less than 25 px- it is out of cell- stop tracking it.
                        if cx < 60:
                            continue

                        # Finding the distance between cnt_id from tracking list to new_cnt in curr frame
                        cnt_index += [j]
                        for cnt_id in tracking_objects_copy:
                            pre_cx, pre_cy = tracking_position[cnt_id][-1][0:2]

                            delta_dist_df.loc[j, cnt_id] = round(math.dist([cx, cy], [pre_cx, pre_cy]), 2) # Euclidean distance

            # If there are more cnts than track:
            # maybe a new cnt - > cx>2000
            # or maybe this "new" cnt is a split of old one...

            row_ind, col_ind = linear_sum_assignment(delta_dist_df)
            row_index = [cnt_index[x] for x in row_ind]
            col_name = [tracking_objects_copy[x] for x in col_ind]
            #print(f'tracking_obj: {tracking_objects_copy}')
            #print(f'j: {row_index}')
            #print(f'track id: {col_name}')
            #print(f'cnt_index(all j):{cnt_index}')
            for row, col in zip(row_index, col_name):
                if delta_dist_df.loc[row, col] < min_dst + \
                        velocity_tracking[col][-1]*(current_frame-frame_tracking[col][-1])*time_interval:  # object exist:
                    update_dic(track_id=col, corr_cnt_id=row, frame_contours=contours, curr_frame=current_frame, new_obj=False)

                    # Subtracting object contours from following images if they are not moving.
                    if len(frame_tracking[col]) > 50:
                        std_distance = np.std(distance_tracking[col][-40:])
                        #print(f'avg_distance:{avg_distance }')
                        if (std_distance < 5) and (col not in subst_contours.keys()):
                            # subtract_ids += [track_id]
                            subst_contours[col] = [contours[row]]
                            # removing the stuck fiber from tracking_object dic
                            tracking_objects.remove(col)

                else:  # object are too far away- make a new obj_id
                    # Get the contour cx, cy:
                    # check her1!!!!!
                    cx, cy = find_center_cnt(contours[row])
                    if cx > 1950:
                        tracking_objects += [obj_id]
                        update_dic(track_id=obj_id, corr_cnt_id=row, frame_contours=contours, curr_frame=current_frame, new_obj=True)
                        obj_id += 1
                    # if cx<190 what to do??? try to understand if the new cnt is a part of old one that got
                    # split...

            # add new contours found that are not related to prev tracking objects:
            new_rows = [x for x in cnt_index if x not in row_index]
            for new_row in new_rows:
                cx, cy = find_center_cnt(contours[new_row])
                if cx > 1950:
                    tracking_objects += [obj_id]
                    update_dic(track_id=obj_id, corr_cnt_id=new_row, frame_contours=contours, curr_frame=current_frame, new_obj=True)

                    obj_id += 1


                ## Need to understand when and how to revive the dead!
                ## maybe if cnt is found in the middle of the cell
            for track_id in tracking_objects:
                # if tracking obj was not detected in last 25 frames- stop tracking it:
                if frame_tracking[track_id][-1] < current_frame - 25:
                    tracking_objects.remove(track_id)


            # Add new IDs found.
            # It is possible to add another limitation
            # new objects can be detected only if there cx is near the entrance (2048)
            # fibers can not be created in the cell, they supposed to appear somewhere before..



            if (i - 99) % 100 == 0 or i == max_frames:
                end = time.time()
                elapsed_time = end - st
                print('Frame num.:', current_frame, 'i:', i)
                print('Execution time of 100 frames:', elapsed_time, 'seconds')

                # Save csv every 500 frames

                if save_csv:
                    dictionaries = [tracking_position,
                                    area_tracking, perimeter_tracking,
                                    fiber_length_tracking,
                                    min_length_tracking,
                                    min_width_tracking,
                                    theta_deg_tracking,
                                    fiber_straightness_tracking,
                                    fiber_curl_major_tracking,
                                    fiber_curl_minor_tracking,
                                    velocity_tracking,
                                    velocity_x_tracking,
                                    velocity_y_tracking,
                                    distance_tracking,
                                    frame_tracking,
                                    ]
                    dictionaries_names = "tracking_position, area_tracking, perimeter_tracking, " \
                                         "fiber_length_tracking, min_length_tracking, min_width_tracking, " \
                                         "theta_deg_tracking, fiber_straightness_tracking, " \
                                         "fiber_curl_major_tracking, fiber_curl_minor_tracking, " \
                                         "velocity_tracking, velocity_x_tracking, velocity_y_tracking, " \
                                         "distance_tracking, frame_tracking"
                    dictionaries_names_lst = dictionaries_names.split(", ")
                    # make csv file for every object_id
                    for item in range(0, obj_id):
                        # If item still under tracking: continue
                        if item in tracking_objects and i != max_frames:
                            continue
                        elif item not in saved_items:  # make csv file for every object_id that wasn't saved by now
                            item_dic = {}
                            for j, d in enumerate(dictionaries):
                                item_dic[dictionaries_names_lst[j]] = d[item]

                            # open file for writing("w"), named with object_id
                            path = os.path.join(dir_data, str(item) + '.csv')
                            w = csv.writer(open(path, "w"))

                            # loop over dictionary keys and values
                            for key, val in item_dic.items():
                                # write every key and value to file
                                w.writerow([key, val])
                            saved_items[item] = 'True'
                            print(f'saved csv files of object_id {item}')

                        # Make csv file with object id that are still under tracking_objects:
                        # open file for writing("w"), named with object_id
                        path = os.path.join(dir_data, 'tracking_objects.csv')
                        w = csv.writer(open(path, "w"))

                        # loop over dictionary keys and values
                        for key in tracking_objects:
                            # write every key and value to file
                            w.writerow([key])


