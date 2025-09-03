'''
Read nd2 video file and
Choose binary threshold based on one image- plot the images and return the threshold
    The image is in gray range.
    For thresholding: if the pixel value is smaller than the threshold, it is set to 0 (black).
    Images have 16 bit depth: 65,536 colors.
    Then it need to be scales to [0,1]
    Lastly, we subtract the first image from every other images.
'''
from nd2reader import ND2Reader
import cv2
from matplotlib import pyplot as plt
import random
from skimage.morphology import skeletonize

from ParticleDetection.clean_iso_pixels import clean_iso_pixels


def choose_binary_threshold(filename,
                            threshold1,  # for first frame only
                            threshold2,  # for all frames, except the first one
                            images_amount=9,
                            clean_isolated_px=True,
                            delete_first_frame=True,
                            skeleton=False,
                            first_img=1,
                            img_lst=None,  # list of indexes of frames
                            ):

    with ND2Reader(filename) as frames:

        num_frames = frames.metadata['num_frames']
        print(f'Number of frames: {num_frames}')
        if img_lst is None:
            frames_index = random.sample(range(1, num_frames), images_amount)
        else:
            frames_index = img_lst

        #print(type(frames[0]))
        _, frame_0 = cv2.threshold(frames[first_img], threshold1, 65535, cv2.THRESH_BINARY)
        #print(type(frame_0))
        for i in frames_index:
            print(f'current frame: {i}')
            frame = frames[i].copy()
            _, frame = cv2.threshold(frame, threshold2, 65535, cv2.THRESH_BINARY)

            if delete_first_frame:
                delta_frame = frame_0 - frame
            else:
                delta_frame = frame
            #_, delta_frame = cv2.threshold(delta_frame, 50000, 65535, cv2.THRESH_BINARY)

            # cleaning image from isolated pixels:
            if clean_isolated_px:
                delta_frame = clean_iso_pixels(delta_frame)

            if skeleton:
                # Dilation and erosion to "bond" 2 close contours that probably got split...
                delta_frame = cv2.dilate(delta_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)), iterations=2)
                # Skeletonize
                delta_frame = skeletonize(delta_frame)
                # Dilate again the skeleton to ~1um diameter fiber
                delta_frame = cv2.dilate(delta_frame.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                                       iterations=1)


            # Output Images
            fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
            axs[0, 0].imshow(frames[0], 'gray')
            axs[0, 0].title.set_text(f'First frame ({threshold1})')
            axs[1, 0].imshow(frame_0, 'gray')

            axs[0, 1].imshow(frames[i], 'gray')
            axs[0, 1].title.set_text(f'Frame {i} thr:({threshold2})')
            axs[1, 1].imshow(frame, 'gray')

            axs[0, 2].imshow(delta_frame, 'gray')
            axs[0, 2].title.set_text('frame0-frame->thr')
            axs[1, 2].imshow(frame_0-frame, 'gray')
            axs[1, 2].title.set_text('thr->frame0-frame')
            plt.show()



