'''
Read nd2 video file
'''
import cv2
from nd2reader import ND2Reader
from matplotlib import pyplot as plt
from ParticleDetection.choose_binary_threshold import choose_binary_threshold
from ParticleDetection.object_tracking import object_tracking
from ParticleDetection.save_bin_frames import save_bin_frames


#with ND2Reader(filename) as frames:
#    _, frame = cv2.threshold(frames[10], 0, 65535, cv2.THRESH_BINARY)
 #   plt.imshow(frame, 'gray')
    #plt.hist(frames[2].ravel(), 1000)
#    plt.show()
#    num_frames = frames.metadata['num_frames']
#    width, height = (frames.metadata['width'], frames.metadata['height'])
#    pixel_size = frames.metadata['pixel_microns']
#    print(f'Num of frames: {num_frames}, width and height: {width, height},  pixel_size: {pixel_size}')

filename = r"\\132.68.85.6\pmvlab_users\Nirrit\PPfibers\020823\PPfibers_beads_tween0.001_0.8ulmin011.nd2"
# Choose binary threshold:

threshold1 = 400 # first frame - fibers:11500
threshold2 = 400  # other frames -  fibers:10800
# threshold = choose_binary_threshold(filename, threshold1, threshold2, images_amount=1, clean_isolated_px=False, skeleton=False,
#                                     first_img=0, delete_first_frame=False, img_lst=None)
#
# #
# # Save binary frames:
frames_directory = save_bin_frames(filename, threshold1, threshold2, init_frame=1, max_frames=2252,
                                   first_img=0, clean_isolated_px=False, delete_first_frame=False,skeleton=False,
                                   path=r"\\132.68.85.6\pmvlab_users\Nirrit\PPfibers\020823\PPfibers_beads_tween0.001_0.8ulmin011"
                                   )

pixel_size = 0.645  # um/pixel
time_interval = 0.0229  # sec

# frames_directory = r"\\132.68.85.2\Nirit\PPfibers\020823\b\PPfibers_beads_tween0.001_0.8ulmin_b009\frames_200_None"
# object_tracking(dir_data_frames=frames_directory,  # directory where the (binary) frames where saved.
#                 time_interval=time_interval,  # time between frames
#                 pixel_size=pixel_size,  # micron to pixel ratio
#                 init_frame=0,  # first image to start, default is the first.
#                 max_frames=None,  # last image to finish, default is the last - None.
#                 min_dst=100,  # distance between the object in the following frame: 100 fibers
#                 save_csv=True,
#                 fiber_min_dia=1,  # unit: micron
#                 )

# Need to save max_frames from num_frames when reading nd2.



