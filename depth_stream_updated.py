from playsound import playsound
from threading import Thread
from multiprocessing import Process
import pyrealsense2 as rs
import numpy as np
import time
import math
import cv2

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 3 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

kernel = np.ones((5,5),np.uint8)

key = 1
cx = cy = extTop = extBot = dist = pi = 0

def calculateDistance(x1,y1,x2,y2):
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return dist

def music():
    playsound("cut3.mp3")


time.sleep(3)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 1, 255, 0)

        thresh = cv2.blur(thresh, (5,5))
        # thresh = cv2.dilate(thresh,kernel,iterations=1)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        colored_mask = np.zeros(color_image.shape, np.uint8)
        colored_mask[:] = (0,255,0)


        try:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]

            M = cv2.moments(cnt)

            if(key==1):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                l = tuple(cnt[cnt[:, :, 0].argmin()][0])
                r = tuple(cnt[cnt[:, :, 0].argmax()][0])
                dist1 = calculateDistance(cx, cy, l[0], l[1])
                dist2 = calculateDistance(cx, cy, r[0], r[1])
                dist = max(dist1,dist2)
                # m = Thread(target = music)
                # m.start()
                Process(name="playsound", target=music).start()
                key = 0


            extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
            extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])


            myradians1 = math.atan2((cy - extLeft[1]), (extLeft[0] - cx))
            myradians2 = math.atan2((cy - extRight[1]), (extRight[0] - cx))

            mydegrees1 = math.degrees(myradians1)
            mydegrees2 = math.degrees(myradians2)

            print mydegrees1, mydegrees2, dist

            hull = []
            hull.append(cv2.convexHull(cnt, False))

            color_image = cv2.bitwise_and(colored_mask, colored_mask, mask=thresh)

            cv2.rectangle(color_image,(extLeft[0]-20,extLeft[1]-20),(extLeft[0]+20,extLeft[1]+20),(0,0,255),2)
            cv2.rectangle(color_image,(extRight[0]-20,extRight[1]-20),(extRight[0]+20,extRight[1]+20),(0,0,255),2)
            cv2.circle(color_image,(cx,cy), 10, (255,255,255), -1)

            # cv2.drawContours(color_image, hull, -1, (0,255,0), 3)

            cv2.imshow('Align Example', color_image)


        except:
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
