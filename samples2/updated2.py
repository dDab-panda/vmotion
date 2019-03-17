from playsound import playsound
from threading import Thread
from multiprocessing import Process
import pyrealsense2 as rs
import numpy as np
import time
import math
import cv2

import pygame
pygame.init()

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
cx = cy = 0
lock1 = lock2 = 0
value1 = value2 = 1

def calculateDistance(x1,y1,x2,y2):
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return dist

cap = cv2.VideoCapture('video.mp4')
frame = np.zeros((1200,700,3), np.uint8)
# pygame.mixer.music.load("1.wav")
# pygame.mixer.music.play()

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
                cx = 320; cy = 240; dist = 220
                x1 = int(cx - dist * 0.500) - 50; y1 =  int(cy + dist * 0.866)
                x2 = int(cx - dist * 0.939) - 50; y2 = int(cy + dist * 0.342);
                x3 = int(cx - dist * 0.939) - 50; y3 = int(cy - dist * 0.342);
                x4 = int(cx - dist * 0.500) - 50; y4 = int(cy - dist * 0.866);

                x5 = int(cx + dist * 0.500) + 50; y5 =  int(cy + dist * 0.866)
                x6 = int(cx + dist * 0.939) + 50; y6 = int(cy + dist * 0.342);
                x7 = int(cx + dist * 0.939) + 50; y7 = int(cy - dist * 0.342);
                x8 = int(cx + dist * 0.500) + 50; y8 = int(cy - dist * 0.866);



            if(cap.isOpened()):
                ret, frame = cap.read()


            extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
            extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])

            d1 = calculateDistance(x1, y1, extLeft[0], extLeft[1])
            d2 = calculateDistance(x2, y2, extLeft[0], extLeft[1])
            d3 = calculateDistance(x3, y3, extLeft[0], extLeft[1])
            d4 = calculateDistance(x4, y4, extLeft[0], extLeft[1])

            list1 = [d1,d2,d3,d4]

            d5 = calculateDistance(x5, y5, extRight[0], extRight[1])
            d6 = calculateDistance(x6, y6, extRight[0], extRight[1])
            d7 = calculateDistance(x7, y7, extRight[0], extRight[1])
            d8 = calculateDistance(x8, y8, extRight[0], extRight[1])

            list2 = [d5,d6,d7,d8]

            dist1 = calculateDistance(cx, cy, extLeft[0], extLeft[1])
            dist2 = calculateDistance(cx, cy, extRight[0], extRight[1])

            ind1 = np.argmin(np.asarray(list1))
            ind2 = np.argmin(np.asarray(list2))

            # print ind1, ind2

            hull = []
            hull.append(cv2.convexHull(cnt, False))
            color_image = cv2.bitwise_and(colored_mask, colored_mask, mask=thresh)

            cv2.rectangle(color_image,(extLeft[0]-20,extLeft[1]-20),(extLeft[0]+20,extLeft[1]+20),(0,0,255),2)
            cv2.rectangle(color_image,(extRight[0]-20,extRight[1]-20),(extRight[0]+20,extRight[1]+20),(0,0,255),2)


            cv2.circle(color_image,(x1,y1), 20, (0,0,255), 2)
            cv2.circle(color_image,(x2,y2), 20, (0,0,255), 2)
            cv2.circle(color_image,(x3,y3), 20, (0,0,255), 2)
            cv2.circle(color_image,(x4,y4), 20, (0,0,255), 2)
            cv2.circle(color_image,(x5,y5), 20, (0,0,255), 2)
            cv2.circle(color_image,(x6,y6), 20, (0,0,255), 2)
            cv2.circle(color_image,(x7,y7), 20, (0,0,255), 2)

            cv2.circle(color_image,(x8,y8), 20, (0,0,255), 2)
            cv2.drawContours(color_image, hull, -1, (255,255,255), 3)


            if(dist1 > dist):
                cv2.circle(color_image,(extLeft[0], extLeft[1]), 20, (255,255,255), -1)
                # cv2.line(color_image, (extLeft[0], extLeft[1]), (extLeft[0], extLeft[1] + 100), (255,255,255), 5)
            if(dist2 > dist):
                cv2.circle(color_image,(extRight[0], extRight[1]), 20, (255,255,255), -1)
                # cv2.line(color_image, (extRight[0], extRight[1]), (extRight[0], extRight[1] + 100), (255,255,255), 5)
            inv = cv2.flip(color_image, 1)
            inv = cv2.resize(inv, (1280, 720))

            gray1 = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
            ret1, thresh1 = cv2.threshold(gray1, 1, 255, 0)
            thresh2 = cv2.bitwise_not(thresh1)
            frame1 = cv2.bitwise_and(frame, frame, mask=thresh2)
            frame2 = cv2.bitwise_and(inv, inv, mask=thresh1)
            frame = cv2.add(frame1, frame2)

            cv2.imshow('Align Example', frame)

            # print dist1, dist2
            if(dist1>130 and ind1==0):
                lock1 = 1 ; lock2 = 0 ; lock3 = 0; lock4=0 ; value2 = 1 ; value3=1 ; value4 =1;
                print "Lock1 acquired"


            if(dist1>150 and ind1==1):
                lock1 = 0 ; lock2 = 1 ; lock3 = 0; lock4=0 ; value1 = 1 ; value3=1 ; value4 =1;
                print "Lock2 acquired"

            if(dist1>220 and ind1==2):
                lock1 = 0 ; lock2 = 0 ; lock3 = 1; lock4=0 ; value2 = 1 ; value1=1 ; value4 =1;
                print "Lock3 acquired"

            if(dist1>220 and ind1==3):
                lock1 = 0 ; lock2 = 0 ; lock3 = 0; lock4=1 ; value1 = 1 ; value3=1 ; value2 =1;
                print "Lock4 acquired"


            if(lock1 == 1 and value1 == 1):
                value1 = 0
                print "startprocess1"
                pygame.mixer.music.load("1.wav")
                pygame.mixer.music.play()

            if(lock2 == 1 and value2 == 1):
                value2 = 0
                print "startprocess2"
                pygame.mixer.music.load("2.wav")
                pygame.mixer.music.play()

            if(lock3 == 1 and value3 == 1):
                value3 = 0
                print "startprocess3"
                pygame.mixer.music.load("3.wav")
                pygame.mixer.music.play()

            if(lock4 == 1 and value4 == 1):
                value4 = 0
                print "startprocess4"
                pygame.mixer.music.load("4.wav")
                pygame.mixer.music.play()


            if(dist2>130 and ind2==0):
                lock5 = 1 ; lock6 = 0 ; lock7 = 0; lock8=0 ; value6 = 1 ; value7=1 ; value8 =1;
                print "lock5 acquired"


            if(dist2>150 and ind2==1):
                  lock5 = 0 ; lock6 = 1 ; lock7 = 0; lock8=0 ; value5 = 1 ; value7=1 ; value8 =1;
                  print "lock6 acquired"

            if(dist2>220 and ind2==2):
                  lock5 = 0 ; lock6 = 0 ; lock7 = 1; lock8=0 ; value6 = 1 ; value5=1 ; value8 =1;
                  print "lock7 acquired"

            if(dist2>220 and ind2==3):
                  lock5 = 0 ; lock6 = 0 ; lock7 = 0; lock8=1 ; value5 = 1 ; value7=1 ; value6 =1;
                  print "lock8 acquired"


            if(lock5 == 1 and value5 == 1):
                  value5 = 0
                  print "startprocess5"
                  pygame.mixer.music.load("5.wav")
                  pygame.mixer.music.play()

            if(lock6 == 1 and value6 == 1):
                  value6 = 0
                  print "startprocess6"
                  pygame.mixer.music.load("6.wav")
                  pygame.mixer.music.play()

            if(lock7 == 1 and value7 == 1):
                  value7 = 0
                  print "startprocess7"
                  pygame.mixer.music.load("7.wav")
                  pygame.mixer.music.play()

            if(lock8 == 1 and value8 == 1):
                  value8 = 0
                  print "startprocess8"
                  pygame.mixer.music.load("8.wav")
                  pygame.mixer.music.play()

        except:
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
