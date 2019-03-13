from Tkinter import *
import os
import pygame
import tkFileDialog
import numpy as np
import time

master = Tk()
master.minsize(360,70)
master.title("vmotion")

trkno = 0
songs = []

pygame.init()

dir = "/home/crypto/Music"
os.chdir(dir)
for i in os.listdir(dir):
    if i.endswith(".mp3"):
        songs.append(i)
print songs
pygame.mixer.init()
pygame.mixer.music.load(songs[0])

def play():
    global trkname
    global trkno
    global pi
    if pi==0:
        pygame.mixer.music.play()
        trkname.set((str(songs[trkno])).replace(".mp3",""))
    else:
        pygame.mixer.music.unpause()

def stop():
    global pi
    pygame.mixer.music.pause()
    pi=1


def nexttrk():
    global trkname
    global trkno
    global songs
    trkno += 1
    pygame.mixer.music.load(songs[trkno])
    pygame.mixer.music.play()
    trkname.set((str(songs[trkno])).replace(".mp3",""))


def prevtrk():
    global trkname
    global trkno
    global songs
    trkno -= 1
    pygame.mixer.music.load(songs[trkno])
    pygame.mixer.music.play()
    trkname.set((str(songs[trkno])).replace(".mp3",""))


def voli():
    global vol
    vol=pygame.mixer.music.get_volume()
    vol += 0.1
    if vol>=0.9921875:
        vol = 0.9921875
    pygame.mixer.music.set_volume(vol)


def vold():
    global vol
    vol=pygame.mixer.music.get_volume()
    vol -= 0.1
    if vol<0:
        vol = 0.0
    pygame.mixer.music.set_volume(vol)




pi = 0
vol = 1

trkname=StringVar()

play()

print "Done"
time.sleep(5000)

nexttrk()

master.mainloop()
