import glob
import cv2
import os
import WordVideo
import dlib
import time
import ROI
import Gabor
import Features
a=time.time()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # path of "shape_predictor_68_face_landmarks.dat"
VideoPath ='D:/Handcrafted/bbae1a.mpg'  # video path
Frame = 'D:/Handcrafted/Picture/'# path to store frames
MouthPath = 'D:/Handcrafted/mouth/'  # path to store mouth region
GaborPath = 'D:/Handcrafted/Gabor/'#path to store Gabor features
SheetPath = 'D:/Handcrafted/Sheet/' # path to store lip features
FeaturesPath = 'D:/Handcrafted/Features/'  # path to store lip features


WordVideo.Frame(detector,predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath)

b=time.time()
print("Time = ",b-a)
