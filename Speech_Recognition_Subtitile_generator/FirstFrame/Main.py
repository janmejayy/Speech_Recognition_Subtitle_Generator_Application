import glob
import cv2
import os
import WordVideo
import dlib
import time
import ROI
import TPE
import Gabor
import Features
a=time.time()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Video_path = r'D:/FeatureExtractionAuto/H/*.mpg'
VideoPath ='D:/FirstFrame/**/*.mpg'
Frame = 'D:/FirstFrame/Picture/'# path to store pictures
MouthPath = 'D:/FirstFrame/mouth/'  # path to store mouth
GaborPath = 'D:/FirstFrame/Gabor/'#path to store Gabor features
SheetPath = 'D:/FirstFrame/Sheet/' # path to storSheetPath
FeaturesPath = 'D:/FirstFrame/Features/'  # path to store sheets


WordVideo.Frame(detector,predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath)

b=time.time()
print("Time = ",b-a)
