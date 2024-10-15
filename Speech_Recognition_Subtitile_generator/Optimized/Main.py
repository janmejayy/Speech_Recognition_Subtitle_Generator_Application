# import glob
# import cv2
# import os
# import WordVideo
# import dlib
# import time
# import ROI
# import TPE
# import Gabor
# import Features
# a=time.time()
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # path of "shape_predictor_68_face_landmarks.dat"
# VideoPath ='/Users/jay/Documents/ba1.mp4'  # video path
# #VideoPath ='/Users/jay/Documents/DisserationProject/lrs2_v1_partaa/mvlrs_v1/main/**/*.mp4'
# Frame = '/Users/jay/Documents/FeatureExtraction/Picture/'# path to store pictures
# MouthPath = '/Users/jay/Documents/FeatureExtraction/mouth/'  # path to store mouth region
# GaborPath = '/Users/jay/Documents/FeatureExtraction/Gabor/'# path to store Gabor features
# SheetPath = '/Users/jay/Documents/FeatureExtraction/Sheet/' # path to store lip features
# FeaturesPath = '/Users/jay/Documents/FeatureExtraction/Features/'  # path to store lip features


# WordVideo.Frame(detector,predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath)


# b=time.time()
# print("Time = ",b-a)





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
import importlib
import configparser

a = time.time()

# Read the config file
config = configparser.ConfigParser()
config.read('config.ini')

# Get the paths from the config file
video_path = config.get('Paths', 'VideoPath')
frame_path = config.get('Paths', 'Frame')
mouth_path = config.get('Paths', 'MouthPath')
gabor_path = config.get('Paths', 'GaborPath')
sheet_path = config.get('Paths', 'SheetPath')
features_path = config.get('Paths', 'FeaturesPath')

# Get the predictor model path
predictor_model_path = config.get('Predictor', 'ModelPath')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_model_path)

WordVideo.Frame(detector, predictor, video_path, frame_path, mouth_path, gabor_path, sheet_path, features_path)

b = time.time()
print("Time =", b - a)
