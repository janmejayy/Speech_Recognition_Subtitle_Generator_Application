from configparser import ConfigParser

config =ConfigParser()

config['Paths'] = {
    #'VideoPath': '/Users/jay/Documents/00023.mp4',
    'VideoPath': '/Users/jay/Documents/DisserationProject/lrs2_v1_partaa/mvlrs_v1/main/**/*.mp4',
    'Frame': '/Users/jay/Documents/FeatureExtraction/Picture/',
    'MouthPath': '/Users/jay/Documents/FeatureExtraction/mouth/',
    'GaborPath': '/Users/jay/Documents/FeatureExtraction/Gabor/',
    'SheetPath': '/Users/jay/Documents/FeatureExtraction/Sheet/',
    'FeaturesPath': '/Users/jay/Documents/FeatureExtraction/Features/'
}

config['Predictor'] = {
    'ModelPath': 'shape_predictor_68_face_landmarks.dat'
}

with open('config.ini', 'w') as configfile:
    config.write(configfile)
