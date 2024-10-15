# Gabor-based lipreading systems
This repository provides the code for lip feature extraction systems described . We also offer an updated version of the lip feature extraction system: 'FirstFrame' Gabor-based lip feature extraction system.

+ [Gabor features](#gabor-features)
+ [Environment](#environment)
+ [Gabor-Based Lip feature extraction system](#gabor-based-lip-feature-extraction-system)
  - [1. Gabor Based Lipreading with a New Audiovisual Mandarin Corpus](#1-gabor-based-lipreading-with-a-new-audiovisual-mandarin-corpus)
  - [2. Gabor-based Audiovisual Fusion for Mandarin Chinese Speech Recognition](#2-gabor-based-audiovisual-fusion-for-mandarin-chinese-speech-recognition)
  - [3. 'FirstFrame' Gabor-based lip feature extraction system](#3-firstframe-gabor-based-lip-feature-extraction-system)
+ [Citation](#citation)
+ [License](#license)
+ [Contact](#contact)

### Gabor features
The first GIF displays the original video where the speaker says "bin blue at e one again". The second GIF demonstrates the Gabor features of the changes in the lip area during the utterance of "bin blue at e one again". The image presents the curve of the lip area changes during the pronunciation of "bin blue at e one again".

<div style="display:flex;justify-content:space-between;">
    <img src="https://github.com/YX536/Gabor-based-lipreading-system/blob/main/bbae1a.gif" width="300" />
    <img src="https://github.com/YX536/Gabor-based-lipreading-system/blob/main/Area.gif" width="300" />
   <img src="https://github.com/YX536/Gabor-based-lipreading-system/blob/main/Area.png" width="300" />
</div>

### Environment
All package versions are recorded in the "Packages.txt" file.

### Gabor-Based Lip feature extraction system
#### 1. Gabor Based Lipreading with a New Audiovisual Mandarin Corpus
Authors: [Janmejay yadav](https://scholar.google.com/citations?user=2ICpMSsAAAAJ&hl=zh-CN&oi=sra),[Andrew Abel](https://pureportal.strath.ac.uk/en/persons/andrew-abel)
- Handcrafted Gabor-based lip feature extraction system
![Handcrafted Gabor-based lip feature extraction system](https://github.com/YX536/Gabor-based-lipreading-system/blob/main/Handcrafted.png)

  1). Main.py. 
  ```python 
       #Change path. Please modify the path in the code to your local path.
  12   predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # path of "shape_predictor_68_face_landmarks.dat"
  13   VideoPath ='D:/Handcrafted/*.mpg'  # video path 
  14   Frame = 'D:/Handcrafted/Picture/'# path to store frames
  15   MouthPath = 'D:/Handcrafted/mouth/'  # path to store mouth region
  16   GaborPath = 'D:/Handcrafted/Gabor/'#path to store Gabor features
  17   SheetPath = 'D:/Handcrafted/Sheet/' # path to store lip features
  18   FeaturesPath = 'D:/Handcrafted/Features/'  # path to store lip features
  ```
  2). Frame.py. Cut frames from one video. (Frames are stored in 'Picture' folder)

  3). ROI.py. Choose the region of interest (ROI) using Dlib 68 point. (Mouth pictures are stored in 'Mouth' folder)

  4). Gabor.py .Manually adjust Gabor parameters and generate Gabor features. (Gabor features are stored in 'Gabor' folder)

  5). Features.py.  Obtian 7 lip features: Width, height, area, intensity, orientation, the x-value of central point and the y value of central point. (Lip features are instored in 'Feature' folder and 'Sheet' folder)

To run the system, execute "Main.py".

##### Audiovisual Mandarin Chinese (AVMC) dataset:
This corpus is a labelled video corpus of distinct Chinese characters. 162 Chinese characters were collected from the general specification table published by the Chinese ministry of education. These characters are chosen with a reasonable distribution of initials, finals and tones. 10 native Mandarin Chinese volunteers were used, 5 of them are females, and 5 are males. Most of them are aged 22 and do not have a heavy accent. This corpus was recorded during the course of this research project, using the professional recording studio facilities at Xi'an Jiaotong-Liverpool University. The data capture devices are a Cannon camera 5D4 and a Sony microphone. The frame rate is set as 50. 

The data acquisition procedure for each volunteer was: (1) sign participant information and consent form; (2) read caption list; (3) practice recording for 1–2 min; (4) record for all captions and repeat 3 times. During recording, volunteers were asked to pause between each word, and if they made mistakes, paused and repeated. Mistakes not identified during recording were identified later in the editing process. This produced 30 videos, each being a volunteer reciting all 162 characters in a quiet environment, with a plain blue screen as background. To ensure they were looking directly at the screen, a teleprompter was used. As some Chinese characters have the same pronunciation, there are in total 158 types of pronunciations including both correct and wrong utterances. The video was recorded at a resolution of 1920 $\times$ 1080, at 50 fps, and the audio was recorded at 48 kHz. The processed videos are stored in the format of “mp4”. The example video is provided in ['ba1.mpg'](https://github.com/YX536/Gabor-based-lipreading-system/blob/main/ba1.mp4).

Please contact us if you are intersted in it: yan.xu[at]xjtlu.edu.cn, andrew.abel[at]strath.ac.uk.

![ChineseDataset](https://github.com/YX536/Gabor-based-lipreading-system/blob/main/AVMC.jpg)

#### 2. Gabor-based Audiovisual Fusion for Mandarin Chinese Speech Recognition
Authors: [Yan Xu](https://scholar.google.com/citations?user=2ICpMSsAAAAJ&hl=zh-CN&oi=sra), Hongce Wang, Zhongping Zhong, Yuexuan Li, [Andrew Abel](https://pureportal.strath.ac.uk/en/persons/andrew-abel) 
- Optimized Gabor-based lip feature extraction system (code is in "Optimized" folder)
![Optimized Gabor-based lip feature extraction system](https://github.com/YX536/Gabor-based-lipreading-system/blob/main/optimization.png)

  1). Main.py. 

  2). Frame.py. Cut frames from one video. (Frames are stored in 'Picture' folder)

  3). ROI.py. Choose the ROI using Dlib 68 point. (Mouth picture are stored in 'Mouth' folder)

  4). TPE.py. Find the most suitable Gabor parameters and iterating them for all frames. 
  ```python
         #You can change the path of test Gabor path to your local path.
  76     Gabor_Path = './TPEH.jpg'
  ```

  ```python
         #You can change the "Search space", 'iteration times' and 'best loss' according to your requirement.
         #Search range of Gabor parameters
  166        search_spaceH = {
  167        "Hkernel_size": hyperopt.hp.quniform('Hkernel_size', 5, 20, 1),
  168        "Hwavelength": hyperopt.hp.quniform('Hwavelength', 5, 20, 1),
  169        "Hsig": hyperopt.hp.quniform('Hsig', 3, 7, 1),
  170       "Hgamma": hyperopt.hp.quniform('Hgamma', 0.3, 0.7,0.1)
  171    }

  175    while True:
  176        try:
  177            trials = hyperopt.Trials()
  178            Hbest = hyperopt.fmin(
  179                        fn=HGabor,
  180                        space=search_spaceH,
  181                        algo=hyperopt.tpe.suggest,
  182                        max_evals=150,             #iteration times
  183                        trials=trials
  184                    )


  187            trial_loss = np.asarray(trials.losses(), dtype=float)
  188            best_loss = min(trial_loss)
  189            print('best loss: ', best_loss) 
  190            if best_loss>10:                           #best loss
  191                continue
  ```
  5). Gabor.py uses best Gabor parameters to generate gabor features. (Gabor features are stored in 'Gabor' folder)

  6). Features.py.  Obtian 7 lip features: Width, height, area, intensity, orientation, the x-value of central point and the y value of central point. (Lip features are instored in 'Feature' folder and 'Sheet' folder)

To run the system, execute "Main.py".

#### 3. 'FirstFrame' Gabor-based lip feature extraction system 

  1). Main.py. 

  2). Frame.py. Cut frames from one video. (Frames are stored in 'Picture' folder)

  3). ROI.py. Choose the ROI using Dlib 68 point. (Mouth picture are stored in 'Mouth' folder)

  4). TPE.py. Find the most suitable Gabor parameters and iterating them for first frames. 

  5). Gabor.py uses best Gabor parameters to generate gabor features. (Gabor features are stored in 'Gabor' folder)

  6). Features.py.  Obtian 7 lip features: Width, height, area, intensity, orientation, the x-value of central point and the y value of central point. (Lip features are instored in 'Feature' folder and 'Sheet' folder)

To run the system, execute "Main.py".


### License
The source code for the site is licensed under the MIT license, which you can find in the [LICENSE](https://github.com/JanmejayYadav/Gabor-based-lipreading-system) file.

### Contact
janmejay.yadav.1234@gmail.com
