# Passive-Liveliness Detector
Results from my attempt at making a passive liveliness detector as part of my summer internship.

# Contents
1. [Dependencies](#Dependencies)
2. [Datasets](#Datasets)
3. [Training and Testing](#TrainingAndTesting)
4. [Pipeline](#Pipeline)
5. [Misc](#MISC)

## Dependencies 
This project is based on a liveliness net 3 layered CNN model to detect fake-face submissions (see inside liveliness folder)
External libraries used include:
- `Face Detectors`: Multiple face detectors have been applied to cut 32x32 face ROIs:
                    i)   a *caffe model* face detector
                    ii)  a *face_recognition* library
                    iii) a *retina-face* from dlib 
- `Tensorflow`    : 2.9.2 (latest version required only for training, testing can be achieved on lower models; see imports needed for testing)
- `Keras`         : 2.9.0
- `Sklearn`       : 1.1.1
- Others - check imports in train/test file for smaller imported libs

## Datasets
- A huge hurdle in the project was dataset geneartion for machine learning. Real dataset was acquired from an exisiting database at my internship. 
- **Fake images** were generated by clubbing real images into a 1 FPS video that was recorded under different circumstances and using different equipment to ensure diversity. The fake images were then cut on each frame, tagged with a name (which was stored when generating the real video)
- Care was taken to make sure that there was ample test_data different from the train/validation data. 
- Sample of datasets can be found in liveliness folder. Entire database has not been uploaded.
- There also exist Fail_cases; a sample is present in the liveliness folder. These were used to improve model accuracy on each attempt.

## Training and Testing

### Training
- After applying face detectors, the processed dataset was used to train the **model classifier** with 2 pre-dominant labels - REAL and FAKE (see train files for different types of training - with face detection and without)
- One-hot label encoder is used (see le.pickle file)
- Classification report, training plot and corresponding model is then produced and stored (samples available)

### Testing
- The model was then tested against a fresh test_dataset (sample given) and accuracy
was checked. (notes.txt)
- Different conditions such as 'Initial Learning Rate' and 'Epochs' were altered
- Fail data can be recorded and viewed during the process

## Pipeline
- Finally a containerised (docker image) pipeline was developed that used an external API to test the model on incoming images. This has been put into place using the detect and test_detect python files. 
- Results are stored under publish folder

## Misc - What's next?
- Before applying the ANN approach, I explored other approaches such as **Monocular depth estimation** and **Multiple image disparity map** generation (check disparity folder) but this seemed to be the most fruitful approach. 
- Code to generate fake datasets etc. can be found outside the liveliness folder
- **JPGvsPNG** : while JPG images are lighter it seems there is very slight/minial loss in accuracy when using jpg for training data. Preffered form remains to be PNG images
- This repo does not hold the entire codebase and other plots (from differnet trainings). These can be provided on request. 
- The model is still improving. Currently progress depends on better input images. Work is being done to enhance input data quality. 

__This project has taught me technicals such as image processing and AI/ML using CV2, Python, Sklearn, Tensorflow/Keras, Docker as well as working in a company structure and getting hands on industry experience__

I would like to thank __Dr. Maria Valera-Espina__,__Som Raj__, __Amit Jain__, __Devendra Kumar__ ,__Abhishek Agarwal__ and the entire team at __ElixirCT__ for their guidance and support.
