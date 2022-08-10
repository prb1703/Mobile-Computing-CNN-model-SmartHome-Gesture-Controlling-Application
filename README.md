SMART HOME GESTURE RECOGNITION APPLICATION

This application uses TensorFlow, OpenCV,and Kera with python to develop a smart home gesture recognition application. It uses 51 precaptured videos of 13 different gestures to train the CNN model and create labels for given test videos.

The gestures that are used in this application are Num0 to Num10, FanOn, FanOff, LightsOn, LightsOff, FanUp, FanDown,SetThermo.

The python application extract the middleframe of the 51 precaptured videos and train the model using those frame and once the CNN model is trained it extracts the middle frame of the test videos and uses cosine similarity to recognize the gesture.

The direction to start thinking about the project is mentioned below:

STEP 1: to generate the penultimate layer for the training set:

Extract the middle frames of all the training gesture videos.

For each gesture video, you will have one frame extract the hand shape feature by calling the get_Intsance() method of the HandShapeFeatureExtractor class. (HandShapeFeatureExtractor class uses CNN model that is trained for alphabet gestures)

For each gesture, extract the feature vector.

Feature vectors of all the gestures is the penultimate layer of the training set.

STEP 2: Follow the steps for Step 1 to get the penultimate layer of the test dataset.

STEP 3: 

Apply cosine similarity between the vector of the gesture video and the penultimate layer of the training set. Corresponding gesture of the training set vector with minimum cosine difference is the recognition of the gesture.

Save the gesture number to the results.csv

Recognize the gestures for all the test dataset videos and save the results to the results.csv file.

