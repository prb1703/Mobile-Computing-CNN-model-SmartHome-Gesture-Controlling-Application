# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import glob
import os
import cv2
import frameextractor
from handshape_feature_extractor import HandShapeFeatureExtractor
import numpy as np

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
# your code goes here
videopath = os.path.join('traindata')
framepath = os.path.join(os.getcwd(), "trainingframes")
vpath = os.path.join(videopath, "*.MOV")
vid = glob.glob(vpath)
Tlabellist = []
cntr = 0

for each in vid:
    frameextractor.frameExtractor(each, framepath, cntr)
    cntr += 1

path = os.path.join(framepath, "*.png")
frame = glob.glob(path)
frame.sort()
flist = frame
model = HandShapeFeatureExtractor.get_instance()
vec = []
for i in flist:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = model.extract_feature(img)
    vec.append(out)

vec = np.asarray(vec)
trainvector = vec.reshape((-1, vec.shape[2]))

for i in vid:
    label = i.split('_')
    label = label[0].split('\\')
    if label[1].startswith("Num"):
        label[1] = label[1].replace('Num', "")
    if label[1] == 'FanDown':
        label[1] = 10
    if label[1] == 'FanOn':
        label[1] = 11
    if label[1] == 'FanOff':
        label[1] = 12
    if label[1] == 'FanUp':
        label[1] = 13
    if label[1] == 'LightsOff':
        label[1] = 14
    if label[1] == 'LightsOn':
        label[1] = 15
    if label[1] == 'SetThermo':
        label[1] = 16
    Tlabellist.append(label[1])

labelsTrain = np.asarray(Tlabellist).reshape(-1, 1)


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
videopathtest = os.path.join('test')
framepathtest = os.path.join(os.getcwd(), "testingframe")
vpathtest = os.path.join(videopathtest, "*.mp4")

vid = glob.glob(vpathtest)
cntr = 0

for i in vid:
    frameextractor.frameExtractor(i, framepathtest, cntr)
    cntr += 1

pathtest = os.path.join(framepathtest, "*.png")
frame = glob.glob(pathtest)
frame.sort()
flist = frame
model = HandShapeFeatureExtractor.get_instance()
vec = []
for i in flist:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = model.extract_feature(img)
    vec.append(output)

vec= np.asarray(vec)
testvector = vec.reshape((-1, vec.shape[2]))

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
res= []
for i in testvector:
    flabel = []
    for j in trainvector:
        flabel.append(np.dot(i, j)/(np.linalg.norm(i)*np.linalg.norm(j)))
    flabel = np.array(flabel)
    idx = np.argmax(flabel)
    vallabel = labelsTrain[idx]
    res.append(vallabel)

res = np.array(res).reshape(-1, 1)

with open('Results.csv', 'w') as f:
    res = res.reshape(1, -1)
    for each in res:
        np.savetxt("Results.csv", each, delimiter=",", fmt='%s')