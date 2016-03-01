# -*- coding: utf-8 -*-

import cv2
import cv
import os
import shutil


"""
directory = "/informatik2/wtm/home/barros/Pictures/Datasets/Dynamic Gestures Dataset/video/"
directorySave = "/informatik2/wtm/home/barros/Pictures/Datasets/Dynamic Gestures Dataset/images/30.04.2014/"

imgBase = cv2.imread("/informatik2/wtm/home/barros/Pictures/Datasets/Dynamic Gestures Dataset/out0104.png")
imgBase = cv2.cvtColor(imgBase, cv2.COLOR_BGR2GRAY)
histBase = cv2.calcHist([imgBase],[0],None,[256],[0,256])
histBase = cv2.normalize(histBase,histBase,0,255,cv2.NORM_MINMAX)

subjects = os.listdir(directory)
for s in subjects:    
    gestures = os.listdir(directory+"/"+s+"/rgb/")    
    for g in gestures:
       if os.path.isdir(directory+"/"+s+"/rgb/"+"/"+g):
        sequence = 0
        cut = False        
        files = os.listdir(directory+"/"+s+"/rgb/"+"/"+g) 
        for f in files:
            
            img = cv2.imread(directory+"/"+s+"/rgb/"+"/"+g + "/" + f)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        
            hist = cv2.calcHist([img],[0],None,[256],[0,256])
            hist = cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
            
            compareValue = cv2.compareHist(hist, histBase, cv.CV_COMP_CORREL)
            if compareValue <= 0.40:
                if(cut == True):
                    sequence = sequence+1
                    cut=False
                    
                if not os.path.exists(directorySave + "/"+s+"/"+g+"/"+"/"+str(sequence)+"/"): os.makedirs(directorySave + "/"+s+"/"+g+"/"+"/"+str(sequence)+"/")
                print  directorySave + "/"+s+"/"+g+"/"+"/"+str(sequence)+"/"+f + " ---- " + str(compareValue)
                shutil.copyfile(directory+"/"+s+"/rgb/"+"/"+g + "/" + f,directorySave + "/"+s+"/"+g+"/"+"/"+str(sequence)+"/"+f)
                lastHistogram = 0
            else:    
                cut = True  
            
"""