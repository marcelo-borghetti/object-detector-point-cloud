import cv2
import numpy
import os
from subprocess import call


directoryFrom = "/informatik2/wtm/home/barros/Documents/Experiments/cam3D/imagesRaw/"
directoryTo = "/informatik2/wtm/home/barros/Documents/Experiments/cam3D/testingClassification/"
i = 0
classes = os.listdir(directoryFrom)
for c in classes:
    #sequences = os.listdir(directoryFace+"/"+c)    
    #for s in sequences:
        #files = os.listdir(directoryFace+"/"+c+"/"+s)            
        sequences = os.listdir(directoryFrom+"/"+c+"/")            
        for s in sequences:
            print directoryFrom+"/"+c+"/"+"/"+s+"/images/"
            files = os.listdir(directoryFrom+"/"+c+"/"+"/"+s+"/images/") 
                    
            
            if not os.path.exists(directoryTo+"/"+str(c)+"/"+s+"/"):            
                        os.makedirs(directoryTo+"/"+str(c)+"/"+s+"/")      
            
            video = directoryFrom+"/"+c+"/"+"/"+s+"/colour.avi"                
            vidcap = cv2.VideoCapture(video)

            frames = []
            
            success,image = vidcap.read()
            
            count = 0;
            while success:
              success,image = vidcap.read()
              
              frames.append(image)
              cv2.imwrite(directoryTo+"/"+str(c)+"/"+s+"/"+"/"+"frame%d.jpg" % count, image)     # save frame as JPEG file
              if cv2.waitKey(10) == 27:                     # exit if Escape is hit
                  break
              count += 1
              print "ffmpeg -framerate 1/5 -i "+directoryTo+"/"+str(c)+"/"+s+"/"+"frame%d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p "+directoryTo+"/"+str(c)+"/colour_label.mp4"
              call("ffmpeg -framerate 1/5 -i "+directoryTo+"/"+str(c)+"/"+s+"/"+"frame%d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p "+directoryTo+"/"+str(c)+"/colour_label.mp4")
# -*- coding: utf-8 -*-

