# -*- coding: utf-8 -*-
import cv2
import os
import datetime
import ImageProcessingUtil
import Image
import numpy
import pyttsx
from espeak import espeak


gesture = "stand"
sequence = "/informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/sequencesNimbro//"+gesture+"/"
motion = "/informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/motionNimbro/"+gesture+"/"


cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)
#vc2 = cv2.VideoCapture(1)
#vc2.set(4,640)
#vc2.set(5,480)
vc.set(11,3)
#vc.set(12,0)
#vc.set(18,0)

if vc.isOpened(): # try to get the first frame
    rval, f = vc.read()
 #   rval2, f2 = vc2.read()
else:
    rval = False

images = []
sequenceNumber = str(datetime.datetime.now())
putMotion = False
pastMotion = f
label = "Point Right"
#engine = pyttsx.init()
#engine.setProperty('rate', 50)
#voices = engine.getProperty('voices')
#engine.say("Point Right, Point Left, Circle, Stand, Turn") 
#'espeak.synth("Turn Left!")

#for voice in voices:
#    print "Using voice:", repr(voice)
#    engine.setProperty('voice', voice.id)
#    engine.say("Point Right, Point Left, Circle, Stand, Turn")    
#engine.runAndWait()


while rval:                            
    
    imageName =str(datetime.datetime.now())+".jpg"               
    #cv2.imwrite(sequence+"/"+str(sequenceNumber)+"/"+imageName,f)
    #frame = ImageProcessingUtil.detectSkin(frame)
    frame = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    
    #frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    images.append(frame)
        
    if not os.path.exists(sequence+"/"+str(sequenceNumber)+"/"):            
                        os.makedirs(sequence+"/"+str(sequenceNumber)+"/") 
    
    imageName =str(datetime.datetime.now())+".jpg"               
   # cv2.imwrite(sequence+"/"+str(sequenceNumber)+"/"+imageName,frame)
    
    
    newx,newy = 640,480 #new size (w,h)
    f = cv2.resize(f,(newx,newy))
    
    blank_image = Image.new("RGB", (1800, 1024))
    blank_image.paste(Image.fromarray(f), (1024,0))
    blank_image.paste(Image.fromarray(f), (1024,480))
    #print f2
    #cv2.imshow("preview2", f2)
    #blank_image.paste(f2, (768,480))
    if putMotion == True:
        blank_image.paste(Image.fromarray(pastMotion), (0,100))
    #print "Image:", blank_image

     
    if(len(images)==60):        
        sequenceNumber = str(datetime.datetime.now())
        motionImage = ImageProcessingUtil.doConvShadow(images) 
        cv2.imwrite("/informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/save.png",motionImage)
        motionImage = cv2.imread("/informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/save.png")
        motionImage = cv2.cvtColor(motionImage, cv2.COLOR_BGR2GRAY)
        if not os.path.exists(motion+"/"):            
                        os.makedirs(motion+"/") 
        imageName =str(datetime.datetime.now())+".jpg"   
        putMotion = True        
        newx,newy = 1024,768 #new size (w,h)
        pastMotion = cv2.resize(motionImage,(newx,newy))        
        
        #cv2.imwrite(motion+"/"+imageName,motionImage)
                                    
       # cv2.imshow("preview2", motionImage)
        #cv2.imwrite("//informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/save.png",motionImage)
        #motionImage = cv2.imread("//informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/save.png")
        #motionImage = cv2.cvtColor(motionImage, cv2.COLOR_BGR2GRAY)
 
        #cv2.imshow("preview2", motionImage)
        #image = MCNN.getOutputImage(outputConvLayers,batch_size,motionImage,imageSize,channels,len(conParams)) 
        
        #image = numpy.array(motionImage)
        #newx,newy = (100,100) #new size (w,h)
        #image = cv2.resize(image,(newx,newy))
        #print "Shape:",  image.shape
        #features = image.view(numpy.ndarray)
        #features.shape = -1
        
        #result = MCNN.classify(classifier,features,batch_size)
        #print result
        #if result[0] == 0:
        #    result = "Ok!"
       # elif result[0] == 1:
       #     result = "Not Ok!"
        #elif result[0]==2:
        #    result = "Point"
        #elif result[0]==3:
        #    result = "Stop"
                                
        #cv2.putText(motionImage, "Class: "+ str(result), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255))
       
        #cv2.imshow("preview3", motionImage)
       
        images = []          
          
      
    #cv2.imshow("preview", frame)                
    blank_image = numpy.array(blank_image)
    cv2.putText(blank_image, str(len(images)), (20,900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))  
    cv2.putText(blank_image, label, (20,1000), cv2.FONT_HERSHEY_SIMPLEX , 3, (255,0,0), 4)                    
    cv2.imshow("preview1", blank_image)
    rval, f = vc.read()
  #  rval2, f2 = vc2.read()
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break                

