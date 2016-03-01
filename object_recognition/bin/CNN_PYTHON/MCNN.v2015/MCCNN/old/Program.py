# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import MCNN
import LogUtil
import DataUtil
import datetime
import numpy
import cv2
import ImageProcessingUtil
import Image
import time as timeThread
import os
from subprocess import call
from naoqi import ALProxy
import sys
import almath
#import pyttsx
#from espeak import espeak

#import roslib
#import sys
#import rospy
#from std_msgs.msg import String
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge, CvBridgeError


def speak(IP, PORT, String):
    tts = ALProxy("ALTextToSpeech", IP, PORT)
    tts.setLanguage("English")
    tts.say(String)
    
def goPosture(IP,PORT, COMMAND):
    
    try:
        motionProxy = ALProxy("ALMotion", IP, PORT)
    except Exception,e:
        print "Could not create proxy to ALMotion"
        print "Error was: ",e
        sys.exit(1)        
    

    motionProxy.setStiffnesses("RArm", 1.0)
    
    motionProxy.setStiffnesses("Head", 1.0)

    # Simple command for the HeadYaw joint at 10% max speed
    names            = COMMAND[0]
    angles           = COMMAND[1]
    fractionMaxSpeed = 0.2
    motionProxy.setAngles(names,angles,fractionMaxSpeed)

    speak(IP,PORT,COMMAND[4])
    
    timeThread.sleep(2.0)
    
    
    names            = COMMAND[2]
    angles           = COMMAND[3]
    fractionMaxSpeed = 0.2
    motionProxy.setAngles(names,angles,fractionMaxSpeed)

    timeThread.sleep(2.0)
    
    motionProxy.setStiffnesses("RArm", 0.0)
    motionProxy.setStiffnesses("Head", 0.0)

def callbackFromTreatRequest(data):
    request = 0; 
  
def listener():
    rospy.Subscriber("/RequestFromInterfaceToCNN", String, callbackFromTreatRequest)

def MCNN3ChannelsExperiment(experimentName,baseDirectory, testImageDirectory, repetitions, featuresFileName, train, inputType, channelsTopology, params, timeStep, hintonDiagrams, createOutputImages, liveClassification, createConvFeatures, nimbroClassification, createFeaturesFile, videoClassification, crossValidation,crossValidationSets, leaveOneOutJaffeExperiment, visualizeTrain):      
    
    
    
    
    
    pointingRight = [["RShoulderRoll", "RShoulderPitch","RElbowYaw", "RElbowRoll","RWristYaw", "RHand", "HeadYaw", "HeadPitch"],
                 [-67.8*almath.TO_RAD,59.2*almath.TO_RAD,6*almath.TO_RAD,8.9*almath.TO_RAD,-4.7*almath.TO_RAD,0.67, -53.2*almath.TO_RAD,12.0*almath.TO_RAD ],
                 ["RShoulderRoll", "RShoulderPitch","HeadYaw", "HeadPitch"],
                 [-8.6*almath.TO_RAD,86.7*almath.TO_RAD,-0.9*almath.TO_RAD,6.8*almath.TO_RAD],
                 "Pointing to the right."]


    pointingLeft = [["LShoulderRoll", "LShoulderPitch","LElbowYaw", "LElbowRoll","LWristYaw", "LHand", "HeadYaw", "HeadPitch"],
                     [67.8*almath.TO_RAD,59.2*almath.TO_RAD,6*almath.TO_RAD,-8.9*almath.TO_RAD,-4.7*almath.TO_RAD,0.67, 53.2*almath.TO_RAD,12.0*almath.TO_RAD ],
                     ["LShoulderRoll", "LShoulderPitch","HeadYaw", "HeadPitch"],
                     [-8.6*almath.TO_RAD,86.7*almath.TO_RAD,-0.9*almath.TO_RAD,6.8*almath.TO_RAD],
                     "Pointing to the left."]
                     
     
    stop = [["RShoulderRoll", "RShoulderPitch","RElbowYaw", "RElbowRoll","RWristYaw", "RHand", "HeadYaw", "HeadPitch"],
                     [-10*almath.TO_RAD,-3*almath.TO_RAD,28.5*almath.TO_RAD,36.4*almath.TO_RAD,35.7*almath.TO_RAD,0.90, -0.2*almath.TO_RAD,-6.9*almath.TO_RAD ],
                     ["RShoulderRoll", "RShoulderPitch","HeadYaw", "HeadPitch"],
                     [-8.6*almath.TO_RAD,86.7*almath.TO_RAD,-0.9*almath.TO_RAD,6.8*almath.TO_RAD],
                     "Stop now!"]             




    if inputType == DataUtil.INPUT_TYPE["3D"]:
        imagesDirectory = baseDirectory + "/sequences"
    else:
        imagesDirectory = baseDirectory + "/images"
        
    featuresDirectory = baseDirectory + "/features/"
    featuresFile =  featuresDirectory + featuresFileName
    
    
    
    baseDirectory = baseDirectory + "/experiments/"+experimentName+"/"
    
    metricsDirectory = baseDirectory+"/metrics/"
    metricsFile = metricsDirectory+"/Metrics_"+experimentName+".txt"
    convFeaturesDirectory=baseDirectory+"/convFeatures/"
    modelDirectory=baseDirectory+"/model/"
    hintonDiagram=baseDirectory+"/hintonDiagram/"
    outputImages=baseDirectory+"/outputImages/"
    saveHistoryImageFiltersDirectory = hintonDiagram+"/"+"FiltersDuringTraining"
    
    
    log = LogUtil.LogUtil()
    log.createLog(experimentName,metricsDirectory)
    
    log.createFolder(metricsDirectory)
    log.createFolder(convFeaturesDirectory)
    log.createFolder(modelDirectory)
    log.createFolder(hintonDiagram)          
    log.createFolder(saveHistoryImageFiltersDirectory)
    log.createFolder(outputImages)     
    
    
    precisionMicro = []
    precisionMacro = []
    precisionWeighted = []
    
    recallMicro = [] 
    recallMacro = []
    recallWeighted = []
        
    fScoreMicro = []
    fScoreMacro = []
    fScoreWeighted = []

    accuracy = []

    precisionsPerClass = []
    recallsPerClass = []
    fScoresPerClass = [] 
    
    
    classesInEachSet = []    
    
    trainingTime = []
    recognitionTime = []                
     
    conParams = params[0] 
    unitsInHiddenLayer = params[1]
    outputUnits = params[2]
    learningRate = params[3]        
    batch_size = params[4]
    imageSize = params[5]
    epochs = params[6]
    trainDataBaseRate = params[7][0]
    testDataBaseRate = params[7][1]
    
    log.startNewStep("Network Parameters")
    log.printMessage(("Learning Rate: ", learningRate))
    log.printMessage(("Batch Size: ", batch_size))
    log.printMessage(("Input image size: ", imageSize))
    log.printMessage(("Convolution Layers Parameters: ", conParams))
    log.printMessage(("Units Hidden Layer: ", unitsInHiddenLayer))
    log.printMessage(("Units Output Layer: ", outputUnits))
    
    
    if createFeaturesFile:
            log.startNewStep(("Creating features file."))                
                        
            ImageProcessingUtil.createFeatureFile(imageSize,imagesDirectory,featuresDirectory,featuresFileName, log, inputType)        
            
            log.startNewStep(("File created: ", featuresFileName ))       
        

    for i in range(repetitions): 
        if not crossValidation:
            crossValidationSets = 1
        for crossIndex in range(crossValidationSets):              
             
            if crossValidation:
                log.startNewStep(("Starting repetition: "+str(i)+" -------  "+str(crossValidationSets)+"-fold cross validation:  "+ str(crossIndex)))                        
            else:
                log.startNewStep(("Starting Repetition "+ str(i)))                        
            
            savedState = modelDirectory +"model_"+experimentName+"_"+featuresFileName+"_Repetition_"+str(i)+".save"
            if train:
                loadFrom = ""
            else:
                loadFrom = modelDirectory +"model_"+experimentName+"_"+featuresFileName+"_Repetition_"+str(i)+".save"
                #loadFrom = modelDirectory +"nn.save"
                                   
            if leaveOneOutJaffeExperiment:
               datasets = DataUtil.createFeatureFileJaffeExperiment(imagesDirectory,log,imageSize)              
            elif crossValidation:
                dataSets = DataUtil.loadDataCrossValidation(log,featuresFile, crossValidationSets,"Static")
                trainingInput = []
                trainingOutput = []
                for l in range(crossValidationSets):
                    if not l == crossIndex:                    
                        trainingInput.extend(dataSets[0][l])
                        trainingOutput.extend(dataSets[1][l])
                        
                sharedTraining = DataUtil.shared_dataset((trainingInput,trainingOutput))
                sharedTesting = DataUtil.shared_dataset((dataSets[0][crossIndex],dataSets[1][crossIndex]))
                datasets = [sharedTraining,sharedTraining,sharedTesting]
            else:
                datasets = DataUtil.loadData(log,featuresFile, trainDataBaseRate,testDataBaseRate,timeStep,inputType)                    
                
                
                
            time = datetime.datetime.now()
                  
                                                                                               
            dataX,dataY,outputConvLayers,convolutionOutputLayer,classifier = MCNN.MCNN3Channels(channelsTopology,inputType,datasets,train,loadFrom,savedState,conParams,unitsInHiddenLayer,outputUnits, imageSize, epochs, learningRate, batch_size,log,  saveHistoryImageFiltersDirectory, i, timeStep, visualizeTrain)                      
            #dataX,dataY,outputConvLayers,convolutionOutputLayer,classifier = MCNN.MCNN3Channels(channels,datasets,train,loadFrom,savedState,conParams,unitsInHiddenLayer,outputUnits, imageSize, epochs, learningRate, batch_size,log, useUniversalFilters, saveHistoryImageFiltersDirectory, i, useInhibition, useColorImage)                      
            
            log.startNewStep("Metrics")
           
            time = (datetime.datetime.now()-time).total_seconds()
                    
            trainingTime.append(time)        
            
            trueData = []     
            for value in dataY.eval():
                trueData.append(value)
             
            predictedData = []
                  
                 
            for value in dataX.get_value(borrow=True):                
                timeR = datetime.datetime.now()
                predictedData.append(MCNN.classify(classifier,value,batch_size)[0])         
                timeR = (datetime.datetime.now() - timeR).total_seconds()
                recognitionTime.append(timeR)
              
                         
            MCNN.getClassificationReport(trueData,predictedData,metricsFile,metricsDirectory,experimentName,i,log)
            
            accuracy.append(MCNN.getAccuracy(trueData,predictedData))            
            
            precisionMicro.append(MCNN.getPrecision(trueData,predictedData,"micro"))
            
            recallMicro.append(MCNN.getRecall(trueData,predictedData,"micro"))
            fScoreMicro.append(MCNN.getFScore(trueData,predictedData,"micro"))
            
            precisionMacro.append(MCNN.getPrecision(trueData,predictedData,"macro"))
            recallMacro.append(MCNN.getRecall(trueData,predictedData,"macro"))
            fScoreMacro.append(MCNN.getFScore(trueData,predictedData,"macro"))
            
            precisionWeighted.append(MCNN.getPrecision(trueData,predictedData,"weighted"))
            recallWeighted.append(MCNN.getRecall(trueData,predictedData,"weighted"))
            fScoreWeighted.append(MCNN.getFScore(trueData,predictedData,"weighted"))                  
            
            log.printMessage(("Accuracy:", accuracy[len(accuracy)-1])) 
            log.printMessage(("Training time:", time))
            
                    
            classesPerSet = []
            for t in trueData:
                if not t in classesPerSet:
                    classesPerSet.append(t)
                    
            classesInEachSet.append(classesPerSet)
            
            precisionsPerClass.append(MCNN.getPrecision(trueData,predictedData,None))
            recallsPerClass.append(MCNN.getRecall(trueData,predictedData,None))
            fScoresPerClass.append(MCNN.getFScore(trueData,predictedData,None))
        
              
        
    averagePrecisionMicro = numpy.mean(precisionMicro)
    averageRecallMicro = numpy.mean(recallMicro)
    averageFScoreMicro = numpy.mean(fScoreMicro)
    
    averagePrecisionMacro = numpy.mean(precisionMacro)
    averageRecallMacro = numpy.mean(recallMacro)
    averageFScoreMacro = numpy.mean(fScoreMacro)
    
    averagePrecisionWeighted = numpy.mean(precisionWeighted)
    averageRecallWeighted = numpy.mean(recallWeighted)
    averageFScoreWeighted = numpy.mean(fScoreWeighted)
    
    trainingTimeAvg = numpy.mean(trainingTime)
    recognitionTimeAvg = numpy.mean(recognitionTime)
    accuracyAverage = numpy.mean(accuracy)

    
    
    setPrecisionPerClasses = []
    setRecallPerClasses = []
    setFScorePerClasses = []
    for c in range(outputUnits):
        setPrecisionPerClasses.append([])
        setRecallPerClasses.append([])
        setFScorePerClasses.append([])
        
    h = 0        
    for c in classesInEachSet:        
        u = 0
        for i in c:            
            setPrecisionPerClasses[i].append(precisionsPerClass[h][u])
            setRecallPerClasses[i].append(recallsPerClass[h][u])
            setFScorePerClasses[i].append(fScoresPerClass[h][u])
            u = u+1
        h = h+1
    
    
    
    averagePrecisionPerClasses = []
    averageRecallPerClasses = []
    averageFScorePerClasses = []
    
    for a in range(len(setPrecisionPerClasses)):                
        averagePrecisionPerClasses.append(numpy.mean(setPrecisionPerClasses[a]))
        averageRecallPerClasses.append(numpy.mean(setRecallPerClasses[a]))
        averageFScorePerClasses.append(numpy.mean(setFScorePerClasses[a]))
        
#    for c in range(outputUnits):
#        averageClass = []
#        h = 0
#        for p in precisionsPerClass:
#            if c in classesInEachSet[h]:          
#                averageClass.append(p[c])
#            h=h+1                        
#        averagePrecisionPerClasses.append(numpy.mean(averageClass))    
            
#    averageRecallPerClasses = []
#    for c in range(outputUnits):
#        averageClass = []
#        for p in recallsPerClass:
#            if c in classesInEachSet[p]:          
#                averageClass.append(p[c])                        
#        averageRecallPerClasses.append(numpy.mean(averageClass))    
#        
#    averageFScorePerClasses = []
#    for c in range(outputUnits):
#        averageClass = []
#        for p in fScoresPerClass:
#            if c in classesPerSet[p]:          
#                averageClass.append(p[c])                        
#        averageFScorePerClasses.append(numpy.mean(averageClass))    
    
    if crossValidation:
        log.startNewStep("Final Metrics for "+str(repetitions) + " repetitions with "+str(crossValidationSets)+"-fold fross validation")  
    else:
        log.startNewStep("Final Metrics for "+str(repetitions) + " repetitions.")  
        
    log.printMessage("Training time:" + str(trainingTimeAvg) + " -  Standard Deviation: " + str(numpy.std(trainingTime)))
    log.printMessage("Recognition time:" + str(recognitionTimeAvg) + " -  Standard Deviation: " + str(numpy.std(recognitionTime)))


    
    log.startNewStep("Accuracy")        
    log.printMessage("Accuracy:" + str(accuracyAverage) + " -  Standard Deviation: " + str(numpy.std(accuracy)))
    
    log.startNewStep("Precision")
    log.printMessage("Precision Weighted:" + str(averagePrecisionWeighted) + " -  Standard Deviation: " + str(numpy.std(precisionWeighted)))
    #log.printMessage("Precision Micro:" + str(averagePrecisionMicro) + " -  Standard Deviation: " + str(numpy.std(precisionMicro)))
    #log.printMessage("Precision Macro:" + str(averagePrecisionMacro) + " -  Standard Deviation: " + str(numpy.std(precisionMacro)))
    
    
    log.printMessage("Precision Per class:")    
    i = 1
    for c in averagePrecisionPerClasses:
       log.printMessage("      Class "+str(i)+": "+str(c))
       i = i+1    
         
    log.startNewStep("Recall")
    log.printMessage("Recall Weighted:" + str(averageRecallWeighted) + " -  Standard Deviation: " + str(numpy.std(recallWeighted)))
    #log.printMessage("Recall Micro:" + str(averageRecallMicro) + " -  Standard Deviation: " + str(numpy.std(recallMicro)))
    #log.printMessage("Recall Macro:" + str(averageRecallMacro) + " -  Standard Deviation: " + str(numpy.std(recallMacro)))
    
 
    log.printMessage("Recall Per class:")    
    i = 1
    for c in averageRecallPerClasses:
       log.printMessage("      Class "+str(i)+": "+str(c))
       i = i+1    
                  
    
    log.startNewStep("F1 Score")
    log.printMessage("F1 Score Weighted:" + str(averageFScoreWeighted) + " -  Standard Deviation: " + str(numpy.std(fScoreWeighted)))
    #log.printMessage("F1 Score Micro:" + str(averageFScoreMicro) + " -  Standard Deviation: " + str(numpy.std(fScoreMicro)))
    #log.printMessage("F1 Score Macro:" + str(averageFScoreMacro) + " -  Standard Deviation: " + str(numpy.std(fScoreMacro)))
    
 
    log.printMessage("F1 Score Per class:")    
    i = 1
    for c in averageFScorePerClasses:
       log.printMessage("      Class "+str(i)+": "+str(c))
       i = i+1          
    
    if(createConvFeatures):
        log.startNewStep("Creating ConvFeatures")
        features = MCNN.getConvolutionalFeatures(convolutionOutputLayer,imagesDirectory,batch_size,imageSize)
        DataUtil.writeSingleFile(features, convFeaturesDirectory+"ConvFeatures_"+experimentName+".txt")           
        log.printMessage(("ConvFeatures created at : ", convFeaturesDirectory+"ConvFeatures_"+experimentName+".txt"))    
    
    if(hintonDiagrams):
        log.startNewStep("Creating Hinton Diagrams")        
            
        MCNN.createHintonDiagram(savedState,hintonDiagram,len(conParams),len(channelsTopology),len(channelsTopology)*len(conParams) +2)
    
    if(createOutputImages):
            log.startNewStep("Showing OutputImages")
            #MCNN.showOutputImages(outputConvLayers,batch_size,imagesDirectory,imageSize,channels,len(conParams))
            log.startNewStep("Creating OutputImages")
            outputIndex = 0
            
                        
            for c in range(len(channelsTopology)):
                                
                if channelsTopology[c][0] == DataUtil.CHANNEL_TYPE["SobelX"]:
                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/Input(SobelX)/"
                    log.printMessage(("Creating Output in: ", directoryToSaveImage))                    
                    MCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
                    outputIndex = outputIndex+1 
                                
                elif channelsTopology[c][0] == DataUtil.CHANNEL_TYPE["SobelY"]:
                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/Input(SobelY)/"
                    log.printMessage(("Creating Output in: ", directoryToSaveImage))                    
                    MCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
                    outputIndex = outputIndex+1
                else:
                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/Input/"
                    log.printMessage(("Creating Output in: ", directoryToSaveImage))                    
                    MCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
                    outputIndex = outputIndex+1  
                        
                for l in range(len(conParams)):                                    
                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/layer"+str(l)+"/"+"Conv/"
                    log.printMessage(("Creating Output in: ", directoryToSaveImage))
                    MCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
                    outputIndex = outputIndex+1
                    
                    directoryToSaveImage =  outputImages+"/Channel "+str(c)+"/layer"+str(l)+"/"+"MaxPooling/"
                    log.printMessage(("Creating Output in: ", directoryToSaveImage))
                    MCNN.createOutputImages(outputConvLayers[outputIndex],batch_size, imagesDirectory, directoryToSaveImage, imageSize, inputType)
                    outputIndex = outputIndex+1
    
#    if (nimbroClassification):
#          ic = nimbroImageReceiver(classifier,batch_size,imageSize)
#          rospy.init_node('image_converter', anonymous=True)
#          try:
#            rospy.spin()
#          except KeyboardInterrupt:
#            print "Shutting down"          
    
    if(liveClassification):
        listener()
                
        #IP = "192.168.104.112"
        #PORT = 9559
        #camProxy = ALProxy("ALVideoDevice", IP, PORT)
        #resolution = 1    # VGA
        #colorSpace = 11   # RGB
        images = []
        while (100):
	    rospy.spin()
	    if ( request == 1 ):
		print "Inside the loop...\n"
		#videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)
	    
		  # Get a camera image.
		  # image[6] contains the image data passed as an array of ASCII chars.
		#naoImage = camProxy.getImageRemote(videoClient)
		#camProxy.unsubscribe(videoClient)
		#imageWidth = naoImage[0]
		#imageHeight = naoImage[1]
		#array = naoImage[6]
		#imageWidth = naoImage[0]
		#imageHeight = naoImage[1]
		#array = naoImage[6]
		#frame = Image.fromstring("RGB", (imageWidth, imageHeight), array)              
		
		#imageWidth = 100
		#imageHeight = 100
		frame = misc.imread(testImageDirectory+"1.png")
		
		#im = Image.fromstring("RGB", (imageWidth, imageHeight), array)
		
		im = numpy.array(frame)
		print "before:", im.shape
		im = im[:, :, ::-1].copy()            
		#im = ImageProcessingUtil.grayImage(im,imageSize ,False,"")
			    
		im = ImageProcessingUtil.resize(im,imageSize)
		im = im.view(numpy.ndarray)            
		im = ImageProcessingUtil.whiten(im)
		#im.shape = -1
		
		im = numpy.swapaxes(im, 2,1)
		im = numpy.swapaxes(im, 0,1)
		
		im = numpy.reshape(im, (3, imageSize[0]*imageSize[1]))
		
		print "after:", im.shape
		result = MCNN.classify(classifier,im,batch_size)
		print result
		if result[0] == 0:
		    result = "Circle"
		if result[0] == 1:
		    result = "Point Left"
		elif result[0] == 2:
		    result = "Point Right"
		elif result[0]==3:
		    result = "Standing"
		elif result[0]==4:
		    result = "Stop"
		elif result[0]==5:
		    result = "Turn"
		    
		cv2.putText(frame, str(result), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
		cv2.imshow("preview3", frame)
		    
		#frame = numpy.array(im)
		
		
		#images.append(im)
		frame = numpy.array(frame)
		frame = frame[:, :, ::-1].copy()            
		cv2.putText(frame, str(len(images)), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))
		cv2.imshow("preview2", frame)
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
			break   
		
		#if(len(images)>=10):
		    
		    
    #               if result !=  "Standing":                    
    #                     speak(IP,PORT,result)
      #                  print "Sleeping"
      #                   timeThread.sleep(3)
															  
		    #images = []
    #            
    #                key = cv2.waitKey(20)
    #                if key == 27: # exit on ESC
    #                    break      
		
		  # Create a PIL Image from our pixel array.

		
		
		
    #                print "Aqui!"
    #                cv2.namedWindow("preview")
    #                vc = cv2.VideoCapture(0)
    #                
    #                if vc.isOpened(): # try to get the first frame
    #                    rval, frame = vc.read()
    #                else:
    #                    rval = False
    #                            
    #                while rval:                            
    #                    print "aqui"
    #                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #                    #images.append(frame)
    #                    print "aqui"                                    
    #                    cv2.imshow("preview2", frame)
    #                    #image = MCNN.getOutputImage(outputConvLayers,batch_size,motionImage,imageSize,channels,len(conParams)) 
    #                    print "aqui"
    #                    image = numpy.array(frame)
    #                    newx,newy = (100,100) #new size (w,h)
    #                    image = cv2.resize(image,(newx,newy))
    #                    print image.shape
    #                    features = image.view(numpy.ndarray)
    #                    features.shape = -1
    #                    
    #                    result = MCNN.classify(classifier,features,batch_size)
    #                    print result
    #                    if result[0] == 0:
    #                        result = "Ok!"
    #                    elif result[0] == 1:
    #                        result = "Not Ok!"
    #                    elif result[0]==2:
    #                        result = "Point"
    #                    elif result[0]==3:
    #                        result = "Stop"
    #                                            
    #                    cv2.putText(frame, "Class: "+ str(result), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255))
    #                   
    #                    cv2.imshow("preview3", frame)
    #                    
    #                                                                   
    #                                    
    #                    
    #                    rval2, frame = vc.read()
    #                
    #                    key = cv2.waitKey(20)
    #                    if key == 27: # exit on ESC
    #                        break      
	    elif ( request == 0 ):
	      print "Waiting..."
                     
   
#    if(videoClassification):
#        directoryFrom = "/informatik2/wtm/home/barros/Documents/Experiments/cam3D/imagesRaw/"
#        directoryTo = "/informatik2/wtm/home/barros/Documents/Experiments/cam3D/testingClassification/"
#        i = 0
#        trueDataVideoClassification = []
#        predictedDataVideoClassification = []
#        
#        classes = os.listdir(directoryFrom)
#        for c in classes:
#            #sequences = os.listdir(directoryFace+"/"+c)    
#            #for s in sequences:
#                #files = os.listdir(directoryFace+"/"+c+"/"+s)            
#                sequences = os.listdir(directoryFrom+"/"+c+"/")            
#                for s in sequences:
#
#                        
#                    print directoryFrom+"/"+c+"/"+"/"+s+"/images/"
#                                                                    
#                    if not os.path.exists(directoryTo+"/"+str(c)+"/"+s+"/"):            
#                                os.makedirs(directoryTo+"/"+str(c)+"/"+s+"/")      
#                    
#                    video = directoryFrom+"/"+c+"/"+"/"+s+"/colour.avi"                
#                    vidcap = cv2.VideoCapture(video)                    
#                    print "Video:", video
#                    success,frame = vidcap.read()                    
#                        
#                    count = 0;
#                    while success:
#                      success,frame = vidcap.read()
#                      try:
#                          frame = numpy.array(frame)
#                          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                          image = numpy.array(frame)
#                          newx,newy = (imageSize) #new size (w,h)
#                          image = cv2.resize(image,(newx,newy))
#                          #print "Shape:",  image.shape0
#                          features = image.view(numpy.ndarray)
#                          features.shape = -1
#                          
#                          result = MCNN.classify(classifier,features,batch_size)
#                          if(c == "Agreeing"): trueDataVideoClassification.append(0)  
#                          if(c == "Bored"): trueDataVideoClassification.append(1)  
#                          if(c == "Disagreeing"): trueDataVideoClassification.append(2)  
#                          if(c == "Disgusted"): trueDataVideoClassification.append(3)  
#                          if(c == "Excited"): trueDataVideoClassification.append(4)  
#                          if(c == "Happy"): trueDataVideoClassification.append(5)  
#                          if(c == "Interested"): trueDataVideoClassification.append(6)  
#                          if(c == "Neutral"): trueDataVideoClassification.append(7)  
#                          if(c == "Sad"): trueDataVideoClassification.append(8)  
#                          if(c == "Surprised"): trueDataVideoClassification.append(9)  
#                          if(c == "Thinking"): trueDataVideoClassification.append(10)  
#                           
#                          predictedDataVideoClassification.append(result[0])
#                          
#                          
#                          #print trueDataVideoClassification
#                          #print predictedDataVideoClassification
#                          
#                          if(result[0] == 0):
#                              result = "Agreeing"                              
#                          if(result[0] == 1):
#                              result = "Bored"
#                          if(result[0] == 2):
#                              result = "Disagreeing"
#                          if(result[0] == 3):
#                              result = "Disgusted"
#                          if(result[0] == 4):
#                              result = "Excited"                              
#                          if(result[0] == 5):
#                              result = "Happy"
#                          if(result[0] == 6):
#                              result = "Interested"
#                          if(result[0] == 7):
#                              result = "Neutral"
#                          if(result[0] == 8):
#                              result = "Sad"                              
#                          if(result[0] == 9):
#                              result = "Surprised"
#                          if(result[0] == 10):
#                              result = "Thinking"
#                              
#                          
#                          cv2.putText(frame, str(result), (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,0), 1)                    
#                          
#                          
#                          cv2.imwrite(directoryTo+"/"+str(c)+"/"+s+"/"+"/"+"frame%d.jpg" % count, frame)     # save frame as JPEG file
#                          if cv2.waitKey(10) == 27:                     # exit if Escape is hit
#                              break
#                          count += 1
#                      except Exception as e:
#                        print  "Error:", e         
#                      
#        log.startNewStep("Metrics for Video Classification")
#        MCNN.getClassificationReport(trueDataVideoClassification,predictedDataVideoClassification,metricsFile,metricsDirectory,experimentName,i,log)  
#        
#               
#        precisionMicroVideo = MCNN.getPrecision(trueDataVideoClassification,predictedDataVideoClassification,"micro")
#        recallMicroVideo = MCNN.getRecall(trueDataVideoClassification,predictedDataVideoClassification,"micro")
#        fScoreMicroVideo = MCNN.getFScore(trueDataVideoClassification,predictedDataVideoClassification,"micro")
#                
#        precisionPerClassVideo = MCNN.getPrecision(trueDataVideoClassification,predictedDataVideoClassification,None)
#        recallsPerClassVideo  = MCNN.getRecall(trueDataVideoClassification,predictedDataVideoClassification,None)
#        fScoresPerClassVideo = MCNN.getFScore(trueDataVideoClassification,predictedDataVideoClassification,None)
#        
#        log.startNewStep("Precision")
#        log.printMessage("Precision Micro:" + str(precisionMicroVideo))        
#     
#        log.printMessage("Precision Per class:")    
#        i = 1
#        for c in precisionPerClassVideo:
#           log.printMessage("      Class "+str(i)+": "+str(c))
#           i = i+1    
#             
#        log.startNewStep("Recall")
#        log.printMessage("Recall Micro:" + str(recallMicroVideo))
#             
#        log.printMessage("Recall Per class:")    
#        i = 1
#        for c in recallsPerClassVideo:
#           log.printMessage("      Class "+str(i)+": "+str(c))
#           i = i+1    
#                          
#        log.startNewStep("F1 Score")
#        log.printMessage("F1 Score Micro:" + str(fScoreMicroVideo))              
#     
#        log.printMessage("F1 Score Per class:")    
#        i = 1
#        for c in fScoresPerClassVideo:
#           log.printMessage("      Class "+str(i)+": "+str(c))
#           i = i+1                  
                
                    #call("ffmpeg -framerate 1/5 -i "+directoryTo+"/"+str(c)+"/"+s+"/"+"frame%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+directoryTo+"/"+str(c)+"colour_label.mp4")

    if(videoClassification):
        directoryFrom = "/informatik2/wtm/home/barros/Documents/Experiments/FABO/BodyLabeled/"
        directoryTo = "/informatik2/wtm/home/barros/Documents/Experiments/FABO/BodyClassified/"
        i = 0
        trueDataVideoClassification = []
        predictedDataVideoClassification = []
        
        classes = os.listdir(directoryFrom)
        for c in classes:
            #sequences = os.listdir(directoryFace+"/"+c)    
            #for s in sequences:
                #files = os.listdir(directoryFace+"/"+c+"/"+s)            
                sequencesFiles = os.listdir(directoryFrom+"/"+c+"/")            
                for s in sequencesFiles:

                        
                    print directoryFrom+"/"+c+"/"+"/"+s+"/images/"
                                                                    
                    if not os.path.exists(directoryTo+"/"+str(c)+"/"+s+"/"):            
                                os.makedirs(directoryTo+"/"+str(c)+"/"+s+"/")      
                    
                    video = directoryFrom+"/"+c+"/"+"/"+s               
                    vidcap = cv2.VideoCapture(video)                    
                    print "Video:", video
                    success,frame = vidcap.read()                    
                        
                    count = 0;
                    while success:
                      success,frame = vidcap.read()
                      try:
                          frame = numpy.array(frame)
                          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                          image = numpy.array(frame)
                          newx,newy = (imageSize) #new size (w,h)
                          image = cv2.resize(image,(newx,newy))
                          #print "Shape:",  image.shape0
                          features = image.view(numpy.ndarray)
                          features.shape = -1
                          
                          result = MCNN.classify(classifier,features,batch_size)
                          if(c == "ANGER"): trueDataVideoClassification.append(0)  
                          if(c == "AXIETY"): trueDataVideoClassification.append(1)  
                          if(c == "BOREDOM"): trueDataVideoClassification.append(2)  
                          if(c == "DISGUST"): trueDataVideoClassification.append(3)  
                          if(c == "FEAR"): trueDataVideoClassification.append(4)  
                          if(c == "HAPPINESS"): trueDataVideoClassification.append(5)  
                          if(c == "NEUTRAL"): trueDataVideoClassification.append(6)                            
                          if(c == "NGT SRP"): trueDataVideoClassification.append(7)  
                          if(c == "PST SRP"): trueDataVideoClassification.append(8)  
                          if(c == "PUZZLEMENT"): trueDataVideoClassification.append(9)  
                          if(c == "SADNESS"): trueDataVideoClassification.append(10)  
                          if(c == "UNCERTAINTY"): trueDataVideoClassification.append(11)  
                           
                          predictedDataVideoClassification.append(result[0])
                          
                          
                          #print trueDataVideoClassification
                          #print predictedDataVideoClassification
                          
                          if(result[0] == 0):
                              result = "ANGER"                              
                          if(result[0] == 1):
                              result = "AXIETY"
                          if(result[0] == 2):
                              result = "BOREDOM"
                          if(result[0] == 3):
                              result = "DISGUST"
                          if(result[0] == 4):
                              result = "FEAR"                              
                          if(result[0] == 5):
                              result = "HAPPINESS"
                          if(result[0] == 6):
                              result = "NEUTRAL"
                          if(result[0] == 7):
                              result = "NGT SRP"                              
                          if(result[0] == 8):
                              result = "PST SRP"    
                          if(result[0] == 9):
                              result = "PUZZLEMENT"   
                          if(result[0] == 10):
                              result = "SADNESS"    
                          if(result[0] == 11):
                              result = "UNCERTAINTY"    
                          
                          cv2.putText(frame, str(result), (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,0), 1)                    
                          
                          
                          cv2.imwrite(directoryTo+"/"+str(c)+"/"+s+"/"+"/"+"frame%d.jpg" % count, frame)     # save frame as JPEG file
                          if cv2.waitKey(10) == 27:                     # exit if Escape is hit
                              break
                          count += 1
                      except Exception as e:
                        print  "Error:", e         
                      
        log.startNewStep("Metrics for Video Classification")
        MCNN.getClassificationReport(trueDataVideoClassification,predictedDataVideoClassification,metricsFile,metricsDirectory,experimentName,i,log)  
        
               
        precisionMicroVideo = MCNN.getPrecision(trueDataVideoClassification,predictedDataVideoClassification,"micro")
        recallMicroVideo = MCNN.getRecall(trueDataVideoClassification,predictedDataVideoClassification,"micro")
        fScoreMicroVideo = MCNN.getFScore(trueDataVideoClassification,predictedDataVideoClassification,"micro")
                
        precisionPerClassVideo = MCNN.getPrecision(trueDataVideoClassification,predictedDataVideoClassification,None)
        recallsPerClassVideo  = MCNN.getRecall(trueDataVideoClassification,predictedDataVideoClassification,None)
        fScoresPerClassVideo = MCNN.getFScore(trueDataVideoClassification,predictedDataVideoClassification,None)
        
        log.startNewStep("Precision")
        log.printMessage("Precision Micro:" + str(precisionMicroVideo))        
     
        log.printMessage("Precision Per class:")    
        i = 1
        for c in precisionPerClassVideo:
           log.printMessage("      Class "+str(i)+": "+str(c))
           i = i+1    
             
        log.startNewStep("Recall")
        log.printMessage("Recall Micro:" + str(recallMicroVideo))
             
        log.printMessage("Recall Per class:")    
        i = 1
        for c in recallsPerClassVideo:
           log.printMessage("      Class "+str(i)+": "+str(c))
           i = i+1    
                          
        log.startNewStep("F1 Score")
        log.printMessage("F1 Score Micro:" + str(fScoreMicroVideo))              
     
        log.printMessage("F1 Score Per class:")    
        i = 1
        for c in fScoresPerClassVideo:
           log.printMessage("      Class "+str(i)+": "+str(c))
           i = i+1                
                      
#    if(liveClassification):
#
#                #cv2.namedWindow("preview")
#                vc = cv2.VideoCapture(0)
#                vc.set(11,3)
#           
#                
#                if vc.isOpened(): # try to get the first frame
#                    rval, f = vc.read()
#                else:
#                    rval = False
#                
#                images = []
#                putMotion = False
#                pastMotion = f
#                result = "Stand"
#                #engine = pyttsx.init()
#                #engine.setProperty('rate', 50)   
#                espeak.set_voice("english")
#                espeak.set_parameter(1,100)
#                espeak.set_parameter(3,100)
#                recognized = False                
#                
#                while rval:       
#                    frame = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
#                    images.append(frame)                     
#                    
#                   # cv2.putText(f, "Class: "+ str(len(images)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))  
#                    #cv2.imshow("preview", f)  
#                    
#                    
#                    newx,newy = 640,480 #new size (w,h)
#                    f = cv2.resize(f,(newx,newy))
#                    
#                    blank_image = Image.new("RGB", (1800, 1024))
#                    blank_image.paste(Image.fromarray(f), (1024,0))
#                    
#                    if putMotion == True:
#                          blank_image.paste(Image.fromarray(pastMotion), (0,100))
#                                    
#                
#                    if(len(images)==60):
#                        motionImage = ImageProcessingUtil.doConvShadow(images) 
#                        cv2.imwrite("//informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/save.png",motionImage)
#                        motionImage = cv2.imread("//informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/save.png")
#                        motionImage = cv2.cvtColor(motionImage, cv2.COLOR_BGR2GRAY)
#                       # cv2.imshow("preview2", motionImage)
#                        #image = MCNN.getOutputImage(outputConvLayers,batch_size,motionImage,imageSize,channels,len(conParams)) 
#                        
#                        image = numpy.array(motionImage)
#                        newx,newy = (100,100) #new size (w,h)
#                        image = cv2.resize(image,(newx,newy))
#                        #print "Shape:",  image.shape0
#                        features = image.view(numpy.ndarray)
#                        features.shape = -1
#                        
#                        result = MCNN.classify(classifier,features,batch_size)
#                        print result
#                        if result[0] == 0:
#                            result = "Circle"
#                        elif result[0] == 1:
#                            result = "Point Left"
#                        elif result[0]==2:
#                            result = "Point Right"
#                        elif result[0]==3:
#                            result = "Stand"
#                        elif result[0]==4:
#                            result = "Stop"
#                        elif result[0]==5:
#                            result = "Turn"                            
#                        print result                
#                        putMotion = True        
#                        newx,newy = 1024,768 #new size (w,h)
#                        pastMotion = cv2.resize(motionImage,(newx,newy))    
#                        recognized = True                        
#                       
#                       # cv2.imshow("preview3", motionImage)
#                        images = []          
#                          
#                                  
#                                    
#                    
#                    blank_image = numpy.array(blank_image)
#                    cv2.putText(blank_image, str(len(images)), (20,900), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))  
#                    
#                    if not result == "Stand":
#                        cv2.putText(blank_image, str(result), (20,1000), cv2.FONT_HERSHEY_SIMPLEX , 3, (255,0,0), 4)                    
#                        
#                    if recognized:
#                        recognized = False
#                        if not result == "Stand":                                                                          
#                            
#                            espeak.synth(result)
#                            #engine.say(result)
#                            
#                            
#                        
#                            
#                    cv2.imshow("preview1", blank_image)                    
#                   # cv2.imshow("preview2", blank_image)   
#                    rval2, f = vc.read()
#                
#                    key = cv2.waitKey(20)
#                    if key == 27: # exit on ESC
#                        break                
   
                #engine.runAndWait()
    #extractBodyShapesConvolutionalFeatures(convolutionOutputLayer, batch_size,directoryImages, directoryConvFeatures)
 


def experiments():
    
   
    #Directory where the trained filters are located.
    CAEDirectory = "/informatik/isr/wtm/home/barros/Desktop/ijcnnExperiments/jaffe_cae_GPU0/20/params"
    

    #Parameters about network visualization options
    hintonDiagrams = False
    createOutputImages = False
    createConvFeatures = False
    
    #Parameters about liveClassification
    liveClassification = True 
    nimbroClassification = False
    videoClassification = False
    
    #Parameters about experiment Data
    leaveOneOutJaffeExperiment = False
    crossValidation = (False,10)
        
        
    #Parameters about Network Topology  
    inputType = DataUtil.INPUT_TYPE["Common"]   
    imageSize = (100,100)
    timeStep = 2
    
    #----Channels topology
    "Channel Type, Inhibition"
    channel0 = (DataUtil.CHANNEL_TYPE["Common"],False, CAEDirectory)
    channel1 = (DataUtil.CHANNEL_TYPE["SobelX"],False, CAEDirectory)    
    channel2 = (DataUtil.CHANNEL_TYPE["SobelY"],False, CAEDirectory)
    #channel3 = (DataUtil.CHANNEL_TYPE["CAE"],False, CAEDirectory)
    
    channelsTopology = []
    channelsTopology.append(channel0)
    channelsTopology.append(channel1)
    channelsTopology.append(channel2)
    #channelsTopology.append(channel3)
    
    
    #----Layers topology
    param1 = [10,3,3,2,2]    
    param2 = [20,5,5,2,2]
    param2 = [5,5,5,4,4]
    
    conParams = []
    conParams.append(param1)
    conParams.append(param2)      
    
    #----Classifier topology
    
    unitsInHiddenLayer = 100
    outputUnits = 6    
    
    
    #Parameters about Network training
    visualizeTrain = False
    train = False
    batchSize = 10
    learningRate = 0.1
    epochs = 100
    dataSetTrainDivision = (60,20,20)        
    
    params = []
    params.append(conParams)
    params.append(unitsInHiddenLayer)
    params.append(outputUnits)
    params.append(learningRate)
    params.append(batchSize)
    params.append(imageSize)
    params.append(epochs)
    params.append(dataSetTrainDivision)
    

    #Parameters about Experiment settings
    repetitions = 1
    baseDirectory = "/informatik2/wtm/home/barros/Documents/Experiments/LenaRecording/"    
    
    experimentName = "Experiment1"
    experimentName += "_ImageSize:"+ str(imageSize)
    experimentName += "_Input:"+ str(inputType)
    experimentName += "_NetworkTopology: "
    for c in channelsTopology:
        experimentName += str(c[0])+"_Inhibition:"+str(c[1])+"__"
        
    createFeaturesFile = True  
    featuresFileName = "Features_"+str(imageSize)+"_"+str(inputType)+"_.txt"


    #featuresFileName = "256x256_all_STDNormalization.txt"
    
    
    MCNN3ChannelsExperiment(experimentName,baseDirectory,repetitions, featuresFileName, train, inputType, channelsTopology, params, timeStep, hintonDiagrams, createOutputImages,liveClassification, createConvFeatures, nimbroClassification, createFeaturesFile, videoClassification, crossValidation[0] ,crossValidation[1], leaveOneOutJaffeExperiment, visualizeTrain)

def experimentsNAO():
    
   
    #Directory where the trained filters are located.
    CAEDirectory = "/informatik/isr/wtm/home/borghetti/Desktop/TESTECNN/"
    

    #Parameters about network visualization options
    hintonDiagrams = False
    createOutputImages = False
    createConvFeatures = False
    
    #Parameters about liveClassification
    liveClassification = True
    nimbroClassification = False
    videoClassification = False
    
    #Parameters about experiment Data
    leaveOneOutJaffeExperiment = False
    crossValidation = (False,10)
        
        
    #Parameters about Network Topology  
    inputType = DataUtil.INPUT_TYPE["Color"]   
    imageSize = (100,100) # change this information here
    timeStep = 10
    
    #----Channels topology
    "Channel Type, Inhibition"
    channel0 = (DataUtil.CHANNEL_TYPE["Common"],False, CAEDirectory)
    channel1 = (DataUtil.CHANNEL_TYPE["SobelX"],False, CAEDirectory)    
    channel2 = (DataUtil.CHANNEL_TYPE["SobelY"],False, CAEDirectory)
    #channel3 = (DataUtil.CHANNEL_TYPE["CAE"],False, CAEDirectory)
    
    channelsTopology = []
    channelsTopology.append(channel0)
    channelsTopology.append(channel1)
    channelsTopology.append(channel2)
    #channelsTopology.append(channel3)
    
    
    #----Layers topology
    param1 = [10,3,3,2,2]    
    param2 = [20,3,3,2,2]
    param2 = [5,5,5,4,4]
    
    conParams = []
    conParams.append(param1)
    conParams.append(param2)      
    
    #----Classifier topology    
    unitsInHiddenLayer = 100
    outputUnits = 2    
    
    
    #Parameters about Network training
    visualizeTrain = False
    train = False
    batchSize = 20
    learningRate = 0.1
    epochs = 100
    dataSetTrainDivision = (60,20,20)        
    
    params = []
    params.append(conParams)
    params.append(unitsInHiddenLayer)
    params.append(outputUnits)
    params.append(learningRate)
    params.append(batchSize)
    params.append(imageSize)
    params.append(epochs)
    params.append(dataSetTrainDivision)
    

    #Parameters about Experiment settings
    repetitions = 1
    baseDirectory = "/informatik/isr/wtm/home/borghetti/Desktop/TESTECNN/"
    testImageDirectory = "/informatik/isr/wtm/home/borghetti/Desktop/TESTECNN/testImage/"    
    
    experimentName = "Sequences_6_Gestures"
    experimentName += "_ImageSize:"+ str(imageSize)
    experimentName += "_Input:"+ str(inputType)
    experimentName += "_NetworkTopology: "
    for c in channelsTopology:
        experimentName += str(c[0])+"_Inhibition:"+str(c[1])+"__"
    
        
    createFeaturesFile = True  
    featuresFileName = "Features_"+str(imageSize)+"_"+str(inputType)+"_.txt"


    #featuresFileName = "256x256_all_STDNormalization.txt"
    
    
    MCNN3ChannelsExperiment(experimentName,baseDirectory,testImageDirectory, repetitions, featuresFileName, train, inputType, channelsTopology, params, timeStep, hintonDiagrams, createOutputImages,liveClassification, createConvFeatures, nimbroClassification, createFeaturesFile, videoClassification, crossValidation[0] ,crossValidation[1], leaveOneOutJaffeExperiment, visualizeTrain)



request = 1
rospy.init_node('CNN', anonymous=True)
experimentsNAO()
rospy.spin()
