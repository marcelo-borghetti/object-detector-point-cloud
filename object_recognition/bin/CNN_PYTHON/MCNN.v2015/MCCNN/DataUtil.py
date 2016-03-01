# -*- coding: utf-8 -*-

import numpy
import cPickle
import os
import theano
import theano.tensor as T
import random
import pylab
import itertools
import cv2


INPUT_TYPE =  {"Common": "common", "Color":"color", "3D":"3d"}
CHANNEL_TYPE =  {"Common": "common", "SobelX":"sobelX", "SobelY":"sobelY", "CAE":"cae"}


 
#Method that reads the file containing the features.
# The feature file must be formated as follows:
    #<label>,<featuress eparated with a coma>
    #<label>,<featuress eparated with a coma>
    #<label>,<featuress eparated with a coma>
    #etc...
#This method separate the data in two lists:
#one with the label and one with the features.
#These two lists are passed to the separateSets method.    
def readFeatureFile(featuresDirectory, randomized,  percentTrain, percentValid, useColor):
        
    print "USe color:", useColor    
    directory = featuresDirectory        
    f = open(directory, 'r')        
    inputs = []
    outputs = []            
    for line in f:
        li = line.split(",")            
        outputs.append(int(li[0]))
        li.remove(li[0])
        features = [] 
        colorNumber = 0
        color = []
        for i in li:   
            if not i == 0:
                try:
                    if useColor:
                        color.append(i)
                        colorNumber = colorNumber +1                        
                        if(colorNumber==3):                        
                            features.append(color)                            
                            colorNumber = 0                        
                            color = []
                    else:
                        features.append(i)
                except:
                     pass
        
        features = numpy.swapaxes(features, 0,1)        
        inputs.append(features)    
        
    f.close()    



 
    if randomized:
        return randomizeSet(inputs,outputs)
    else:
        return separateSets(inputs,outputs,percentTrain,percentValid)
    #return (numpy.array(inputs),numpy.array(outputs))



def readFeatureFileColor(featuresDirectory, randomized,  percentTrain, percentValid, color):
        
    directory = featuresDirectory        
    f = open(directory, 'r')        
    inputs = []
    outputs = []            
    for line in f:
        li = line.split(",")            
        outputs.append(int(li[0]))
        li.remove(li[0])
        features = []       
        color = []
        colorNumber = 0
        for i in li:   
            if not i == 0:
                try:          
                    color.append(i)
                    colorNumber = colorNumber +1
                    if(colorNumber==3):
                        features.append(colorNumber)
                        colorNumber = 0
                except:
                     pass
        inputs.append(features)    
        
    f.close()    
    
    
 
    
    if randomized:
        return randomizeSet(inputs,outputs)
    else:
        return separateSets(inputs,outputs,percentTrain,percentValid)
    #return (numpy.array(inputs),numpy.array(outputs))

def randomizeSet(inputs, outputs):
    
    
    positions = []
    for p in range(len(inputs)):
        positions.append(p)
        
    random.shuffle(positions)
    
    
    
    newInputs = []
    newOutputs = []
    for p in positions:
        newInputs.append(inputs[p])
        newOutputs.append(outputs[p])
        
    return (newInputs,newOutputs)


#This method separates the set in trhee sub-sets with 
#shufled and ramdonly chose values. Each subset has a 
# pre-defined amount of values, passed as a parameter.
# For each subset a list of positions is created
#and  then filled with sorted values from the original set.
# After the amount of values in each list is reached, the values
# of each position are copied to a final list.
def separateSets(inputSet,outputSet, pTrain, pValid):
    

    
   
   
    positionsSetTrain = []
    positionsSetValidate = []
    positionsSetTest = []
    
    patterns = []    
        
    for o in outputSet:
        if not o in patterns:
            patterns.append(o)
     
    for c in patterns:
       outputsInThisClass = []
       for i in range(len(outputSet)):           
           if( outputSet[i] == c):
               outputsInThisClass.append(i)
               
       positionsTrainSet = []
       positionsValidateSet = []
       positionsTestSet = []
       percentTest = len(outputsInThisClass)* ( 100-pTrain-pValid)/100
       percentTrain = len(outputsInThisClass) * pTrain/100       
       percentValid = len(outputsInThisClass) * pValid/100       
             
           
       
       while len(outputsInThisClass) >0:
           
           #print "Class:", c
           #print "Train:", positionsTrainSet
           #print "Valid:", positionsValidateSet
           #print "Test:", positionsTestSet, " percent test:", percentTest, " lenPositionsTestSet", len(positionsTestSet), " outputInThisClass: ", len(outputsInThisClass)
           if(len(positionsTrainSet) <= percentTrain):    
               rnd = random.randint(0,len(outputsInThisClass)-1)
               positionsTrainSet.append(outputsInThisClass[rnd])
               outputsInThisClass.remove(outputsInThisClass[rnd])
           
           if(len(positionsValidateSet) <= percentValid):           
               rnd = random.randint(0,len(outputsInThisClass)-1)
               positionsValidateSet.append(outputsInThisClass[rnd])
               outputsInThisClass.remove(outputsInThisClass[rnd])
               
           if(len(positionsTestSet) <= percentTest):
               rnd = random.randint(0,len(outputsInThisClass)-1)
               positionsTestSet.append(outputsInThisClass[rnd])
               outputsInThisClass.remove(outputsInThisClass[rnd])           
                  
      
       for i in positionsTrainSet:
           positionsSetTrain.append(i)
           
           
       for i in positionsValidateSet:
           positionsSetValidate.append(i)
           
       for i in positionsTestSet:
           positionsSetTest.append(i)
           
    inputSetTrain = []
    outputSetTrain = []

    inputSetValidate = []
    outputSetValidate = []

    inputSetTest = []
    outputSetTest = []   
    
    
    random.shuffle(positionsSetTrain)
    random.shuffle(positionsSetValidate)
    random.shuffle(positionsSetTest)
    
    for i in positionsSetTrain:
        inputSetTrain.append(inputSet[i])
        outputSetTrain.append(outputSet[i])
    for i in positionsSetValidate:
        inputSetValidate.append(inputSet[i])
        outputSetValidate.append(outputSet[i])    
    for i in positionsSetTest:
        inputSetTest.append(inputSet[i])
        outputSetTest.append(outputSet[i])  
                
    
    return ( (numpy.array(inputSetTrain),numpy.array(outputSetTrain)), (numpy.array(inputSetValidate),numpy.array(outputSetValidate)), (numpy.array(inputSetTest),numpy.array(outputSetTest)))


#Method that saves the state of a MCNN
def saveState(params,directory):    
        
    f = file(directory, 'wb')
    for obj in params:                    
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

#Method that load a state of a MCNN
def loadState(directory,parametersToLoad):
    
    f = file(directory, 'rb')
    loaded_objects = []
    for i in range(parametersToLoad):                
        loaded_objects.append(cPickle.load(f))
    f.close()

    return loaded_objects    

#Method that load a state of a MCNN
def loadStates(directory,layers):
    
    print "Loading ", layers, " state from state in :", directory
    
    f = file(directory, 'rb')
    loaded_objects = []
    for i in range(layers):                
        loaded_objects.append(cPickle.load(f))
    f.close()

    return loaded_objects 


def createTrainingSequence(data,timeStep):
    sequencesX = []    
    for h in range(len(data[0])):
        sequenceX = []
        for i in range(timeStep):
            sequenceX.append(data[0][h])        
        sequencesX.append(sequenceX)        
            
      
    return (sequencesX,data[1])    
            

def createSequence(data, timeStep):
    classes = []
    sequencesX = []
    sequencesY = []
         
    for i in range(len(data[1])):
        if(not data[1][i] in classes):
            classes.append(data[1][i])
            classSequence = []                        
            for j in range(len(data[1])):            
                if(data[1][j] == data[1][i]):                      
                    classSequence.append(data[0][j])                        
            #print "Class: ", data[1][i], " - Elements:", len(classSequence)
            for subset in itertools.combinations(classSequence,timeStep):
                sequencesX.append(subset)                
                sequencesY.append(data[1][i])                
    
    #print "Classes: ", sequencesY
    positions = []
    for i in range(len(sequencesX)):              
        positions.append(i)
    

    random.shuffle(positions)
    
    sequencesXShuffled = []
    sequencesYShuffled = []
    for i in positions:
        sequencesXShuffled.append(sequencesX[i])
        sequencesYShuffled.append(sequencesY[i])
    
    return (sequencesXShuffled,sequencesYShuffled)


def readFeatureFileSequence(featuresDirectory, percentTrain, percentValid, timeStep):
        
    directory = featuresDirectory        
    f = open(directory, 'r')        
    inputs = []
    outputs = []    
    
    sequences = []        
    sequenceNumber = 0
    
    for line in f:
        li = line.split(",")
        output = int(li[0])
        li.remove(li[0])
        features = []
        for i in li:   
            if not i == 0:
                try:
                    features.append(float(i))
                except:
                     pass
        sequences.append(features)
        sequenceNumber = sequenceNumber+1
                            
        if (sequenceNumber % timeStep) == 0 and sequenceNumber != 0:            
            inputs.append(sequences)            
            outputs.append(output)        
            sequences = []
            
    return separateSets(inputs,outputs,percentTrain,percentValid)    

   
   
   

def loadDataCrossValidation(log, featuresDirectory, datasetDivision, dataType):
    
    log.startNewStep("Loading Data "+str(datasetDivision)+"-Cross Validation")

    # Load the dataset
    log.printMessage(("Data Type:", dataType))
    log.printMessage(("Loading from: ", featuresDirectory))
    inputs = []
    outputs = []
    
    if(dataType == "Static"):
        inputSet,outputSet = readFeatureFile(featuresDirectory,True,0,0)
        
        intensEachSet = int(len(inputSet)/datasetDivision)        
        
        
        for i in range(datasetDivision):
            
            
            if(i == datasetDivision-1):
                
                newSetInput = inputSet[i*intensEachSet:]
                newSetOutput = outputSet[i*intensEachSet:]
                
                inputs.append(newSetInput)
                outputs.append(newSetOutput)
            else:           
                
                newSetInput = inputSet[i*intensEachSet:i*intensEachSet+intensEachSet]
                newSetOutput = outputSet[i*intensEachSet:i*intensEachSet+intensEachSet]
                inputs.append(newSetInput)
                outputs.append(newSetOutput)
                
               
    #print "Input:", inputs[0]
    
    log.printMessage(("Numbers of sets: ", datasetDivision))        
    log.printMessage(("Elements in each input set: ", len(inputs[0])))        
    log.printMessage(("--- Each Element in each input set: ", len(inputs[0][0])))                     
    log.printMessage(("Elements in each output set: ", len(outputs[0])))        
    
            
            
    return (inputs,outputs)

#input is an numpy.ndarray of 2 dimensions (a matrix)
#witch row's correspond to an example. target is a
#numpy.ndarray of 1 dimensions (vector)) that have the same length as
#the number of rows in the input. It should give the target
#target to the example with the same index in the input.

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """    
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    
    return shared_x, T.cast(shared_y, 'int32')  
   
#Metod used for loading the features for the network. After obtain the
# three lists, training, validation and testing, it transform the lists 
#in theano shared variables.
def loadData(log, featuresDirectory, percentTrain, percentValid, timeStep, dataType):    

    log.startNewStep("Loading Data")

    # Load the dataset
    log.printMessage(("Data Type:", dataType))
    log.printMessage(("Loading from: ", featuresDirectory))
    
#    if(dataType == "Static"):
#        train_set, valid_set, test_set = readFeatureFile(featuresDirectory,False, percentTrain, percentValid, color)
#        log.printMessage(("Elements in train Data: ", len(train_set[0])))        
#        log.printMessage(("--- Each Element in train Data: ", len(train_set[0][0])))
#        log.printMessage(("Elements in validation Data: ", len(valid_set[0])))
#        log.printMessage(("--- Each Element in validation Data: ", len(valid_set[0][0])))
#        log.printMessage(("Elements in test Data: ", len(test_set[0])))
#        log.printMessage(("--- Each Element in test Data: ", len(test_set[0][0])))    
#        
#        print train_set[0].shape
                       
#    elif(dataType == "StaticSequence"):
#        train_set, valid_set, test_set = readFeatureFile(featuresDirectory,False, percentTrain, percentValid, color)
#        train_set = createSequence(train_set,timeStep)
#        #train_set = createTrainingSequence(train_set,timeStep)
#        valid_set = createTrainingSequence(valid_set,timeStep)
#        test_set = createTrainingSequence(test_set,timeStep)   
#        log.printMessage(("Elements in train Data: ", len(train_set[0])))
#        log.printMessage(("--- Each Sequence Element in train Data: ", len(train_set[0][0])))
#        log.printMessage(("------ Each Element in a sequence in train Data: ", len(train_set[0][0][0])))
#        log.printMessage(("Elements in validation Data: ", len(valid_set[0])))
#        log.printMessage(("--- Each Sequence Element in validation Data: ", len(valid_set[0][0])))
#        log.printMessage(("------ Each Element in a sequence in train Data: ", len(valid_set[0][0][0])))
#        log.printMessage(("Elements in test Data: ", len(test_set[0])))
#        log.printMessage(("--- Each Sequence Element in test Data: ", len(test_set[0][0])))
#        log.printMessage(("------ Each Element in a sequence in test Data: ", len(test_set[0][0][0])))
                       
    if(dataType == INPUT_TYPE["3D"]):
        print "TimeStep:", timeStep
        train_set, valid_set, test_set = readFeatureFileSequence(featuresDirectory,percentTrain, percentValid, timeStep)
        log.printMessage(("Elements in train Data: ", len(train_set[0])))
        log.printMessage(("--- Each Sequence Element in train Data: ", len(train_set[0][0])))
        log.printMessage(("------ Each Element in a sequence in train Data: ", len(train_set[0][0][0])))
        log.printMessage(("Elements in validation Data: ", len(valid_set[0])))
        log.printMessage(("--- Each Sequence Element in validation Data: ", len(valid_set[0][0])))
        log.printMessage(("------ Each Element in a sequence in train Data: ", len(valid_set[0][0][0])))
        log.printMessage(("Elements in test Data: ", len(test_set[0])))
        log.printMessage(("--- Each Sequence Element in test Data: ", len(test_set[0][0])))
        log.printMessage(("------ Each Element in a sequence in test Data: ", len(test_set[0][0][0])))
    else:            
        train_set, valid_set, test_set = readFeatureFile(featuresDirectory,False, percentTrain, percentValid, dataType == INPUT_TYPE["Color"])
        log.printMessage(("Elements in train Data: ", len(train_set[0])))        
        log.printMessage(("--- Each Element in train Data: ", len(train_set[0][0])))
        log.printMessage(("Elements in validation Data: ", len(valid_set[0])))
        log.printMessage(("--- Each Element in validation Data: ", len(valid_set[0][0])))
        log.printMessage(("Elements in test Data: ", len(test_set[0])))
        log.printMessage(("--- Each Element in test Data: ", len(test_set[0][0])))   
           

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
   
    return rval
    

def createFolder(directory):
    if not os.path.exists(directory): os.makedirs(directory)


def _blob(x,y,area,colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = numpy.sqrt(area) / 2
    xcorners = numpy.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = numpy.array([y - hs, y - hs, y + hs, y + hs])
    pylab.fill(xcorners, ycorners, colour, edgecolor=colour)

def saveHintonDiagram(W, directory):
    maxWeight = None
    #print "Weight: ", W
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    if pylab.isinteractive():
        pylab.ioff()
    pylab.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**numpy.ceil(numpy.log(numpy.max(numpy.abs(W)))/numpy.log(2))

    pylab.fill(numpy.array([0,width,width,0]),numpy.array([0,0,height,height]),'gray')
    pylab.axis('off')
    pylab.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    if reenable:
        pylab.ion()
    #pylab.show()
    pylab.savefig(directory)
    

def createFeatureVector(imagesList, imageSize):
    
     
    inputs = []
    outputs = []
    for i in imagesList:
        
        img = cv2.imread(i[0])
        img = numpy.array(img) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img - img.mean()
        img = img / numpy.std(img)
        

        #img = numpy.array(img)
        newx,newy = imageSize #new size (w,h)
        img = cv2.resize(img,(newx,newy))
        cv2.imwrite("//informatik2/wtm/home/barros/Documents/Experiments/JaffeDataset/test.jpg",img)
         
        outputs.append(int(i[1])) 
        
        imageFeatures = []
        for x in img:
           for y in x:
                imageFeatures.append(y)                                
        
        inputs.append(imageFeatures)
    
    return (inputs,outputs)


def createFeatureFileJaffeExperiment(directory, log, imageSize):


    log.startNewStep("Loading Data Leave one out experiment ")

    # Load the dataset    
    log.printMessage(("Loading from: ", directory))
    
    testingSet = []
    trainingSet = []
    
    classes = os.listdir(directory)
    classIndex = 0
    for c in classes:
        persons = []
        images = os.listdir(directory+"/"+c+"/")
        for i in images:
            name = i[0:2]
            if not name in persons:
                persons.append(name)

        imagesPerPerson = []                
        for i in range(len(persons)):
            imagesPerPerson.append([])
                
        for i in images:
            personIndex = 0
            for p in persons:

                if p in i:                                        
                    imagesPerPerson[personIndex].append((directory+"/"+str(c)+"/"+i,classIndex))

                    break
                personIndex = personIndex+1        
        
        for iP in range(len(imagesPerPerson)):            
             random.shuffle(imagesPerPerson[iP])             
             testingSet.append(imagesPerPerson[iP][+len(imagesPerPerson[iP])-1])
             imagesPerPerson[iP].pop()
             trainingSet.extend(imagesPerPerson[iP])
        
        classIndex = classIndex+1
        
    random.shuffle(trainingSet)    
    random.shuffle(testingSet)
    
    train_set = createFeatureVector(trainingSet,imageSize)
    test_set = createFeatureVector(testingSet,imageSize)

    
    print train_set[0][0]
    log.printMessage(("Elements in train Data: ", len(train_set[0])))        
    log.printMessage(("--- Each Element in train Data: ", len(train_set[0][0])))    
    log.printMessage(("Elements in test Data: ", len(test_set[0])))
    log.printMessage(("--- Each Element in test Data: ", len(test_set[0][0])))
        
    test_set_x, test_set_y = shared_dataset(test_set)    
    train_set_x, train_set_y = shared_dataset(train_set)

    return [(train_set_x, train_set_y), (train_set_x, train_set_y),
            (test_set_x, test_set_y)]        



def writeSingleFile(features, location, color):
       # print features        
        f = open(location,"w")
        for featureSet in features:         
           # print "Cada FeatureSet tem:", len(featureSet)
            featureNumber = 0
            
            for feature in featureSet:               
#                print "Cada feature tem:", len(feature)  
                if featureNumber == 0:  
                    f.write(str(feature))
                    featureNumber = featureNumber+1
                    f.write(",")
                    
                else:
                    if color:                    
                        for c in feature:
                            f.write(str(c))
                            
                            featureNumber = featureNumber+1
                            if(featureNumber < len(featureSet)*3):
                                f.write(",")   
                            
                    else:        
                        f.write(str(feature))
                        
                        featureNumber = featureNumber+1
                        if(featureNumber < len(featureSet)):
                            f.write(",")    
 
                
            f.write("\n")
        f.close()