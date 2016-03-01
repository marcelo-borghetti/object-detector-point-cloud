# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import theano.tensor.nnet.conv3d2d

import numpy
import Image

import DataUtil
import ImageProcessingUtil
import cv2

from sklearn.metrics import classification_report, confusion_matrix, precision_score,recall_score,f1_score,accuracy_score
import pylab
import matplotlib.pyplot as plt
import matplotlib
import os

#import datetime


universalFeatures = "/informatik/isr/wtm/home/barros/Desktop/ijcnnExperiments/params.save"


class FirstLayer(object):
    def __init__(self, input):
        
       self.input = input
       self.output = input       
 

class LayerSobel(object):
    def __init__(self, rng,direction, input, filter_shape, image_shape):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
 
 
        #image_shape=(batch_size, 1, 28, 28),
        #filter_shape=(nkerns[0], 1, 5, 5)
        #print "Image Shape:", image_shape
       # print "Filter Shape:", filter_shape
        print "Filter shape:", filter_shape
        print "Image shape:", image_shape
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
                                
        Ww = numpy.asarray(
             rng.uniform(low=-0, high=0, size=filter_shape),
             dtype=theano.config.floatX)
             
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        #print Ww
        
        self.b = theano.shared(value=b_values, borrow=True)
        
        if len(image_shape) == 4:
             
            
            if(direction == "x"):
                for i in range(len(Ww)):
                    for u in range(len(Ww[i])):
                        Ww[i][u][0] = [1,0,-1]
                        Ww[i][u][1] = [2,0,-2]
                        Ww[i][u][2] = [1,0,-1]
            else:
                for i in range(len(Ww)):
                    for u in range(len(Ww[i])):
                        Ww[i][u][0] = [1,2,1]
                        Ww[i][u][1] = [0,0,0]
                        Ww[i][u][2] = [-1,-2,-1]   
                        
            self.W = theano.shared(Ww,
                              borrow=True)   
                              
            conv_out = conv.conv2d(input=input, filters= self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        else:
            
            if(direction == "x"):
                for i in range(len(Ww)):
                    for u in range(len(Ww[i])):
                        for k in range(len(Ww[i][u])):                        
                            Ww[i][u][k][0] = [0,1,0]
                            Ww[i][u][k][1] = [1,-4,1]
                            Ww[i][u][k][2] = [0,1,0]
            else:
                for i in range(len(Ww)):
                    for u in range(len(Ww[i])):
                        for k in range(len(Ww[i][u])):
                            Ww[i][u][k][0] = [-2,-1,0]
                            Ww[i][u][k][1] = [-1,1,1]
                            Ww[i][u][k][2] = [0,1,2]  
            self.W = theano.shared(Ww,
                          borrow=True) 
                          
            conv_out = T.nnet.conv3d2d.conv3d(signals=input, filters= self.W,filters_shape=filter_shape, signals_shape=image_shape,border_mode='valid')    
                    
        """
        if(direction == "x"):
                    for i in range(len(Ww)):
                        for u in range(len(Ww[i])):
                            for k in range(len(Ww[i][u])):                        
                                Ww[i][u][0] = [0,0,0]
                                Ww[i][u][1] = [-1,1,0]
                                Ww[i][u][2] = [0,0,0]
        else:
                    for i in range(len(Ww)):
                        for u in range(len(Ww[i])):
                            for k in range(len(Ww[i][u])):
                                Ww[i][u][0] = [0,1,0]
                                Ww[i][u][1] = [1,1,1]
                                Ww[i][u][2] = [0,1,2]                              
        """ 
     
        # the bias is a 1D tensor -- one bias per output feature map
        
        #print "Bias:", b_values
        

        # convolve input feature maps with filters
        
       
        
        
        # downsample each feature map individually, using maxpooling
      ##  pooled_out = downsample.max_pool_2d(input=conv_out,
#                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
      
        # store parameters of this layer
        self.params = [self.W, self.b]


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self,useInhibition, rng, universalFeatures, layerOrder, loadFrom , parametersToLoad, input, filter_shape, image_shape, poolsize):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
 
 
        #image_shape=(batch_size, 1, 28, 28),
        #filter_shape=(nkerns[0], 1, 5, 5)
        print "Filter shape:", filter_shape
        print "Image shape:", image_shape
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        #print "Filter shape:", filter_shape
        fan_in = numpy.prod(filter_shape[1:])
        #print "filter_shape[1:]", filter_shape[1:]
        #print "filter_shape[2:]", filter_shape[2:]
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        #print "Filter_shape2:", numpy.prod(filter_shape[2:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        #print "Fan_out:", fan_out
        # initialize weights with random weights
        
        
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        #W_bound = numpy.sqrt(fan_in)
        
        rng = numpy.random.RandomState(None)
        
        if(universalFeatures[0]):            
                loadedParams = DataUtil.loadState(universalFeatures[1],1)                
                
                Ww = loadedParams[0][0].astype(theano.config.floatX)
                b_values = loadedParams[0][1].astype(theano.config.floatX)
        elif(loadFrom != ""):
                loadedParams = DataUtil.loadState(loadFrom,parametersToLoad)                
                
                Ww = loadedParams[layerOrder][0].get_value()
                b_values = loadedParams[layerOrder][1].get_value()
        else:    
            
            Ww = numpy.asarray(
                 rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                 dtype=theano.config.floatX)
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
                        
            
        W_bound_Inhibitory = numpy.asarray(
                 rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                 dtype=theano.config.floatX)
                 
        b_values_Inhibitory = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            
        #print "Pesos:", Ww.shape[0]    
        
        self.W = theano.shared(Ww,
                              borrow=True)   
        self.wInhibitory = theano.shared(W_bound_Inhibitory,
                              borrow=True)   
                                      
        #print "Dec: ", dec
        #print "B_Inhibitory:", b_values_Inhibitory
        self.decayTerm = rng.uniform(low=0,high=1,size=[1])
        
        # the bias is a 1D tensor -- one bias per output feature map
        
        #print "Bias:", b_values
        self.b = theano.shared(value=b_values, borrow=True)
        self.bInhibitory = theano.shared(value=b_values_Inhibitory, borrow=True)
        
        
        
        # convolve input feature maps with filters
        
        print "Use Inhibition:" +   str( useInhibition      )
        if useInhibition:        
            
            newW = self.W / (self.decayTerm + self.wInhibitory) 
            #newW = self.wInhibitory / (self.decayTerm + self.W) 
            
                        
            if len(image_shape) == 4:
                conv_out = conv.conv2d(input=input, filters= newW , filter_shape=filter_shape, image_shape=image_shape)
            else:
                conv_out = T.nnet.conv3d2d.conv3d(signals=input, filters=newW,filters_shape=filter_shape, signals_shape=image_shape,border_mode='valid')    
                
            pooled_out = downsample.max_pool_2d(input=conv_out,ds=poolsize, ignore_border=True)
            
            #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) 
            self.output = T.maximum(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'),0) 
            
            #self.outputConv = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')) 
            self.outputConv = T.maximum(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'),0) 
            
            self.params = [self.W, self.b, self.wInhibitory]            
            print "Params" +   str( self.params      )
            
#            newW = T.add(self.W,self.wInhibitory)
#            newW = self.W / (self.decayTerm + self.wInhibitory)        
#          
#            print "newW:", newW.shape.eval()
#            print "W:", self.W.shape.eval()
#            inibitoryOut = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
#            pooled_out = downsample.max_pool_2d(input=inibitoryOut,ds=poolsize, ignore_border=True)
#            
#            #pooled_out_Inhibition = downsample.max_pool_2d(input=conv_out,ds=poolsize, ignore_border=True)
#            
#            #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) / (self.decayTerm  + T.tanh(pooled_out_Inhibition + self.bInhibitory.dimshuffle('x', 0, 'x', 'x'))) 
#            
#            self.output = T.tanh(inibitoryOut + self.b.dimshuffle('x', 0, 'x', 'x' ))
#            
#            self.outputConv = T.tanh(inibitoryOut + self.b.dimshuffle('x', 0, 'x', 'x' ))
#            
#            self.params = [self.W, self.b]
            
            #conv_out = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')) / (self.decayTerm  + T.tanh(inibitoryOut + self.bInhibitory.dimshuffle('x', 0, 'x', 'x'))) 
            #conv_out = (conv_out) / ((inibitoryOut)) 
        else:            
            
            print "ImageShape: " , len(image_shape)
            
            if len(image_shape) == 4:
                conv_out = conv.conv2d(input=input, filters= self.W , filter_shape=filter_shape, image_shape=image_shape)
            else:
                conv_out = T.nnet.conv3d2d.conv3d(signals=input, filters=self.W,filters_shape=filter_shape, signals_shape=image_shape,border_mode='valid')    
                
            pooled_out = downsample.max_pool_2d(input=conv_out,ds=poolsize, ignore_border=True)
            self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) 
            self.outputConv = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')) 
            
           # self.output = T.maximum(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'),0) 
           # self.outputConv = T.maximum(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'),0) 
            self.params = [self.W, self.b]
            
            
                
                
        # downsample each feature map individually, using maxpooling

    

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) 

       

#        self.output = T.maximum(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'),0)
        
        #self.outputConv = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        #self.outputConv = T.maximum(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), 0)
        
        
        #self.output = realOutput

        # store parameters of this layer
            
 

class HiddenLayer(object):
    def __init__(self, rng, layerOrder,loadFrom, parametersToLoad,input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        
        if(loadFrom != ""):
            
            loadedParams = DataUtil.loadState(loadFrom,parametersToLoad)
            W_values = loadedParams[layerOrder][0].get_value()
            b_values = loadedParams[layerOrder][1].get_value()
        else:    
            
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)        
            
        if W is None:
            
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
     
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, layerOrder, parametersToLoad, loadFrom, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """


        if(loadFrom != ""):
            loadedParams = DataUtil.loadState(loadFrom,parametersToLoad)
            W_values = loadedParams[layerOrder][0].get_value()
            b_values = loadedParams[layerOrder][1].get_value()
        else:    
            
            W_values = numpy.zeros((n_in, n_out),dtype=theano.config.floatX)
            b_values = numpy.zeros(n_out,dtype=theano.config.floatX)
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=W_values,
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=b_values,
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b] 

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example andCHANNEL_TYPE
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type)) 
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            

def MCNN3Channels(channelsTopology,inputType, datasets, training,loadFrom, savedState, conLayersParams, unitsInHiddenLayer,outputUnits, imageSize, n_epochs, learning_rate, batch_size, log, saveHistoryImageFiltersDirectory, repetitionNumber, timeStep, visualizeTrain):
    """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type conLayersParams: numpy.random.RandomState
        :param conLayersParams: Parameters for the convolutionalLayers. Each parameter is related to one
                                layer. Each parameter is composed by:  [numberOfFilters,receptiveFieldsX,
                                                                        receptiveFieldsY,poolSieX,poolSizeY]

        """
       
    log.startNewStep("Generating the Model")
    rng = numpy.random.RandomState(23455)    
                
    inputImages = 1 
    
    
    if inputType == DataUtil.INPUT_TYPE["Common"]:        
        x = T.matrix('x')
        inputImages = 1
        
        firstLayerInput = x.reshape((batch_size,inputImages,imageSize[0],imageSize[1]))         
        
        imageShape = (batch_size, inputImages, imageSize[0], imageSize[1])
        filterShape=(1, inputImages, 3, 3)
        
    elif inputType == DataUtil.INPUT_TYPE["Color"]:
        x = T.tensor3('x')# the data is presented as color images
        inputImages = 3   
        
        firstLayerInput = x.reshape((batch_size,inputImages,imageSize[0],imageSize[1])) 
        
        imageShape = (batch_size, inputImages, imageSize[0], imageSize[1])
        filterShape=(1, inputImages, 3, 3)
        
    elif inputType == DataUtil.INPUT_TYPE["3D"]:
        x = T.matrix('x').reshape((batch_size, timeStep,imageSize[0]*imageSize[1]))
        inputImages = 1
        
        firstLayerInput = x.reshape((batch_size, timeStep, 1,imageSize[0],imageSize[1]))
                
        imageShape = (batch_size, timeStep, inputImages, imageSize[0], imageSize[1])        
        filterShape=(1, timeStep, 1, 3, 3)
        
                
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch    
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
                        
    parametersToLoad = len(channelsTopology) * len(conLayersParams) +2

    
     

    firstLayer = FirstLayer(firstLayerInput)
    
    layerSobelX = LayerSobel(rng, "x", input=firstLayerInput,
        image_shape=imageShape,
        filter_shape=filterShape)
        
    layerSobelY = LayerSobel(rng, "y", input=firstLayerInput,
        image_shape=imageShape,
        filter_shape=filterShape)

        
    
    #MAYBE CHANGE THE BATCH SIZE HERE!                         
    

    #firstLayer =  FirstLayer(imageSize[0],imageSize[1],firstLayerInput)
    
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # 100-12+1 = 89
    # 89/2 = 44
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    
            
    
    channels = []
    loadPosition = 0
    channelsOutputs = []
    for channel in range(len(channelsTopology)):        
        log.printMessage(("Channel: ", channel))
        filtersInLayers = []
        receptiveFiltersInLayer=[]
        poolSizeInLayers = []
        
        for param in conLayersParams:
            filtersInLayers.append(param[0])
            receptiveFiltersInLayer.append((param[1],param[2]))
            poolSizeInLayers.append((param[3],param[4]))
            
        
        imageWidth = imageSize[0]
        imageHeight = imageSize[1]
        
        #Parameters for the hidden layer    
        
        #Parameters for the classification Layer    
            
                     
    # nkerns = number of features units (neurons) in each layer
    
        """ Demonstrates lenet on MNIST dataset
    
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
    
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
    
        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)
    
        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """        
    
        #ishape = (28, 28)  # this is the size of MNIST images
    
        ######################
        # BUILD ACTUAL MODEL #
        ######################        
        
        conLayers = []
        # Reshape matrix of rasterized images of shape  to a 4D tensor, compatible with our LeNetConvPoolLayer
        
        if channelsTopology[channel][0] ==  DataUtil.CHANNEL_TYPE["SobelX"]:
            conlayersInput = layerSobelX.output
            conlayerOutput = [(imageHeight-3+1) ,(imageWidth-3+1)]
            
        elif channelsTopology[channel][0] ==  DataUtil.CHANNEL_TYPE["SobelY"]:
            conlayersInput = layerSobelY.output
            conlayerOutput = [(imageHeight-3+1) ,(imageWidth-3+1)] 
        else:        
            conlayersInput = firstLayer.output
            conlayerOutput = [imageHeight ,imageWidth]           
        
                          
                        
        log.printMessage(( "--- Output Filter Layer Shape: ", conlayerOutput))
                
        inhibition = channelsTopology[channel][1]        
        

        if channelsTopology[channel][0] == DataUtil.CHANNEL_TYPE["CAE"]:  
            CAE = (True,channelsTopology[channel][2] )
        else:                            
            CAE = (False,"" )
        
        
        test_set_x, test_set_y = datasets[2]
        model_predict = theano.function([x], conlayersInput)
        classified = classify(model_predict,test_set_x.get_value(borrow=True)[0],batch_size)
        print "Stack1 output: ", classified.shape
        
        if inputType == DataUtil.INPUT_TYPE["3D"]:
            
            if channelsTopology[channel][0] ==  DataUtil.CHANNEL_TYPE["SobelX"] or channelsTopology[channel][0] ==  DataUtil.CHANNEL_TYPE["SobelY"]:
                imageShape=(batch_size, inputImages, conlayerOutput[0], conlayerOutput[1])
                conlayersInput = conlayersInput.reshape((batch_size, inputImages, conlayerOutput[0], conlayerOutput[1]))                
                filterShape=(filtersInLayers[0], inputImages, receptiveFiltersInLayer[0][0], receptiveFiltersInLayer[0][1])
            else:
                imageShape=(batch_size, timeStep, inputImages, conlayerOutput[0], conlayerOutput[1])
                filterShape=(filtersInLayers[0], timeStep, inputImages, receptiveFiltersInLayer[0][0], receptiveFiltersInLayer[0][1])
        elif inputType == DataUtil.INPUT_TYPE["Color"] and (channelsTopology[channel][0] ==  DataUtil.CHANNEL_TYPE["SobelX"] or channelsTopology[channel][0] ==  DataUtil.CHANNEL_TYPE["SobelY"]):            

             imageShape=(batch_size, 1, conlayerOutput[0], conlayerOutput[1])
             filterShape=(filtersInLayers[0], 1, receptiveFiltersInLayer[0][0], receptiveFiltersInLayer[0][1])
        else:
            imageShape=(batch_size, inputImages, conlayerOutput[0], conlayerOutput[1])
            filterShape=(filtersInLayers[0], inputImages, receptiveFiltersInLayer[0][0], receptiveFiltersInLayer[0][1])
                    
        
        
        

        
        
        layer0 = LeNetConvPoolLayer(inhibition, rng, CAE,  loadPosition, loadFrom,parametersToLoad, input=conlayersInput,
                    image_shape=imageShape,
                    filter_shape=filterShape, poolsize=(poolSizeInLayers[0][0], poolSizeInLayers[0][1]))   

      
        functionSobel= theano.function([x],conlayersInput)
        valid_set_x, valid_set_y = datasets[1]
        
        result = classify(functionSobel, valid_set_x.get_value(borrow=True)[0], batch_size)
        print "Channel:",channel," Layer1. Image Shape: ", result.shape
        print "Channel:",channel," Layer1. Input Shape: ",batch_size, timeStep, 1, conlayerOutput[0], conlayerOutput[1]
        
#    if channelsTopology[channel] == CHANNEL_TYPE["CAE"]:
#        
#    else:
#        layer0 = LeNetConvPoolLayer(inhibition, rng, (False,""),  loadPosition, loadFrom,parametersToLoad, input=conlayersInput,
#                image_shape=(batch_size, inputImages, conlayerOutput[0], conlayerOutput[1]),
#                filter_shape=(filtersInLayers[0], inputImages, receptiveFiltersInLayer[0][0], receptiveFiltersInLayer[0][1]), poolsize=(poolSizeInLayers[0][0], poolSizeInLayers[0][1]))               
                
                    
        loadPosition = loadPosition+1
        
        

        conlayerOutput = [(conlayerOutput[0]-receptiveFiltersInLayer[0][0]+1)/poolSizeInLayers[0][0] ,(conlayerOutput[1]-receptiveFiltersInLayer[0][0]+1)/poolSizeInLayers[0][1]]    
        log.printMessage(( "--- Output 0 ConLayer : ", conlayerOutput   ))
        
        conLayers.append(layer0)
        
        previousLayer = 0
        for k in range(len(conLayersParams)-1):            
            i = k+1 
            
            if k == 0 and inputType == DataUtil.INPUT_TYPE["3D"]:
                x.reshape((batch_size, timeStep, 1,imageSize[0],imageSize[1]))
                conInput = conLayers[previousLayer].output.reshape((batch_size,filtersInLayers[previousLayer],conlayerOutput[0],conlayerOutput[1]))                
                
            else:
                conInput = conLayers[previousLayer].output            
                                          
            layer1 = LeNetConvPoolLayer(False, rng, (False,""), loadPosition,loadFrom,parametersToLoad, input=conInput,
                image_shape=(batch_size, filtersInLayers[previousLayer], conlayerOutput[0], conlayerOutput[1]),
                filter_shape=(filtersInLayers[i],filtersInLayers[previousLayer], receptiveFiltersInLayer[i][0], receptiveFiltersInLayer[i][1]), poolsize=(poolSizeInLayers[i][0], poolSizeInLayers[i][1]))
            
            
            loadPosition = loadPosition+1
            
            conlayerOutput = [(conlayerOutput[0]-receptiveFiltersInLayer[i][0]+1)/poolSizeInLayers[i][0],
                             (conlayerOutput[1]-receptiveFiltersInLayer[i][1]+1)//poolSizeInLayers[i][1]]  

            
            log.printMessage(( "--- Output",i," ConLayer : ", conlayerOutput   ))
            previousLayer +=1   
            conLayers.append(layer1)              
        channelsOutputs.append(conlayerOutput)
        channels.append(conLayers)
        
    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
        
    
  
  #  value = valid_set_x.get_value(borrow=True)[0 * batch_size: (0 + 1) * batch_size]
    #print value.shape    
    
    
    #print "Len Con layers:", len(conLayersParams)
    #print "Con layers:", conLayersParams
    
    outputs = []
    for c in range(len(channels)):
        output = channels[c][len(conLayersParams)-1].output.flatten(2)
        outputs.append(output)
        
    layer2Outputs = theano.tensor.concatenate(outputs,1)
    
#    outputChannel0 = channels[0][1].output.flatten(2)    
#    model_predict = theano.function([x], outputChannel0)
#    classified = classify(model_predict,test_set_x.get_value(borrow=True)[0],batch_size)
#    print "Channel 0 output: ", classified.shape
#    
#    
#    outputChannel1 = channels[1][1].output.flatten(2)    
#    model_predict = theano.function([x], outputChannel1)
#    classified = classify(model_predict,test_set_x.get_value(borrow=True)[0],batch_size)
#    print "Channel 1 output: ", classified.shape
#    
#    
#    
#    outputChannel2 = channels[2][1].output.flatten(2)    
    
#    stack1 = theano.tensor.concatenate((outputChannel0,outputChannel1,outputChannel2),1)
    
    

    
#    model_predict = theano.function([x], layer2Outputs)
#    classified = classify(model_predict,test_set_x.get_value(borrow=True)[0],batch_size)
#    print "Stack1 output: ", classified.shape
#    outputs = channels[0][len(conLayersParams)-1].output.flatten(2)
#    for channel in range(channelsNumbers-1):
#        c = channel + 1
#        newOutput = channels[channel][c-1].output.flatten(2)
#                
#        model_predict = theano.function([x], newOutput)
#        classified = classify(model_predict,test_set_x.get_value(borrow=True)[0],batch_size)
#        print "Model Channel new: ", classified.shape
#        
#        outputs = theano.tensor.stack(newOutput)
#        outputs = outputs.flatten(2)
#        
#        model_predict = theano.function([x], outputs)
#        classified = classify(model_predict,test_set_x.get_value(borrow=True)[0],batch_size)
#        print "Model Channel outputs: ", classified.shape
                                  
        #data = encapsulateData([valid_set_x.get_value(borrow=True)[0]])          
        
       
       # outputs = newOutput
        
#        
    #outputs = channels[0][len(conLayers)-1].output
    
    #outputs = outputs.flatten(2)
    #stack1 = outputs.flatten(2)
    
    

    #print "Shape:", outputs.eval().shape
    #layer2_input = channels[0][len(conLayers)-1].output.flatten(2)
    layer2_input = layer2Outputs
    
    #for channel in range(channelsNumbers):
    #    print "ChannelNumbers", channel
    #    for conLayer in range(len(channels[channel])):    
    #        print "ConLayer: ", conLayer
    #        print "Output: ", channels[channel][conLayer].output
    #        layer2_input += channels[channel][conLayer].output
        
    #layer2_input = layer2_input.flatten(2)    
    
    # construct a fully-connected sigmoidal layer
    n_in = 0
    for a in range(len(channelsOutputs)):        
        n_in = n_in + filtersInLayers[len(conLayers)-1]*channelsOutputs[a][0]*channelsOutputs[a][1]
              
    
    #n_in=filtersInLayers[len(conLayers)-1] * conlayerOutput[0]    
    
    layer2 = HiddenLayer(rng, loadPosition, loadFrom,parametersToLoad, input=layer2_input, n_in=n_in,
                         n_out=unitsInHiddenLayer, activation=T.tanh)        
    log.printMessage(( "Hidden Layer Input: ", n_in))      
    loadPosition = loadPosition+1                        
    # classify the values of the fully-connected sigmoidal layer
    log.printMessage(( "Hidden Layer Output: ", unitsInHiddenLayer ))
    layer3 = LogisticRegression(loadPosition,parametersToLoad, loadFrom, input=layer2.output, n_in=unitsInHiddenLayer, n_out=outputUnits)
                       
    
    #print "BatchSize:", batch_size                    
            
    log.startNewStep("Creating Training Strategies")      
    train_set_x, train_set_y = datasets[0]    
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
        

   
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size    
    
        
    
    if(training):        
        # the cost we minimize during training is the NLL of the model
        cost = layer3.negative_log_likelihood(y)    
        # create a function to compute the mistakes that are made by the model
        test_model = theano.function([index], layer3.errors(y),
                 givens={
                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                    y: test_set_y[index * batch_size: (index + 1) * batch_size]})
    
        validate_model = theano.function([index], layer3.errors(y),
                givens={
                    x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                    y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
    
       
       # print "Example:", valid_set_x.get_value(borrow=True)[0].shape[0]
       # model_predict = theano.function(inputs=[x], outputs=layer3.y_pred)    
        
        #print "Classify:", classify(model_predict, [train_set_x.get_value(borrow=True)[0]])
        #data = encapsulateData(valid_set_x.get_value(borrow=True)[0])
        #print "Classify:", model_predict(data.get_value(borrow=True))
        #print "Classify", classify(model_predict,valid_set_x.get_value(borrow=True)[0])
        # create a list of all model parameters to be fit by gradient descent
                
        
        params = layer3.params + layer2.params
        
        for channel in range(len(channelsTopology)):
            isFirstLayer = True
            for layer in channels[channel]:
                
                if channelsTopology[channel][0] == DataUtil.CHANNEL_TYPE["CAE"] and isFirstLayer == True:                    
                    
                    if channelsTopology[channel][1]:
                        tensor = []
                        tensor.append(layer.params[2])
                        params += tensor
                    isFirstLayer = False
                else:    
                    params += layer.params            
                    
        
        # create a list of gradients for all model parametersprint "Outputs 3D:", numpy.array(image3D).shape
        grads = T.grad(cost, params)
    
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i],grads[i]) pairs.
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))
    
        train_model = theano.function([index], cost, updates=updates,
              givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    
        ###############
        # TRAIN MODEL #
        ###############
        log.startNewStep("Training Model")      
        # early-stopping parameters
        patience = 100000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        last_validations = []
        best_validation_loss = numpy.inf
        test_score = 0.
    
        epoch = 0
        done_looping = False        
        
        filtersInLayers = []
        receptiveFiltersInLayer=[]
        poolSizeInLayers = []
        
        for param in conLayersParams:
            filtersInLayers.append(param[0])
            receptiveFiltersInLayer.append((param[1],param[2]))
            poolSizeInLayers.append((param[3],param[4]))
        
        if visualizeTrain:
            filtersMapsImages = []
            for c in range(len(channels)):            
                for i in range(len(channels[c])):         
                    rows = int((n_epochs/10)+1) * 52                        
                    collumns = 52 * filtersInLayers[i]
                    
                    if channelsTopology[c][0] and i == 0:                    
                        for a in range(2):
                             if inputType == DataUtil.INPUT_TYPE["3D"] and i ==0:
                                 for a in range(timeStep):
                                     image = Image.new('RGB', (rows,collumns))
                                     filtersMapsImages.append(image)
                             else:
                                "Lateral Inhibition + Final filter"                        
                                image = Image.new('RGB', (rows,collumns))                        
                                filtersMapsImages.append(image)

                    
                    "Common Filters"                        
                    if inputType == DataUtil.INPUT_TYPE["3D"] and i ==0:
                        for a in range(timeStep):
                            image = Image.new('RGB', (rows,collumns))
                            filtersMapsImages.append(image)
                    else:                        
                        image = Image.new('RGB', (rows,collumns))
                        filtersMapsImages.append(image)
        
       
                       
        showImage = True  
        updatesLearningRate = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
                        
            lastEpoch = 0
            for minibatch_index in xrange(n_train_batches):
                
                #print "Ntrain batches:", n_train_batches
                #print "Minibatch Index:", minibatch_index
                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                if iter % 100 == 0:
                    log.printMessage(("Training iteration:", iter ))  
                    
                if visualizeTrain:                     
                    if showImage:
                        showImage = False
                        lastEpoch  = epoch
                        imageIndex = 0                        
                        for c in range(len(channels)):
                            for i in range(len(channels[c])):                                        
                                
                                "Lateral Inhibition Visualization"
                                if channelsTopology[c][1] and i == 0:   
                                    
                                    "3DInput"    
                                    if inputType == DataUtil.INPUT_TYPE["3D"] and i == 0:
                                       filterMaps = channels[c][i].params[2].get_value(borrow=True)                                          
                                       filterMaps = numpy.swapaxes(filterMaps,0,1)
                                       
                                       for a in range(timeStep):   
                                                                          
                                           if epoch == 1:
                                               posX = 0
                                           else:
                                               posX = int((epoch / 10)) * 52
                                               
                                           indexFilterMap = 0
                                           #print "Shape Filter3d:", numpy.array(filterMaps[a]).shape
                                           for w in filterMaps[a]:
                                                                                   
                                               posY = indexFilterMap * 52
                                               indexFilterMap = indexFilterMap+1                                    
                                               if inputType == DataUtil.INPUT_TYPE["Color"] and i == 0:
                                                   img = numpy.swapaxes(w, 0,1) 
                                                   img = numpy.swapaxes(img,1,2)      
                                                   
                                                   img[:,:,0] = ImageProcessingUtil.convertFloatImage(img[:,:,0])
                                                   img[:,:,1] = ImageProcessingUtil.convertFloatImage(img[:,:,1])
                                                   img[:,:,2] = ImageProcessingUtil.convertFloatImage(img[:,:,2])
                                                   img = ImageProcessingUtil.resize(img,(50,50))
                                                   
                                                   fi = Image.fromarray(img.astype('uint8')).convert('RGBA')                                        
                                               else:                                                
                                                   img = ImageProcessingUtil.convertFloatImage(w[0])
                                                   img = ImageProcessingUtil.resize(img,(50,50))
                                                   fi = Image.fromarray(img.astype('uint8')).convert('L')
                                                                                   
                                               filtersMapsImages[imageIndex].paste(fi,(posX,posY))                                    
                                           imageIndex = imageIndex +1
                                    else:       
                                            "Lateral Inhibition Filters"
                                            filterMaps = channels[c][i].params[2].get_value(borrow=True)
                                            
        
                                            img = ImageProcessingUtil.convertFloatImage(filterMaps)
                                            
                                            if epoch == 1:
                                                posX = 0
                                            else:
                                                posX = int((epoch / 10)) * 52
                                                
                                            indexFilterMap = 0
                                            for w in filterMaps:        
                                                posY = indexFilterMap * 52
                                                indexFilterMap = indexFilterMap+1                                        
                                                if inputType == DataUtil.INPUT_TYPE["Color"] and i == 0:
                                                    img = numpy.swapaxes(w, 0,1) 
                                                    img = numpy.swapaxes(img,1,2)                                    
                                                    img[:,:,0] = ImageProcessingUtil.convertFloatImage(img[:,:,0])
                                                    img[:,:,1] = ImageProcessingUtil.convertFloatImage(img[:,:,1])
                                                    img[:,:,2] = ImageProcessingUtil.convertFloatImage(img[:,:,2])
                                                    img = ImageProcessingUtil.resize(img,(50,50))
                                                    fi = Image.fromarray(img.astype('uint8')).convert('RGBA')
                                                   
                                                else:
                                                    img = ImageProcessingUtil.convertFloatImage(w[0])
                                                    img = ImageProcessingUtil.resize(img,(50,50))
                                                    fi = Image.fromarray(img.astype('uint8')).convert('L')
                                                                                                        
                                                filtersMapsImages[imageIndex].paste(fi,(posX,posY))
                                            imageIndex = imageIndex +1
                                            
                                            
                                    if inputType == DataUtil.INPUT_TYPE["3D"] and i == 0:
                                        
                                       filterMapsInhibition = channels[c][i].params[2].get_value(borrow=True)
                                       filterMaps = channels[c][i].params[0].get_value(borrow=True)
                                       finalFilterMap = filterMaps / (channels[c][i].decayTerm + filterMapsInhibition)                                       
                                       filterMaps = numpy.swapaxes(filterMaps,0,1)
                                       
                                       for a in range(timeStep):   
                                                                          
                                           if epoch == 1:
                                               posX = 0
                                           else:
                                               posX = int((epoch / 10)) * 52
                                               
                                           indexFilterMap = 0
                                          # print "Shape Filter3d:", numpy.array(filterMaps[a]).shape
                                           for w in filterMaps[a]:
                                                                                   
                                               posY = indexFilterMap * 52
                                               indexFilterMap = indexFilterMap+1                                    
                                               if inputType == DataUtil.INPUT_TYPE["Color"] and i == 0:
                                                   img = numpy.swapaxes(w, 0,1) 
                                                   img = numpy.swapaxes(img,1,2)      
                                                   
                                                   img[:,:,0] = ImageProcessingUtil.convertFloatImage(img[:,:,0])
                                                   img[:,:,1] = ImageProcessingUtil.convertFloatImage(img[:,:,1])
                                                   img[:,:,2] = ImageProcessingUtil.convertFloatImage(img[:,:,2])
                                                   img = ImageProcessingUtil.resize(img,(50,50))
                                                   
                                                   fi = Image.fromarray(img.astype('uint8')).convert('RGBA')                                        
                                               else:                                                
                                                   img = ImageProcessingUtil.convertFloatImage(w[0])
                                                   img = ImageProcessingUtil.resize(img,(50,50))
                                                   fi = Image.fromarray(img.astype('uint8')).convert('L')
                                                                                   
                                               filtersMapsImages[imageIndex].paste(fi,(posX,posY))                                    
                                           imageIndex = imageIndex +1
                                           
                                    else:
                                        "Final Filter, after the application of the lateral Inhibition"
                                        filterMapsInhibition = channels[c][i].params[2].get_value(borrow=True)
                                        filterMaps = channels[c][i].params[0].get_value(borrow=True)
                                        finalFilterMap = filterMaps / (channels[c][i].decayTerm + filterMapsInhibition) 
                                                                             
                                        
                                        if epoch == 1:
                                            posX = 0
                                        else:
                                            posX = int((epoch / 10)) * 52
                                            
                                        indexFilterMap = 0
                                        
                                        for w in finalFilterMap:
                                                                  
                                            posY = indexFilterMap * 52
                                            indexFilterMap = indexFilterMap+1
                                            if inputType == DataUtil.INPUT_TYPE["Color"] and channelsTopology[c][0] ==  DataUtil.CHANNEL_TYPE["Common"] and i == 0:
                                                img = numpy.swapaxes(w, 0,1) 
                                                img = numpy.swapaxes(img,1,2)                                    
                                                img[:,:,0] = ImageProcessingUtil.convertFloatImage(img[:,:,0])
                                                img[:,:,1] = ImageProcessingUtil.convertFloatImage(img[:,:,1])
                                                img[:,:,2] = ImageProcessingUtil.convertFloatImage(img[:,:,2])
                                                img = ImageProcessingUtil.resize(img,(50,50))
                                                fi = Image.fromarray(img.astype('uint8')).convert('RGBA')
                                                
                                            else:
                                                img = ImageProcessingUtil.convertFloatImage(w[0])
                                                img = ImageProcessingUtil.resize(img,(50,50))
                                                fi = Image.fromarray(img.astype('uint8')).convert('L')
                                                                   
                                            
                                            filtersMapsImages[imageIndex].paste(fi,(posX,posY))
    
                                        imageIndex = imageIndex +1
                
                   
                                filterMaps = channels[c][i].params[0].get_value(borrow=True)
                                                                
                                
                                if inputType == DataUtil.INPUT_TYPE["3D"] and i == 0:
                                    filterMaps = numpy.swapaxes(filterMaps,0,1)
                                    print "TimeStep:", timeStep
                                    for a in range(timeStep):   
                                                                       
                                        if epoch == 1:
                                            posX = 0
                                        else:
                                            posX = int((epoch / 10)) * 52
                                            
                                        indexFilterMap = 0
                                       # print "Shape Filter3d:", numpy.array(filterMaps[a]).shape
                                        for w in filterMaps[a]:
                                                                                
                                            posY = indexFilterMap * 52
                                            indexFilterMap = indexFilterMap+1                                    
                                            if inputType == DataUtil.INPUT_TYPE["Color"] and i == 0:
                                                img = numpy.swapaxes(w, 0,1) 
                                                img = numpy.swapaxes(img,1,2)      
                                                
                                                img[:,:,0] = ImageProcessingUtil.convertFloatImage(img[:,:,0])
                                                img[:,:,1] = ImageProcessingUtil.convertFloatImage(img[:,:,1])
                                                img[:,:,2] = ImageProcessingUtil.convertFloatImage(img[:,:,2])
                                                img = ImageProcessingUtil.resize(img,(50,50))
                                                
                                                fi = Image.fromarray(img.astype('uint8')).convert('RGBA')                                        
                                            else:                                                
                                                img = ImageProcessingUtil.convertFloatImage(w[0])
                                                img = ImageProcessingUtil.resize(img,(50,50))
                                                fi = Image.fromarray(img.astype('uint8')).convert('L')
                                                                                
                                            filtersMapsImages[imageIndex].paste(fi,(posX,posY))                                    
                                        imageIndex = imageIndex +1
                                else:
                                    
                                        if epoch == 1:
                                            posX = 0
                                        else:
                                            posX = int((epoch / 10)) * 52
                                            
                                        indexFilterMap = 0
                                        #print "Second layer Shape Filter3d:", numpy.array(filterMaps).shape
                                        for w in filterMaps:
                                                                                
                                            posY = indexFilterMap * 52
                                            indexFilterMap = indexFilterMap+1                                    
                                            if inputType == DataUtil.INPUT_TYPE["Color"] and channelsTopology[c][0] ==  DataUtil.CHANNEL_TYPE["Common"] and i == 0:
                                                img = numpy.swapaxes(w, 0,1) 
                                                img = numpy.swapaxes(img,1,2)      
                                                
                                                img[:,:,0] = ImageProcessingUtil.convertFloatImage(img[:,:,0])
                                                img[:,:,1] = ImageProcessingUtil.convertFloatImage(img[:,:,1])
                                                img[:,:,2] = ImageProcessingUtil.convertFloatImage(img[:,:,2])
                                                img = ImageProcessingUtil.resize(img,(50,50))
                                                
                                                fi = Image.fromarray(img.astype('uint8')).convert('RGBA')                                        
                                            else:                                                
                                                img = ImageProcessingUtil.convertFloatImage(w[0])
                                                img = ImageProcessingUtil.resize(img,(50,50))
                                                fi = Image.fromarray(img.astype('uint8')).convert('L')
                                                                                
                                            filtersMapsImages[imageIndex].paste(fi,(posX,posY))                                    
                                        imageIndex = imageIndex +1
                                    
                        
                        showIndex = 0                                
                        for c in range(len(channels)):
                            for i in range(len(channels[c])):  
                              if channelsTopology[c][1]  and i == 0:                 
                                  
                                  if inputType == DataUtil.INPUT_TYPE["3D"] and i == 0:
                                    for a in range(timeStep):   
                                      img = numpy.array(filtersMapsImages[showIndex])                              
                                      cv2.imshow("C: "+ str(c+1)+ " L: "+ str(i)+" I: "+ str(a) + "-LI",  img)      
                                      cv2.imwrite(saveHistoryImageFiltersDirectory+"/"+"C_"+ str(c+1)+ "_L_"+ str(i)+"_LI_"+str(repetitionNumber)+".png",img)
                                      showIndex = showIndex+1
                                      key = cv2.waitKey(20)   
                                      
                                      img = numpy.array(filtersMapsImages[showIndex])
                                      cv2.imshow("C: "+ str(c+1)+ " L: "+ str(i)+" I: "+ str(a)+" -FF",  img)                                    
                                      cv2.imwrite(saveHistoryImageFiltersDirectory+"/"+"C_"+ str(c+1)+ "_L_"+ str(i)+"_FF_"+str(repetitionNumber)+".png",img)
                                      showIndex = showIndex+1
                                      key = cv2.waitKey(20)                                 
                                  else:
                                      img = numpy.array(filtersMapsImages[showIndex])                              
                                      cv2.imshow("C: "+ str(c+1)+ " L: "+ str(i)+" - LI",  img)      
                                      cv2.imwrite(saveHistoryImageFiltersDirectory+"/"+"C_"+ str(c+1)+ "_L_"+ str(i)+"_LI_"+str(repetitionNumber)+".png",img)
                                      showIndex = showIndex+1
                                      key = cv2.waitKey(20)   
                                      
                                      img = numpy.array(filtersMapsImages[showIndex])
                                      cv2.imshow("C: "+ str(c+1)+ " L: "+ str(i)+" -FF",  img)                                    
                                      cv2.imwrite(saveHistoryImageFiltersDirectory+"/"+"C_"+ str(c+1)+ "_L_"+ str(i)+"_FF_"+str(repetitionNumber)+".png",img)
                                      showIndex = showIndex+1
                                      key = cv2.waitKey(20)                                 
                                      
                               
                              if inputType == DataUtil.INPUT_TYPE["3D"] and i == 0:
                                    for a in range(timeStep):   
                                      img = numpy.array(filtersMapsImages[showIndex])
                                      cv2.imshow("C: "+ str(c+1)+ " L: "+ str(i)+" I: "+ str(a),  img)      
                                      
                                      cv2.imwrite(saveHistoryImageFiltersDirectory+"/"+"C_"+ str(c+1)+ "_L_"+ str(i)+"_"+str(repetitionNumber)+".png",img)
                                      showIndex = showIndex+1
                                      key = cv2.waitKey(20)   
                              else:
                                  img = numpy.array(filtersMapsImages[showIndex])
                                  cv2.imshow("C: "+ str(c+1)+ " L: "+ str(i),  img)      
                                  
                                  cv2.imwrite(saveHistoryImageFiltersDirectory+"/"+"C_"+ str(c+1)+ "_L_"+ str(i)+"_"+str(repetitionNumber)+".png",img)
                                  showIndex = showIndex+1
                                  key = cv2.waitKey(20)  
                                    
                          
                if epoch % 10 == 0 and lastEpoch != epoch:                    
                    showImage = True
                    
                train_model(minibatch_index)
               
                
                if (iter + 1) % validation_frequency == 0:
    
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    log.printMessage((" Epoch:",epoch, " minibatch ",minibatch_index + 1,"/",n_train_batches, " . Validation Error: ", this_validation_loss * 100. ))
                   
                    if len(last_validations) == 15:
                        last_validations.pop(0)  

                    last_validations.append(this_validation_loss)
                                        
                    unchangedValidations = 0
                    last = last_validations[0]                    
                    for ah in last_validations:                        
                        if last == ah:
                            unchangedValidations = unchangedValidations+1
                        else:   
                             unchangedValidations = unchangedValidations-1
                        last = ah
                        
                    #print "Last validations:", last_validations
                    #print "UnchangedValidations: ", unchangedValidations                   
                    if unchangedValidations >= 10:
                        updatesLearningRate = updatesLearningRate + 1
                        print "Update learning rate --- from: ", learning_rate, " to: ", (learning_rate/2)
                        learning_rate = learning_rate /2
                        last_validations = []
                        # train_model is a function that updates the model parameters by
                        # SGD Since this model has many parameters, it would be tedious to
                        # manually create an update rule for each model parameter. We thus
                        # create the updates list by automatically looping over all
                        # (params[i],grads[i]) pairs.
                        updates = []
                        for param_i, grad_i in zip(params, grads):
                            updates.append((param_i, param_i - learning_rate * grad_i))
                    
                        train_model = theano.function([index], cost, updates=updates,
                              givens={
                                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                y: train_set_y[index * batch_size: (index + 1) * batch_size]})
                        
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
    
                        # test it on the test set
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)
                        log.printMessage((" Epoch:",epoch, " minibatch ",minibatch_index + 1,"/",n_train_batches, " . Test Error: ", test_score * 100. ))
       
                if updatesLearningRate == 10:
                    done_looping = True
                    break
                
                if patience <= iter:
                    done_looping = True
                    break
    
        log.startNewStep("Saving Network")
        params1 = []        
        positionSave = 0
        for channel in channels:
            for layer in channel:
                
                params1.append(layer.params)        
                positionSave = positionSave+1
                        
        params1.append(layer2.params)
        
        positionSave = positionSave+1
        
        params1.append(layer3.params)    
        
        DataUtil.saveState(params1,savedState)
        log.printMessage( "Model Saved in :"+ savedState)
    #print "Params Real: ", layer0.W.get_value() 
    
    #layer0Output = theano.function([x], layer0.output)
    
    
    
    outputsConvLayers = []
    
    
    for c in range(len(channelsTopology)):
        if channelsTopology[c][0] == DataUtil.CHANNEL_TYPE["SobelX"]:
                outputsConvLayers.append(theano.function([x],layerSobelX.output))                
                                
        elif channelsTopology[c][0] == DataUtil.CHANNEL_TYPE["SobelY"]:
                outputsConvLayers.append(theano.function([x],layerSobelY.output))
        else:
                outputsConvLayers.append(theano.function([x],firstLayer.output))                
                    
        for l in range(len(conLayersParams)):
            outputsConvLayers.append(theano.function([x],channels[c][l].outputConv))
            outputsConvLayers.append(theano.function([x],channels[c][l].output))
              
    
    layer2Output = theano.function([x], layer2.input)
    modelOutput = theano.function([x], layer3.y_pred) 
    
    return (test_set_x,test_set_y,outputsConvLayers, layer2Output,modelOutput)


def classify(predictModel, array, batchSize):
        
    newArray = []
    for i in range(batchSize):
        newArray.append(array)      

    return predictModel(newArray)
                               
def getClassificationReport(trueData,predictData,directory,metricsDirectory,experimentName,repetition,log):
         
     #print "Calculating Metrics...."     
     metrics =  (classification_report(trueData,predictData))    
     cM = confusion_matrix(trueData,predictData)  
     log.startNewStep("Confusion Matrix")
     log.printMessage(cM)
     pylab.matshow(cM)
     pylab.title('Confusion matrix')
     pylab.colorbar()
     pylab.ylabel('True label')
     pylab.xlabel('Predicted label')
     metricsDirectory = metricsDirectory + "/ConfusionMatrix/"
     DataUtil.createFolder(metricsDirectory)
     pylab.savefig(metricsDirectory+experimentName+"_"+str(repetition)+"_confusionMatrix.png")    
         
     log.printMessage(metrics)
  
def getPrecision(trueData, predictData,average):
    return precision_score(trueData,predictData,average=average)

    
def getAccuracy(trueData, predictData):
    return accuracy_score(trueData,predictData,normalize= True)    
    
def getRecall(trueData, predictData,average):
    return recall_score(trueData,predictData,average=average)
    
def getFScore(trueData, predictData,average):
    return f1_score(trueData,predictData,average=average)       


def getConvolutionalFeatures(outputModel,directoryImages, batchSize,imageSize):
    
    convFeatures = []
    classesPath = os.listdir(directoryImages)          
    classNumber = 0
    for classs in classesPath:  
        files = os.listdir(directoryImages+os.sep+classs+os.sep)                    
        for image in files:
            imageFeatures = []
            img = cv2.imread(directoryImages+os.sep+classs+os.sep+image)
                                
            features = ImageProcessingUtil.grayImage(img,imageSize ,False,"")
            features = features.view(numpy.ndarray)
            features.shape = -1
            cFeature = classify(outputModel,features,batchSize)[0]
            #print "Shape Cfeature:", cFeature.shape
            for f in cFeature:                                    
                imageFeatures.append(f)
            imageFeatures.append(classNumber)            
            
            convFeatures.append(imageFeatures)            
        classNumber = classNumber+1
    return convFeatures


def getOutputImage(outputModels,batchSize, img,imageSize,channels,layers):
    
    grayScale = ImageProcessingUtil.grayImage(img,imageSize ,False,"")
    #grayScale = ImageProcessingUtil.resize(img,imageSize )
    
    features = grayScale.view(numpy.ndarray)
    features.shape = -1    
    
    outputIndex = 0      
    pylab.gray()
    fig = plt.figure()       
    fig.subplots_adjust(left=0.3, wspace=0.3, hspace=0.3)

    ax1 = fig.add_subplot(1+layers*2,channels,1)                
    ax1.imshow(grayScale[:, :])                
    ax1.set_title("C1: Gray")
    ax1.axis('off')                

    ax1 = fig.add_subplot(1+layers*2,channels,2)                
    sobelX = classify(outputModels[outputIndex],features,batchSize)[0][0]
    ax1.imshow(sobelX[:, :])                
    ax1.set_title("C2: SX")
    ax1.axis('off')         

    ax1 = fig.add_subplot(1+layers*2,channels,3)
    outputIndex = outputIndex+1
    sobelY = classify(outputModels[outputIndex],features,batchSize)[0][0]
    ax1.imshow(sobelY[:, :])                
    ax1.set_title("C3: SY")
    ax1.axis('off')                         
    outputIndex = outputIndex+1
    for c in range(channels):
        layersConvOut = []
        layersMaxOut = []                    
        for l in range(layers):       
           convImages = classify(outputModels[outputIndex],features,batchSize)[0]
           outputIndex = outputIndex+1
           maxPoolingImages = classify(outputModels[outputIndex],features,batchSize)[0]
           outputIndex = outputIndex+1
           
           layerConvOut = None
           layerMaxOut = None
           for i in range(len(convImages)):
               img = convImages[i]
               if layerConvOut == None:
                   layerConvOut = img
               else:    
                   layerConvOut = numpy.hstack((layerConvOut,img))
               
               img = maxPoolingImages[i]
               if layerMaxOut == None:
                   layerMaxOut = img
               else:    
                   layerMaxOut = numpy.hstack((layerMaxOut,img))
                   
           layersConvOut.append(layerConvOut)
           layersMaxOut.append(layerMaxOut)
        posConv = channels+c+1                    
        posMax = 2*channels+c+1
        for l in range(layers):
            #print str(1+layers*2)+","+str(channels)+","+str(posConv)
            ax1 = fig.add_subplot(1+layers*2,channels,posConv)
            ax1.imshow(layersConvOut[l][:, :])
            ax1.set_title("C:"+ str(c)+ "_L:"+str(l)+" Conv")
            ax1.axis('off')                    
            
            #print str(1+layers*2)+","+str(channels)+","+str(posMax)
            ax1 = fig.add_subplot(1+layers*2,channels,posMax)
            ax1.imshow(layersMaxOut[l][:, :])                        
            ax1.set_title("C:"+ str(c)+ "_L:"+str(l)+" Max")
            ax1.axis('off')                        
            
            posConv = posConv+channels*2
            posMax = posMax+channels*2    
    return ImageProcessingUtil.fig2data(fig) 
                

def showOutputImages(outputModels,batchSize,directoryImages,imageSize,channels,layers):
    directory = directoryImages    
    classesPath = os.listdir(directory)            
        
    for classs in classesPath:      
            files = os.listdir(directory+os.sep+classs+os.sep)                        
            for image in files:
                img = cv2.imread(directory+os.sep+classs+os.sep+image)
                                
                fig = getOutputImage(outputModels,batchSize, img,imageSize,channels,layers)                
                cv2.imshow('image',fig)
                #cv2.imwrite("/informatik2/wtm/home/barros/Documents/Experiments/Cambridge/100x100MotionWithShadows_SET1/Final.png", fig)
                key = cv2.waitKey(20)
                #break
            #break        
    cv2.destroyAllWindows()
    
def createOutputImagesSequence(outputModel, batchSize, directoryImages, directorySave, imageSize):
    
    directory = directoryImages
    featuresDirectory = directorySave       
    classesPath = os.listdir(directory)            
    gesture = 0
    
    for classs in classesPath:      
        sequencesNumber = 0                  
        sequences = os.listdir(directory+os.sep+classs+os.sep)
        for sequence in sequences:
            files = os.listdir(directory+os.sep+classs+os.sep+sequence+os.sep)            
            #files = os.listdir(directory+os.sep+classs+os.sep)            
            frameNumber = 0
            
            images = []
            for image in files:                                    
                img = cv2.imread(directory+os.sep+classs+os.sep+sequence+os.sep+image)
                #img = cv2.imread(directory+os.sep+classs+os.sep+image)                   
                              
                features = ImageProcessingUtil.resize(img,imageSize)  
                features = features.view(numpy.ndarray)
                                
                features = ImageProcessingUtil.grayImage(features,imageSize ,False,"")
                features = ImageProcessingUtil.whiten(features)         
                
                features.shape = -1
              
                frameNumber = frameNumber+1  
                images.append(features)
                                            
                
            cFeature = classify(outputModel,images,batchSize)
            
            
            cFeature = cFeature[0]
            #print "Shape cFeature:", numpy.array(cFeature).shape
            if len(numpy.array(cFeature).shape) == 4:
                if not numpy.array(cFeature).shape[0] == 1:
                    cFeature = numpy.swapaxes(cFeature, 0,1)
                cFeature = cFeature[0]   
                    
                
            
            o = 0
            DataUtil.createFolder(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep)
            #print "Shape cFeature:", numpy.array(cFeature).shape
            for im in cFeature:        
                
                img = im
                
               # print img                                      
                img = Image.fromarray(ImageProcessingUtil.convertFloatImage(img),"L") #Image.fromarray(img, "L")
                #img = Image.fromarray(img,"L")
                pylab.gray()
                                    
                #if not os.path.exists(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep): os.makedirs(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep)            
                
                
                #img.save(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg") 
                #print "Image Name: ", featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+image+"_"+str(o)+"_.jpeg"
                pylab.imsave(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+image+"_"+str(o)+"_.jpeg", im)
                #img.save(featuresDirectory+os.sep+classs+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg")      
                o= o+1                    
                
            sequencesNumber = sequencesNumber +1
            
            if sequencesNumber >3:
                break          
     
               
            gesture = gesture+1    
    
def createOutputImages(outputModel,batchSize, directoryImages, directorySave, imageSize, inputType):
        
    if  inputType == DataUtil.INPUT_TYPE["3D"]:
        createOutputImagesSequence(outputModel,batchSize, directoryImages, directorySave, imageSize)
    else:
        directory = directoryImages
        featuresDirectory = directorySave       
        classesPath = os.listdir(directory)            
        gesture = 0
        
        for classs in classesPath:                        
            #sequences = os.listdir(directory+os.sep+classs+os.sep)
            #for sequence in sequences:
               #files = os.listdir(directory+os.sep+classs+os.sep+sequence+os.sep)            
                files = os.listdir(directory+os.sep+classs+os.sep)            
                frameNumber = 0
                imagesNumber = 0
                for image in files:                                    
                    #img = cv2.imread(directory+os.sep+classs+os.sep+sequence+os.sep+image)
                    img = cv2.imread(directory+os.sep+classs+os.sep+image)                   
                                  
                    features = ImageProcessingUtil.resize(img,imageSize)  
                    features = features.view(numpy.ndarray)
                    
                    if inputType == DataUtil.INPUT_TYPE["Color"]:
                        features = numpy.swapaxes(features, 2,1)
                        features = numpy.swapaxes(features, 0,1)
                        
                        features = numpy.reshape(features, (3, imageSize[0]*imageSize[1]))
                        
                    else:                
                        features = ImageProcessingUtil.grayImage(features,imageSize ,False,"")
                        features = ImageProcessingUtil.whiten(features)         
                        
                        features.shape = -1
                  
                    frameNumber = frameNumber+1                                          
                    
                    cFeature = classify(outputModel,features,batchSize)
                                    
                    cFeature = cFeature[0]
                    
                    o = 0
                    DataUtil.createFolder(featuresDirectory+os.sep+classs+os.sep+str(frameNumber)+os.sep)
                    for im in cFeature:                    
                        img = im
                        
                       # print img                                      
                        img = Image.fromarray(ImageProcessingUtil.convertFloatImage(img),"L") #Image.fromarray(img, "L")
                        #img = Image.fromarray(img,"L")
                        pylab.gray()
                                            
                        #if not os.path.exists(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep): os.makedirs(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep)            
                        
                        
                        #img.save(featuresDirectory+os.sep+classs+os.sep+sequence+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg") 
                        
                        pylab.imsave(featuresDirectory+os.sep+classs+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg", im)
                        #img.save(featuresDirectory+os.sep+classs+os.sep+str(frameNumber)+os.sep+image+"_"+str(o)+"_.jpeg")      
                        o= o+1                    
                        
                    imagesNumber = imagesNumber +1
                    
                    if imagesNumber >3:
                        break          
         
                   
                gesture = gesture+1    
             

def createHintonDiagram(directory, saveDirectory,convLayers,channels,totalLayers):    
        
    lines = 5
    rows = 0
       
    loadedParams = DataUtil.loadState(directory,totalLayers)
        
    index = 0
    
    for channel in range(channels):
        
        for layer in range(convLayers):
            layerImage = []
            for a in range(len(loadedParams[index][0].get_value())):                
                
                
                f = loadedParams[index][0].get_value()[a][0].astype(theano.config.floatX)
                img = ImageProcessingUtil.convertFloatImage(f)
                #img = Image.fromarray(f, "L")
                #pylab.imsave(finalDirectory, img)
                layerImage.append(img)
                finalDirectory = saveDirectory + "Channel_"+str(channel)+"/Layer_"+str(layer)+"/"
                DataUtil.createFolder(finalDirectory)  
                finalDirectory = finalDirectory +"_filter_"+str(a)+"_"+str(index)+".png"
                #img = ImageProcessingUtil.grayImage(img)
                matplotlib.image.imsave(finalDirectory,img, cmap=matplotlib.cm.gray)                
                #plt.imshow(img)
               # plt.savefig(finalDirectory)
                #img.save(finalDirectory)
                
                
                #DataUtil.saveHintonDiagram(loadedParams[index][0].get_value()[f][0],finalDirectory)
            rows = (len(layerImage)/lines)
            
            kernelSize = Image.fromarray(layerImage[0]).size
            #print kernelSize
            new_im = Image.new('RGB', (lines*kernelSize[0],rows*kernelSize[1]))
            imgIndex = 0
            for i in range(lines):
                for j in range(rows):                
                    #paste the image at location i,j:
                    posL = kernelSize[0]*i 
                    posR = kernelSize[0]*j 
                    new_im.paste(Image.fromarray(layerImage[imgIndex]), (posL,posR))
                    imgIndex = imgIndex+1
                    
            finalDirectory = saveDirectory + "Channel_"+str(channel)+"/Layer_"+str(layer)            
            DataUtil.createFolder(finalDirectory)    
            finalDirectory = finalDirectory +"_filter_"+str(index)+".png"      
            new_im.save(finalDirectory)   
            #plt.imshow(new_im)
            #plt.savefig(finalDirectory)
            #new_im = ImageProcessingUtil.convertFloatImage(new_im)
            #matplotlib.image.imsave(finalDirectory,new_im)                
            index = index+1                   
    
    
    
    
    
    #loadedParams = DataUtil.loadState(directory,totalLayers)
        
#    index = 0
#    
#    for channel in range(channels):
#        
#        for layer in range(convLayers):
#            for f in range(len(loadedParams[index][0].get_value())):                
#                finalDirectory = saveDirectory + "Channel_"+str(channel)+"/Layer_"+str(layer)
#                DataUtil.createFolder(finalDirectory)
#                
#                finalDirectory = finalDirectory +"/filter_"+str(f)+".png"                
#                DataUtil.saveHintonDiagram(loadedParams[index][0].get_value()[f][0],finalDirectory)
#                   
#            index = index+1
#    
#    finalDirectory = saveDirectory + "Hidden"+"/"
#    DataUtil.createFolder(finalDirectory)    
#    finalDirectory = finalDirectory + "hiddenLayer.png"    
#    DataUtil.saveHintonDiagram(loadedParams[index][0].get_value(),finalDirectory)
#    index = index+1
#    
#    finalDirectory = saveDirectory + "Output"+"/"
#    DataUtil.createFolder(finalDirectory)    
#    finalDirectory = finalDirectory + "outputLayer.png"    
#    DataUtil.saveHintonDiagram(loadedParams[index][0].get_value(),finalDirectory)


"""
trueData = []
for i in range(6):
    for h in range(30):
        trueData.append(i)
 
trueData = numpy.array(trueData)
#trueData = numpy.ravel(trueData)
#print trueData.shape
       
predictData = [0,4,5,0,5,0,0,0,0,0,5,5,0,0,0,5,0,0,5,0,5,0,4,0,0,4,0,0,4,0,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,3,1,1,1,3,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,3,3,1,3,1,3,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,0,3,3,4,3,4,4,4,4,4,4,4,1,3,4,4,4,4,3,4,1,4,1,4,1,4,4,4,5,5,5,5,5,5,5,4,5,0,5,4,5,5,5,5,5,5,5,5,5,4,4,5,5,4,5,5,4,5]
predictData = numpy.array(predictData)
#predictData = numpy.ravel(predictData)
#print predictData.shape
cM = confusion_matrix(trueData,predictData)  
     
labels = ["Circle", "P. Left", "P. Right", "Stand", "Stop", "Turn"]    


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cM, cmap=pylab.cm.Greys)
pylab.title('Confusion matrix')

for i, cas in enumerate(cM):
    for j, c in enumerate(cas):
        if c>0:
            plt.text(j-.2, i+.2, c, fontsize=14, color='red')

fig.colorbar(cax)
pylab.gray()
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pylab.ylabel('True label')
pylab.xlabel('Predicted label')
pylab.savefig("/informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/NimbroLiveTest/Turn/_confusionMatrix.png")    

pylab.matshow(cM)
pylab.title('Confusion matrix')
pylab.colorbar()
pylab.ylabel('True label')
pylab.xlabel('Predicted label')
pylab.set_xticklabels([''] + labels)
pylab.set_yticklabels([''] + labels)
pylab.savefig("/informatik2/wtm/home/barros/Documents/Experiments/Dynamic Gesture Dataset/testMotion/NimbroLiveTest/Turn/_confusionMatrix.png")    

print getFScore(trueData,predictData,"micro")
print getFScore(trueData,predictData,None)
print cM
#print f1_score(trueData,trueData,"micro")  
"""