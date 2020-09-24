# point-cloud-object-detector
Sample codes for object recog and neural networks


*************************************************************
* GENERAL INFO		 	 

This repository has 2 directories:

1) src/ contains all source codes for RGB-D capture. To use this source code, put src/ in a catkin environment and compile them with 'catkin_make'.

2) bin/ has some scripts and other development files that should work with the binaries built from the source code. These files should be relocated to the same directory of binary files created by 'catkin_make'. The code of the CNN is also in tis directory.

*************************************************************
* USING THE RGB-D capture		  	    


To use the RGB-D capture type

	./obj_recognition.sh -on -cnn <label> <dir_results>

	-on: online capture activated

	-cnn: CNN neural recognition (you should use this option, but CNN online recognition is currently disabled) 

	<label>: category of the object captured (use a generic name when no special category is captured)

	<dir_results>: directory to store the point clouds captured

To use the capture, you should press "space". The point clouds will be stored in a directory "new_trainning_data/" inside <dir_results>.

*************************************************************
* USING THE Neural Network	           

To use the code, type:

	cd bin/CNN_PYTHON/MCNN.v2015/MCCNN/
	
	python Program.py




