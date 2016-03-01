#!/bin/bash
createNewResultDir()
{
    ############################################
    # Creating Base directory for OUTPUT files
    ############################################
    BASE=Results/
    if [ ! -d $BASE ] 
    then
      mkdir $BASE
    fi

    ##################################################
    # Counting the number of directory already created
    # One for each execution
    #################################################
    FILESinBASE=Results/*
    FILECOUNT=0
    DIRCOUNT=0
    for item in $FILESinBASE
    do
    if [ -f "$item" ]
	then
	    FILECOUNT=$[$FILECOUNT+1]
	elif [ -d "$item" ]
	    then
	    DIRCOUNT=$[$DIRCOUNT+1]
    fi
    done
    DIRCOUNT=$[$DIRCOUNT+1]
    ######################################################
    # Creating the new directory to store the new results
    #####################################################
    mkdir $BASE$DIRCOUNT
}


offlineProcedure()
{
    OPTION=$1
    ARG=$2
    INPUTDIR=$3
    createNewResultDir
    ######################################################
    # Iterating among the input directory (with known labels)
    # TODO: change for automatic label recognition
    #       In this way, strings like "/Box/", "/Can/", etc
    #       should be automatically recognitized
    #####################################################
    for item in $INPUTDIR*
    do
    if [ -d "$item" ]
	then
	    #echo "Processing $item..."
	    y=${item##*/}
	    z=${y%*}         
	    
	    LABEL_OBJECT=$z         
	    #echo "Label: $LABEL_OBJECT"
	    FILES=$item/*
	    #echo "Files: $FILES"
	    OUTPUT=$BASE$DIRCOUNT/$LABEL_OBJECT".txt"
	    echo "OUTPUT: $OUTPUT"

	    for f in $FILES
	    do    
		echo "Processing $f file..."
		./segment_cluster $OPTION $ARG $f $BASE$DIRCOUNT $OUTPUT
	    done         
    fi
    done
}