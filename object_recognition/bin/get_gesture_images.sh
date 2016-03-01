#!/bin/bash
source include/offline.sh
source include/online.sh

OPTION=$1
if [ "$OPTION" = "-off" ]
then
	if [ "$2" = "-mlp" ] 
	then
	    ARG=$2	
	    SOURCE_DATA=$3
	    if [ "$SOURCE_DATA" = "" ]
	    then
		echo "Database path is required as input."
	    else
		offlineProcedure $OPTION $ARG $SOURCE_DATA
	    fi
	elif [ "$2" = "-cnn" ] 
	then
	    ARG=$2	
	    SOURCE_DATA=$3
	    if [ "$SOURCE_DATA" = "" ]
	    then
		echo "Database path is required as input."
	    else
		offlineProcedure $OPTION $ARG $SOURCE_DATA
	    fi
	else
	  echo -e "\n\tUSAGE: ./obj_recognition.sh -off <-mlp/cnn/gng> <DATABASE>\n"
	fi
elif [ "$OPTION" = "-on" ] 
then
	if [ "$2" = "-v" ] 
	then
	    ARG=$2
	    INPUT_FILE=$3
	    if [ "$INPUT_FILE" = "" ]
	    then
		echo "Input file is required as input."
	    else
		onlineProcedureView $OPTION $ARG $INPUT_FILE
	    fi
	elif [ "$2" = "-mlp" ] 
	then
	    ARG=$2
	    LABEL=$3
	    OUTPUT_DIR=$4
	    onlineProcedureCapture $OPTION $ARG $LABEL $OUTPUT_DIR
	elif [ "$2" = "-cnn" ] 
	then
	    ARG=$2
	    LABEL=$3
	    OUTPUT_DIR=$4
	    onlineProcedureCapture $OPTION $ARG $LABEL $OUTPUT_DIR
	else
	  echo -e "\n\tUSAGE: ./obj_recognition.sh -on <-mlp/cnn/gng> <LABEL-OBJECT> <OUTPUT_DIR>\n"
	fi
elif [ "$OPTION" = "-h" ]
then
	ARG=$1
        onlineProcedureHelp $ARG
else
      echo "You must choose one of the options -off, -on or -h"
fi
      
      