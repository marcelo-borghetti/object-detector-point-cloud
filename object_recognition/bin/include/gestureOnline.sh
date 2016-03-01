#!/bin/bash
source include/offline.sh

onlineProcedureCapture_gestures()
{
   OPTION=$1 
   ARG=$2
   LABEL=$3
   OUTPUT_DIR=$4
   if [ "$OUTPUT_DIR" = "" ]
   then
      createNewResultDir
      OUTPUT_DIR=$BASE$DIRCOUNT 
   fi 
   echo $OUTPUT_DIR/"out"$LABEL".txt"
   ./get_gestures $OPTION $ARG $OUTPUT_DIR/"out"$LABEL".txt" $OUTPUT_DIR 
}

onlineProcedureView_gestures()
{
   OPTION=$1 
   ARG=$2
   INPUT_FILE=$3
   ./get_gestures $OPTION $ARG $INPUT_FILE
}

onlineProcedureHelp_gestures()
{
   ARG=$1
   ./get_gestures $ARG
}