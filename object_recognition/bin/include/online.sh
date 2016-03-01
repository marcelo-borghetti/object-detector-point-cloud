#!/bin/bash
source include/offline.sh

onlineProcedureCapture()
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
   ./segment_cluster $OPTION $ARG $OUTPUT_DIR/"out"$LABEL".txt" $OUTPUT_DIR 
}

onlineProcedureView()
{
   OPTION=$1 
   ARG=$2
   INPUT_FILE=$3
   ./segment_cluster $OPTION $ARG $INPUT_FILE
}

onlineProcedureHelp()
{
   ARG=$1
   ./segment_cluster $ARG
}