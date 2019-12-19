#!/bin/bash


################################################################
## Predict RNA base pair
## Analyze the total f1-score
## Generate new test and training set --> 
##		based on previous f1-score
## Use cs_f1_score_analyze.py
################################################################

python cs_f1_score_analyze.py

## Generate new test and training set
sed 1d average_f1.csv | awk '{$1=""; print}' | sort -n -k 1


