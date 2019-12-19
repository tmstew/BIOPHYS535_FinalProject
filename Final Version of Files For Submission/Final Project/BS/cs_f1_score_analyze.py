## Import Module
import pandas as pd
import numpy as np


################################################################
## This is used to analyze the f1-score result.
## LOO model:
##	1. Drop Chemical Shift
##	2. Drop 1 RNA
##	3. A total of 104 * 19 data
## File: all_score_cs.csv
## Create new csv file:  
################################################################


################################################################
## Read in and process all f1-score data 
################################################################

print(f"[INFO]: Now processing the f1-score data")

## Process data
df = pd.read_csv('all_score_cs.csv', sep=" ")
if 'Unnamed: 0' in df.columns:
	df = df.drop(columns = ['Unnamed: 0'])
df = df.drop(columns = ['Average of f1 score', 'Std of f1 score'])
f1 = df.values
pdbid = list(df.columns) 
f1_avg = np.around(np.average(f1, axis = 0), decimals = 4)

## Creat new dataframe
new_df = pd.DataFrame(f1_avg, dtype = 'object', columns = ['f1'])
new_df['pdbid'] = pdbid
new_df.to_csv('average_f1.csv', sep=' ')

