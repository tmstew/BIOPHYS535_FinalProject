## Import Module
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


################################################################
## This is used to search the best number of neighboring base 
##	chemical shift used in this study.
## Number of neighboring pair is set from 0 to 5 (by bash script)
## LOO model
## sklearn: LinearRegression Ridge
## r2-score is saved under the folder output/seraching_neighbors
################################################################


################################################################
## Read in and pre-process chemical shift data
################################################################

c = pd.read_csv('new.csv', sep=" ")
cs_data = c.rename(columns = {'H8\t' : 'H8'})
if 'Unnamed: 0' in cs_data.columns:
	cs_data = cs_data.drop(columns = ['Unnamed: 0'])

################################################################
## Global variables and functions
################################################################

NUMBER_CHEMICAL_SHIFT_TYPE = 19
neighbors = np.loadtxt("neighbors", dtype=int)

def get_cs_all(cs_all, id):
	'''
	This function gets chemical shifts for a particular RNA.
	'''
	return(cs_all[cs_all.id == id])

def get_cs_residues(cs_i, resid, dummy=0):
	'''
	This function return an array contining the chemical shifts 
	for a particular residues in an RNA.
	'''
	cs_tmp=cs_i[(cs_i.resid == resid)].drop(['id', 'resid', 'resname', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All'], axis=1)
	info_tmp=cs_i[(cs_i.resid == resid)]
	if (cs_tmp.shape[0] != 1):
		return(dummy*np.ones(shape=(1, NUMBER_CHEMICAL_SHIFT_TYPE)))
	else:
		return(cs_tmp.values)

def get_resnames(cs_i, resid, dummy="UNK"):
	'''
	This function returns the residue name for specified residue (resid)
	'''
	cs_tmp=cs_i[(cs_i.resid == resid)]
	if (cs_tmp.shape[0] != 1):
		return(dummy)
	else:
		return(cs_tmp['resname'].values[0])

def get_cs_features(cs_i, resid, neighbors):
	'''
	This function return chemical shifts and resnames for 
	residues (resid) and its neighbors
	'''
	cs=[]
	resnames=[]
	for i in range(resid-neighbors, resid+neighbors+1):
		cs.append(get_cs_residues(cs_i, i))
		resnames.append(get_resnames(cs_i, i))
	return(resnames, np.array(cs))

def get_columns_name(neighbors=3, chemical_shift_types = NUMBER_CHEMICAL_SHIFT_TYPE):
	'''
	Helper function that writes out the required column names
	'''
	#tmp=2*neighbors+1
	#neighbors=1
	columns=['id', 'resname', 'resid', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All']
	for i in range(0, neighbors*NUMBER_CHEMICAL_SHIFT_TYPE):
		columns.append(i)
	return(columns)

def write_out_resname(neighbors=1):
	''' 
	Helper function that writes out the column names associated 
	resnames for a given residue and its neighbors
	'''  
	colnames = []
	for i in range(1-neighbors-1, neighbors+1):
		if i < 0: 
			colnames.append('R%s'%i)
		elif i > 0: 
			colnames.append('R+%s'%i)
		else: 
			colnames.append('R')
	return(colnames)    

def get_cs_features_rna(cs, neighbors=1, retain = ['id', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'resid']):
	'''    
	This function generates the complete required data frame an RNA    
	'''
	all_features = []
	all_resnames = []
	for resid in cs['resid'].unique():
		resnames, features = get_cs_features(cs, resid, neighbors)
		all_features.append(features.flatten())
		all_resnames.append(resnames)

	all_resnames = pd.DataFrame(all_resnames, dtype='object', columns = write_out_resname(neighbors))
	all_features = pd.DataFrame(all_features, dtype='object')
	info = pd.DataFrame(cs[retain].values, dtype='object', columns = retain)
	return(pd.concat([info, all_resnames, all_features], axis=1))

def get_cs_features_rna_all(cs, neighbors):  
	'''    
	This function generate a pandas dataframe containing training data for all RNAs
	Each row in the data frame should contain the class and chemical shifts for given residue and neighbors in a given RNA.
	'''  
	cs_new=pd.DataFrame()
	for pdbid in cs['id'].unique()[0 :]:
		tmp=get_cs_features_rna(get_cs_all(cs, id=pdbid), neighbors)
		cs_new=pd.concat([cs_new, tmp], axis=0)
	return(cs_new)

################################################################
## Build model and test
################################################################

cs_all = get_cs_features_rna_all(cs_data, neighbors = neighbors)
drop_names = ['id', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'resid']
target = ['sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All']
score = []
score_itself = []
all_prediction = []
col = 2*neighbors + 1
print(f"[SET UP DATA]: The amount of the neighboring bases used is {neighbors}")

for pdbid in cs_all['id'].unique()[0 :]:
	print(f"[INFO]: Now predict RNA --> {pdbid}")
	
	## Prepare test set
	test = cs_all[cs_all.id == pdbid]
	tmp = test.drop(drop_names, axis=1)
	tmp_testX = tmp.iloc[:, col :]
	tmp_testy = pd.DataFrame(test[target].values, dtype = 'object', columns = target)
	testX = tmp_testX.values
	testy = tmp_testy.values
	id = pd.unique(test.id)
	print(f"[INFO]: Test set --> {id}")

	## Prepare training set
	train = cs_all[cs_all.id != pdbid]
	tmp = train.drop(drop_names, axis=1)
	tmp_trainX = tmp.iloc[:, col :]
	tmp_trainy = pd.DataFrame(train[target].values, dtype = 'object', columns = target)
	trainX = tmp_trainX.values
	trainy = tmp_trainy.values
	id = pd.unique(train.id)
	print(f"[INFO]: Test set --> {id}")

	## Normalization of the features 
	scaler = StandardScaler()
	scaler.fit(trainX)
	trainX_scaled = scaler.transform(trainX)
	testX_scaled = scaler.transform(testX)
	print(f"[INFO]: Scale the features")



	## Normalization of the features 
	scaler = StandardScaler()
	scaler.fit(trainy)
	trainy_scaled = scaler.transform(trainy)
	testy_scaled = scaler.transform(testy)
	print(f"[INFO]: Scale the features")



	## Train and test model
	reg = Ridge(alpha = 5, tol = 0.0001).fit(trainX_scaled, trainy_scaled)
	predicted = reg.predict(testX_scaled)
	if pd.unique(test.id) == '1A60':
		all_prediction = predicted
	else:
		all_prediction = np.concatenate((all_prediction, predicted), axis = 0)
	print(f"[INFO]: Running prediction")

	## R2-score analysis
	score.append(r2_score(testy_scaled, predicted, multioutput = 'raw_values'))

	## Test model with training set
	predicted = reg.predict(trainX_scaled)
	score_itself.append(r2_score(trainy_scaled, predicted, multioutput = 'raw_values'))
	print(f"[INFO]: Test model with the training set")

	print(" ") 

print("[INFO]: Finish prediction")
print(" ")
print(" ")

################################################################
## Prediction analysis 
################################################################

## Running analysis
average_r2 = np.average(score, axis = 0)
std_r2 = np.std(score, axis = 0)
tmp_truey = pd.DataFrame(cs_all[target].values, dtype = 'object', columns = target)
truey = tmp_truey.values
fig = plt.figure(figsize = (25,15))

## Plot result
plt.subplot(151)							## sasa-All-atoms
plt.scatter(truey[:, 0], all_prediction[:, 0], color = 'black')
plt.plot(truey[:, 0], truey[:, 0], color = 'blue', linewidth = 5)
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.title(target[0])

plt.subplot(152)							## sasa-Total-Side
plt.scatter(truey[:, 1], all_prediction[:, 1], color = 'black')
plt.plot(truey[:, 1], truey[:, 1], color = 'blue', linewidth = 5)
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.title(target[1])

plt.subplot(153)							## sasa-Main-Chain
plt.scatter(truey[:, 2], all_prediction[:, 2], color = 'black')
plt.plot(truey[:, 2], truey[:, 2], color = 'blue', linewidth = 5)
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.title(target[2])

plt.subplot(154)							## sasa-Non-polar
plt.scatter(truey[:, 3], all_prediction[:, 3], color = 'black')
plt.plot(truey[:, 3], truey[:, 3], color = 'blue', linewidth = 5)
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.title(target[3])

plt.subplot(155)							## sasa-All
plt.scatter(truey[:, 4], all_prediction[:, 4], color = 'black')
plt.plot(truey[:, 4], truey[:, 4], color = 'blue', linewidth = 5)
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.title(target[4])

plt.savefig("r2_score.png")

## Save r2-score data to a csv
r2_data = pd.DataFrame(score, dtype = 'object', columns = target)
r2_data.to_csv('r2.csv', sep = ' ')

## Output analysis result
print(f"[PREDICTION RESULT]: LOO model")
print(f"[PREDICTION RESULT]: The average r2-score in predicting {target[0]} is: {average_r2[0]} +/- {std_r2[0]}")
print(f"[PREDICTION RESULT]: The average r2-score in predicting {target[1]} is: {average_r2[1]} +/- {std_r2[1]}")
print(f"[PREDICTION RESULT]: The average r2-score in predicting {target[2]} is: {average_r2[2]} +/- {std_r2[2]}")
print(f"[PREDICTION RESULT]: The average r2-score in predicting {target[3]} is: {average_r2[3]} +/- {std_r2[3]}")
print(f"[PREDICTION RESULT]: The average r2-score in predicting {target[4]} is: {average_r2[4]} +/- {std_r2[4]}")
print(" ")

################################################################
## Model analysis with training set 
################################################################

average_r2 = np.average(score_itself, axis = 0)
std_r2 = np.std(score_itself, axis = 0)
print(f"[MODEL RESULT]: LOO model")
print(f"[MODEL RESULT]: The average r2-score in predicting {target[0]} is: {average_r2[0]} +/- {std_r2[0]}")
print(f"[MODEL RESULT]: The average r2-score in predicting {target[1]} is: {average_r2[1]} +/- {std_r2[1]}")
print(f"[MODEL RESULT]: The average r2-score in predicting {target[2]} is: {average_r2[2]} +/- {std_r2[2]}")
print(f"[MODEL RESULT]: The average r2-score in predicting {target[3]} is: {average_r2[3]} +/- {std_r2[3]}")
print(f"[MODEL RESULT]: The average r2-score in predicting {target[4]} is: {average_r2[4]} +/- {std_r2[4]}")
print(" ")





