## Import Module
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler


################################################################
## This is used to search the best number of neighboring base 
##	chemical shift used in this study.
## Number of neighboring pair is set from 0 to 5 (by bash script)
## LOO model
## sklearn: clf = svm.SVC(gamma='auto')
## f1-score is saved under the fold output/seraching_neighbors
################################################################


################################################################
## Read in and pre-process chemical shift data
################################################################

c=pd.read_csv('final_training.csv', sep=" ")
cs_data=c.replace({'stack':1})			# Replace stack with 1
cs_data=cs_data.replace({'non-stack':0})	# Replace non-stack with 0
cs_data=cs_data.drop(columns=['base_pairing', 'orientation', 'sugar_puckering', 'pseudoknot'])
if 'Unnamed: 0' in cs_data.columns:
	cs_data=cs_data.drop(columns=['Unnamed: 0'])
cs_data=cs_data.rename(columns={'stacking':'class'})

################################################################
## Global variables and functions
################################################################
NUMBER_CHEMICAL_SHIFT_TYPE=19
neighbors=np.loadtxt("neighbors", dtype=int)

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
	cs_tmp=cs_i[(cs_i.resid == resid)].drop(['id', 'resid', 'resname', 'class'], axis=1)
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
	columns=['id', 'resname', 'resid', 'class']
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

def get_cs_features_rna(cs, neighbors=1, retain = ['id', 'class', 'resid']):
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

cs_all = get_cs_features_rna_all(cs_data, neighbors=neighbors)
drop_names = ['id','class', 'resid']
target_name = 'class'
score = []
col = 2*neighbors + 1
print(f"[SET UP DATA]: The amount of the neighboring bases used is {neighbors}")

for pdbid in cs_all['id'].unique()[0 :]:
	print(f"[INFO]: Now predict RNA --> {pdbid}")
	
	## Prepare test set
	test = cs_all[cs_all.id == pdbid]
	tmp = test.drop(drop_names, axis=1)
	tmp_testX = tmp.iloc[:, col :]
	tmp_testy = test[target_name]
	testX = tmp_testX.values
	testy = tmp_testy.values
	id = pd.unique(test.id)
	print(f"[INFO]: Test set --> {id}")

	## Prepare training set
	train = cs_all[cs_all.id != pdbid]
	tmp = train.drop(drop_names, axis=1)
	tmp_trainX = tmp.iloc[:, col :]
	tmp_trainy = train[target_name]
	trainX = tmp_trainX.values
	trainy = tmp_trainy.values
	id = pd.unique(train.id)
	print(f"[INFO]: Test set --> {id}")

	## Normalization of the training set and test set
	scaler = StandardScaler()
	scaler.fit(trainX)
	trainX_scaled = scaler.transform(trainX)
	testX_scaled = scaler.transform(testX)
	print(f"[INFO]: Scale the features")

	## Train model
	clf = svm.SVC(gamma='auto')
	clf.fit(trainX_scaled, np.int_(trainy))

	## Test model
	predicted = clf.predict(testX_scaled)
	print(f"[INFO]: Running prediction")

	## f1-score
	score.append(metrics.f1_score(np.int_(testy),predicted))
	print(" ")

################################################################
## Prediction analysis 
################################################################
average_f1 = np.average(score)
std = np.std(score)
print(f"[ANALYSIS RESULT]: LOO model result")
print(f"[ANALYSIS RESULT]: The average f1-score is: {average_f1} +/- {std}")

