## Import Module
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score 


################################################################
## This is used to search the critical chemical shift for 
##	prediciting SASA. 
## neighbors = 1 (best r2-score) : 0.9229 +/- 0.0851
## LOO model
## sklearn: LinearRegression Ridge
## r2-score is saved under folder output/searching_cs
################################################################


################################################################
## Read in and pre-process chemical shift data
################################################################

c = pd.read_csv('new.csv', sep=" ")
cs_data = c.rename(columns = {'H8\t' : 'H8'})
if 'Unnamed: 0' in cs_data.columns:
	cs_data=cs_data.drop(columns=['Unnamed: 0'])

## Creat chemical shift list
cs_list = list(cs_data.columns)
cs_list.remove('id')
cs_list.remove('resid')
cs_list.remove('resname')
cs_list.remove('sasa-All-atoms')
cs_list.remove('sasa-Total-Side')
cs_list.remove('sasa-Main-Chain')
cs_list.remove('sasa-Non-polar')
cs_list.remove('sasa-All')

################################################################
## Global variables and functions
################################################################
NUMBER_CHEMICAL_SHIFT_TYPE = 18
neighbors = np.loadtxt("neighbors", dtype = int)

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

drop_names = ['id', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'resid']
target = ['sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All']
col = 2*neighbors + 1

## Create`empty total-r2 score
totalr2_1 = [] 				## sasa-All-atoms
totalr2_2 = []				## sasa-Total-Side
totalr2_3 = []				## sasa-Main-Chain
totalr2_4 = []				## sasa-Non-polar
totalr2_5 = []				## sasa-All

for atom in cs_list:
	print(f"[SET UP DATA]: The chemical shift dropped is {atom}")
	tmp_c = cs_data.drop(atom, axis = 1)
	cs_all = get_cs_features_rna_all(tmp_c, neighbors=neighbors)
	score = []
	r2_1 = []			## sasa-All-atoms
	r2_2 = []			## sasa-Total-Side
	r2_3 = []			## sasa-Main-Chain
	r2_4 = []			## sasa-Non-polar
	r2_5 = []			## sasa-All

	for pdbid in cs_all['id'].unique()[0 :]:
		print(f"[INFO]:  Now predict RNA --> {pdbid}")
	
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

		## Train and test model
		reg = RandomForestRegressor(n_estimators = 100, max_features = "sqrt")
		reg.fit(trainX_scaled, trainy)
		predicted = reg.predict(testX_scaled)
		print(f"[INFO]: Running prediction")

		## R2-score
		tmp_score = r2_score(testy, predicted, multioutput = 'raw_values')
		r2_1.append(tmp_score[0])
		r2_2.append(tmp_score[1])
		r2_3.append(tmp_score[2])
		r2_4.append(tmp_score[3])
		r2_5.append(tmp_score[4])
		print(f"[INFO]: Generate r2-score")
		print(" ")

	## Total f1-score
	if atom == 'C1p':
		totalr2_1 = r2_1
		totalr2_2 = r2_2
		totalr2_3 = r2_3
		totalr2_4 = r2_4
		totalr2_5 = r2_5
	else:
		totalr2_1 = np.concatenate((totalr2_1, r2_1), axis = 0)
		totalr2_2 = np.concatenate((totalr2_2, r2_2), axis = 0)
		totalr2_3 = np.concatenate((totalr2_3, r2_3), axis = 0)
		totalr2_4 = np.concatenate((totalr2_4, r2_4), axis = 0)
		totalr2_5 = np.concatenate((totalr2_5, r2_5), axis = 0)
	print(f"[INFO]: Now appending r2-score to total score")
	print(" ") 
	print(" ") 

################################################################
## Process predicition data 
################################################################

r2 = pd.read_csv('r2.csv', sep = ' ')
if 'Unnamed: 0' in r2.columns:
	r2 = r2.drop(columns = ['Unnamed: 0'])
ref_r2_1 = r2[target[0]].values
ref_r2_2 = r2[target[1]].values
ref_r2_3 = r2[target[2]].values
ref_r2_4 = r2[target[3]].values
ref_r2_5 = r2[target[4]].values

pdbid_list = cs_data['id'].unique()[0 :]
num = len(pdbid_list)

################################################################
## Plot heatmap 
################################################################

print(f"[INFO]: Plotting heatmap")

totalr2_1 = totalr2_1.reshape(num, -1)		## sasa-All-atoms
totalr2_1 = np.transpose(totalr2_1)
delta_r2_1 = totalr2_1 - ref_r2_1
delta_r2_1 = delta_r2_1 * 100
plt.figure(figsize = (30,15))
ax = sns.heatmap(delta_r2_1, center = 0, xticklabels = pdbid_list, yticklabels = cs_list)
ax.set_title("sasa all atoms heatmap")
plt.savefig("sasa-All-atoms.png")

totalr2_2 = totalr2_2.reshape(num, -1)		## sasa-Total-Side
totalr2_2 = np.transpose(totalr2_2)
delta_r2_2 = totalr2_2 - ref_r2_2
delta_r2_2 = delta_r2_2 * 100
plt.figure(figsize = (30,15))
ax = sns.heatmap(delta_r2_2, center = 0, xticklabels = pdbid_list, yticklabels = cs_list)
ax.set_title("sasa total side heatmap")
plt.savefig("sasa-Total-Side.png")

totalr2_3 = totalr2_3.reshape(num, -1)		## sasa-Main-Chain
totalr2_3 = np.transpose(totalr2_3)
delta_r2_3 = totalr2_3 - ref_r2_3
delta_r2_3 = delta_r2_3 * 100
plt.figure(figsize = (30,15))
ax = sns.heatmap(delta_r2_3, center = 0, xticklabels = pdbid_list, yticklabels = cs_list)
ax.set_title("sasa main chain heatmap")
plt.savefig("sasa-Main-Chain.png")

totalr2_4 = totalr2_4.reshape(num, -1)		## sasa-Non-polar
totalr2_4 = np.transpose(totalr2_4)
delta_r2_4 = totalr2_4 - ref_r2_4
delta_r2_4 = delta_r2_4 * 100
plt.figure(figsize = (30,15))
ax = sns.heatmap(delta_r2_4, center = 0, xticklabels = pdbid_list, yticklabels = cs_list)
ax.set_title("sasa non polar heatmap")
plt.savefig("sasa-Non-polar.png")

totalr2_5 = totalr2_5.reshape(num, -1)		## sasa-All
totalr2_5 = np.transpose(totalr2_5)
delta_r2_5 = totalr2_5 - ref_r2_5
delta_r2_5 = delta_r2_5 * 100
plt.figure(figsize = (30,15))
ax = sns.heatmap(delta_r2_5, center = 0, xticklabels = pdbid_list, yticklabels = cs_list)
ax.set_title("sasa all heatmap")
plt.savefig("sasa-All.png")

print(" ")

################################################################
## Analyze prediction result 
################################################################

print(f"[ANALYSIS RESULT]: LOO model result")
i = 0
num = len(cs_list)

average_r2_1 = []
average_r2_2 = []
average_r2_3 = []
average_r2_4 = []
average_r2_5 = []

std_r2_1 = []
std_r2_2 = []
std_r2_3 = []
std_r2_4 = []
std_r2_5 = []

while i < num :
	tmp_r2_1 = totalr2_1[i,:]
	tmp_r2_2 = totalr2_2[i,:]
	tmp_r2_3 = totalr2_3[i,:]
	tmp_r2_4 = totalr2_4[i,:]
	tmp_r2_5 = totalr2_5[i,:]

	average_r2_1.append(np.average(tmp_r2_1))
	average_r2_2.append(np.average(tmp_r2_2))
	average_r2_3.append(np.average(tmp_r2_3))
	average_r2_4.append(np.average(tmp_r2_4))
	average_r2_5.append(np.average(tmp_r2_5))

	std_r2_1.append(np.std(tmp_r2_1))
	std_r2_2.append(np.std(tmp_r2_2))
	std_r2_3.append(np.std(tmp_r2_3))
	std_r2_4.append(np.std(tmp_r2_4))
	std_r2_5.append(np.std(tmp_r2_5))

	print(f"[INFO]: The chemical shift {cs_list[i]} is dropped -->") 
	print(f"[ANALYSIS RESULT]: The average r2-score in predicting {target[0]} is: {average_r2_1[i]} +/- {std_r2_1[i]}")
	print(f"[ANALYSIS RESULT]: The average r2-score in predicting {target[1]} is: {average_r2_2[i]} +/- {std_r2_2[i]}")
	print(f"[ANALYSIS RESULT]: The average r2-score in predicting {target[2]} is: {average_r2_3[i]} +/- {std_r2_3[i]}")
	print(f"[ANALYSIS RESULT]: The average r2-score in predicting {target[3]} is: {average_r2_4[i]} +/- {std_r2_4[i]}")
	print(f"[ANALYSIS RESULT]: The average r2-score in predicting {target[4]} is: {average_r2_5[i]} +/- {std_r2_5[i]}")
	print(" ") 
	i += 1

################################################################
## Save r2-score data to a csv 
################################################################

## Save sasa-All-atoms r2-score data to a csv
print(f"[INFO]: Save sasa-All-atoms r2-score data")
tmp_score = pd.DataFrame(totalr2_1, dtype = 'object', columns = pdbid_list, index = cs_list)
tmp_average = pd.DataFrame(average_r2_1, dtype = 'object', columns = ['Average of r2-score'], index = cs_list)
tmp_std = pd.DataFrame(std_r2_1, dtype = 'object', columns = ['Std of r2-score'], index = cs_list)
tmp = pd.concat([tmp_score, tmp_average], axis=1)
all_r2 = pd.concat([tmp, tmp_std], axis=1)
all_r2.to_csv('sasa-All-atoms.csv', sep=' ')

## Save sasa-Total-Side r2-score data to a csv
print(f"[INFO]: Save sasa-Total-Side r2-score data")
tmp_score = pd.DataFrame(totalr2_2, dtype = 'object', columns = pdbid_list, index = cs_list)
tmp_average = pd.DataFrame(average_r2_2, dtype = 'object', columns = ['Average of r2-score'], index = cs_list)
tmp_std = pd.DataFrame(std_r2_2, dtype = 'object', columns = ['Std of r2-score'], index = cs_list)
tmp = pd.concat([tmp_score, tmp_average], axis=1)
all_r2 = pd.concat([tmp, tmp_std], axis=1)
all_r2.to_csv('sasa-Total-Side.csv', sep=' ')

## Save sasa-Main-Chain r2-score data to a csv
print(f"[INFO]: Save sasa-Main-Chain r2-score data")
tmp_score = pd.DataFrame(totalr2_3, dtype = 'object', columns = pdbid_list, index = cs_list)
tmp_average = pd.DataFrame(average_r2_3, dtype = 'object', columns = ['Average of r2-score'], index = cs_list)
tmp_std = pd.DataFrame(std_r2_3, dtype = 'object', columns = ['Std of r2-score'], index = cs_list)
tmp = pd.concat([tmp_score, tmp_average], axis=1)
all_r2 = pd.concat([tmp, tmp_std], axis=1)
all_r2.to_csv('sasa-Main-Chain.csv', sep=' ')

## Save sasa-Non-polar r2-score data to a csv
print(f"[INFO]: Save sasa-Non-polar r2-score data")
tmp_score = pd.DataFrame(totalr2_4, dtype = 'object', columns = pdbid_list, index = cs_list)
tmp_average = pd.DataFrame(average_r2_4, dtype = 'object', columns = ['Average of r2-score'], index = cs_list)
tmp_std = pd.DataFrame(std_r2_4, dtype = 'object', columns = ['Std of r2-score'], index = cs_list)
tmp = pd.concat([tmp_score, tmp_average], axis=1)
all_r2 = pd.concat([tmp, tmp_std], axis=1)
all_r2.to_csv('sasa-Non-polar.csv', sep=' ')

## Save sasa-All r2-score data to a csv
print(f"[INFO]: Save sasa-All r2-score data")
tmp_score = pd.DataFrame(totalr2_5, dtype = 'object', columns = pdbid_list, index = cs_list)
tmp_average = pd.DataFrame(average_r2_5, dtype = 'object', columns = ['Average of r2-score'], index = cs_list)
tmp_std = pd.DataFrame(std_r2_5, dtype = 'object', columns = ['Std of r2-score'], index = cs_list)
tmp = pd.concat([tmp_score, tmp_average], axis=1)
all_r2 = pd.concat([tmp, tmp_std], axis=1)
all_r2.to_csv('sasa-All.csv', sep=' ')


