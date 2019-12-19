## Import Module
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import io
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score


################################################################
## This is used to search the critical chemical shift for 
##	prediciting base pair.
## neighbors = 1 (best f1-score) : 0.9229 +/- 0.0851
## LOO model
## sklearn: clf = svm.SVC(gamma='auto')
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

## Creat chemical shift list
cs_list = list(cs_data.columns)
cs_list.remove('id')
cs_list.remove('resid')
cs_list.remove('resname')
cs_list.remove('class')
num = len(cs_list)

################################################################
## Global variables and functions
################################################################
NUMBER_CHEMICAL_SHIFT_TYPE = 18
neighbors = np.loadtxt("neighbors", dtype = int)
f1 = np.loadtxt("f1", dtype = float)
recall = np.loadtxt("recall", dtype = float)
precision = np.loadtxt("precision", dtype = float)

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

drop_names = ['id', 'class', 'resid']
target_name = 'class'
col = 2*neighbors + 1
totalscore = []
totalrecall = []
totalprecision = []

for atom in cs_list:
	print(f"[SET UP DATA]: The chemical shift dropped is {atom}")
	tmp_c = cs_data.drop(atom, axis=1)
	cs_all = get_cs_features_rna_all(tmp_c, neighbors=neighbors)
	score = []
	recall = []
	precision = []

	for pdbid in cs_all['id'].unique()[0 :]:
		print(f"[INFO]:  Now predict RNA --> {pdbid}")
	
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

		## Recall
		recall.append(recall_score(np.int_(testy), predicted))

		## Precision
		precision.append(precision_score(np.int_(testy), predicted))

		## f1-score
		score.append(metrics.f1_score(np.int_(testy),predicted))
		print(" ")

	## Total f1-score
	totalscore.append(score)
	totalrecall.append(recall)
	totalprecision.append(precision)
	print(f"[INFO]: Now appending recall to total recall")
	print(f"[INFO]: Now appending f1-score to total score")
	print(f"[INFO]: Now appending precision to total precision")
	print(" ") 
	print(" ") 

################################################################
## Prediction analysis 
################################################################

## Prepare data
totalscore = np.asarray(totalscore)
totalrecall = np.asarray(totalrecall)
totalprecision = np.asarray(totalprecision)

totalscore = totalscore.reshape(num,-1)
totalrecall = totalrecall.reshape(num,-1)
totalprecision = totalprecision.reshape(num,-1)

pdbid_list = cs_data['id'].unique()[0 :]
average_name = ['Average of f1 score']
std_name = ['Std of f1 score']

## Analyze prediction result
print(f"[ANALYSIS RESULT]: LOO model result")
i = 0
average_f1 = []
average_recall = []
average_precision = []
std_f1 = []
std_recall = []
std_precision = []

while i < num :
	score = totalscore[i,:]
	recall = totalrecall[i,:]
	precision = totalprecision[i,:]

	average_f1.append(np.average(score))
	average_recall.append(np.average(recall))
	average_precision.append(np.average(precision))

	std_f1.append(np.std(score))
	std_recall.append(np.std(recall))
	std_precision.append(np.std(precision))

	print(f"[INFO]: The chemical shift {cs_list[i]} is dropped -->") 
	print(f"[ANALYSIS RESULT]: The average f1-score is: {average_f1[i]} +/- {std_f1[i]}")
	print(f"[ANALYSIS RESULT]: The average recall is: {average_recall[i]} +/- {std_recall[i]}")
	print(f"[ANALYSIS RESULT]: The average precision is: {average_precision[i]} +/- {std_precision[i]}")
	print(" ") 
	i += 1

## Save f1-score data to a csv
print(f"[INFO]: Save f1-score data")
tmp_score = pd.DataFrame(totalscore, dtype = 'object', columns = pdbid_list, index = cs_list)
tmp_average = pd.DataFrame(average_f1, dtype = 'object', columns = ['Average of f1-score'], index = cs_list)
tmp_std = pd.DataFrame(std_f1, dtype = 'object', columns = ['Std of f1-score'], index = cs_list)
tmp = pd.concat([tmp_score, tmp_average], axis=1)
all_score = pd.concat([tmp, tmp_std], axis=1)
all_score.to_csv('all_score.csv', sep=' ')

## Save recall data to a csv
print(f"[INFO]: Save recall data")
tmp_recall = pd.DataFrame(totalrecall, dtype = 'object', columns = pdbid_list, index = cs_list)
tmp_average = pd.DataFrame(average_recall, dtype = 'object', columns = ['Average of recall'], index = cs_list)
tmp_std = pd.DataFrame(std_recall, dtype = 'object', columns = ['Std of recall'], index = cs_list)
tmp = pd.concat([tmp_score, tmp_average], axis=1)
all_recall = pd.concat([tmp, tmp_std], axis=1)
all_recall.to_csv('all_recall.csv', sep=' ')

## Save precision data to a csv
print(f"[INFO]: Save recall data")
tmp_recall = pd.DataFrame(totalprecision, dtype = 'object', columns = pdbid_list, index = cs_list)
tmp_average = pd.DataFrame(average_precision, dtype = 'object', columns = ['Average of precision'], index = cs_list)
tmp_std = pd.DataFrame(std_precision, dtype = 'object', columns = ['Std of precision'], index = cs_list)
tmp = pd.concat([tmp_score, tmp_average], axis=1)
all_precision = pd.concat([tmp, tmp_std], axis=1)
all_precision.to_csv('all_precision.csv', sep=' ')

## Plot heatmap
print(f"[INFO]: Plotting heatmap")

delta_score = totalscore - f1
delta_score = delta_score * 100
plt.figure(figsize = (30,15))
ax = sns.heatmap(delta_score, center = 0, xticklabels = pdbid_list, yticklabels = cs_list)
ax.set_title("f1 heatmap")
plt.savefig("f1.png")

delta_recall = totalrecall - recall
delta_recall = delta_recall * 100
plt.figure(figsize = (30,15))
ax = sns.heatmap(delta_recall, center = 0, xticklabels = pdbid_list, yticklabels = cs_list)
ax.set_title("recall heatmap")
plt.savefig("recall.png")

delta_precision = totalprecision - precision
delta_precision = delta_precision * 100
plt.figure(figsize = (30,15))
ax = sns.heatmap(delta_precision, center = 0, xticklabels = pdbid_list, yticklabels = cs_list)
ax.set_title("precision heatmap")
plt.savefig("precision.png")


