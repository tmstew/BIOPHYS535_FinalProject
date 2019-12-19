#!/bin/bash


################################################################
## Predict RNA base pair
## Amount of neighboring pair is set from 0 to 5
## LOO model
## sklearn:  clf = svm.SVC(gamma='auto')
################################################################

rm -r output/searching_neighbors
rm -r output/searching_cs
mkdir -p output/searching_neighbors
mkdir -p output/searching_cs

for i in `seq 0 5`; do

	echo $i
	echo $i > neighbors
	python bp_searching_neighbors.py > output/searching_neighbors/output_${i}

	grep recall output/searching_neighbors/output_${i} | awk '{print $7}' > recall 
	grep f1-score output/searching_neighbors/output_${i} | awk '{print $7}' > f1 
	grep precision output/searching_neighbors/output_${i} | awk '{print $7}' > precision 

	mkdir -p output/searching_cs/$i
	python bp_searching_cs.py > output/searching_cs/$i/output
	mv f1 recall precision f1.png recall.png precision.png all* output/searching_cs/$i

	done


