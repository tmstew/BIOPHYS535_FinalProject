#!/bin/bash


################################################################
## Predict RNA base pair
## Amount of neighboring pair is set from 0 to 5
## LOO model
## sklearn:  clf = svm.SVC(gamma='auto')
################################################################

rm -r output/searching_neighbors
mkdir -p output/searching_neighbors

for i in `seq 0 5`; do
	echo $i > neighbors
	python bp_searching_neighbors.py > output/searching_neighbors/output_${i}
	echo $i
	done


