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

for i in `seq 0 0`; do

	echo $i
	echo $i > neighbors
	mkdir -p output/searching_neighbors/$i
	python sasa_searching_neighbors.py > output/searching_neighbors/$i/output

	mkdir -p output/searching_cs/$i
	python sasa_searching_cs.py > output/searching_cs/$i/output

	mv r2.csv r2_score.png output/searching_neighbors/$i
	mv sasa-* output/searching_cs/$i

	done


