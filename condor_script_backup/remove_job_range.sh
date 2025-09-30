#!/bin/bash

echo "Removing jobs $2 to $3 non-inclusive from cluster $1"

for ((i = ${2} ; i < ${3} ; i++ )); 
do
	condor_rm ${1}.${i};
done


