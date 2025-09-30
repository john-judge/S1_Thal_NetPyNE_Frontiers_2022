#!/bin/bash

# pull from Github
#git clone -4 https://github.com/john-judge/S1_Thal_NetPyNE_Frontiers_2022.git
cd S1_Thal_NetPyNE_Frontiers_2022
git pull
cd ..

# compress
tar -czf S1_Thal_NetPyNE_Frontiers_2022.tar.gz S1_Thal_NetPyNE_Frontiers_2022

cp S1_Thal_NetPyNE_Frontiers_2022.tar.gz /staging/jjudge3/


