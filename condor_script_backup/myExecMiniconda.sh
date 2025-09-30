#!/bin/bash
# My CHTC job
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job $1. Proceeding to run workload..."

# clone from Github
git clone https://github.com/john-judge/S1_Thal_NetPyNE_Frontiers_2022.git

# un-tar and move input to the repository subdirectory
cp /staging/jjudge3/in-silico-hVOS-env.tar.gz ./

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=in-silico-hVOS-env
# if you need the environment directory to be named something other than the environment name, change this line
export ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# Command for myprogram, which will use files from the working directory

cd S1_Thal_NetPyNE_Frontiers_2022
cd sim
nrnivmodl mod .
echo "Finished nrnivmodl. Running batch.py..."
python batch.py
#mpiexec -n 8 nrniv -python -mpi init.py
cd ..
cd ..

# Before the script exits, make sure to remove the file(s) from the working directory
#rm an171923_2012_06_04_data_struct.tar.gz
#rm Test_sim_Svoboda-judge_data.tar.gz
#rm "./RealisticBarrel/Input data/an171923_2012_06_04_data_struct.mat"

# tar output directory
tar -czvf S1-Thal-output.tar.gz "S1_Thal_NetPyNE_Frontiers_2022"
mv S1-Thal-output.tar.gz /staging/jjudge3/
