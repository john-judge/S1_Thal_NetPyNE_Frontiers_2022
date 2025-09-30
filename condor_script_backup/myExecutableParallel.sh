#!/bin/bash
# My CHTC job
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job $1. Proceeding to run workload..."

# clone from Github
git clone https://github.com/john-judge/in-silico-hVOS.git

# un-tar and move input to the repository subdirectory
cp /staging/jjudge3/in-silico-hVOS-env-parallel.tar.gz ./
pwd

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=in-silico-hVOS-env-parallel
# if you need the environment directory to be named something other than the environment name, change this line
export ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# Command for myprogram, which will use files from the working directory

cd in-silico-hVOS
cp /staging/jjudge3/NMC_model.tar.gz ./
tar -xzf NMC_model.tar.gz

python network_script_parallel.py 8
cd ..

# Before the script exits, make sure to remove the file(s) from the working directory
#rm an171923_2012_06_04_data_struct.tar.gz
#rm Test_sim_Svoboda-judge_data.tar.gz
#rm "./RealisticBarrel/Input data/an171923_2012_06_04_data_struct.mat"

# tar output directory
tar -czvf in-silico-hVOS-parallel-output.tar.gz "in-silico-hVOS"
mv in-silico-hVOS-parallel-output.tar.gz /staging/jjudge3/
rm "in-silico-hVOS/*"
