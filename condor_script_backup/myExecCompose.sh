#!/bin/bash
# My CHTC job
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job $1. Proceeding to run workload..."

# clone from Github

# un-tar and move input to the repository subdirectory
cp /staging/jjudge3/in-silico-hVOS-env.tar.gz ./
cp /staging/jjudge3/S1_Thal_NetPyNE_Frontiers_2022.tar.gz ./
mkdir analyze_output
cd analyze_output
cp /staging/jjudge3/output_dir_*.tar.gz ./
cd ..
tar -xvsf S1_Thal_NetPyNE_Frontiers_2022.tar.gz

# have job exit if any command returns with non-zero exit status (aka failure)
#set -e

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

mkdir mpl_writeable/
export MPLCONFIGDIR="mpl_writeable/"
#git clone -4 https://github.com/john-judge/S1_Thal_NetPyNE_Frontiers_2022.git

cd S1_Thal_NetPyNE_Frontiers_2022
python compose_parallel.py
cd ..

tar -czvf composed_output.tar.gz composed_output
cp composed_output.tar.gz /staging/jjudge3/

