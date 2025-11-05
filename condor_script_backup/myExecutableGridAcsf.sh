#!/bin/bash
# My CHTC job
# print a 'hello' message to the job's terminal output:
echo "Hello CHTC from Job $1. Proceeding to run workload..."

# clone from Github

# un-tar and move input to the repository subdirectory
#cp /staging/jjudge3/in-silico-hVOS-env.tar.gz ./

# have job exit if any command returns with non-zero exit status (aka failure)
#set -e

# replace env-name on the right hand side of this line with the name of your conda environment
#ENVNAME=in-silico-hVOS-env
# if you need the environment directory to be named something other than the environment name, change this line
#export ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
#export PATH
#mkdir $ENVDIR
#tar -xzf $ENVNAME.tar.gz -C $ENVDIR
#. $ENVDIR/bin/activate

# Command for myprogram, which will use files from the working directory
cp /staging/jjudge3/NMC_model.tar.gz ./
tar -xvsf NMC_model.tar.gz

cp /staging/jjudge3/S1_Thal_NetPyNE_Frontiers_2022.tar.gz ./
tar -xvsf S1_Thal_NetPyNE_Frontiers_2022.tar.gz
rm S1_Thal_NetPyNE_Frontiers_2022.tar.gz
#git clone -4 https://github.com/john-judge/S1_Thal_NetPyNE_Frontiers_2022.git
#cp /staging/jjudge3/S1_Thal_NetPyNE_Frontiers_2022.tar.gz ./
#tar -xvsf S1_Thal_NetPyNE_Frontiers_2022.tar.gz


cd S1_Thal_NetPyNE_Frontiers_2022
git pull

# continue tuning from saved file
#cp /staging/jjudge3/tune.tar.gz ./
#tar -xvf tune.tar.gz

cd sim
nrnivmodl mod .
echo "Finished nrnivmodl. Running grid-acsf.py..."
#python write_batch_parameters.py "$1"
python grid_acsf.py "$1"
python grid_acsf_analysis.py "$1"
#mpiexec -n 8 nrniv -python -mpi init.py
#mpiexec -n 8 nrniv -python -mpi init.py
cd ..
cd ..

# Before the script exits, make sure to remove the file(s) from the working directory
#rm an171923_2012_06_04_data_struct.tar.gz
#rm Test_sim_Svoboda-judge_data.tar.gz
#rm "./RealisticBarrel/Input data/an171923_2012_06_04_data_struct.mat"

# tar output directory
#tar -czvf tune.tar.gz --exclude="S1_Thal_NetPyNE_Frontiers_2022/data/optuna_tuning/gen*" S1_Thal_NetPyNE_Frontiers_2022/data
tar -czvf "grid${1}.tar.gz" --include="*all_cells_rec_acsf_trial*" "S1_Thal_NetPyNE_Frontiers_2022/data"
mv "grid${1}.tar.gz" /staging/jjudge3/

tar -czvf "grid_acsf_map${1}.tar.gz" "grid_acsf_map.pkl"
mv "grid_acsf_map${1}.tar.gz" /staging/jjudge3/


