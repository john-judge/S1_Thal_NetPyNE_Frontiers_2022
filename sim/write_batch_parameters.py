''' Write an import file for cfg.py to read to know which parameters to use for this simulation. 
    This file will be written to the directory where the simulation is run. 
    The file is called recordTraceBatchSettings.py and contains the following code: 
    record_trace_setting = {'compartment': 'soma', 'cell_num_start': 0, 'cell_num_end': 50}
    This will be run from myExecutable.sh and feed job id as command line arg.
    This script then assigns compartment, start_no (optional), and end_no (optional) based on job ID.
    '''
import sys
import os

n_compartment_ids_per_job = 10
n_jobs = 42 # (IDs 0-41)

if len(sys.argv) > 1:
    job_id = int(sys.argv[1])
job_id %= n_jobs  # make sure job_id is in range 0 to n_jobs-1

# soma: only 1 job: 
# axons: 1 job
# apic: <180 segments, 9 jobs (IDs 11-19)
# dend: <220 segments, 11 jobs (IDs 0-10)

job_id_to_task_map = {
    n_jobs: {'compartment': 'soma', 'cell_num_start': None, 'cell_num_end': None},
    n_jobs-1: {'compartment': 'axon', 'cell_num_start': 0, 'cell_num_end': 10},
}

# for apic_0 to apic_40, increments of n_compartment_ids_per_job=10
for i in range(0, 4):
    job_id_to_task_map[i] = {'compartment': 'apic', 'cell_num_start': i*10, 'cell_num_end': (i+1)*10}

# for dend_0 to dend_40, increments of n_compartment_ids_per_job=10
for i in range(4, 8):
    i_dend = i - 4
    job_id_to_task_map[i] = {'compartment': 'dend', 'cell_num_start': i_dend*10, 'cell_num_end': (i_dend+1)*10}

# for apic_40 to apic_100, increments of n_compartment_ids_per_job=20
for i in range(8,11):
    i_apic = i - 8
    job_id_to_task_map[i] = {'compartment': 'apic', 'cell_num_start': i_apic*20, 'cell_num_end': (i_apic+1)*20}

# for dend_40 to dend_100, increments of n_compartment_ids_per_job=20
for i in range(11, 14):
    i_dend = i - 11
    job_id_to_task_map[i] = {'compartment': 'dend', 'cell_num_start': i_dend*20, 'cell_num_end': (i_dend+1)*20}

# for apic_100 to apic_180, increments of n_compartment_ids_per_job=40
for i in range(14, 16):
    job_id_to_task_map[i] = {'compartment': 'apic', 'cell_num_start': (i-14)*40, 'cell_num_end': (i-13)*40}

# for dend_100 to dend_220, increments of n_compartment_ids_per_job=40
for i in range(16, 19):
    job_id_to_task_map[i] = {'compartment': 'dend', 'cell_num_start': (i-16)*40, 'cell_num_end': (i-15)*40}


task_selected = job_id_to_task_map[job_id]
compartment = task_selected['compartment']
cell_num_start = task_selected['cell_num_start']
cell_num_end = task_selected['cell_num_end']

output_file_name = compartment
if cell_num_end is not None:
    output_file_name += '_'+str(cell_num_end)
output_file_name = "S1-Thal-output-"+output_file_name + ".tar.gz"
# create the output file (empty file)
open("../../" + output_file_name, 'w').close()
print('Output file created:', output_file_name)


if os.path.exists('recordTraceBatchSettings.py'):
    os.remove('recordTraceBatchSettings.py')

# write to file
f = open('recordTraceBatchSettings.py', 'w')
f.write('record_trace_setting = {\'compartment\': \''+compartment+'\', \'cell_num_start\': '+str(cell_num_start)+', \'cell_num_end\': '+str(cell_num_end)+'}')
f.close()
print('recordTraceBatchSettings.py written. compartment:', compartment, 'cell_num_start:', cell_num_start, 'cell_num_end:', cell_num_end)
