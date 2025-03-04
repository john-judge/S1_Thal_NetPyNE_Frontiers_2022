''' Write an import file for cfg.py to read to know which parameters to use for this simulation. 
    This file will be written to the directory where the simulation is run. 
    The file is called recordTraceBatchSettings.py and contains the following code: 
    record_trace_setting = {'compartment': 'soma', 'cell_num_start': 0, 'cell_num_end': 50}
    This will be run from myExecutable.sh and feed job id as command line arg.
    This script then assigns compartment, start_no (optional), and end_no (optional) based on job ID.
    '''
import sys
import os

n_compartment_ids_per_job = 50
max_compartment_id = 500
n_jobs = (max_compartment_id // n_compartment_ids_per_job)* 3 + 1  # all somas can go on 1 job

if len(sys.argv) > 1:
    job_id = int(sys.argv[1])

compartment = 'soma'
cell_num_start = None
cell_num_end = None
job_id %= n_jobs

if job_id != 0:
    if job_id % 3 == 0:
        compartment = 'axon'
    elif job_id % 3 == 1:
        compartment = 'dend'
    else:
        compartment = 'apic'
    cell_num_start = (job_id // 3) * n_compartment_ids_per_job
    if job_id % 3 == 0:
        cell_num_start -= n_compartment_ids_per_job  # axon jobs are off by 1 since job_id = 0 is soma
    cell_num_end = cell_num_start + n_compartment_ids_per_job

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
