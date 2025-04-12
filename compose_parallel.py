""" After analyze_hVOS_parallel.py is run on the HTC clusters,
    the result are files output_dir_#.tar.gz files in the directory,
    'analyze_output' where # is the number of the job and also the 
    index of the Cell. 
    This script will extract the files and rename them"""
import os
import gc
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import imageio

from src.hVOS.camera import Camera

# load time
t_steps = 501
delta_t = 0.1
time = np.arange(0, t_steps * delta_t, delta_t)

# input: expects a directory 'analyze_output' with the output_dir_#.tar.gz files
data_dir = '../analyze_output/model_rec_final'
output_dir = '../composed_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
run_id = 2
sparsity = 1.0
should_re_extract = True  # set to True if you want to re-extract the data into existing folders

############################################
# create a 2-d gaussian PSF with a sigma of 1.5 pixels
############################################
psf_2d = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        psf_2d[i, j] = np.exp(-((i - 3) ** 2 + (j - 3) ** 2) / (2 * 1.5 ** 2))

# normalize the PSF
psf_2d /= np.sum(psf_2d)

################################################
# Extract the tar.gz file (result from optical model)
################################################
output_dir_dict = {}
for file in os.listdir(data_dir):
    if file.endswith('.tar.gz'):
        
        output_dir = data_dir + file[:-7] + '/'
        print(data_dir + file)
        if not os.path.exists(output_dir) or should_re_extract:
            if not os.path.exists(output_dir):
                print("\tCreating directory:", output_dir)
                os.makedirs(output_dir)
            result = subprocess.run(['tar', '-xzvf', data_dir + file, "-C", output_dir], 
                                    capture_output=True, text=True, check=True)
        i_output = int(file.replace(".tar.gz", "").split("_")[-1])
        output_dir_dict[i_output] = output_dir

################################################
# map files and collect data into composed arrays of all cells
################################################
comparts = ['axon', 'dend', 'soma', 'apic']
all_cells_rec = {
    p: { 
        c: {
            'syn': None,
            'spk': None,
        } for c in comparts
    } for p in ['no_psf', 'psf']
}

for i_output in output_dir_dict.keys():
    output_dir = output_dir_dict[i_output] + "run" + str(run_id) + "/model_rec_final/"
    if not os.path.exists(output_dir):
        continue
    for file in os.listdir(output_dir):
        if file.endswith('.npy'):
            # open numpy memmap file 
            file_path = output_dir + file
            print(file_path)
            arr = np.memmap(file_path, dtype='float32', mode='r').reshape(-1, 300, 300)

            # use the file name to determine the compartment and type of data
            # e.g. no_psf_cell_8406-syn_rec_dend.npy
            # e.g. psf_cell_8406-spk_rec_soma.npy
            # e.g. psf_cell_8406-syn_rec_apic.npy
            compart_type = file.split("_")[-1].replace(".npy", "")
            psf_type = file.split("_")[0]
            psf_type = 'no_psf' if psf_type == 'no' else 'psf'
            activity_type = file.split("_")[-3]
            activity_type = activity_type.split("-")[1]
            cell_id = file.split("_")[1].split("-")[0]

            print(compart_type, psf_type, activity_type, cell_id)

            if all_cells_rec[psf_type][compart_type][activity_type] is None:
                all_cells_rec[psf_type][compart_type][activity_type] = \
                    np.zeros(arr.shape, dtype='float32')
            
            all_cells_rec[psf_type][compart_type][activity_type] += arr

            del arr
            gc.collect()



###########################################
# Show each result in all_cells_rec and save in output_dir
###########################################
# finally, show each of the results in all_cells_rec
for psf_type in all_cells_rec.keys():
    composed_arr = None
    for compart_type in all_cells_rec[psf_type].keys():
        for activity_type in all_cells_rec[psf_type][compart_type].keys():
            arr = all_cells_rec[psf_type][compart_type][activity_type]
            if arr is None:
                continue
            if composed_arr is None:
                composed_arr = np.zeros(arr.shape, dtype='float32')
            composed_arr += arr

            if psf_type == 'psf':
                plt.clf()
                plt.imshow(-arr[0, :, :], cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f"{psf_type} {compart_type} {activity_type}")
                plt.savefig(output_dir + f"{psf_type}_{compart_type}_{activity_type}_psf.png")
            elif psf_type == 'no_psf':
                # show it side-by-side with the image blurred with the PSF
                # blur the image with the PSF
                blurred_arr = np.zeros(arr.shape, dtype='float32')
                for i in range(arr.shape[0]):
                    blurred_arr[i, :, :] = signal.convolve2d(arr[i, :, :], psf_2d, mode='same')
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.imshow(-arr[0, :, :], cmap='hot', interpolation='nearest')
                plt.title(f"{psf_type} {compart_type} {activity_type}")
                plt.subplot(1, 2, 2)
                plt.imshow(-blurred_arr[0, :, :], cmap='hot', interpolation='nearest')
                plt.title(f"{psf_type} {compart_type} {activity_type} blurred")
                plt.savefig(output_dir + f"{psf_type}_{compart_type}_{activity_type}_blurred.png")

                # save the blurred image

    # show composed_arr
    plt.clf()
    plt.imshow(-composed_arr[0, :, :], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(output_dir + f"{psf_type}_composed.png")

    if psf_type == 'no_psf':
        blurred_composed_arr = np.zeros(composed_arr.shape, dtype='float32')
        for i in range(composed_arr.shape[0]):
            blurred_composed_arr[i, :, :] = signal.convolve2d(composed_arr[i, :, :], psf_2d, mode='same')
        plt.clf()
        plt.imshow(-blurred_composed_arr[0, :, :], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.savefig(output_dir + f"blurred_composed.png")


###############################################
# make each array into an animated gif using Camera.animate_frames_to_video
###############################################
cam = Camera([], [], time, init_dummy=True)
final_arr =  {}
for psf_type in all_cells_rec.keys():
    for compart_type in all_cells_rec[psf_type].keys():
        for activity_type in all_cells_rec[psf_type][compart_type].keys():
            arr = all_cells_rec[psf_type][compart_type][activity_type]
            gif_filename = output_dir + f"{psf_type}_{compart_type}_{activity_type}.gif"
            if arr is None:
                continue
            cam.animate_frames_to_video(arr, gif_filename, frames=(0, 501))

            if psf_type == 'no_psf':
                # blur the image with the PSF
                blurred_arr = np.zeros(arr.shape, dtype='float32')
                for i in range(arr.shape[0]):
                    blurred_arr[i, :, :] = signal.convolve2d(arr[i, :, :], psf_2d, mode='same')
                cam.animate_frames_to_video(blurred_arr, gif_filename.replace('.gif', '_blurred.gif'), frames=(0, 501))

            # store the arr
            if psf_type not in final_arr:
                final_arr[psf_type] = {}
            if compart_type not in final_arr[psf_type]:
                final_arr[psf_type][compart_type] = {}
            if activity_type not in final_arr[psf_type][compart_type]:
                final_arr[psf_type][compart_type][activity_type] = {}
            final_arr[psf_type][compart_type][activity_type] = arr

            if psf_type == 'no_psf':
                if 'blurred_arr' not in final_arr:
                    final_arr['blurred_arr'] = {}
                if compart_type not in final_arr['blurred_arr']:
                    final_arr['blurred_arr'][compart_type] = {}
                if activity_type not in final_arr['blurred_arr'][compart_type]:
                    final_arr['blurred_arr'][compart_type][activity_type] = {}

                final_arr['blurred_arr'][compart_type][activity_type] = blurred_arr

            

###############################################
# build final composed as well
###############################################
final_composed_arr = {}
for psf_type in final_arr.keys():
    if psf_type not in final_composed_arr:
        final_composed_arr[psf_type] = None
    for compart_type in final_arr[psf_type].keys():
        for activity_type in final_arr[psf_type][compart_type].keys():
            arr = final_arr[psf_type][compart_type][activity_type]
            if final_composed_arr[psf_type] is None:
                final_composed_arr[psf_type] = np.zeros(arr.shape, dtype='float32')
            final_composed_arr[psf_type] += arr


###############################################
# build 4 hisograms: # pixels for % compartment contribution for each compartment
# for each of the 4 compartments
# 1. get the max amp for each compartment
# 2. get the sum of the max amps for each compartment
# 3. get the % contribution for each compartment
# 4. plot the histogram for each compartment
# 5. save the histogram as a png file
###############################################
t_stride = 10
for psf_type in final_arr.keys():
    #filter
    if psf_type != 'blurred_arr':
        continue
    images = []
    for t in range(0, len(time), t_stride):
        print("processing t = ", t)
        plt.clf()
        plt.figure(figsize=(10, 6))
        for compart_type in final_arr[psf_type].keys():
            for activity_type in final_arr[psf_type][compart_type]:
                arr = final_arr[psf_type][compart_type][activity_type]
                if arr is None:
                    continue
                as_fraction_arr = arr / final_composed_arr[psf_type]
            
                # discard elements that are not in the range (0, 1]
                as_fraction_arr_t = as_fraction_arr[t, :, :]
                as_fraction_arr_t = as_fraction_arr_t[as_fraction_arr_t > 0]
                as_fraction_arr_t = as_fraction_arr_t[as_fraction_arr_t <= 1]
                # get the histogram of the % contribution for each compartment

            plt.hist(as_fraction_arr_t.flatten(), bins=100, 
                     label=compart_type, histtype='step',
                     linewidth=2,)
        plt.title(f" Contributions of each compartment to pixel (t = {time[t]} ms)")
        plt.xlabel("percent contribution")
        plt.legend()
        plt.ylabel("number of pixels")
        # make log y-scale
        plt.yscale('log')
        filename = f"{psf_type}_{compart_type}_percent_contribution_t_" + str(t) + ".png"
        plt.savefig(output_dir + filename)
        plt.close()

        images.append(imageio.imread(filename))
    imageio.mimsave(output_dir + f"{psf_type}_{compart_type}_percent_contribution.gif", images)
    
        