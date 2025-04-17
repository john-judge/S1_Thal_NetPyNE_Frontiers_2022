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
import shutil
import random

from src.hVOS.camera import Camera
from cam_params import cam_params

# load time
t_max = 999
delta_t = 0.1
time = np.arange(0, t_max * delta_t, delta_t)
cam_width = cam_params['cam_width']
cam_height = cam_params['cam_height']

# input: expects a directory 'analyze_output' with the output_dir_#.tar.gz files
data_dir = '../analyze_output/'
output_dir = '../composed_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
run_id = 2
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
# map files and collect data into composed arrays of all cells
# Delete files as possible to free up disk space
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
biological_sparsity = 1.0  # when sparsity == 1.0, the biological sparsity is 0.6
sparsity_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
sparsity_arr = [np.zeros((cam_width, cam_height), dtype='float32') 
                    for _ in range(len(sparsity_range))]
five_soma_masks = []  # for later analysis
output_dir_dict = {}
for file in os.listdir(data_dir):
    if file.endswith('.tar.gz'):
        i_output = file.replace(".tar.gz", "").split("_")[-1]
        if i_output == '':
            continue
        i_output = int(i_output)
        
        output_dir_extract = data_dir + file[:-7] + '/'
        print(data_dir + file)
        if not os.path.exists(output_dir_extract) or should_re_extract:
            if not os.path.exists(output_dir_extract):
                print("\tCreating directory:", output_dir_extract)
                os.makedirs(output_dir_extract)
            result = subprocess.run(['tar', '-xzvf', data_dir + file, "-C", output_dir_extract], 
                                    capture_output=True, text=True, check=True)
        
        input_dirs = []
        for subdir in os.listdir(output_dir_extract + 'analyze_output/model_rec_final/'):
            print(subdir)
            if 'cell' in subdir:
                cell_id = 'cell_' + subdir.split('cell_')[1]

                output_dir_dict[cell_id] = output_dir_extract + subdir + '/'
                input_dirs.append(output_dir_dict[cell_id])

        ################################################
        for input_dir in input_dirs:
            if not os.path.exists(input_dir):
                print(f"input_dir {input_dir} does not exist. Skipping.")
                continue

            # roll the sparsity dice for this cell. rand float between 0 and 1
            sparsity_dice_rolls = [
                random.random() < sparse for sparse in sparsity_range
            ]
            for file in os.listdir(input_dir):
                if file.endswith('.npy'):
                    # open numpy memmap file 
                    file_path = input_dir + file
                    print(file_path)
                    arr = np.memmap(file_path, dtype='float32', mode='r').reshape(-1, cam_width, cam_height)

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

                    # sparsity data collection
                    for i_sp in range(len(sparsity_range)):
                        if sparsity_dice_rolls[i_sp]:
                            sparsity_arr[i_sp] += np.max(arr, axis=0)

                    print(compart_type, psf_type, activity_type, cell_id)

                    if all_cells_rec[psf_type][compart_type][activity_type] is None:
                        all_cells_rec[psf_type][compart_type][activity_type] = \
                            np.zeros(arr.shape, dtype='float32')
                    
                    all_cells_rec[psf_type][compart_type][activity_type] += arr

                    if len(five_soma_masks) < 5:
                        if compart_type == 'soma' and activity_type == 'syn' and psf_type == 'no_psf':
                            if np.sum(-arr) > 0:
                                # take the first 5 masks
                                five_soma_masks.append(np.max(arr, axis=0))
                    del arr
                    gc.collect()
            
            # delete the input_dir to free up space
            try:
                shutil.rmtree(input_dir)
            except Exception as e:
                print(f"Error deleting {input_dir}: {e}")

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

    if composed_arr is None:
        continue
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
            cam.animate_frames_to_video(arr, gif_filename, frames=(0, t_max),
                                        flip=True)

            if psf_type == 'no_psf':
                # blur the image with the PSF
                blurred_arr = np.zeros(arr.shape, dtype='float32')
                for i in range(arr.shape[0]):
                    blurred_arr[i, :, :] = signal.convolve2d(arr[i, :, :], psf_2d, mode='same')
                cam.animate_frames_to_video(blurred_arr, gif_filename.replace('.gif', '_blurred.gif'), 
                                            frames=(0, t_max), flip=True)

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

##################################################
# make a gif of the final composed arr (blurred)
###############################################
cam.animate_frames_to_video(final_composed_arr['blurred_arr'], 
    output_dir + "final_composed_blurred.gif", frames=(0, t_max),
    flip=True)

###################################################
# choose random ROIs in the final_composed_arr blurred_arr image 
# and plot their optical traces over time (png)
# Also show location of the ROIs overlaid in the image
###################################################
n_rois = 10
roi_diameter_range = [1, 15]  # size of the ROIs, they are distributed uniformly in size
# matplotlib figure
plt.clf()
plt.figure(figsize=(10, 6))
# show image in left subplot
plt.subplot(1, 2, 1)
plt.imshow(-final_composed_arr['blurred_arr'][0, :, :], 
                cmap='gray', interpolation='nearest')
rois = []
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k',
            'orange', 'purple', 'pink']
for i_roi in range(n_rois):
    roi_diameter = np.random.randint(roi_diameter_range[0], roi_diameter_range[1])

    # random location
    roi_x = np.random.randint(roi_diameter//2, 
                final_composed_arr['blurred_arr'].shape[1] - roi_diameter//2)
    roi_y = np.random.randint(roi_diameter//2,
                final_composed_arr['blurred_arr'].shape[2] - roi_diameter//2)

    roi = [[roi_x, roi_y]]
    # add the roi_size nearest pixels to the roi by spiraling outward until limit reached
    if roi_diameter > 1:
        for i in range(max(0, i-roi_diameter//2), 
                        min(final_composed_arr['blurred_arr'].shape[1], i+roi_diameter//2)):
            for j in range(max(0, j-roi_diameter//2),
                            min(final_composed_arr['blurred_arr'].shape[2], j+roi_diameter//2)):
                if np.sqrt((i - roi_x)**2 + (j - roi_y)^2) <= roi_diameter // 2:
                    if not (i == roi_x and j == roi_y):
                        roi.append([i, j])
    
    # for each pixel in roi, shade the pixel in the image
    for pixel in roi:
        plt.scatter(pixel[0], pixel[1], color=colors[i_roi % len(colors)], s=5,
                    alpha=0.25)
    rois.append(roi)

# now plot the optical trace for each roi
last_headspace = 0
plt.subplot(1, 2, 2)
for i_roi in range(n_rois):
    roi = rois[i_roi]
    roi_trace = None
    for px in roi:
        if roi_trace is None:
            roi_trace = final_composed_arr['blurred_arr'][:, px[0], px[1]]
        else:
            roi_trace += final_composed_arr['blurred_arr'][:, px[0], px[1]]

    roi_trace /= len(roi)
    plt.plot(time, roi_trace + last_headspace, 
                color=colors[i_roi % len(colors)])

    last_headspace += roi_trace.max() * 1.1

plt.xlabel("Time (ms)")
plt.ylabel("Optical Trace")
plt.savefig(output_dir + "roi_traces.png")

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
                as_fraction_arr[final_composed_arr[psf_type] == 0] = 0

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

        images.append(imageio.imread(output_dir + filename))
    imageio.mimsave(output_dir + f"{psf_type}_{compart_type}_percent_contribution.gif", images)
    
###################################################
# Soma location optical traces versus non-soma
################################################
# use five_soma_masks for non-soma rois
# use all-soma no-psf composition for a mask of all non-soma areas
# and then plot the optical traces for each cell including only pixels inside the mask
# and save the image

all_non_soma = (final_arr['no_psf']['soma']['syn'] == 0)[0, :, :]  # True if not soma
avg_soma_size = np.mean([mask.sum() for mask in five_soma_masks])
avg_soma_diameter = np.sqrt(avg_soma_size) / np.pi

five_non_soma_masks = []
for i_non_soma in range(50):
    five_non_soma_masks.append(np.zeros(all_non_soma.shape, dtype=bool))
    # sample a random point from all_non_soma
    non_soma_x = np.random.randint(0, all_non_soma.shape[0])
    non_soma_y = np.random.randint(0, all_non_soma.shape[1])

    for i in range(max(0, int(non_soma_x - avg_soma_diameter // 2)), 
                        min(all_non_soma.shape[1], int(non_soma_x + avg_soma_diameter // 2))):
        for j in range(max(0, int(non_soma_y - avg_soma_diameter // 2)),
                        min(all_non_soma.shape[2], int(non_soma_y + avg_soma_diameter // 2))):
            if abs(i - non_soma_x) + abs(j - non_soma_y) <= avg_soma_diameter // 2:
                if not (i == non_soma_x and j == non_soma_y):
                    if all_non_soma[i, j]:
                        five_non_soma_masks[i_non_soma][i, j] = True
                        all_non_soma[i, j] = True

plt.clf()
plt.figure(figsize=(10, 6))
# show image in left subplot
plt.subplot(1, 2, 1)
plt.imshow(-final_composed_arr['blurred_arr'][0, :, :], 
                cmap='gray', interpolation='nearest')

# plot the rois in the image
for i_roi in range(min(5, len(five_non_soma_masks))):
    roi = five_non_soma_masks[i_roi]
    # for each pixel in roi, shade the pixel in the image
    for pixel in roi:
        plt.scatter(pixel[0], pixel[1], color=colors[i_roi % len(colors)], s=5,
                    alpha=0.25)

# now plot the optical trace for each roi
last_headspace = 0
plt.subplot(1, 2, 2)
leg_handles = []
# plot non-soma traces
for i_soma, five_masks in enumerate([five_non_soma_masks, five_soma_masks]):
    label = "Non-soma"
    color = 'tab:orange'
    l1 = None
    if i_soma == 0:
        label = "Soma"
        color = 'tab:green'
    for i_roi in range(len(five_non_soma_masks)):
        roi_mask = five_non_soma_masks[i_roi]
        roi_trace = None
        for i in range(roi_mask.shape[0]):
            for j in range(roi_mask.shape[1]):
                if roi_mask[i, j]:
                    if roi_trace is None:
                        roi_trace = final_composed_arr['blurred_arr'][:, i, j]
                    else:
                        roi_trace += final_composed_arr['blurred_arr'][:, i, j]
        if roi_trace is None:
            continue
        roi_trace /= len(roi)
        l1 = plt.plot(time, roi_trace + last_headspace, 
                    color=color,
                    label=label)

        last_headspace += roi_trace.max() * 1.1
    if l1 is not None:
        leg_handles.append(l1[0])


plt.legend(handles=leg_handles, loc='upper right')
plt.xlabel("Time (ms)")
plt.ylabel("Optical Trace")
plt.savefig(output_dir + "dend_v_soma_roi_traces.png")

##################################################
# Sparsity analysis: single-cell crosstalk
###############################################
# convolve each of the sparse images
for i_sp in range(len(sparsity_range)):
    sparsity_arr[i_sp][:, :] = signal.convolve2d(sparsity_arr[i_sp][:, :], psf_2d, mode='same')
crosstalk_fractions = [[] for _ in range(len(sparsity_range))]
for i_sp, sparsity in enumerate(sparsity_range):
    # reconduct the same compose analysis but only sampling SPARSITY % of the cells

    # use the five_soma_masks to analyze the crosstalk
    # at each soma roi, get the fraction of the signal that is from the cell
    # whose soma resides there divided by the total signal from all cells.
    for soma_max in five_soma_masks:
        # get the signal from all cells in the soma mask
        soma_mask = (soma_max != 0)
        all_signal = sparsity_arr[i_sp] * soma_mask
        # get the signal from the cell whose soma resides there
        cell_soma_signal = -signal.convolve2d(
                    soma_max, 
                    psf_2d, mode='same')
        # get the crosstalk fraction
        crosstalk_fraction = np.sum(cell_soma_signal) / np.sum(all_signal)
        crosstalk_fractions[i_sp].append(1-crosstalk_fraction)

bio_sparsity = [biological_sparsity * sp for sp in sparsity_range]
print(crosstalk_fractions)
# plot average crosstalk fraction verssus sparsity
plt.clf()
plt.figure(figsize=(10, 6))
# scatter plot with error bars
crosstalk_fractions_std = [np.std(crosstalk) for crosstalk in crosstalk_fractions]
crosstalk_fractions = [np.mean(crosstalk) for crosstalk in crosstalk_fractions]

plt.errorbar(bio_sparsity, crosstalk_fractions, yerr=crosstalk_fractions_std, fmt='o')

plt.xlabel("Sparsity")
plt.ylabel("Average Crosstalk Fraction at Soma location")
plt.savefig(output_dir + "crosstalk_sparsity.png")
print(crosstalk_fractions)
print(crosstalk_fractions_std)