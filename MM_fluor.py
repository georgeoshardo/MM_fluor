from helpers import *
import time
import os
import glob
from PIL import Image, ImageEnhance
import numpy as np
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import mahotas as mh
from scipy import ndimage as ndi
from skimage.morphology import watershed as sk_watershed
import skimage
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
import pandas as pd
from joblib import Parallel, delayed
from skimage.measure import regionprops
from tqdm import tqdm

experiment_directory = "/home/camillo/Desktop/cm967/3n1_sfGFP/"
experiment_files = os.listdir(experiment_directory)
experiment_files.sort()



channels = ["GFP", "RFP"]
grid = generate_grid_xy(100)
FOVs = []
for x in range(len(grid)):
    if len(glob.glob(str(experiment_directory)+"/*"+str(grid[x])+"*")) > 0:
        FOVs.append(grid[x])
    else:
        pass


for f in range(len(FOVs)):
    if f == None: # If some FOVs are obscured
        pass
    else:
        all_cells = []
        image_dirs = get_img_dirs(experiment_directory, FOVs[f], "RFP")
        image_dirs_analyse = get_img_dirs(experiment_directory, FOVs[f], "GFP")
        image_dirs.sort()
        image_dirs_analyse.sort()
        def get_timelapse_mother_coordinate_centroid(i):
            image = Image.open(image_dirs[i])
            image = np.array(image)
            image = crop_image(image, [130, 480, 0, image.shape[1]])
            watershed_segmentation = watershed_from_image(image)
            cell_centroids, cell_coordinates = get_centroids_from_watershed(watershed_segmentation)
            mother_cell_centroids, mother_cell_coordinate = get_mother_from_coords(cell_centroids, cell_coordinates)
            #mother_cell_labels = []
            #for x in range(len(mother_cell_coordinate)):
            #    mother_cell_labels.append(get_mother_label_from_coordinate(mother_cell_coordinate[x], watershed_segmentation))
 
            ## A bit to return YFP channel
            image_GFP = Image.open(image_dirs_analyse[i])
            image_GFP = np.array(image_GFP)
            image_GFP = crop_image(image_GFP, [130, 480, 0, image.shape[1]])
 
            return mother_cell_coordinate, mother_cell_centroids, watershed_segmentation, image_GFP
 
 
 
        output = Parallel(n_jobs=-1)(delayed(get_timelapse_mother_coordinate_centroid)(i) for i in tqdm(range(len(image_dirs))))
 
        output = np.array(output)
 
 
        all_images = []
        for x in range(len(image_dirs)):
            all_images.append([output[x][2], output[x][3]])
 
        timelapse_mother_cell_coordinate = []
        timelapse_mother_cell_centroids = []
        for x in range(len(image_dirs)):
            timelapse_mother_cell_coordinate.append(output[x][0])
            timelapse_mother_cell_centroids.append(output[x][1])
 
        mother_cell_centroids_flattened = []
        for x in range(len(timelapse_mother_cell_centroids)):
            for y in range(len(timelapse_mother_cell_centroids[x])):
                mother_cell_centroids_flattened.append(timelapse_mother_cell_centroids[x][y])
        mother_cell_centroids_flattened = np.array(mother_cell_centroids_flattened)
 
        mother_cell_coordinate_flattened = []
        for x in range(len(timelapse_mother_cell_coordinate)):
            for y in range(len(timelapse_mother_cell_coordinate[x])):
                mother_cell_coordinate_flattened.append(timelapse_mother_cell_coordinate[x][y])
        mother_cell_coordinate_flattened = np.array(mother_cell_coordinate_flattened)
 
        X = mother_cell_centroids_flattened
        clustering = DBSCAN(eps=16, min_samples=5).fit(X)
        clustering.labels_
        clustering
 
 
        to_append = [X, mother_cell_coordinate_flattened, clustering.labels_, FOVs[f], "RFP", all_images]
 
        all_cells.append(to_append)

    all_cells = pd.DataFrame(all_cells)
    all_cells.columns = ["centroids", "coordinates", "mother_labels", "FOV", "channel", "images"]

    #go through each FOV
    all_intensity = []
    for ff in range(len(all_cells)):
        mother_labels = all_cells.loc[ff]["mother_labels"]
        mother_labels = np.unique(mother_labels)
        # go through mother by mother, iterating over all FOVs 
        FOV_intensity = []
        for m in range(len(mother_labels)):
            current_mother = mother_labels[m]
            mother_cell_coordinates = all_cells["coordinates"][ff][all_cells.loc[ff]["mother_labels"] == current_mother]
            intensity = []
            filled_area = []
            num_timepoints = len(all_cells["images"][ff])
            for t in range(num_timepoints):
                current_bin_image = all_cells["images"][ff][t][0]
                current_raw_image = all_cells["images"][ff][t][1]

                mother_cell_bin_label = []
                for x in range(len(mother_cell_coordinates)):
                    m_labels = current_bin_image[mother_cell_coordinates[x][1]][mother_cell_coordinates[x][0]]
                    if m_labels == 0:
                        pass
                    else:
                        mother_cell_bin_label.append(m_labels)
                mother_cell_bin_label = np.unique(np.array(mother_cell_bin_label))
                if len(mother_cell_bin_label) > 1:
                    pass
                    #print("We got an error")
                    #print(mother_cell_bin_label)
                elif len(mother_cell_bin_label) == 0:
                        mother_cell_bin_label = 0
                else:
                    mother_cell_bin_label = mother_cell_bin_label[0]
                    mother_cell_bin_image = (current_bin_image == mother_cell_bin_label) * 1
                    current_intensity = regionprops(label_image = mother_cell_bin_image, intensity_image = current_raw_image)[0]["mean_intensity"]
                    current_filled_area = regionprops(label_image = mother_cell_bin_image)[0]["filled_area"]
                    intensity.append(current_intensity)
                    filled_area.append(current_filled_area)
            FOV_intensity.append([mother_cell_bin_label, all_cells["FOV"][ff], intensity, filled_area])
        all_intensity.append(FOV_intensity)


    all_trajectories = pd.DataFrame()
    for x in range(len(all_intensity)):
        for y in range(len(all_intensity[x])):
            if len(all_intensity[x][y][2]) > 1:
                a = pd.DataFrame(all_intensity[x][y]).T
                all_trajectories = all_trajectories.append(a)

    all_trajectories.columns = ["mother cell bin label", "FOV", "timeseries intensity", "timeseries filled area"]
    all_trajectories.reset_index(inplace=True)
    all_trajectories.to_pickle("/home/georgeos/Documents/RFP_growth_{}.pkl".format(FOVs[f]))

print("Done")
end = time.time()
print(end - start)