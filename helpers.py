### Imports
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

### Definitions 
 
start = time.time()
 
def get_img_dirs(experiment_directory, FOV, channel):
    img_dirs = glob.glob(str(experiment_directory)+"/*"+str(FOV)+"_T*_"+channel+"*")
    img_dirs.sort()
    return img_dirs
 
def crop_image(image, crop):
    return image[crop[0]:crop[1],crop[2]:crop[3]]
 
def erode(image, rounds):
    for x in range(rounds):
        image = mh.morph.erode(image)
    return image
 

def watershed_from_image(image):
    bin_image = threshold_niblack(image)
    bin_image = image > bin_image/0.8
    bin_image = mh.label(bin_image)[0]
 
 
    sizes = mh.labeled.labeled_size(bin_image)
    bin_image = mh.labeled.remove_regions_where(bin_image, sizes < 50)
    bin_image = (bin_image > 0) * 1
 
    distance = ndi.distance_transform_edt(bin_image)
 
    imagef = mh.gaussian_filter(image.astype(float), 2)
 
 
    maxima = mh.regmax(mh.stretch(imagef))
    maxima,_= mh.label(maxima)
 
 
    markers = erode(bin_image,1)
    markers = mh.label(markers+1)[0]
 
    labels = sk_watershed(-distance, maxima, watershed_line=1)
 
    watershed = labels * bin_image
    return watershed
def get_centroids_from_watershed(watershed):
    centroids = []
    coords = []
    num_cells = len(np.unique(watershed))
    for x in range(num_cells-1):
        centroids.append(skimage.measure.regionprops(watershed)[x]["centroid"])
        coords.append(skimage.measure.regionprops(watershed)[x]["coords"])
    return centroids, coords
 

def get_trenches_from_coords(centroids, coords):
    y_coords = []
    x_coords = []
    first_y_coord = []
    first_x_coord = []
    for x in range(len(centroids)):
        y_coords.append(centroids[x][0])
        x_coords.append(centroids[x][1])
        first_y_coord.append(coords[x][0][0])
        first_x_coord.append(coords[x][0][1])
    X = np.array(x_coords)
    X = np.reshape(X, (-1, 1))
 
 
    ms = MeanShift(bandwidth=20, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    trenches = []
    for x in range(len(y_coords)):
        trenches.append([labels[x], x_coords[x], y_coords[x], first_x_coord[x], first_y_coord[x]])
 
    trenches = pd.DataFrame(trenches)
    trenches.columns = ["trench_ID", "x", "y", "first_x", "first_y"]
    trenches = trenches.sort_values(by=["trench_ID"]).reset_index(drop=True)
    return trenches
 
def get_mother_from_coords(centroids, coords):
    trenches = get_trenches_from_coords(centroids, coords)
    trench_IDs = np.unique(np.array(trenches["trench_ID"]))
    mother_centroids = []
    mother_coordinate = []
    for z in range(len(trench_IDs)):
        mother_y = min(trenches[trenches["trench_ID"] == trench_IDs[z]]["y"])
        mother_x = trenches["x"][trenches["y"] == mother_y].iloc[0]
        mother_centroids.append([mother_x, mother_y])
 
        mother_first_x = trenches["first_x"][trenches["y"] == mother_y].iloc[0]
        mother_first_y = trenches["first_y"][trenches["y"] == mother_y].iloc[0]
        mother_coordinate.append([mother_first_x, mother_first_y])
    return mother_centroids, mother_coordinate
 
def get_mother_label_from_coordinate(coordinate, watershed):
    return watershed[coordinate[1], coordinate[0]]

def generate_grid():
    grid = []
    letters = "ABCDEFG"
    for x in range(len(letters)):
        for y in range(30):
            grid.append(letters[x]+str(y).zfill(2))
    return grid

def generate_grid_xy(number):
    grid = []
    prefix = "xy"
    for x in range(number):
        grid.append(prefix+str(x).zfill(2))
    return grid