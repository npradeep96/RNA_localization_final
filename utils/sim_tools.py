#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules are used to

    1. Nucleate seed of protein condensates
    2. Create circular meshes
    3. Read out radius of protein condensate
"""
import numpy as np
from fipy import Gmsh2D
import cv2 as cv
from sets import Set
import math
import pandas as pd
import itertools as it
from mpl_toolkits import mplot3d



from sets import Set


class condensateProperties(object):
    """Class to calculate k-means clusters in mesh"""
    
    # Instantiate
    def __init__(self, data, phi_p, k, dx, protein_threshold = 0.3, distance_threshold=1, max_iters=1000, seed=0):
        
    
        self.phi_p = phi_p
        self.data = data
        self.data_thresh = self.data.cellCenters.value[:, self.phi_p.value > protein_threshold] 
        self.k = k
        self.dx = dx
        self.dimension = self.data_thresh.shape[0]
        self.distance_threshold = distance_threshold
        self.protein_threshold = protein_threshold
        self.overlap = True
        self.convergence = False
        self.labels = None
        self.areas = []
        self.vacuoles = []
        self.max_iters = max_iters
        np.random.seed(seed)
        
        
    def myPerms(self):
    # find permutations of a 
        f_k=math.factorial(self.k)
        A=np.empty((self.k,f_k))
        for i,perm in enumerate(it.permutations(range(self.k))):
            A[:,i] = perm
        return A
    
    def isBetween(self, a, b, c, epsilon=1.):
        """Function to check if point c lies between a and b"""
        aligned_cross = np.cross(b-a, c-a)
        dotproduct = np.dot(b-a, c-a)
        ab_length = np.linalg.norm(b-a)

        if abs(aligned_cross.sum()) > epsilon:
            return False
        if dotproduct < 0.:        
            return False
        if dotproduct < ab_length:
            return False

        return True
    
    # Initialize random points
    def initialize(self):
        if self.data_thresh.shape[1] == 0:
            self.convergence = True
            self.overlap = False
            return None
        idx = np.random.choice(self.data_thresh.shape[1], self.k, replace=False)
        return self.data_thresh[:, idx].T
    
    # Calculate distance matrix
    def calculate_distances(self):
        for i, centroid in enumerate(self.selected_centroids):
            dist = np.linalg.norm(centroid -  self.data_thresh.T, axis=1)
            if i == 0:
                distances = dist
            else:
                distances = np.vstack([distances, dist])    
        return distances
    
    def calculate_centroids(self, labels):
        subsets = []
        for l in np.unique(labels):
            subset_l = np.argwhere(labels == l).flatten()
            subset_l = self.data_thresh[:,subset_l]
            subsets.append(subset_l.mean(axis=1))
            
        if np.all(self.selected_centroids == np.array(subsets)):
            self.convergence = True
            
        self.selected_centroids = np.array(subsets)
            
    def check_overlap(self):
        self.overlap = False
        # from provided centroids, check whether two coordinates are close such that it is one body
        rows_ = self.myPerms()[:2].T.astype(int)
        # iterate over selected columns and check distances
        
        for row in rows_:
            selected_rows = self.distances[row]
            difference = np.abs(selected_rows[0] - selected_rows[1]) # calculate difference
            sum_ = np.sum(difference < self.distance_threshold)
            if sum_ > 0: # how many values are below the threshold
                self.overlap = True
                break
    
    def perform_kmeans(self):
        while self.overlap == True: # reduce K 
            self.convergence = False # reset convergence to false
            self.k -= 1 # reduce number of centroids
            self.selected_centroids = self.initialize() # randomize centroids
            if self.convergence == True:
                break
            i = 0
            while self.convergence == False and i < self.max_iters:
                
                # print('selected_centroid shape {}'.format(self.selected_centroids.shape))
                # calculate distances between centroids and datapoints
                self.distances = self.calculate_distances()
                # print('distances shape {}'.format(self.distances.shape))

                # calculate nearest cluster
                if self.k != 1:
                    labels_ = np.argmin(self.distances,axis=0)
                else:
                    labels_ = np.zeros(len(self.distances))
                

                self.labels = labels_
                # recalculate clusters
                self.calculate_centroids(self.labels)
                i += 1
                            
            # if only one centroid is left we don't need to check overlap
            if self.k == 1:
                break
            
            # check overlap
            self.check_overlap()
            
            
    def retrieveAreas(self):
        if self.labels is None:
            self.areas = [0]
            self.selected_centroids = [np.nan]
        else:
            self.areas = pd.value_counts(self.labels).values * self.dx ** self.dimension
        
    def retrieveVacuoles(self):
        
        if np.sum(self.areas) == 0:
            return [np.nan]
        
        idx_vacuoles = []
        
        # iterate over centroids
        for i, centroid in enumerate(self.selected_centroids):
            
            idx_vacuole = [] # will contain all indexes that belong within a vacuole
            idx_protein = np.argwhere(self.phi_p > self.protein_threshold).flatten()
            idx_condensate = np.argwhere(self.labels == i).flatten()
            idx_vicinity = [] # will contain indexs that are nearby the condensate
            
            # retrieve the distance of the protein farthest away from the centroid
            if self.k != 1:
                distance_max = np.max(self.distances[i][idx_condensate])
                idx_large_distance = np.argsort(self.distances[i][idx_condensate])[-200:]
            else:
                distance_max = np.max(self.distances[idx_condensate])
                idx_large_distance = np.argsort(self.distances[idx_condensate])[-200:]
            
            dist = np.linalg.norm(self.data.cellCenters.value.T - centroid, axis=1)
            idx_vicinity = np.argwhere(dist <= distance_max).flatten()
            
            

            # select indexes that are in vicinity but not in idx_protein
            idx_vacuole_candidate = Set(list(idx_vicinity)) - Set(list(idx_protein))
            coords = self.data_thresh.T[idx_large_distance] # coordinates of a single condensate
            vecfunc = np.vectorize(self.isBetween)
            
            idx_vacuoles.append(idx_vacuole_candidate)
            #print(len(idx_vacuole_candidate))
            #for j_ in idx_vacuole_candidate:
            #    if j_ not in idx_vacuole:
            #        for i_, coord in enumerate(coords):
            #            if self.isBetween(coord, centroid, self.data.cellCenters.value.T[j_]):
            #                idx_vacuole.append(j_)
            #        
            #                    
            #idx_vacuoles.append(idx_vacuole)
        
        self.vacuoles = [len(x) * self.dx ** self.dimension for x in idx_vacuoles]

    def retrieveProperties(self):
        self.retrieveAreas()
        self.retrieveVacuoles()
    
    def calculateDistancetoSource(self, sourceCoordinates):
        
        if np.isnan(np.sum(self.selected_centroids)):
            dists_ = [np.nan]
            return dists_
        
        dists_ = []
        for sourceC in sourceCoordinates:
            d_ = np.linalg.norm(self.selected_centroids - sourceC, axis=1)
            dists_.append(d_)
            
        return dists_
            
            
        
        
        
def center_and_eccentricity(M, nx):
    """Given the dictionary of image moments extracted from a contour,
    calculate the centroid of the object and 
    eccentricity of the shape"""
    xc_mesh, yc_mesh = nx / 2, nx / 2
    
    try:
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        e = np.abs(((M['mu20'] - M['mu02']) ** 2 - 4 * M['mu11'] ** 2) / (M['mu20'] + M['mu02']) ** 2)
    except ZeroDivisionError:
        cx, cy, e = np.nan, np.nan, np.nan
    return (cx - xc_mesh, cy - yc_mesh), e




def fetch_areas(image_src, contours, nx, protein_limit, coordinates):
        
    
    filled_areas = []
    areas = []
    vacuole_area = []
    vacuole_num = 0

    # iterate over contours and fill objects
    for c in contours:
        img = np.zeros((nx,nx))
        c_reshaped = c.reshape(-1,2)
        p = cv.fillPoly(img, pts =[c_reshaped], color=(1))
        filled_areas.append(p)
        areas.append(np.sum(p))

    # if no contours, ciao
    if len(filled_areas) == 0:
        return 0, [np.nan], np.nan, vacuole_num, vacuole_area

    # Remove stupid contours
    for i, f in enumerate(filled_areas):
        # contour area is too small to consider legit
        if np.sum(f) < 2:
            del filled_areas[i]

    # Check again if the filtering actually removed all the contours... 
    if len(filled_areas) == 0:
        return 0, [np.nan], np.nan, vacuole_num, vacuole_area
    
    ################################
    # If there are multiple contours
    if len(filled_areas) > 1:

        # Calculate overlap matrix
        overlap_matrix = np.zeros((len(filled_areas), len(filled_areas)))
        for i, a in enumerate(filled_areas):
            for i2, a2 in enumerate(filled_areas):
                area_overlap = np.sum(a * a2)
                overlap_matrix[i,i2] = area_overlap
                
        print('OVERLAP MATRIX')
        print(overlap_matrix)
                
        # Find vacuoles and record their areas
        vacuole_idx = []
        for i, row in enumerate(overlap_matrix):
            for j, col in enumerate(row):
                
                # skip if it is a diagonal element 
                if i == j: 
                    continue
                    
                if col == overlap_matrix[j,j]:
                    vacuole_area.append(col)
                    vacuole_idx.append(j)
                    
        # remove vacuoles from filled areas
        for idx in sorted(np.unique(vacuole_idx), reverse=True):
            print(idx)
            del filled_areas[idx]
                
        
        

        filtered_areas = []
        distance_to_source_list = []
        eccentricity_list = []
        

        for f in filled_areas:
            # get back contour from filled area
            ret, thresh = cv.threshold(f, protein_limit, 1,0)
            thresh = np.uint8(thresh)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            contours = contours[0]
            
            # calculate image moments
            image_moments = cv.moments(contours)
            centers_, eccentricity_ = center_and_eccentricity(image_moments, nx) # retrieve centroid and eccentricity
            
            print('CALCULATED CENTROID')
            print(centers_)
            
            if centers_[0] == np.nan:
                continue
            
            # for each rna source calculate the distance to the center of the protein condensate
            dist_list = []
            for c_ in coordinates:
                dist = np.linalg.norm(np.array(c_) - np.array(centers_))
                print('DISTANCE')
                print(dist)
                dist_list.append(dist)
            
            distance_to_source_list.append(dist_list)
            
            calculated_sum = np.sum(f * image_src > protein_limit)
            filtered_areas.append(calculated_sum)
            eccentricity_list.append(eccentricity_)
        
        try:
            selected_index = np.argmax(filtered_areas) # retrieve the index of the biggest area
            filled_areas = filtered_areas[selected_index]
            distance_to_source = distance_to_source_list[selected_index]
            eccentricity = eccentricity_list[selected_index]
            vacuole_num = len(vacuole_area)
            
        except:
            print('EXCEPTION RAISED')
            pass
#             print('FILLED AREA')
#             print(filled_areas)
#             print('\nSELECTED INDEX and FILTERED AREA')
#             print(selected_index) 
#             print(filtered_areas)


    #######################################        
    # If there was only one contour, calculate the sum of its area when multiplied by the source
    elif len(filled_areas) == 1:
        
        # get back contour from filled area
        ret, thresh = cv.threshold(filled_areas[0], protein_limit, 1,0)
        thresh = np.uint8(thresh)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours = contours[0]
        
        filled_areas = np.sum(filled_areas[0] * image_src > protein_limit)
        
        # calculate image moments
        image_moments = cv.moments(contours)
        centers_, eccentricity = center_and_eccentricity(image_moments, nx) # retrieve centroid and eccentricity
        
        if eccentricity == np.nan:
            return filled_areas, [np.nan], np.nan, vacuole_num, vacuole_area

        # for each rna source calculate the distance to the center of the protein condensate
        distance_to_source = []
        if isinstance(coordinates, list):
            for c_ in coordinates:
                dist = np.linalg.norm(np.array(c_) - np.array(centers_))
                distance_to_source.append(dist)            
                    
        
    #######################################
    # If there is no contour
    else:
        filled_areas = 0.
        distance_to_source = [np.nan]
        eccentricity = np.nan
        
    return filled_areas, distance_to_source, eccentricity, vacuole_num, vacuole_area


def retrieve_condensate_properties(image_src, nx, coordinates, protein_limit=0.3):
    """Function to return list of unique protein condensate areas
    Args:
    image_src: numpy Array. 
    protein_limit: float. Corresponds to the minimum quantity of protein to be considered"""
    
    ret, thresh = cv.threshold(image_src, protein_limit, 1,0)
    thresh = np.uint8(thresh)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    
    filled_areas, distance_to_source, eccentricity, vacuole_num, vacuole_areas = fetch_areas(image_src, contours, nx, protein_limit, coordinates)
    
    return filled_areas, distance_to_source, eccentricity, vacuole_num, vacuole_areas

def get_radius(phi,mesh,threshold=0.25,dimension=2):
    """
    **Input**

    -   phi     =   Phase-field variable
    -   mesh    =   Mesh variable
    -   threshold   =   Threshold value of phase-field variable to count as dense phase
    -   dimension   =   Dimension of grid

    **Output**

        Returns radius
    """
    if dimension==2:
        Area = np.sum(mesh.cellVolumes[np.where(phi.value>threshold)[0]]);
        R = (np.sqrt(Area/np.pi))
    elif dimension==3:
        V = np.sum(len(np.where(phi.value>threshold)[0]));
        R = (np.power(3*V/(4*np.pi),1/3.0));
    return(R)

def nucleate_seed(mesh,phi_a,phia_value,nucleus_size=5.0,dimension=2, location=(0,0)):
    """
    Function nucleates spherical nucleus of condensate into mesh

    **Input**

    -   phi_a   =   Phase-field variable
    -   mesh    =   Mesh variable
    -   phia_value  =   Value of dense phase to nucleate
    -   nucleus_size   =   Radius of initial nucleus
    -   dimension   =   Dimension of grid
    """
    a=(mesh.cellCenters)

    xc = (min(mesh.x) + max(mesh.x))*0.5
    yc = (min(mesh.y) + max(mesh.y))*0.5
    
    # modify xc and yc with location coords
    xc += location[0]
    yc += location[1]
    
    if dimension==3:
        zc = (min(mesh.z) + max(mesh.z))*0.5;

    for i in np.arange(a.shape[1]):
        if dimension==2:
            dist = np.sqrt((a.value[0][i]-xc)**2 + (a.value[1][i]-yc)**2)
        elif dimension==3:
            dist = np.sqrt((a.value[0][i]-xc)**2 + (a.value[1][i]-yc)**2 + (a.value[2][i]-zc)**2)

        if (dist<=nucleus_size):
            phi_a.value[i] = phia_value

def create_circular_mesh(radius,cellSize):
    """
    Function creates circular 2D mesh

    **Input**

    -   radius   =   Radius of mesh
    -   cellSize    =   Size of unit cell

    *Note* : No support for 3D meshes currently and **requires GMSH**
    """

    mesh = Gmsh2D('''
                     cellSize = %g;
                     radius = %g;
                     Point(1) = {0, 0, 0, cellSize};
                     Point(2) = {-radius, 0, 0, cellSize};
                     Point(3) = {0, radius, 0, cellSize};
                     Point(4) = {radius, 0, 0, cellSize};
                     Point(5) = {0, -radius, 0, cellSize};
                     Circle(6) = {2, 1, 3};


                    Circle(7) = {3, 1, 4};
                    Circle(8) = {4, 1, 5};
                    Circle(9) = {5, 1, 2};
                    Line Loop(10) = {6, 7, 8, 9};
                    Plane Surface(11) = {10};

       '''%(cellSize,radius)) # doctest: +GMSH


    return(mesh)
