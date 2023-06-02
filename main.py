# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:44:51 2023

@author: domer
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from reconstruction import *

red_factor = 0.4

images_front, images_left, images_right = read_images('chess', red_factor=red_factor) 

num_pts = 54
num_img = len(images_left)

# Real Dimensions of our Checkboard
rows = 6 
columns = 9
scale = 55 # in mm

h,w = images_front[0].shape[:2]

#%% GET CHESS POINTS
# Gets the object points and image points:
#   obj : List of cells of calibration pattern points in the calibration pattern 
#          coordinate space [[[x,y,z], ..], ...].
#   crd : List of cells of the projections of calibration pattern points 
#          [[[x,y], ..], ...].

obj_pts_l = []
obj_pts_f = []
obj_pts_r = []

img_pts_l = []
img_pts_f = []
img_pts_r = []

corners_l = []
corners_f = []
corners_r = []

for i in range(num_img):
    print('Getting chess points in image ' + str(i))
    img_l = images_left[i]
    obj, crd = get_chess_points_(img_l, rows, columns, scale) 
    obj_pts_l.append(obj)
    img_pts_l.append(crd)
    corners_l.append(cv.drawChessboardCorners(img_l, (rows,columns), crd, True))
    
    img_f = images_front[i]
    obj, crd = get_chess_points_(img_f, rows, columns, scale) 
    obj_pts_f.append(obj)
    img_pts_f.append(crd)
    corners_f.append(cv.drawChessboardCorners(img_f, (rows,columns), crd, True))
    
    img_r = images_right[i]
    obj, crd = get_chess_points_(img_r, rows, columns, scale) 
    obj_pts_r.append(obj)
    img_pts_r.append(crd)
    corners_r.append(cv.drawChessboardCorners(img_r, (rows,columns), crd, True))

save_images(corners_l, './corners/Left')
save_images(corners_f, './corners/Front')
save_images(corners_r, './corners/Right')

#%% CAMERA CALIBRATION
# Gets the camera matrix, and distortion coefficients:
#   cm_  : 3x3 floating-point camera matrix = [fx 0 cx; 0 fy cy; 0 0 1].
#   dist_: Vector of distortion coefficients.

print('\nCalibrating cameras')
cm_l, dist_l = calibrate_individual_camera_(obj_pts_l, img_pts_l, w, h)
cm_f, dist_f = calibrate_individual_camera_(obj_pts_f, img_pts_f, w, h)
cm_r, dist_r = calibrate_individual_camera_(obj_pts_r, img_pts_r, w, h)


#%% FUNDAMENTAL MATRIX
# Gets the rotation matrix, translation vector, essential matrix and 
# fundamental matrix: 
#   R_: 3x3 rotation matrix between 1st and 2nd camera coordinate systems.
#   T_: 3x1 translation vector between the coordinate systems of the cameras.
#   E_: 3x3 essential matrix.
#   F_: 3x3 fundamental matrix.

print('\nCalculating fundamental matrix')
R_lf, T_lf, E_lf, F_lf = stereo_calibrate_(obj_pts_l, obj_pts_f, img_pts_l, img_pts_f, cm_l, cm_f, dist_l, dist_f, w, h)
R_fr, T_fr, E_fr, F_fr = stereo_calibrate_(obj_pts_f, obj_pts_r, img_pts_f, img_pts_r, cm_f, cm_r, dist_f, dist_r, w, h)
R_lr, T_lr, E_lr, F_lr = stereo_calibrate_(obj_pts_l, obj_pts_r, img_pts_l, img_pts_r, cm_l, cm_r, dist_l, dist_r, w, h)


#%% EPILINES
# Find the epilines corresponding to the chess board points in each camera
#   img: width x height image with corresponding epilines

print('\nDrawing epilines')
epilines_left = []
epilines_front = []
epilines_right = []

for i in range(len(images_left)):
    img = draw_epilines_(img_pts_l[i], img_pts_f[i], images_left[i], F_lf, 2)
    epilines_left.append(img)
    img = draw_epilines_(img_pts_f[i], img_pts_l[i], images_front[i], F_lf, 1)
    img = draw_epilines_(img_pts_f[i], img_pts_r[i], images_front[i], F_fr, 2)
    epilines_front.append(img)
    img = draw_epilines_(img_pts_r[i], img_pts_f[i], images_right[i], F_fr, 1)
    epilines_right.append(img)

save_images(epilines_left, './epilines/Left')
save_images(epilines_front, './epilines/Front')
save_images(epilines_right, './epilines/Right')


#%% HOMOGRAPHIC MATRIX
# Gets the homography matrices to rectify the images:
#   h_: 3x3 rectification homography matrix

print('\nCalculating homographic matrix')
h_l, h_f = calculate_homographic_(img_pts_l, img_pts_f, F_lf, w, h)
_, h_r = calculate_homographic_(img_pts_f, img_pts_r, F_fr, w, h)


#%% VALIDATION
# Gets the average euclidean distance between the image points (ground truth) 
# and the epiline intersections of the other two cameras:
#   d_: float value

print('\nValidation')
d_l, d_f, d_r = epilines_intersections(img_pts_l, img_pts_f, img_pts_r, F_lf, F_fr, F_lr)


#%% READ SCENE IMAGES

sc_f, sc_l, sc_r = read_images('scene', red_factor=red_factor)
num_img = len(sc_f)

#%% RECTIFICATION
# Rectify the images with the epilines drawed to verify the lines are now 
# paralel to the camera plane

print('\nRectifying images')
img_l_rect, img_f_rect = rectify_images_(images_left[0], images_front[0], h_l, h_f, w, h)
_, img_r_rect = rectify_images_(images_front[0], images_right[0], h_f, h_r, w, h)

save_images([img_l_rect, img_f_rect, img_r_rect], './rect/epilines', ['left', 'front', 'right'])


images_l_rect = []
images_f_rect = []
images_r_rect = []

for i in range(num_img):
    img_l = sc_l[i]
    img_f = sc_f[i]
    img_r = sc_r[i]
    
    img_l_rect, img_f_rect = rectify_images_(img_l, img_f, h_l, h_f, w, h)
    _, img_r_rect = rectify_images_(img_f, img_r, h_f, h_r, w, h)
    images_l_rect.append(img_l_rect)
    images_f_rect.append(img_f_rect)
    images_r_rect.append(img_r_rect)
    
save_images([images_l_rect[0], images_f_rect[0], images_r_rect[0]], './rect/all', ['left', 'front', 'right'])
save_images([images_l_rect[1], images_f_rect[1], images_r_rect[1]], './rect/lamp', ['left', 'front', 'right'])
save_images([images_l_rect[2], images_f_rect[2], images_r_rect[2]], './rect/office', ['left', 'front', 'right'])


#%% DISPARITY MAP
# Creates a disparity map between both images with the modified H. Hirschmuller 
# algorithm:
#   disparity_: width x height disparity map.

print('\nComputing disparity map')
disparity_l = compute_disparity_(images_l_rect[0], images_f_rect[0], 'left')
disparity_r = compute_disparity_(images_f_rect[0], images_r_rect[0], 'left')

save_images([disparity_l, disparity_r, (disparity_l+disparity_r)/2], './disparity', ['left', 'right', 'combined'])


#%% PERSPECTIVE TRANSFORMATION MATRIX
# Calculates the perspective transformation matrix Q to convert pixels with 
# disparity value into the corresponding [x, y, z]:
#   Q_: 4x4 perspective transformation matrix = 
#       [[1    0    0           -0.5width              ] 
#        [0    1    0           -0.5height             ] 
#        [0    0    0            focal-lenght          ] 
#        [0    0   -1/baseline   x_translation/baseline]]

print('\nCalculating depth')
Q_l = calculate_depth_(cm_l, T_lf, w, h)
Q_r = calculate_depth_(cm_f, T_fr, w, h)


#%% 3D RECOVERING
# Reprojects a disparity image to 3D space. Generates a ply file that contains
# a 3-channel floating-point image of the same size as disparity. Each element 
# of im3d(x,y,ch) contains 3D coordinates of the point (x,y) computed from the 
# disparity map.

print('\n3D Reconstruction')
reproject_stereo_3D_(sc_f[0], disparity_l, disparity_r, Q_l, Q_r)








