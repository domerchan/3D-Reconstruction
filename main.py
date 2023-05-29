# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:44:51 2023

@author: domer
"""

from reconstruction import *

num_pts = 54
num_img = 6
red_factor = 0.4

# Reads the calibration images
images_front, images_left, images_right = read_images('chess', red_factor=red_factor) # <- Set the name of the folder that contains the images (must have internal Left, Front, Right folders)

h,w = images_front[0].shape[:2]


#%% CALIBRATION

# Gets the object points, image points, camera matrix, and distortion 
# coefficients for left, front and right images:
#   op_  : List of cells of calibration pattern points in the calibration pattern 
#          coordinate space [[[x,y,z], ..], ...].
#   ip_  : List of cells of the projections of calibration pattern points 
#          [[[x,y], ..], ...].
#   m_   : 3x3 floating-point camera matrix = [fx 0 cx; 0 fy cy; 0 0 1].
#   dist_: Vector of distortion coefficients.
op_l, op_f, op_r, ip_l, ip_f, ip_r, m_l, m_f, m_r, dist_l, dist_f, dist_r = get_chess_points(images_left, images_front, images_right)


# Gets the rotation matrix, translation vector, essential matrix and fundamental
# matrix fot he left, front and right images: 
#   r_: 3x3 rotation matrix between 1st and 2nd camera coordinate systems.
#   t_: 3x1 translation vector between the coordinate systems of the cameras.
#   e_: 3x3 essential matrix.
#   f_: 3x3 fundamental matrix.
r_lf, t_lf, e_lf, f_lf, r_fr, t_fr, e_fr, f_fr = stereo_calibrate(op_l, op_f, op_r, ip_l, ip_f, ip_r, m_l, m_f, m_r, dist_l, dist_f, dist_r, w, h)


# Find the epilines corresponding to the chess board points in each camera and
# save them in the epiline/ folder.
draw_epilines(ip_l, ip_f, ip_r, images_left, images_front, images_right, f_lf, f_fr)

# Gets the homography matrices to rectify the images:
#   h_: 3x3 rectification homography matrix
h_l, h_fl, h_fr, h_r = calculate_homographic(ip_l, ip_f, ip_r, f_lf, f_fr, w, h)



#%% RECTIFICATION 

# Rectify the images with the epilines drawed to verify the lines are now 
# paralel to the camera plane
l_rect, f_rect, r_rect = rectify_images(images_left[0], images_front[0], images_right[0], h_l, h_fl, h_fr, h_r, w, h, 'epilines')


# Reads the scene images
sc_f, sc_l, sc_r = read_images('scene', red_factor=red_factor)


# Rectify the scene images
img_l_rect, img_f_rect, img_r_rect = rectify_images(sc_l[0], sc_f[0], sc_r[0], h_l, h_fl, h_fr, h_r, w, h, 'all_lights')


# Creates a disparity map between both images with the modified H. Hirschmuller 
# algorithm:
#   disparity_: width x height disparity map.
disparity_l = compute_disparity(img_l_rect, img_f_rect, 'left')
disparity_r = compute_disparity(img_f_rect, img_r_rect, 'right')


# Calculates the perspective transformation matrix Q to convert pixels with 
# disparity value into the corresponding [x, y, z]:
#   Q_: 4x4 perspective transformation matrix = 
#       [[1    0    0           -0.5width              ] 
#        [0    1    0           -0.5height             ] 
#        [0    0    0            focal-lenght          ] 
#        [0    0   -1/baseline   x_translation/baseline]]
Q_l, Q_r = calculate_depth(m_l, m_r, t_lf, t_fr, w, h)


# Reprojects a disparity image to 3D space. Generates a ply file that contains
# a 3-channel floating-point image of the same size as disparity. Each element 
# of im3d(x,y,ch) contains 3D coordinates of the point (x,y) computed from the 
# disparity map.
reprojection3D_multi(sc_f[0], disparity_l, disparity_r, Q_l, Q_r)
