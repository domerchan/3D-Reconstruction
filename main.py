# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:44:51 2023

@author: domer
"""

from reconstruction import *

num_pts = 54
num_img = 6
red_factor = 0.4

# Set the name of the folder that contains the images (must have internal Left, Front, Right folders)
images_front, images_left, images_right = read_images('chess', red_factor=red_factor) 

h,w = images_front[0].shape[:2]

# Gets the object points, image points, camera matrix, and distortion 
# coefficients for left, front and right images:
#   op_  : List of cells of calibration pattern points in the calibration pattern 
#          coordinate space [[[x,y,z], ..], ...].
#   ip_  : List of cells of the projections of calibration pattern points 
#          [[[x,y], ..], ...]
#   m_   : 3x3 floating-point camera matrix = [fx 0 cx; 0 fy cy; 0 0 1]
#   dist_: Vector of distortion coefficients
op_l, op_f, op_r, ip_l, ip_f, ip_r, m_l, m_f, m_r, dist_l, dist_f, dist_r = get_chess_points(images_left, images_front, images_right)


# Gets the rotation matrix, translation vector, essential matrix and fundamental
# matrix fot he left, front and right images: 

r_lf, t_lf, e_lf, f_lf, r_fr, t_fr, e_fr, f_fr = stereo_calibrate(op_l, op_f, op_r, ip_l, ip_f, ip_r, m_l, m_f, m_r, dist_l, dist_f, dist_r, w, h)

draw_epilines(ip_l, ip_f, ip_r, images_left, images_front, images_right, f_lf, f_fr)

h_l, h_fl, h_fr, h_r = calculate_homographic(ip_l, ip_f, ip_r, f_lf, f_fr, w, h)

# Rectify the images with the epilines drawed to verify the lines are now paralel to the camera plane
#l_rect, f_rect, r_rect = rectify_images(images_left[0], images_front[0], images_right[0], h_l, h_fl, h_fr, h_r, w, h, 'epilines')

#ip_l_rect, ip_f_rect, ip_r_rect = get_corners(l_rect, f_rect, r_rect)

# Next we will work with the scene images
sc_f, sc_l, sc_r = read_images('scene', red_factor=red_factor)

img_l_rect, img_f_rect, img_r_rect = rectify_images(sc_l[0], sc_f[0], sc_r[0], h_l, h_fl, h_fr, h_r, w, h, 'all_lights')

#disparity_l, disparity_r = compute_disparity(img_l_rect, img_f_rect, img_r_rect)

disparity_l = compute_disparity2(img_l_rect, img_f_rect, 'left')
disparity_r = compute_disparity2(img_f_rect, img_r_rect, 'right')

#depth_l, depth_r, R1_l, P1_l, R1_r, P1_r= depth_mapping2(m_l, dist_l, m_fl, dist_fl, m_r, dist_r, w, h, r_lf, r_fr, t_lf, t_fr)

depth_l, depth_r = calculate_depth(m_l, m_r, t_lf, t_fr, w, h)

#disparity_l_inv = inverse_rectify(disparity_l, m_l, dist_l, R1_l, P1_l, w, h)
#disparity_r_inv = inverse_rectify(disparity_r, m_r, dist_r, R1_r, P1_r, w, h)

reprojection3D_multi(sc_f[0], disparity_l, disparity_r, depth_l, depth_r)
