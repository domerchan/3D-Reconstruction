# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:43:01 2023

@author: domer
"""

# opencv contrib package needed:
# pip uninstall opencv-contrib-python opencv-python
# pip install opencv-contrib-python

from matplotlib import pyplot as plt
import cv2 as cv
import glob
import numpy as np
import math
import os
import copy
import sys

#################### Do not change next values
num_pts = 54
num_img = 6
img_size = (3000,2700)
####################

def read_images(path, red_factor = 1):
    print('\nReading images in ' + path + '...\n')
    
    if red_factor > 1:
        print('-- ERROR: reduction factor can not be higher than 1, setting to 1 instead..\n')
        red_factor = 1
        
    images_left = []
    for imname in sorted(glob.glob('./' + path + '/Left/*')):
        im = cv.imread(imname, 1)
        im = im[1500:4200, 2000:5000]
        im = cv.resize(im, tuple(np.multiply(img_size,red_factor).astype(int)))
        images_left.append(im)    

    images_front = []
    for imname in sorted(glob.glob('./' + path + '/Front/*')):
        im = cv.imread(imname, 1)
        im = im[1500:4200, 1500:4500]
        im = cv.resize(im, tuple(np.multiply(img_size,red_factor).astype(int)))
        images_front.append(im)    

    images_right = []
    for imname in sorted(glob.glob('./' + path + '/Right/*')):
        im = cv.imread(imname, 1)
        im = im[1000:3700, 1000:4000]
        im = cv.resize(im, tuple(np.multiply(img_size,red_factor).astype(int)))
        images_right.append(im)  
    
    return images_front, images_left, images_right



def get_chess_points(img_l, img_f, img_r):
    print('\nGetting chess board coordinates...\n')
    # Using cv Functions (Criteria) to detect Checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Real Dimensions of our Checkboard
    rows = 6 
    columns = 9
    scale = 55 # in mm
    
    # World Space 
    world = np.zeros((rows*columns,3), np.float32)
    world[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    world = scale* world
     
    # Checkerboard Coordinates
    img_pts_l = [] # Points on Image Plane (2D)
    obj_pts_l = [] # Points in World Space (3D)
    img_pts_f = []
    obj_pts_f = []
    img_pts_r = []
    obj_pts_r = []
    
    for i in range(len(img_l)):
        frame_l = img_l[i]
        frame_f = img_f[i]
        frame_r = img_r[i]
        gray_l = cv.cvtColor(frame_l, cv.COLOR_BGR2GRAY)
        gray_f = cv.cvtColor(frame_f, cv.COLOR_BGR2GRAY)
        gray_r = cv.cvtColor(frame_r, cv.COLOR_BGR2GRAY)
     
        # Locate Checkerboard
        ret_l, corners_l = cv.findChessboardCorners(gray_l, (rows, columns), None)
        ret_f, corners_f = cv.findChessboardCorners(gray_f, (rows, columns), None)
        ret_r, corners_r = cv.findChessboardCorners(gray_r, (rows, columns), None)
     
        if ret_l != True or ret_f != True or ret_r != True:
            print('\n -- ERROR: we could not find all the chess board coordinates in image ' + str(i+1) + '...')
            print(' -- try with better quality images (increase reduction factor)..')
            sys.exit()
        else:
            print('Saving coordinates for image ' + str(i+1) + '...\n')
            # Convolution for Corner Detection 
            conv_size = (4, 4)
             
            # CV Criteria trys Optimizes Corner Detection
            corners_l = cv.cornerSubPix(gray_l, corners_l, conv_size, (-1, -1), criteria)
            corners_f = cv.cornerSubPix(gray_f, corners_f, conv_size, (-1, -1), criteria)
            corners_r = cv.cornerSubPix(gray_r, corners_r, conv_size, (-1, -1), criteria)
            obj_pts_l.append(world)
            img_pts_l.append(corners_l)
            obj_pts_f.append(world)
            img_pts_f.append(corners_f)
            obj_pts_r.append(world)
            img_pts_r.append(corners_r)
            
            if not os.path.exists('./corners'):
                os.makedirs('./corners/Left')
                os.makedirs('./corners/Front')
                os.makedirs('./corners/Right')
                
            img = cv.drawChessboardCorners(frame_l, (rows,columns), corners_l, True)            
            cv.imwrite('./corners/Left/' + str(i) + '.jpg', img)
            img = cv.drawChessboardCorners(frame_f, (rows,columns), corners_f, True)            
            cv.imwrite('./corners/Front/' + str(i) + '.jpg', img)
            img = cv.drawChessboardCorners(frame_r, (rows,columns), corners_r, True)            
            cv.imwrite('./corners/Right/' + str(i) + '.jpg', img)
            
    print('\nCalibrating cameras...\n')
    ret, m_l, dist_l, rvecs, tvecs = cv.calibrateCamera(obj_pts_l, img_pts_l, gray_l.shape[::-1], None, None)
    ret, m_f, dist_f, rvecs, tvecs = cv.calibrateCamera(obj_pts_f, img_pts_f, gray_f.shape[::-1], None, None)
    ret, m_r, dist_r, rvecs, tvecs = cv.calibrateCamera(obj_pts_r, img_pts_r, gray_r.shape[::-1], None, None)
    
    return obj_pts_l, obj_pts_f, obj_pts_r, img_pts_l, img_pts_f, img_pts_r, m_l, m_f, m_r, dist_l, dist_f, dist_r



def stereo_calibrate(obj_pts_l, obj_pts_f, obj_pts_r, img_pts_l, img_pts_f, img_pts_r, m_l, m_f, m_r, dist_l, dist_f, dist_r, w, h):
    print('\nCalibrating cameras in pairs...\n')
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    
    print('left-front...')
    ret, cam_mat_l, dist_l, cam_mat_fl, dist_fl, r_lf, t_lf, e_lf, f_lf = cv.stereoCalibrate(obj_pts_l, 
                                                                                             img_pts_l, 
                                                                                             img_pts_f, 
                                                                                             m_l, 
                                                                                             dist_l, 
                                                                                             m_f, 
                                                                                             dist_f, 
                                                                                             (w,h),
                                                                                             criteria = criteria, 
                                                                                             flags = stereocalibration_flags)
    print('front-right')
    ret, cam_mat_fr, dist_fr, cam_mat_r, dist_r, r_fr, t_fr, e_fr, f_fr = cv.stereoCalibrate(obj_pts_f, 
                                                                                             img_pts_f, 
                                                                                             img_pts_r, 
                                                                                             m_f, 
                                                                                             dist_f, 
                                                                                             m_r, 
                                                                                             dist_r, 
                                                                                             (w,h),
                                                                                             criteria = criteria, 
                                                                                             flags = stereocalibration_flags)
    
    print('left-right')
    ret, aux, aux, aux, aux, aux, aux, aux, f_lr = cv.stereoCalibrate(obj_pts_l, 
                                                                        img_pts_l, 
                                                                        img_pts_r, 
                                                                        m_l, 
                                                                        dist_l, 
                                                                        m_r, 
                                                                        dist_r, 
                                                                        (w,h),
                                                                        criteria = criteria, 
                                                                        flags = stereocalibration_flags)
    
    return r_lf, t_lf, e_lf, f_lf, r_fr, t_fr, e_fr, f_fr, f_lr



def drawlines(img, lines, pts):
    im = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    r, c = im.shape
    
    for r, pt in zip(lines, pts.astype(int)):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1]])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img_out = cv.line(img, (x0,y0), (x1,y1), color, 1)
        img_out = cv.circle(img_out, tuple(pt), 5, color, -1)
        
    return img_out



def draw_epilines(pts_l, pts_f, pts_r, img_l, img_f, img_r, fund_l, fund_r):
    print('\nSaving epilines...\n')
    
    if not os.path.exists('./epilines'):
        os.makedirs('./epilines/Left')
        os.makedirs('./epilines/Front')
        os.makedirs('./epilines/Right')
        
    for i in range(len(img_l)):
        print('image ' + str(i+1) + '...\n')
        im_l = img_l[i]
        im_f = img_f[i]
        im_r = img_r[i]
        
        # Find the epilines corresponding to the chess board points in the
        # front image and drawing them in the left image
        lines = cv.computeCorrespondEpilines(pts_f[i].reshape(-1,1,2), 2, fund_l)
        lines = lines.reshape(-1,3)
        im = drawlines(im_l, lines, pts_l[i].reshape(num_pts,2))
        cv.imwrite('./epilines/Left/' + str(i) + '.jpg', im)
        
        # Find the epilines in the left image and drawing them in the front one
        lines = cv.computeCorrespondEpilines(pts_l[i].reshape(-1,1,2), 1, fund_l)
        lines = lines.reshape(-1,3) 
        im = drawlines(im_f, lines, pts_f[i].reshape(num_pts,2))
        cv.imwrite('./epilines/Front/' + str(i) + '.jpg', im)
        # Find the epilines in the right image and drawing them in the front one
        lines = cv.computeCorrespondEpilines(pts_r[i].reshape(-1,1,2), 2, fund_r)
        lines = lines.reshape(-1,3) 
        im = drawlines(im_f, lines, pts_f[i].reshape(num_pts,2))
        cv.imwrite('./epilines/Front/' + str(i+num_img) + '.jpg', im)
        
        # Find the epilines in the front image and drawing them in the right one
        lines = cv.computeCorrespondEpilines(pts_f[i].reshape(-1,1,2), 1, fund_r)
        lines = lines.reshape(-1,3)
        im = drawlines(im_r, lines, pts_r[i].reshape(num_pts,2))
        cv.imwrite('./epilines/Right/' + str(i) + '.jpg', im)



def calculate_homographic(pts_l, pts_f, pts_r, fund_l, fund_r, width, height):
    print('\nCalculating homographic matrices...\n')
    pts = num_pts * num_img
    pts_l = np.int32(pts_l).reshape(pts,2)
    pts_f = np.int32(pts_f).reshape(pts,2)
    pts_r = np.int32(pts_r).reshape(pts,2)
    
    _, h_l, h_f = cv.stereoRectifyUncalibrated(pts_l, pts_f, fund_l, (width, height))
    _, h_f2, h_r = cv.stereoRectifyUncalibrated(pts_f, pts_r, fund_r, (width, height))
    return h_l, h_f, h_f2, h_r



def epilines_intersections(pts_l, pts_f, pts_r, fund_l, fund_r, fund_lr):
    print('\Comparing epilines...\n')
    n_images = len(pts_l)
    distance_r = 0
    distance_f = 0
    distance_l = 0
    for i in range(n_images):
        print('image ' + str(i+1) + '...\n')
        # Find epilines intersection in the right image from left and front
        # points and compare with points in the right image        
        distance_r += intersect(pts_r[i],
                              cv.computeCorrespondEpilines(pts_f[i].reshape(-1,1,2), 1, fund_r),
                              cv.computeCorrespondEpilines(pts_l[i].reshape(-1,1,2), 1, fund_lr))
        
        # Find epilines intersection in the front image from left and right
        # points and compare with points in the right front 
        distance_f += intersect(pts_f[i],
                              cv.computeCorrespondEpilines(pts_r[i].reshape(-1,1,2), 1, cv.findFundamentalMat(pts_r[i], pts_f[i])[0]),
                              cv.computeCorrespondEpilines(pts_l[i].reshape(-1,1,2), 1, fund_l))
        
        # Find epilines intersection in the left image from right and front
        # points and compare with points in the right left 
        distance_l += intersect(pts_l[i],
                              cv.computeCorrespondEpilines(pts_f[i].reshape(-1,1,2), 1, cv.findFundamentalMat(pts_f[i], pts_l[i])[0]),
                              cv.computeCorrespondEpilines(pts_r[i].reshape(-1,1,2), 1, cv.findFundamentalMat(pts_r[i], pts_l[i])[0]))
        
    return distance_l/n_images, distance_f/n_images, distance_r/n_images



def intersect(pts, lines_1, lines_2):
    n = len(pts)
    total_dist = 0 
    for i in range(n):
        l1 = lines_1[i][0]
        l2 = lines_2[i][0] 
        x = (l1[1]*l2[2]-l1[2]*l2[1])/(l1[0]*l2[1]-l1[1]*l2[0])
        y = (l1[0]*x+l1[2])/-l1[1]
        
        total_dist += math.dist([round(x), round(y)], [pts[i][0][0], pts[i][0][1]])
    return total_dist/n



def rectify_images(img_l, img_f, img_r, h_l, h_fl, h_fr, h_r, width, height, light_name = 'light_A'):
    print('\nRectifying images...\n')
    
    path = './rect/'+light_name
    if not os.path.exists(path):
        os.makedirs(path)

    img_l_rect = cv.warpPerspective(img_l, h_l, (width, height))
    img_f_rect = cv.warpPerspective(img_f, h_fl, (width, height))
    img_r_rect = cv.warpPerspective(img_r, h_r, (width, height))
    cv.imwrite(path + '/left_rectified.jpg', img_l_rect)
    cv.imwrite(path + '/front_rectified.jpg', img_f_rect)
    cv.imwrite(path + '/right_rectified.jpg', img_r_rect)
    
    return img_l_rect, img_f_rect, img_r_rect



def compute_disparity(img_l, img_f, name, win_size = 50, block_size = 5, 
                      ratio = 10, disp_max_diff = 12, spakle_range = 32):
    print('\nComputing disparity StereoSGBM '+ name + '...\n')
    window_size = 3
    
    left_matcher = cv.StereoSGBM_create(minDisparity=-1,
                                        numDisparities=5*16,
                                        blockSize=block_size,
                                        P1=8*3*window_size,
                                        P2=32*3*window_size,
                                        disp12MaxDiff=disp_max_diff,
                                        uniquenessRatio=ratio,
                                        speckleWindowSize=win_size,
                                        speckleRange=spakle_range,
                                        preFilterCap=63,
                                        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

    front_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    
    lmbda = 80000
    sigma = 1.2
    
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    
    displ = left_matcher.compute(img_l, img_f)  
    dispf = front_matcher.compute(img_f, img_l) 
    displ = np.int16(displ)
    dispf = np.int16(dispf)
    
    filtered_img_l = wls_filter.filter(displ, img_l, None, dispf)
    _, filtered_img_l = cv.threshold(filtered_img_l , 0, 5*16, cv.THRESH_TOZERO)
    filtered_img_l = (filtered_img_l / 16).astype(np.uint8)
    
    if not os.path.exists('./disparity'):
        os.makedirs('./disparity')
        
    cv.imwrite('./disparity/' + name + '_SGBM.jpg', filtered_img_l)
    
    return filtered_img_l



def calculate_depth(cam_mat_l, cam_mat_r, t_lf, t_fr, width, height):
    baseline_l = np.linalg.norm(t_lf)
    baseline_r = np.linalg.norm(t_fr)
    f_l = cam_mat_l[0,0]
    f_r = cam_mat_r[0,0]
    
    Q_l = np.array([[1, 0, 0, -width/2], [0, 1, 0, -height/2],[0, 0, 0, f_l],[0, 0, -1/baseline_l, t_lf[0][0]/baseline_l]])
    Q_r = np.array([[1, 0, 0, -width/2], [0, 1, 0, -height/2],[0, 0, 0, f_r],[0, 0, -1/baseline_r, t_fr[0][0]/baseline_r]])
    
    return Q_l, Q_r



def reprojection3D_multi(image, disparity1, disparity2, Q1, Q2):
    print('\nGenerating 3D points...\n')
    # generate the 3D points for cam2 image from different pairs
    points_1 = cv.reprojectImageTo3D(disparity1, Q1)
    mask_1 = disparity1 > disparity1.min()
    points_1[~mask_1] = 0

    points_2 = cv.reprojectImageTo3D(disparity2, Q2)
    mask_2 = disparity2 > disparity2.min()
    points_2[~mask_2] = 0


    # compine different pairs construction by smart avergining (by neglicting outlier points in each reconstruction) 
    mask_compined = np.array(mask_1, dtype=np.float16) + np.array(mask_2, dtype=np.float16) 
    points_compine = (points_1+points_2) / np.expand_dims(mask_compined,axis=-1)

    # get the final mask for the compination
    final_mask = np.logical_or(mask_1, mask_2)
    colors = image

    out_points = points_compine[final_mask]
    out_colors = image[final_mask]
    plt.imshow(points_compine[:,:,-1])
    plt.show()
    
    # create the ply file
    verts = out_points.reshape(-1,3)
    colors = out_colors.reshape(-1,3)
    verts = np.hstack([verts, colors])

    print('\nWritting ply file...\n')
    # header of ply file
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''

    with open('./output.ply', 'w') as f:
        f.write(ply_header % dict(vert_num = len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')