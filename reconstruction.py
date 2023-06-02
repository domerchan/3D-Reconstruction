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


def get_chess_points_(img, rows, columns, scale):
    # Using cv Functions (Criteria) to detect Checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # World Space 
    obj_pts = np.zeros((rows*columns,3), np.float32)
    obj_pts[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    obj_pts = scale* obj_pts
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Locate Checkerboard
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
    if ret != True :
        print('\n -- ERROR: we could not find all the chess board coordinates in image...')
        print(' -- try with better quality images (increase reduction factor)..')
        sys.exit()
    else:
        # Convolution for Corner Detection 
        conv_size = (4, 4)
         
        # CV Criteria trys Optimizes Corner Detection
        img_pts = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
    
    return obj_pts, img_pts



def calibrate_individual_camera_(obj_pts, img_pts, width, height):
    ret, cam_mat, dist, rvecs, tvecs = cv.calibrateCamera(obj_pts, img_pts, (width, height), None, None)
    
    return cam_mat, dist



def stereo_calibrate_(obj_pts_l, obj_pts_r, img_pts_l, img_pts_r, m_l, m_r, dist_l, dist_r, w, h):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    ret, cam_mat_l, dist_l, cam_mat_r, dist_r, R, T, E, F = cv.stereoCalibrate(obj_pts_l, 
                                                                               img_pts_l, 
                                                                               img_pts_r, 
                                                                               m_l, 
                                                                               dist_l, 
                                                                               m_r, 
                                                                               dist_r, 
                                                                               (w,h),
                                                                               criteria = criteria, 
                                                                               flags = cv.CALIB_FIX_INTRINSIC)
    return R, T, E, F



def drawlines_(img, lines, pts):
    im = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    r, c = im.shape
    
    for r, pt in zip(lines, pts.astype(int)):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1]])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img_out = cv.line(img, (x0,y0), (x1,y1), color, 1)
        img_out = cv.circle(img_out, tuple(pt), 5, color, -1)
        
    return img_out



def draw_epilines_(pts_img, pts_clc, img, F, index):    
    # Find the epilines corresponding to the chess board points in the
    # front image and drawing them in the left image
    lines = cv.computeCorrespondEpilines(pts_clc.reshape(-1,1,2), index, F)
    lines = lines.reshape(-1,3)
    im = drawlines_(img, lines, pts_img.reshape(num_pts,2))
    
    return im
       

    
def calculate_homographic_(pts_l, pts_r, F, width, height):
    pts = num_pts * num_img
    
    pts_l = np.int32(pts_l).reshape(pts,2)
    pts_r = np.int32(pts_r).reshape(pts,2)
    
    _, h_l, h_r = cv.stereoRectifyUncalibrated(pts_l, pts_r, F, (width, height))
    
    return h_l, h_r


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



def epilines_intersections(pts_l, pts_f, pts_r, fund_l, fund_r, fund_lr):
    n_images = len(pts_l)
    distance_r = 0
    distance_f = 0
    distance_l = 0
    for i in range(n_images):
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



def rectify_images_(img_l, img_r, h_l, h_r, width, height, light_name = 'light_A'):
    img_l_rect = cv.warpPerspective(img_l, h_l, (width, height))
    img_r_rect = cv.warpPerspective(img_r, h_r, (width, height))
    
    return img_l_rect, img_r_rect


def compute_disparity_(img_l, img_r, name, win_size = 50, block_size = 5, 
                      ratio = 10, disp_max_diff = 12, spakle_range = 32,
                      window_size = 3):
    
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

    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    
    lmbda = 80000
    sigma = 1.2
    
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    
    displ = left_matcher.compute(img_l, img_r)  
    dispr = right_matcher.compute(img_r, img_l) 
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    
    filtered_img_l = wls_filter.filter(displ, img_l, None, dispr)
    _, filtered_img_l = cv.threshold(filtered_img_l , 0, 5*16, cv.THRESH_TOZERO)
    filtered_img_l = (filtered_img_l / 16).astype(np.uint8)
    
    return filtered_img_l


def calculate_depth_(cam_mat, T, width, height):
    baseline = np.linalg.norm(T)
    
    f = cam_mat[0,0]
    
    Q = np.array([[1, 0, 0, T[0][0]], [0, 1, 0, T[1][0]],[0, 0, 0, f],[0, 0, -1/baseline, T[0][0]/baseline]])
    
    return Q


def reproject_3D_(image, disparity, Q):
    im3d = cv.reprojectImageTo3D(disparity, Q)
    colors = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    mask = disparity > disparity.min()
    out_points = im3d[mask]
    out_colors = colors[mask]
    
    verts = out_points.reshape(-1,3)
    colors = out_colors.reshape(-1,3)
    verts = np.hstack([verts, colors])
    
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''

    with open('./out.ply', 'w') as f:
        f.write(ply_header % dict(vert_num = len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


def reproject_stereo_3D_(image, disparity_l, disparity_r, Q_l, Q_r):
    im3d_l = cv.reprojectImageTo3D(disparity_l, Q_l)
    im3d_r = cv.reprojectImageTo3D(disparity_r, Q_r)
    colors = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    im3d = (im3d_l + im3d_r) / 2
    disparity = (disparity_l + disparity_r) / 2
    
    mask = disparity > disparity.min()
    im3d[~mask] = 0
    
    verts = im3d.reshape(-1,3)
    colors = colors.reshape(-1,3)
    verts = np.hstack([verts, colors])
    
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''

    with open('./reconstruction.ply', 'w') as f:
        f.write(ply_header % dict(vert_num = len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')
        
        
        
def save_images(images, path, name = []):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(images)):
        if len(name) == 0:
            cv.imwrite(path + '/' + str(i) + '.jpg', images[i])
        else:
            cv.imwrite(path + '/' + name[i] + '.jpg', images[i])