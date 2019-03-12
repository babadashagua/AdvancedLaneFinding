#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:00:11 2019

@author: yu
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import scipy.misc
import time

def camera_cal(image):
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#    image = mpimg.imread('camera_cal/calibration1.jpg')
    objp = np.zeros((5*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:5].T.reshape(-1,2)
    
    objpoints = [] 
    imgpoints = [] 
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find the chessboard corners
    nx = 9
    ny = 5
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    
    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        objpoints.append(objp)
        imgpoints.append(corners)
    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        return mtx, dist
    
    return None

def img_undistort(mtx, dist, img):
        
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)): # thresh=(20, 100)
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
        
    abs_sobel = np.absolute(sobel)
    scale_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    grad_binary = np.zeros_like(scale_sobel)
    grad_binary[(scale_sobel > thresh[0]) & (scale_sobel < thresh[1])] = 1       
    
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel_mag = np.uint8(255*sobel_mag/np.max(sobel_mag))
    
    mag_binary = np.zeros_like(scaled_sobel_mag)
    mag_binary[(scaled_sobel_mag > mag_thresh[0]) & (scaled_sobel_mag < mag_thresh[1])] = 1
    
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
    
    dir_sobel = np.arctan2(sobely, sobelx)
    scaled_dir = np.uint8(255*dir_sobel/np.max(dir_sobel))
    
    dir_binary = np.zeros_like(scaled_dir)
    dir_binary[(scaled_dir > thresh[0]) & (scaled_dir < thresh[1])] = 1
    
    return dir_binary

def s_channel_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
#    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    s_combine = np.zeros_like(s_channel)
    s_combine[(sxbinary == 1) | (s_binary == 1)] = 1
    return s_combine

def region_of_interest(img, vertices):
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# input: thresholded image
def warped_img(img):
    
    img_shape = img.shape
    src = np.float32([[200, img_shape[0]],[570, 460],[620, 460],[1200, img_shape[0]]])
    offset = 200
    dst = np.float32([[offset, img_shape[0]],[offset, 0],[img_shape[0]-offset, 0],[img_shape[1]-offset,img_shape[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(masked_img, M, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return warped, Minv

def find_lane_pixels(binary_warped, line_class):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & 
                          (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & 
                           (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = (np.mean(nonzerox[good_left_inds])).astype(int)
        if len(good_right_inds) > minpix:
            rightx_current = (np.mean(nonzerox[good_right_inds])).astype(int)

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    line_class.detected = True
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    if not line_class.current_fit:
        line_class.current_fit.append(left_fit)
        line_class.current_fit.append(right_fit)
    else:
        line_class.current_fit[0] = left_fit
        line_class.current_fit[1] = right_fit

    return leftx, lefty, rightx, righty #, out_img

def find_lane_pixels_prior(binary_warped, line_class):
    margin = 100
    
    left_fit = line_class.current_fit[0]
    right_fit = line_class.current_fit[1]
    
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*nonzeroy**2+left_fit[1]*nonzeroy+left_fit[2]-margin)) & 
                      (nonzerox < (left_fit[0]*nonzeroy**2+left_fit[1]*nonzeroy+left_fit[2]+margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*nonzeroy**2+right_fit[1]*nonzeroy+right_fit[2]-margin)) & 
                      (nonzerox < (right_fit[0]*nonzeroy**2+right_fit[1]*nonzeroy+right_fit[2]+margin)))
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped, line_class):
    # Find our lane pixels first
    if not line_class.detected: 
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped, line_class)
    else:
        leftx, lefty, rightx, righty = find_lane_pixels_prior(binary_warped, line_class)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    if not line_class.current_fit:
        line_class.current_fit.append(left_fit)
        line_class.current_fit.append(right_fit)
    else:
        line_class.current_fit[0] = left_fit
        line_class.current_fit[1] = right_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
#    try:
#        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#    except TypeError:
#        # Avoids an error if `left` and `right_fit` are still none or incorrect
#        print('The function failed to fit a line!')
#        left_fitx = 1*ploty**2 + 1*ploty
#        right_fitx = 1*ploty**2 + 1*ploty
#
#    ## Visualization ##
#    # Colors in the left and right lane regions
#    out_img[lefty, leftx] = [255, 0, 0]
#    out_img[righty, rightx] = [0, 0, 255]
#
#    # Plots the left and right polynomials on the lane lines
#    plt.plot(left_fitx, ploty, color='yellow')
#    plt.plot(right_fitx, ploty, color='yellow')
#
#    return out_img

    return ploty, left_fit, right_fit

def fit_polynomial_pts(ploty, left_fit, right_fit):
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx

def measure_curvature_pixels(binary_warped):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    ploty, left_fit, right_fit = fit_polynomial(binary_warped)
    
    y_eval = np.max(ploty)
   
    left_curverad = np.power((1+(2*left_fit[0]*y_eval+left_fit[1])**2), 1.5)/np.absolute(2*left_fit[0])
    right_curverad = np.power((1+(2*right_fit[0]*y_eval+right_fit[1])**2), 1.5)/np.absolute(2*right_fit[0])  
    
    return left_curverad, right_curverad

def measure_curvature_meters(binary_warped, line_class):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    if not line_class.detected:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped, line_class)
    else:
        leftx, lefty, rightx, righty = find_lane_pixels_prior(binary_warped, line_class)
      
    ym_per_pix = 30/720 
    xm_per_pix = 3.7/800
    leftx = np.multiply(leftx, xm_per_pix)
    rightx = np.multiply(rightx, xm_per_pix)
    lefty = np.multiply(lefty, ym_per_pix)
    righty = np.multiply(righty, ym_per_pix)
    left_fit_cr = np.polyfit(lefty, leftx, 2)
    right_fit_cr = np.polyfit(righty, rightx, 2)
    
    y_eval_left = np.max(lefty)
    y_eval_right = np.max(righty)
    
    left_curverad = np.power((1+(2*left_fit_cr[0]*y_eval_left+left_fit_cr[1])**2), 1.5)/np.absolute(2*left_fit_cr[0])
    right_curverad = np.power((1+(2*right_fit_cr[0]*y_eval_right+right_fit_cr[1])**2), 1.5)/np.absolute(2*right_fit_cr[0]) 
    
    return left_curverad, right_curverad

def lane_area_drawn(undist, warped_img, Minv, line_class):
    img_shape = warped_img.shape
    
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty, left_fit, right_fit = fit_polynomial(warped_img, line_class)
    sanity_check(ploty, left_fit, right_fit, line_class)
    if not line_class.detected:
        ploty, left_fit, right_fit = fit_polynomial(warped_img, line_class)
        sanity_check(ploty, left_fit, right_fit, line_class)
    left_fitx, right_fitx = fit_polynomial_pts(ploty, line_class.best_fit_left, line_class.best_fit_right)
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_shape[1], img_shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

def sanity_check(ploty, left_fit, right_fit, line_class):
    '''
    
    '''
    y_min = np.min(ploty)
    y_mid = np.median(ploty)
    y_max = np.max(ploty)
    
    x_min_left = left_fit[0]*y_min**2 + left_fit[1]*y_min + left_fit[2]
    x_mid_left = left_fit[0]*y_mid**2 + left_fit[1]*y_mid + left_fit[2]
    x_max_left = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    
    x_min_right = right_fit[0]*y_min**2 + right_fit[1]*y_min + right_fit[2]
    x_mid_right = right_fit[0]*y_mid**2 + right_fit[1]*y_mid + right_fit[2]
    x_max_right = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
    
    dif_min = x_min_right - x_min_left
    dif_mid = x_mid_right - x_mid_left
    dif_max = x_max_right - x_max_left
    
    dif_min_mid = dif_mid - dif_min
    dif_min_max = dif_max - dif_min
    dif_mid_max = dif_max - dif_mid
    
    margin = 30
    margin_width = 500
    
    frame_tol = 3
    
    if (dif_min_mid < margin) & (dif_min_max < margin) & (dif_mid_max < margin):
        line_class.detected = True
        line_class.failure_time = 0
        if len(line_class.recent_fit_left) <= frame_tol:
            line_class.recent_fit_left.append(left_fit)
            line_class.recent_fit_right.append(right_fit)
        else:
            line_class.recent_fit_left.append(left_fit)
            del line_class.recent_fit_left[0]
            line_class.recent_fit_right.append(right_fit)
            del line_class.recent_fit_right[0]
    else:
        line_class.detected = False
        line_class.failure_time += 1
    
    if (dif_min < margin_width) & (dif_mid < margin_width) & (dif_max < margin_width):
        line_class.detected = False
        line_class.failure_time += 1
        
    if line_class.failure_time <= frame_tol:
        if len(line_class.recent_fit_left) <= 1:
            line_class.best_fit_left = left_fit
            line_class.best_fit_right = right_fit
        else:
            recent_fit_left = np.asarray(line_class.recent_fit_left)
            recent_fit_right = np.asarray(line_class.recent_fit_right)
            line_class.best_fit_left = np.average(recent_fit_left, axis = 0)
            line_class.best_fit_right = np.average(recent_fit_right, axis = 0)        
    else:
        line_class.detected = False
        
    
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # failure times 
        self. failure_time = 0
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None
        self.best_fit_right = None
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        # recent polynomial coefficients
        self.recent_fit_left = []
        self.recent_fit_right = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None  

lane_det = Line()
#img_dist = mpimg.imread('test_images/test6.jpg')
start = time.time()
image_cal = mpimg.imread('camera_cal/calibration1.jpg')

video_input = cv2.VideoCapture('project_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_videos/project_video_lane_drawn_usingprior_sanity.mp4', 
                      fourcc, 25.0, (1280, 720)) 

mtx, dist = camera_cal(image_cal)
counter = 0
while video_input.isOpened():
    
    ret, img_dist = video_input.read()
    if ret:
#        print(counter)
        undist = img_undistort(mtx, dist, img_dist)
        sobel_thresh_img = abs_sobel_thresh(undist, thresh=(20, 100))
        schannel_img = s_channel_threshold(undist)
        binary_combine = np.zeros_like(sobel_thresh_img)
        binary_combine[(sobel_thresh_img == 1) | (schannel_img == 1)] = 1
        img_shape = binary_combine.shape
        
        vertices = np.array([[(130, img_shape[0]),(570, 460), (750, 460), (1200, img_shape[0])]], dtype = np.int32)
        masked_img = region_of_interest(binary_combine, vertices)
    
        warped, Minv = warped_img(masked_img)
        
#        left_curverad, right_curverad = measure_curvature_meters(warped, lane_det)
        
        # Combine the result with the original image
        result = lane_area_drawn(undist, warped, Minv, lane_det)
#        counter += 1
        out.write(result)
    else:
        break

video_input.release()
out.release()
print('Finished')
end = time.time()
print('Processing time: ', end-start)
#scipy.misc.imsave('output_images/test6_lane_area_drawn.png', result)

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
#f.tight_layout()
#ax1.imshow(masked_img)
#ax1.set_title('Masked Image', fontsize=20)
#ax2.imshow(warped_img)
#ax2.set_title('Warped Image', fontsize=20)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)



