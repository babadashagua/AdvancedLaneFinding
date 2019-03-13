# **Report: Advanced Lane Finding** 

---

**Finding Lane Lines on the Road -- Advanced**

The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Shortcomings and further improvements on the existing work

### Reflection

### 1. Camera Calibration

I use OpenCV functions to calculate the correct camera matrix and distortion coefficients using the calibration chessboard image 'calibration1.jpg'. The original image (left) and undistorted image (right) are shown below (images are in the folder output_images).
<p float="middle">
  <img src="/output_images/calibration1_undistorted.png" width="800" title="Fig. 1 Camera calibration"/>
</p>

### 2. Creating Thresholded Binary Image

I tried four thresholding methods: sobel operator along x direction, magnitude of the gradient, direction of the gradient, and s channel threshold from HLS color space. Finally, I use a combination of sobel operator along x direction (left) and s channel threshold (right) to create the thresholded binary image. Images are in the folder output_images (test4_sobel_thresh_img.png, test4_s_channle_thresh_img.png, test4_combine_img.png). 
<p float="left">
  <img src="/output_images/test4_sobel_thresh_img.png" width="400" title="Fig. 2 sobel x direction"/>
  <img src="/output_images/test4_s_channle_thresh_img.png" width="400" title="Fig. 2 s channel thresholded"/>
</p>
