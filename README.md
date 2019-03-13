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

I use OpenCV functions to calculate the correct camera matrix and distortion coefficients using the calibration chessboard image 'calibration1.jpg' (implemented by the function camera_cal and img_undistort). The original image (left) and undistorted image (right) are shown below (images are in the folder output_images).
<p float="left">
  <img src="/output_images/calibration1_undistorted.png" width="800" title="Fig. 1 Camera calibration"/>
</p>

### 2. Creating Thresholded Binary Image

I tried four thresholding methods: sobel operator along x direction, magnitude of the gradient, direction of the gradient, and s channel threshold from HLS color space. Finally, I use a combination of sobel operator along x direction (left) and s channel threshold (right) to create the thresholded binary image. Images are in the folder output_images (test4_sobel_thresh_img.png, test4_s_channle_thresh_img.png, test4_combine_img.png). Thresholded image creation is implemented by the functions abs_sobel_thresh and s_channel_threshold in the project code, and combination of the thresholded images is implemeted in the mainstream of the video processing to obtain a combined binary image named binary_combine.
<p float="left">
  <img src="/output_images/test4_sobel_thresh_img.png" width="400" title="Fig. 2 sobel x direction"/>
  <img src="/output_images/test4_s_channle_thresh_img.png" width="400" title="Fig. 2 s channel thresholded"/>
</p>
The combined thresholded binary image is:
<p float="left">
  <img src="/output_images/test4_combine_img.png" width="800" title="Fig. 3 combined thresholded binary image"/>
</p>

### 3. Perspective Transformation ("birds-eye view")

I project the lane area to a birds-eye view using OpenCV functions. This procedure is implemented by the funciton warped_img in the project code. The left image below is the lane area after masking unnecessary details out, and the right image below is the warped image after perspective transformation. The result is named test4_warped_image.png in the output_images folder.
<p float="left">
  <img src="/output_images/test4_warped_image.png" width="800" title="Fig. 4 perspective transformation"/>
</p>

### 4. Lane Line Identification

I use histogram peaks to find the bottom starting point of the left and the right lane lines. I take a histogram along all the columns in the lower half of the birds-eye view image. I add up the pixel values along each column in the image, and the two most prominent peaks in the histogram indicate the x-position of the base of the lane lines. From the two starting points, I use two sliding windows placed around each line centers to find the lane line pixels (histogram peaks and sliding window approach implemented by the function find_lane_pixels). After finding all pixels belonging to each line through the sliding window approach, I fit a polynomial to each line (implemented by the function fit_polynomial). The image below shows sliding windows and fitted ploynomials of each line.
<p float="left">
  <img src="/output_images/test4_slidingwindow_fit.png" width="800" title="Fig. 5 sliding window and polynomial fit"/>
</p>

### 5. Curvature and Vehicle Position

The curvature of lane lines are calculated using the following equation:
<p float="left">
  <img src="/assist_imgs/curvature_equation.png" width="300" title="Equ. 1 curvature calculation"/>
</p>
where A and B are the coefficients of fitted polynomials. The y values of the image increase from top to bottom. To measure the radius of curvature closest to the vehicle, I evaluate the formula at the y value corresponding to the bottom of the image. The lane curvature is calculated by the functions measure_curvature_pixels (unit: pixel) and measure_curvature_meters (unit: meter).

The vehicle position with respect to the center is evaluated by calculating the difference between the center of the image and the center between two lane lines. Assuming the camera is placed at the center of the vehicle, then the vehicle position is the center of the image, which is image.shape[1]//2. The center between two lane lines in birds-eye view can be determined. Its projection coordinates in the original image can be calculated via:
<p float="left">
  <img src="/assist_imgs/perspective_transform.png" width="450" title="Equ. 2 inverse perspective transformation"/>
</p>
Where M is the transformation matrix that project pixels from birds-eye view back to the original image. This is how the OpenCV implements perspective transform for each pixel.
The position of the vehicle is the difference between the 2 x positions. The direction (left or right) depends on the sign of the difference. Vehicle position estimation is calculated by the function image_marking. This function also marks calculated curvatures and vehicle position on the image. The image below shows calculated curvatures (left and right) and vehicle positions and mark them on the image (test4_final_result.png in the folder output_images). 
<p float="left">
  <img src="/output_images/test4_final_result.png" width="800" title="Fig. 6 test4 final result"/>
</p>

### 6. Warp the Detected Lane Boundaries back onto the Original Image
The detected lanes, curvature, and vehicle position from center are marked on the image shown above.

### 7. Pipeline to Process the Project Video
I apply the developed functions to process the project video. I make some modifications on the pipeline for video processing:

Tracking: instead of using sliding window approach for all frames, I use searching from prior when the detected lane change is within a range. This recudes processing time on each frame.

Sanity check: this check evaluate whether current lane detection is correct or not. If current detection is incorrect (fails to pass the sanity check), the current lane detection is replaced by the average of several past detections (in this work I use average of coefficients of 3 most recent fitted polynomials). If incorrect detection occurs for more than n frames (in this work n = 3), the sliding window approach is implemeted for lane search. The sanity check is implemented by comparing the lane line distance at bottom, middle, and top positions. If all the errors between each two of the three distances are within a range (in this work, the margin is 30 pixels, and the sanity check is performed on warped image), the lane detection is considered as correct. In addition, the distance between two lane line should be always large than 500 pixels (on warped image) to avoid narrow lane line detection (distance between two lane lines less than 30 pixels, which is definitely wrong). Sanity check is implemented by the sanity_check function.

Smoothing: to avoid detections jumping around from frame to frame, I take an average over n past measurements to obtain the lane position. In this work, n = 3.

The following is a gif that shows partial result. This link leads to the full video after processing the project vidoe with the developed pipeline: https://youtu.be/wzTiqRVlH-Q (file that exceeds 25MB is not allowed on Github).
<p float="left">
  <img src="project_video_lane_finding_part.gif" width="800" title="Fig. 7 project video"/>
</p>
