**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/original_raw_picture.jpg "Original Picture"
[image2]: ./output_images/original_chessboard.jpg "Original Chessboard Picture"
[image3]: ./output_images/undistorted_and_warped_chessboard.jpg "Undistorted and Unwarped Chessboard"
[image4]: ./output_images/undistorted_picture.jpg "Undistorted Picture"
[image5]: ./output_images/masked_image.jpg "Masked Picture"
[image6]: ./output_images/masked_edges.jpg "Masked Edges"
[image7]: ./output_images/binary_warped.jpg "Binary Warped Picture"
[image8]: ./output_images/lane_detection.jpg "Lane Detected"
[image9]: ./output_images/final_picture.jpg "Final Picture"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "Project 2.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I took average value of calibration and distortion coefficients over the pictures that can be found of all the corners, in order to increase the accuracy of the calibration.
I applied the distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image3]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

**In 1st Code Cell**
I read in a image from "test_images/" folder:
![alt text][image1]

**In 7th Code Cell**
I implemented in the functino `undistort_image()`
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like the one bwlow. I used the calibartion and distortion coefficient that I calculated from Camera Calibration step, and use `cv2.undistort()` function to obtain the result picture above.
![alt text][image4]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

**In 8th Code Cell**
I implemented this in function `region_of_interest()` and `define_vertices`
I applied a mask to the picture, and limited the area to only the road:
![alt text][image5]

**In 9th Code Cell**
I implmented this in the functino `detect_line()`
I used a combination of color and gradient thresholds to generate a binary image 
Here's an example of my output for this step.  
![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

**In 10th Code Cell**
The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

**In 11th Code Cell**
I implement this in the function `find_lane_pixels()` and function `fit_polynomial()`
Then I detect the lane pixels and fit a pllynomial to the lanes. Below is the detailed steps:
1.Take a histogram of the bottom half of the image
2. Find the peak of the left and right halves of the histogram
3. Define the sliding window sizes
4. Search for lane pixels inside the each window, and store then in a list
5. Then use np.polyfit()function to calculate the polynomial coefficients
![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

**In 12th Code Cell**
I implement this in the function `measure_middle()`.
I used the polymonial coefficients I calculated in the step above, and calculated the left and right lane's x values at the bottom of the image.
By taking the average of them, I get the location of the center of the lane.

**In 13th Code Cell**
I implement this in the function `measure_curvature()`
I used the polynomial coefficients I obtained in the step above, and put into the formula taught in class to calculate the curvature of the left and right lane. Since they are supposed to be the same value in real life, I took average of them.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

**In 14th Code Cell**
I implemented this step in the function `determine_test()` and `draw_area()`.  Here is an example of my result on a test image:

![alt text][image9]

Here's a [link to my test images result](./test_images_output/)
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
My pipeline might fail if there a detection of lane is failed (Such as during bad road conditions, where the the white lanes are damaged heavily and almost disappears). I did not have enough failsafe protections taken. Basically any step of the piprline there is a chance of fail to detect/identify/calculate. I think I should put more "If-Else" conditions in my functions. If this step failed, I should use what I have from the last few pictures.
