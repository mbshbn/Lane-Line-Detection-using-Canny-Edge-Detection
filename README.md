# **Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* The output should be two solid lines, one for the right side, and the other for the left side


[//]: # (Image References)

[image1]: ./test_images_output/output.png "detected lanes1"
[image11]: ./test_images_output/output2.png "detected lanes2"
[image2]: ./examples/shortcoming.jpg "shortcoming"
[image3]: ./examples/shortcoming2.jpg "shortcoming"

[image4]: ./test_videos_output/solidWhiteRight_cover_video.png "cover white right video"
[image5]: ./test_videos_output/solidYellowLeft_cover_video.png "cover yellow left video"


---

### Pipeline (single images)

My pipeline consisted of the follwoing steps, and the code is called `Lane_line_finding.py`.
I used the following libraries
```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
```
1. Conversion to grayscale using `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
1. Gaussian smoothing using `cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)`
1. Edge Detection using Canny algorithm, using `cv2.Canny(img, low_threshold, high_threshold)`
1. Defining the region of interest. This is the area in front of the fixed camera that the lanes appear. I defined an array of four points as the vertices of the polygon, and fill the pixels inside the polygon with a color `cv2.fillPoly`. Then the function returns the image only where the mask pixels are nonzero using `cv2.bitwise_and`.
1. Finding line segments in the binary image using the probabilistic Hough transform using `cv2.HoughLinesP`. The inputs to this function are the distance and angular resolution in pixels of the Hough grid. Also, the minimum number of votes (intersections in Hough grid cell), the minimum number of pixels making up a line, and the maximum gap in pixels between connectable line segments. 
      1. mapping out the full extent of the lane to visualize the result, I defined a function called `draw_lines`. To draw a single line on the left and right lanes, it extrapolates the lines using `np.polyfit` and `np.poly1d`. The left and right lines are distinguished using their slope. Usually, the slope is about 0.6 or -0.6. Having this number, to avoid small white and yellow marks on the ground affecting the lines, those who have a slope very different than these usual slopes are ignored. Although this is applied to filter the lines before extrapolating, sometimes the extrapolated line may have a slope very different than the usual slope. To avoid reporting wrong lines, the lines after extrapolation are filtered, and those that do not have a usual slope are ignored.  
1. Combining the original image and the output of the previous step using `cv2.addWeighted`.


Here is the final result, showing the detcetd lines with the red color:

Image 1             |  Image 2
:-------------------------:|:-------------------------:
![alt text][image1]  |  ![alt text][image11] 

### Videos
A video consists of images, so I used the above pipeline and applied it to each image of the video using `clip1.fl_image(Lane_Finding_Pipeline_image)`. Not that this is the same pipeline explained above. To do so, I imported the following libraries:
```
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```
Video 1             |  Video 2
:-------------------------:|:-------------------------:
[![alt text][image4]](https://youtu.be/Nyq5kYjLoSI) |  [![alt text][image5]](https://youtu.be/_ZnMx4tlOKY)

### Potential shortcomings with the current pipeline

One potential shortcoming would happen when the line's curvature is large. This is overcome in another project called [Advanced Lane Line detection](https://github.com/mbshbn/CarND-Advanced-Lane-Lines) in my github repo.

Another shortcoming could be misled with different signs on the street that are ignored. For example the following lines:

Example 1             |  Example 2
:-------------------------:|:-------------------------:
![alt text][image2]  |  ![alt text][image3] 



### Possible improvements to the pipeline

A possible improvement would be to considering higher-order polynomial lines, not only straight lines.

Another potential improvement could be to considering different local signs on the ground to avoid being misled with them.
