#%%
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
%matplotlib inline

#%%
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
# if you wanted to show a single color channel image called 
# 'gray', for example, call as plt.imshow(gray, cmap='gray')

#%%
''' some helper functions '''
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill 
    # the mask with depending on the input image
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

def draw_raw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img 


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    
    left_lines = []
    right_lines = []
   

    size_y, size_x,_ = img.shape
    center_x = size_x / 2
  
    # all lines that start from to the left of the center of the frame might be 
    # left lane lines
    left_lines = lines[lines[:,0,0] <= center_x]
    
    # all lines that end at the right side of the frame are right lane lines
    right_lines = lines[lines[:,0,2] > center_x]
    
    # average out the slope of all line segments in a lane
    # and then draw an line from the bottom of the lane to the top
    left_lane_eq = avg_lane_equation(left_lines)
    right_lane_eq = avg_lane_equation(right_lines)
    
    # extrapolate a line from the bottom of the frame to the center
    # for both left and right lanes

    left_lane = extrapolate_lane(left_lane_eq, size_y , size_y * 0.6)
    right_lane = extrapolate_lane(right_lane_eq, size_y, size_y * 0.6)
        
    cv2.line(img, left_lane[0], left_lane[1], [255,0,0], thickness + 8)
    cv2.line(img, right_lane[0], right_lane[1], [255,0,0], thickness + 8)
    
    return img

def extrapolate_lane(line_eq, frame_bottom, frame_top):
    M, C = line_eq
    
    # Y = MX + C
    # bottom lane line, we know Y = size of frame, find out X
    # X = (Y - C) / M
    
    bottom_pt = (int((frame_bottom - C) / M), int(frame_bottom))
    top_pt = (int((frame_top - C) / M), int(frame_top))
    
    return (bottom_pt, top_pt)

def avg_lane_equation(lines):
    # returns a tuple (M, C) representing line equation Y = mX + C
    # M -> is the slope of a line that is weighted average of all smaller line segments
    # C -> constant in the line equation (y = mx + c)
    
    # Y = MX + C
    # M = (y2 - y1) / (x2 - x1)
    # take a weighted average of all slopes. The bigger the line segment
    # the bigger its weight
    
    slopes = ((lines[:,0,3] - lines[:,0,1]) / (lines[:,0,2] - lines[:,0,0]))
    distance = ((lines[:,0,2] - lines[:,0,0]) ** 2) + ((lines[:,0,3] - lines[:,0,1]) ** 2)
    slope_avg = np.average(slopes, weights = distance)
    
    # C = Y - MX
    # Use the line segment that is biggest in the given lines
    
    biggest_line_segment = np.argmax(distance)
    
    # C = Y - MX
    C = lines[biggest_line_segment,0,3] - slope_avg * lines[biggest_line_segment,0,2]
    return (slope_avg, C)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
     minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def find_color_edges(edges):
    return np.dstack((edges,edges,edges))


def loadImages(dir_path):
    '''loading all images from a directory'''
    images = os.listdir(dir_path)
    load_images = []
    for img in images:
        i = mpimg.imread(dir_path + img)
        load_images.append(i)
    return load_images


#%%
def pipeline_findlane(image,flag):
    test_img = np.copy(image)
    # plt.imshow(test_img)
    imshape = test_img.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] // 2 - 30,
      imshape[0] // 2 + 60), (imshape[1] // 2 + 50, imshape[0] // 2 + 60),
     (imshape[1], imshape[0])]], dtype=np.int32)
    cropped_region = region_of_interest(test_img, vertices)
    # plt.figure()
    # plt.imshow(cropped_region)
    # plt.show()
    gray = grayscale(test_img)
    kernel = 3
    blur_gray = gaussian_blur(gray, kernel)
    low_threshold = 100
    high_threshold = 200
    canny_edge = canny(blur_gray, low_threshold, high_threshold)
    # plt.figure()
    # plt.imshow(canny_edge)
    # plt.show()

   
    masked_region = region_of_interest(canny_edge, vertices)

    # plt.figure()
    # plt.imshow(masked_region)
    # plt.show()

    rho = 2
    theta = np.pi / 180
    threshold = 50
    min_line_len = 50
    max_line_gap = 100
    proc_img = hough_lines(masked_region, rho, theta, threshold,
    min_line_len, max_line_gap)
    color_edges = find_color_edges(canny_edge)
    final_lane_image = weighted_img(color_edges, proc_img)
    lines = cv2.HoughLinesP(masked_region, rho, theta, threshold, np.array([]),
     minLineLength=min_line_len, maxLineGap=max_line_gap)
    if (flag == 0):
        main_lane_image = draw_raw_lines(image, lines)
    elif (flag == 1):
        main_lane_image = draw_lines(image, lines)

    # plt.figure()
    # plt.imshow(main_lane_image)
    # plt.show()
    # plt.imshow(final_lane_image)
    # plt.show()
    return main_lane_image

images = loadImages("test_images/")
pipeline_findlane(images[5],0)

#%%
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#%%
def process_image(image):
    
    return pipeline_findlane(image, 1)
    
def process_image_raw(image):
    
    return pipeline_findlane(image,0) 
    

#%%
white_output = 'test_videos_output/solidWhiteRightRaw.mp4'
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image_raw) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

#%%
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

#%%
white_output = 'test_videos_output/solidWhiteRight.mp4'
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

#%%
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

#%%
yellow_output = 'test_videos_output/solidYellowLeftRaw.mp4'
# clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image_raw)
% time yellow_clip.write_videofile(yellow_output, audio=False)

#%%
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

#%%
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
# clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
% time yellow_clip.write_videofile(yellow_output, audio=False)

#%%
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

#%%
# challenge_output = 'test_videos_output/challenge.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
# clip3 = VideoFileClip('test_videos/challenge.mp4')
# challenge_clip = clip3.fl_image(process_image)
# % time challenge_clip.write_videofile(challenge_output, audio=False)

# #%%
# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(challenge_output))

