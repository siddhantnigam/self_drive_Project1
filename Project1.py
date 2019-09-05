#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# List of images from test_images folder
test_path = "test_images/"
test_output_path="test_images_output/"
list_test_im = os.listdir(test_path)

#reading in an image
image_white = mpimg.imread('test_images/solidWhiteRight.jpg')
image_yellow= mpimg.imread('test_images/solidYellowCurve.jpg')

# Calibration Parameters
cal_kernel=5
cal_low_threshold=70
cal_high_threshold=210
cal_rho=4
cal_theta=np.pi/180
cal_hough_threshold=15
cal_min_line_length=8
cal_max_line_gap=4
cal_vertices=np.array( [[[420,330],[120,539],[905,539],[530,330]]], dtype=np.int32 )

#printing out some stats and plotting
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

# Helper functions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    x_left=[]
    y_left=[]
    x_right=[]
    y_right=[]
    x_half=img.shape[1]/2
    right_line=[]
    left_line=[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            if x2-x1==0:
                slope=100
                slope_val=False
            else:
                slope=(y2-y1)/(x2-x1)
                slope_val=True
            if slope_val is True:
                if x1<x_half and slope<0 and x2<x_half:
                    x_left.append(x1)
                    y_left.append(y1)
                    x_left.append(x2)
                    y_left.append(y2)
                    left_line.append(line)
                if x1>x_half and slope>0 and x2>x_half:
                    x_right.append(x1)
                    x_right.append(x2)
                    y_right.append(y1)
                    y_right.append(y2)
                    right_line.append(line)

    x_left=np.array(x_left)
    y_left=np.array(y_left)
    x_right=np.array(x_right)
    y_right=np.array(y_right)
    [left_slope,left_c]=np.polyfit(x_left,y_left,1)
    [right_slope,right_c] = np.polyfit(x_right, y_right, 1)
    y_top=int(img.shape[0])
    y_start=int(cal_vertices[0][0][1])
    new_left_x1=int((y_start-left_c)/left_slope)
    new_left_x2=int((y_top-left_c)/left_slope)
    new_right_x1=int((y_start-right_c)/right_slope)
    new_right_x2 =int((y_top - right_c) / right_slope)
    cv2.line(img, (new_right_x1, y_start), (new_right_x2, y_top), color, thickness)
    cv2.line(img,(new_left_x1,y_start),(new_left_x2,y_top),color,thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.1):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def image_shape(img):
    x,y,z=img.shape
    return x,y

def avg_height_width(list):
    image_height = []
    image_width = []
    for file in list:
        image_each=mpimg.imread(test_path+file)
        height,width=image_shape(image_each)
        image_height.append(height)
        image_width.append(width)
    image_height=np.array(image_height)
    image_width=np.array(image_width)
    avg_height=np.average(image_height)
    avg_width=np.average(image_width)
    return avg_height,avg_width

def image_resize(img,height=540,width=960):
    image_resize=cv2.resize(img,(width,height))
    return image_resize

def filter_image(image):
    #yellow_im[i][j][0]>5 and yellow_im[i][j][0]<=40 and yellow_im[i][j][1]>sat:
    img=np.copy(image)
    img=region_of_interest(img,cal_vertices)
    lower_white=np.array([200,200,200])
    upper_white=np.array([255,255,255])
    mask_white=cv2.inRange(img,lower_white,upper_white)
    lower_yellow=np.array([80,30,0])
    upper_yellow=np.array([120,255,255])
    image_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask_yellow=cv2.inRange(image_hsv,lower_yellow,upper_yellow)
    comb_mask=cv2.bitwise_or(mask_white,mask_yellow)
    final_image=cv2.bitwise_and(image,image,mask=comb_mask)
    return cv2.addWeighted(image,0.8,final_image,1.0,0.1)

# Pipeline of images
def test_algo_images(list_test_im):
    for i,file in enumerate(list_test_im):
        image_raw=mpimg.imread(test_path+file)
        image_proc=np.copy(image_raw)
        image_proc=filter_image(image_proc)
        image_gray=grayscale(image_proc)
        image_blur=gaussian_blur(image_gray,cal_kernel)
        image_edges=canny(image_blur,cal_low_threshold,cal_high_threshold)
        image_region = region_of_interest(image_edges, cal_vertices)
        image_hough=hough_lines(image_region,cal_rho,cal_theta,cal_hough_threshold,cal_min_line_length,cal_max_line_gap)
        image_weighted=weighted_img(image_hough,image_raw)
        image_weighted_save = cv2.cvtColor(image_weighted, cv2.COLOR_BGR2RGB) # Change in format for saving
        cv2.imwrite(test_output_path + file, image_weighted_save)
        plt.imshow(image_weighted)
        plt.show()
        if cv2.waitKey(0) and 0xFF == ord('q'): # Wait to show image for 5 milliseconds and 'q' tap on keyboard to close and move to next
           cv2.destroyAllWindows()

def process_image(image):
    #image_raw = (mpimg.imread(image)).astype('uint8')
    image_raw=image
    image_proc = np.copy(image_raw)
    image_proc= filter_image(image_proc)
    image_gray = grayscale(image_proc)
    image_blur = gaussian_blur(image_gray, cal_kernel)
    image_edges = canny(image_blur, cal_low_threshold, cal_high_threshold)
    image_region = region_of_interest(image_edges, cal_vertices)
    image_hough = hough_lines(image_region, cal_rho, cal_theta, cal_hough_threshold, cal_min_line_length,
                              cal_max_line_gap)
    image_weighted = weighted_img(image_hough, image_raw)
    return image_weighted
def main():
    test_algo_images(list_test_im)
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    #clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    #%time white_clip.write_videofile(white_output, audio=False)
    white_clip.write_videofile(white_output, audio=False)

    yellow_output = 'test_videos_output/solidYellowLeft.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
    clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)

main()