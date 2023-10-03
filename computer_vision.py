import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as optimize

# Opening image
img = cv.imread("red.png")

# Uncomment this and run the program to make sure the 
# convex_hull_pointing_up algorithm works
# img = cv.rotate(img, cv.ROTATE_180)
  

# OpenCV stores images as BGR by default 
# so the following two lines flip the color channels
# to RGB and HSV
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
  
# Creates the environment 
# of the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()

# Defining threshholds so that we can isolate the HSV pixels that
# have the desired color
img_thresh_low = cv.inRange(img_hsv, np.array([0, 135, 135]), np.array([15, 255, 255]))
img_thresh_high = cv.inRange(img_hsv, np.array([159, 135, 135]), np.array([179, 255, 255]))
# This adds the two throshold maps together
img_thresh = cv.bitwise_or(img_thresh_low, img_thresh_high) 



# Uses erosion followed by dilation to remove noise
kernel = np.ones((5, 5))
img_thresh_opened = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel)


# Blurs the image slightly
img_thresh_blurred = cv.medianBlur(img_thresh_opened, 5)



# Find edges with the Canny edge detection algorithm
img_edges = cv.Canny(img_thresh_blurred, 70, 255)

# Gets contours 
contours, _ = cv.findContours(np.array(img_edges), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
img_contours = np.zeros_like(img_edges)
# Draws contours to a new image
cv.drawContours(img_contours, contours, -1, (255,255,255), 2)


# Approximates or rather simplifies contours using the Douglas-Peucker algorithm
approx_contours = []
for c in contours:
    approx = cv.approxPolyDP(c, 5, closed = True)
    approx_contours.append(approx)
img_approx_contours = np.zeros_like(img_edges)
cv.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)

# Finds convex hulls of the contours
all_convex_hulls = []
for ac in approx_contours:
    all_convex_hulls.append(cv.convexHull(ac))

img_all_convex_hulls = np.zeros_like(img_edges)
cv.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)

# removes any hulls with more than 10 or less than 3 points
convex_hulls_3to10 = []
for ch in all_convex_hulls:
    if 3 <= len(ch) <= 10:
        convex_hulls_3to10.append(cv.convexHull(ch))
img_convex_hulls_3to10 = np.zeros_like(img_edges)
cv.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)




# The following is an function that determines whether a hull is pointing up
# It works by simply finding the middle horizantal line and splitting up the
# points into top and bottom ones with it. It then simply returns a boolean 
# that represents whether all of the top points have x-values that are between 
# the most extreme x-values of the bottom points
def convex_hull_pointing_up(ch:np.ndarray) -> bool:
    points_above_center, points_below_center = [], []
    _, y, _, h = cv.boundingRect(ch) 
    vertical_center = y + h / 2

    for point in ch:
        if point[0][1] < vertical_center: 
            points_above_center.append(point)
        elif point[0][1] >= vertical_center:
            points_below_center.append(point)

    
    x_above, _, w_above, _ = cv.boundingRect(np.array(points_above_center)) 

    x_below, _, w_below, _ = cv.boundingRect(np.array(points_below_center))

    return x_above <= x_below + w_below and x_above + w_above <= x_below + w_below \
        and x_above >= x_below and x_above + w_above >= x_below

for i in convex_hulls_3to10:
    print(i)
    print(convex_hull_pointing_up(i))

cones = []
bounding_rects = []
# Filters out the contours that aren't pointing up
for ch in convex_hulls_3to10:
    if convex_hull_pointing_up(ch):
        cones.append(ch)
        rect = cv.boundingRect(ch)
        bounding_rects.append(rect)
    

img_cones = np.zeros_like(img_edges)
cv.drawContours(img_cones, cones, -1, (255,255,255), 2)


img_res = img_rgb.copy()
cv.drawContours(img_res, cones, -1, (255,255,255), 2)

for rect in bounding_rects:
    cv.rectangle(img_res, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 3)

# for rect in non_cone_bounding_rects:
#     cv.rectangle(img_res, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 1, 255), 3)
# Draws the contours and rectangles and shows them on top of the original RGB image
plt.imshow(img_res)
plt.show()

# This function uses least squares to get the parameters for a 
# best fit line to a given set of points
def least_squares(x, y):

    # Create the least squares objective function.
    def func(x, a, b):
        return a * x + b

    

    popt, pcov = optimize.curve_fit(func, x, y)

    return popt


img_out = img_rgb.copy()
# Gets the points on the left and right sides of the screen
cone_points_left = [(rect[0] + rect[2]/2, rect[1] + rect[3]/2) for rect in bounding_rects if rect[0] + rect[2]/2 < img_res.shape[1]/2]
cone_points_right = [(rect[0] + rect[2]/2, rect[1] + rect[3]/2) for rect in bounding_rects if rect[0] + rect[2]/2 > img_res.shape[1]/2]
# Gets best fit lines for these points
a1, b1 = least_squares(np.array([i[0] for i in cone_points_left]), np.array([i[1] for i in cone_points_left]))
a2, b2 = least_squares(np.array([i[0] for i in cone_points_right]), np.array([i[1] for i in cone_points_right]))

# Draws, displays and saves final output image
cv.line(img_out, [0, int(b1)], [3000, int((3000 * a1) + b1)], (255,1,1), 5)
cv.line(img_out, [0, int(b2)], [3000, int((3000 * a2) + b2)], (255,1,1), 5)
plt.imshow(img_out)
plt.savefig("answer.png")
plt.show()