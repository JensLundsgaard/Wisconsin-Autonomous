import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as optimize

# Opening image
img = cv.imread("red.png")

# img = cv.rotate(img, cv.ROTATE_180)
  
# OpenCV opens images as BRG 
# but we want it as RGB and 
# we also need a grayscale 
# version
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
  
# Creates the environment 
# of the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()

# Defining threshholds 
img_thresh_low = cv.inRange(img_hsv, np.array([0, 135, 135]), np.array([15, 255, 255]))
img_thresh_high = cv.inRange(img_hsv, np.array([159, 135, 135]), np.array([179, 255, 255]))
img_thresh = cv.bitwise_or(img_thresh_low, img_thresh_high) 

#img_thresh = cv.cvtColor(img_thresh, cv.)
# plt.imshow(img_thresh)
# plt.show()


kernel = np.ones((5, 5))
img_thresh_opened = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel)

# plt.imshow(img_thresh_opened)
# plt.show()

img_thresh_blurred = cv.medianBlur(img_thresh_opened, 5)
# plt.imshow(img_thresh_blurred)
# plt.show()


# Apply the Canny edge detector
img_edges = cv.Canny(img_thresh_blurred, 70, 255)
# img_edges = cv.Canny(img_thresh, 0, 255)
# plt.imshow(img_edges)
# plt.show()

contours, _ = cv.findContours(np.array(img_edges), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

img_contours = np.zeros_like(img_edges)
cv.drawContours(img_contours, contours, -1, (255,255,255), 2)
# plt.imshow(img_contours)
# plt.show()


approx_contours = []

for c in contours:
    approx = cv.approxPolyDP(c, 10, closed = True)
    approx_contours.append(approx)
img_approx_contours = np.zeros_like(img_edges)
cv.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)
# plt.imshow(img_approx_contours)
# plt.show()

all_convex_hulls = []
for ac in approx_contours:
# for ac in img_contours:
    all_convex_hulls.append(cv.convexHull(ac))

img_all_convex_hulls = np.zeros_like(img_edges)
cv.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)
# plt.imshow(img_all_convex_hulls)
# plt.show()

convex_hulls_3to10 = []
for ch in all_convex_hulls:
    if 3 <= len(ch) <= 10:
        convex_hulls_3to10.append(cv.convexHull(ch))
img_convex_hulls_3to10 = np.zeros_like(img_edges)
cv.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)
# plt.imshow(img_convex_hulls_3to10)
# plt.show()

print(type(convex_hulls_3to10[0]))
print(convex_hulls_3to10[0])


def convex_hull_pointing_up(ch:np.ndarray) -> bool:
    points_above_center, points_below_center = [], []
    
    _, y, _, h = cv.boundingRect(ch) 
    

    
    
    vertical_center = y + h / 2

    for point in ch:
        if point[0][1] < vertical_center: # если координата y точки выше центра, то добавляем эту точку в список точек выше центра
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

for ch in convex_hulls_3to10:
    if convex_hull_pointing_up(ch):
        cones.append(ch)
        rect = cv.boundingRect(ch)
        bounding_rects.append(rect)
    

img_cones = np.zeros_like(img_edges)
cv.drawContours(img_cones, cones, -1, (255,255,255), 2)
# cv.drawContours(img_cones, bounding_rects, -1, (1,255,1), 2)

# plt.imshow(img_cones)
# plt.show()

img_res = img_rgb.copy()
cv.drawContours(img_res, cones, -1, (255,255,255), 2)

for rect in bounding_rects:
    cv.rectangle(img_res, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 3)

# for rect in non_cone_bounding_rects:
#     cv.rectangle(img_res, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 1, 255), 3)

plt.imshow(img_res)
plt.show()


def least_squares(x, y):

    # Create the least squares objective function.
    def func(x, a, b):
        return a * x + b

    

    popt, pcov = optimize.curve_fit(func, x, y)

    return popt

# def two_lines_least_squares(x, y):
    
#     # Create the least squares objective function.
#     # def func(x_val, a1, b1, a2, b2, m, b):
#     #     lowest_dist = 1000
#     #     lowest_index = 0
#     #     print(list(enumerate(x)))
#     #     for i, val in enumerate(x):
#     #         if(abs(x_val - val) < lowest_dist):
#     #             lowest_dist = abs(x_val - val)
#     #             lowest_index = i
        
#     #     if(m * x_val + b > y[lowest_index]):
#     #         return a1 * x_val + b1
#     #     return a2 * x_val + b2

#     def func(x_val, a1, b1, a2, b2):
#         if((a2 * x_val) + b2 < (a1 * x_val) + b1).all():
#             return (a2 * x_val) + b2
#         return (a1 * x_val) + b1

    

#     popt, pcov = optimize.curve_fit(func, x, y)

#     return popt
    
# cone_points = [(rect[0] + rect[2]/2, rect[1] + rect[3]/2) for rect in bounding_rects]
# a1, b1 = least_squares(np.array([i[0] for i in cone_points]), np.array([i[1] for i in cone_points]))

cone_points_left = [(rect[0] + rect[2]/2, rect[1] + rect[3]/2) for rect in bounding_rects if rect[0] + rect[2]/2 < img_res.shape[1]/2]
cone_points_right = [(rect[0] + rect[2]/2, rect[1] + rect[3]/2) for rect in bounding_rects if rect[0] + rect[2]/2 > img_res.shape[1]/2]

a1, b1 = least_squares(np.array([i[0] for i in cone_points_left]), np.array([i[1] for i in cone_points_left]))
a2, b2 = least_squares(np.array([i[0] for i in cone_points_right]), np.array([i[1] for i in cone_points_right]))


cv.line(img_res, [0, int(b1)], [3000, int((3000 * a1) + b1)], (255,1,1), 5)
cv.line(img_res, [0, int(b2)], [3000, int((3000 * a2) + b2)], (255,1,1), 5)

plt.imshow(img_res)
plt.savefig("sample answer.png")
plt.show()