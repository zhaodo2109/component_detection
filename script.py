from zdlib import *
from tkinter import Tk
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename, asksaveasfilename

import time

print("OpenCV Version:", cv2.__version__)
print("Numpy Version: ", np.__version__)

print("This program takes in a grayscale image and identify its object. "
      "The object detection will be using the connected component by flood fill algorithm while the"
      "stem detection for bananas will be using connected component by union find algorithm.")

Tk().withdraw()
filename = askopenfilename() # open img file
start_time = time.time()

image = cv2.imread(filename, 0)

color_image = cv2.imread(filename, 1)
out_img=color_image.copy()

cv2.imshow("Original Image", image) #show original image
height, width = image.shape #get img dimension
height1, width1,channels = color_image.shape #get img dimension

#Apply Double thresholding
thresh = double_threshold(image, np.zeros((height, width), dtype=np.uint8),95,190)

bananas = np.zeros((height, width), dtype=np.uint8)
# cv2.imshow("tnresn Image", thresh) #show original image

# Apply cleaning operation
clean=clean_image(thresh, np.zeros((height, width), dtype=np.uint8))
cv2.imshow("Clean Threshold Image", clean)

#connected components = cc

#cc-floodfill
proc_img,num_obj,labels=connectedComponentRepeatedFloodFill(clean)
# print("FF",num_obj)
# print("FF label",labels)
cv2.imshow("Connected Components FF", proc_img)

#cc-unionfind
proc_img1,num_obj1,labels1=connected_components_union_find(clean)

# print("UF",num_obj1)
# print("UF labels:",labels1)
cv2.imshow("Connected Components UF", proc_img1)

#initialize a color palette
colors = [(0,255,255), # yellow
          (255,0,0), #blue
          (0,255,0), #green
          (0,0,255), #red
          (255,255,0), #cyan
          (255, 0, 255), #magneta
          (128, 0, 0), #navy blue
          (0, 165, 255), #orange
          (180,105,255), #pink
          (169,171,0), #teal
          (0,0,0)] #black

#object = "fruit"

# wall following and label fruits with axis on cc-floodfill image
for obj in range(num_obj):
    #for label in labels:
    label = labels[obj]
    path = wallfollowing(proc_img,label) #detect object
    moments,central_m,area = region_properties(proc_img,label,num_obj) #moments calculation
    ev1,ev2,theta,major_ax_length,minor_ax_length,eccentricity = pca(moments,central_m)
    #moments = [m00,m01,m10,m11,m02,m20]
    xc, yc = (moments[2] // moments[0], moments[1] // moments[0])
    # print(area)
    length_scale = np.log10(1 +area)
    # print(length_scale)

    #(x1,y1) major axis
    # scale with log scaling with respect to object area
    x1 = int(xc + np.log10(major_ax_length + 1)*length_scale * np.cos(theta))
    y1 = int(yc + np.log10(major_ax_length + 1)*length_scale * np.sin(theta))

    # get symmetrical axis at centroid
    # scale with log scaling with respect to object area

    x1m = int(xc - (x1 - xc))
    y1m = int(yc - (y1 - yc))

    # print(x1,y1)
    # print(x1m,y1m)

    #(x2,y2) minor axis
    # scale with log scaling with respect to object area
    x2 = int(xc + np.log10(major_ax_length + 1)*0.7*length_scale * np.cos(theta+np.pi/2))
    y2 = int(yc + np.log10(major_ax_length + 1)*0.7*length_scale * np.sin(theta+np.pi/2))
    x2m = int(xc - (x2 - xc))
    y2m = int(yc - (y2 - yc))

    # label banana
    if eccentricity > 0.8:
        #object ="banana"
        # better scale as banana has longer shape
        x1 = int(xc + np.log10(major_ax_length + 1) * 2 * length_scale * np.cos(theta))
        y1 = int(yc + np.log10(major_ax_length + 1) * 2 * length_scale * np.sin(theta))
        x1m = int(xc - (x1 - xc))
        y1m = int(yc - (y1 - yc))

        FloodfillSeparate(path[0], proc_img, bananas, 255)

        #draw major axis
        cv2.line(out_img,(xc,yc),(x1,y1),colors[0],2)
        cv2.line(out_img, (xc, yc), (x1m, y1m), colors[0], 2)


        #draw minor axis
        cv2.line(out_img, (xc, yc), (x2, y2), colors[0],2)
        cv2.line(out_img, (xc, yc), (x2m, y2m), colors[0],2)

    # label tangerine
    elif major_ax_length > 2000 and minor_ax_length > 2000:
        #object ="tangerine"
        for pixels in path:
            out_img[pixels] = colors[7]
        #draw major axis
        cv2.line(out_img, (xc, yc), (x1, y1), colors[7],2)
        cv2.line(out_img, (xc, yc), (x1m, y1m), colors[7], 2)

        #draw minor axis
        cv2.line(out_img, (xc, yc), (x2, y2), colors[7], 2)
        cv2.line(out_img, (xc, yc), (x2m, y2m), colors[7],2)
    # label apple
    else:
        #object ="apple"
        for pixels in path:
            out_img[pixels] = colors[3]
        #draw major axis
        cv2.line(out_img, (xc, yc), (x1, y1), colors[3],2)
        cv2.line(out_img, (xc, yc), (x1m, y1m), colors[3], 2)

        #draw minor axis
        cv2.line(out_img, (xc, yc), (x2, y2), colors[3],2)
        cv2.line(out_img, (xc, yc), (x2m, y2m), colors[3],2)

    # print(f"{obj} moments: {moments}")
    # print(f"{obj} central moments: {central_m}")
    # print(f"{label} ev1: {round(ev1, 2)} ev2: {round(ev2, 2)} phase: {round(phase, 2)} "
    #       f"major_ax_length: {round(major_ax_length, 2)} "
    #       f"minor_ax_length: {round(minor_ax_length, 2)} eccentricity: {round(eccentricity, 2)}")

#create a new image to store banana body
banana_body = np.zeros((height, width), dtype=np.uint8)

#create an empty image to perform morphology
temp_img = np.zeros((height, width), dtype=np.uint8)

morphed_img  = erosion(bananas, temp_img.copy())

# erode the banana until the stem is completely removed
for i in range(6):
    morphed_img  = erosion(morphed_img, temp_img.copy())

# restore the banana body
for i in range(8):
    morphed_img  = dilation(morphed_img, temp_img.copy())

# clean the banana body image
banana_body = erosion(morphed_img , banana_body)
banana_body = dilation(banana_body,temp_img.copy())
banana_body = dilation(morphed_img , banana_body)

# create a new image banana_stem that only contain the stem of the banana
banana_stem = bananas - banana_body #remove banana body, keep stem

# clean the banana stem image
banana_stem  = dilation(banana_stem , temp_img.copy())
banana_stem  = erosion(banana_stem , temp_img.copy())
banana_stem  = erosion(banana_stem , temp_img.copy())
banana_stem  = dilation(banana_stem , temp_img.copy())
banana_stem  = dilation(banana_stem , temp_img.copy())

# use cc-union find to detect banana body
body,num_body,labels=connected_components_union_find(banana_body)
#cv2.imshow("label banana body", body) #show banana body labeled image

for num in range(num_body):
    for label in labels:
        body_outline = wallfollowing(body, label)  # detect banana body
        for pixels in body_outline:
            out_img[pixels] = colors[0]

# use cc-union find to detect banana stem
stem,num_stem,labels=connected_components_union_find(banana_stem)
#cv2.imshow("label banana stem", stem) #show banana stem labeled image

for num in range (num_stem):
    for label in labels:
        stem_outline = wallfollowing(stem, label)  # detect stem
        for pixels in stem_outline:
            out_img[pixels] = colors[5]

cv2.imshow("Classified Objects", out_img) #show classified object image

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")

cv2.waitKey(0)
cv2.destroyAllWindows() # close all image windows


