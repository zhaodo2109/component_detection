from tkinter import Tk
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

# Define 4 neighbor points relative to seed point
def neighbor_4(seed_point):
    r = [
        (seed_point[0] + 1, seed_point[1]), #down
        (seed_point[0] - 1, seed_point[1]), #up
        (seed_point[0], seed_point[1] + 1), #left
        (seed_point[0], seed_point[1] - 1)  #right
    ]
    return r

#Define floodfill separate output function
def FloodfillSeparate(seed_point, in_image, out_img, new_color):
    height, width = in_image.shape #get height and width of input image
    frontier = [seed_point] #initialize frontier array with seed point
    old_color = in_image[seed_point[0],seed_point[1]] #get current color of the seed point
    if (old_color == new_color): # if the current color is the same with the new color, return image
        return

    out_img[seed_point[0],seed_point[1]] = new_color #set  color at seed point to new color

    #process each point in frontier array
    while frontier: # while frontier array is not empty
        q = frontier.pop(0) #get next point in array
        for r in neighbor_4(q): #check its 4 neighbors
            if 0 <= r[0] < height and 0 <= r[1] < width: #check if these neighbors are in bounds
                if in_image[r[0],r[1]] == old_color and out_img[r[0],r[1]]!=new_color:
                    # add to frontier the point where old color is present in input pictures and
                    # ouput is not yet floodfilled.
                    frontier.append(r)
                    out_img[r[0],r[1]] = new_color #fill pixel with new color value
    return out_img

def floodfill_f(seed_point, image, new_color):
    height, width = image.shape #get img dimension
    frontier = [seed_point] #initilize frontier array
    old_color = image[seed_point[0], seed_point[1]]

    if old_color == new_color:
        return

    while frontier:
        q = frontier.pop(0)
        image[seed_point[0], seed_point[1]] = new_color
        for r in neighbor_4(q):
            if 0 <= r[0] < height and 0 <= r[1] < width:  # Check bounds
                #if neighbors has old color, add to frontier
                if image[r[0], r[1]] == old_color:
                    frontier.append(r)
                    image[r[0], r[1]] = new_color # set new color value to pixel
    return image

def double_threshold(in_image,out_img,low_thresh,high_thresh):
    height, width = in_image.shape #get input img dimension
    high_thresh_img = np.zeros((height, width), dtype=np.uint8) #create blank high threshold image
    low_thresh_img = np.zeros((height, width), dtype=np.uint8) #create blank low threshold image
    for i in range(height):
        for j in range(width):
            # get high thresh image
            if in_image[i, j] >= high_thresh:
                high_thresh_img[i,j]=255
                #out_img[i, j] = 255
            else:
                high_thresh_img[i, j] = 0

            # get low thresh image
            if in_image[i, j] >= low_thresh:
                low_thresh_img[i,j]=100
                # set to 255 will go against floodfill algorithm as it return low thresh image as new color for
                # output image will be set to 255

            else:
                low_thresh_img[i,j]=0

    # iterate through the image coordinate as they all have same dimension
    for i in range(height):
        for j in range(width):
            # area where the high thresh image is set to 255, perform flood fill using
            # low thresh image as input and output on output image.
            if high_thresh_img[i,j]==255:
                FloodfillSeparate((i, j), low_thresh_img, out_img, 255)

    #Show threshold images
    # cv2.imshow("High Threshold Image", high_thresh_img)
    # cv2.imshow("Low Threshold Image", low_thresh_img)
    return out_img

def histogram_equalization(in_img,out_img):
    height, width = in_img.shape #get img dimension
    total_pixel = height * width #calculate total pixel
    pixel_dict = {i: 0 for i in range(256)} #create a dictionary to store 256 pixel values
    #print(total_pixel)

    for i in range(0,height):
         for j in range(0,width):
            pixel_dict[in_img[i,j]]+=1
    #print(pixel_dict)

    # Calculate the cumulative distribution function
    cumulative_sum = 0
    cdf = {}
    for pixel_value in range(256):
        cumulative_sum += pixel_dict[pixel_value] # add the current pixel value's count to the cumulative sum
        cdf[pixel_value] = cumulative_sum / total_pixel #normalize the intensity by the total pixel
    #print(cdf)

    # equalized picture by mapping pixel value with cdf normalized value
    for i in range(0,height):
         for j in range(0,width):
            out_img[i,j] = int(255*cdf[in_img[i,j]])
    return out_img

def erosion(in_image,output_img):
    height, width = in_image.shape #get img dimension
    for i in range(height):
        for j in range(width):
            seed_point=(i,j) #define seed point

            # if the value of the current pixel and its 4 neighbor are 255, turn the pixel at current pixel
            # at ouput image to ON, otherwise, set it to 0.
            if in_image[seed_point]==255:
                if (in_image[neighbor_4(seed_point)[0]] == 255 and #right
                        in_image[neighbor_4(seed_point)[1]] == 255 and #left
                        in_image[neighbor_4(seed_point)[2]] == 255 and #up
                        in_image[neighbor_4(seed_point)[3]] == 255): #down
                    #print("left",neighbor_4(seed_point)[0],"right",neighbor_4(seed_point)[1],"up",neighbor_4(seed_point)[2],"down",neighbor_4(seed_point)[3])
                    output_img[i,j]=255
                else:
                    #print("left",neighbor_4(seed_point)[0],"right",neighbor_4(seed_point)[1],"up",neighbor_4(seed_point)[2],"down",neighbor_4(seed_point)[3])
                    output_img[i,j]=0
    return output_img

def dilation(in_image, out_img):
    height, width = in_image.shape #get img dimension

    #check each pixel in image, if the pixel in the input image is 255, set the output image at pixel to 255
    #check if neighbors of output image is in range, if they are also turn them to 255.
    for i in range(height):
        for j in range(width):
            if in_image[i, j] == 255:
                out_img[i, j] = 255
                for r in neighbor_4((i, j)):
                    if 0 <= r[0] < height and 0 <= r[1] < width:
                        out_img[r[0], r[1]] = 255
    return out_img

def clean_image(in_image,out_image):
    height, width = in_image.shape
    temp_img = np.zeros((height, width), dtype=np.uint8)

    # Apply dilation first
    morphed_img  = dilation(in_image, temp_img.copy())

    # Apply erosion multiple times
    for i in range(4):
        morphed_img  = erosion(morphed_img , temp_img.copy())

    # Finally dilate to clean up, directly update the output image
    morphed_img = dilation(morphed_img, temp_img.copy())

    out_image = dilation(morphed_img , out_image)

    return out_image

# connected components

def connectedComponentRepeatedFloodFill(thresh_img):
    height,width =thresh_img.shape
    label_image = np.zeros((height,width), dtype=np.uint8)
    label = 70  # Start labeling from 70
    unique_l =[] # label values array
    num_components = 0

    for i in range(height):
        for j in range(width):
            if label_image[i, j] == 0 and thresh_img[i, j] == 255:  # ON state
                FloodfillSeparate((i, j),thresh_img, label_image,  label)
                num_components += 1
                if label >= 255:
                    label = 255
                else:
                    unique_l.append(label)
                    label += 30  # Next label

    return label_image, num_components,unique_l

#equivalent label value
def setequiv(equiv,img1,img2):
    equiv_img1 = getEquiv(equiv,img1)
    equiv_img2 = getEquiv(equiv,img2)

    if equiv_img1 > equiv_img2:
        equiv[equiv_img1]=equiv_img2
    else:
        equiv[equiv_img2]=equiv_img1

def getEquiv(equiv,label):
    if label not in equiv:
        equiv[label] = label  # Initialize if not present

    if equiv[label] != label:
        equiv[label] = getEquiv(equiv, equiv[label])  # Path compression

    return equiv[label]

def connected_components_union_find(thresh_img):
    height, width = thresh_img.shape
    # Initialize intermediate processing arrays with bigger size for label more than 255
    proc_image =np.zeros((height, width), dtype=np.uint32)
    # Initialize label output image
    label_image = np.zeros((height, width), dtype=np.uint8)
    label = 30  # Start labeling from 25
    equivalent_table = {} # To store equivalent labels

    for y in range(height-1):
        for x in range(width-1):
            seed = thresh_img[y,x]

            if seed ==255: #if not background
                if seed == thresh_img[y, x - 1] and seed == thresh_img[y - 1, x]: #check I(P)==I(L) and I(P)==I(U)
                    proc_image[y,x] = proc_image[y - 1,x] #set C(P)=C(U)
                    setequiv(equivalent_table,proc_image[y,x-1],proc_image[y - 1,x]) #Set U,L equivalent
                elif seed == thresh_img[y, x - 1]: #check I(P)==I(L)
                    proc_image[y,x] = proc_image[y,x-1] #set C(P)=C(L)
                elif seed == thresh_img[y - 1, x]: #check I(P)==I(U)
                    proc_image[y,x] = proc_image[y - 1, x] #set C(P)=C(U)
                else:
                    proc_image[y, x] = label #set C(P)=label
                    equivalent_table[label] = label #append label value to equivalent label
                    label += 1 #new label value

    #print("Equivalence table after first pass:", equivalent_table)

    unique_l=[] #unique label value set

    #second pass
    #iterate through image
    for y in range(height):
        for x in range(width):
            proc_image[y, x] = getEquiv(equivalent_table,proc_image[y,x]) #update image with its equivalent label value
            if proc_image[y,x]>0: #if not background
                if proc_image[y, x] not in unique_l: #append only unique label to array
                    unique_l.append(int(proc_image[y, x]))

    num_component = len(unique_l)

    #set processed cc-union find image from type int32 to type int8 as output label
    for label in unique_l:
        label_image[proc_image == label] = label
    #print("unique label",unique_l)
    #print("after 2nd pass",equivalent_table)
    return label_image,num_component,unique_l

#properties

def region_properties(label_img,label,num_obj):
    m00 = m01 = m10 = m11 = m02 = m20 = 0
    u00 = u01 = u10 = u11 = u02 = u20 = 0
    height, width = label_img.shape

    #calculate moments
    for i in range(height):
        for j in range(width):
            if label_img[i,j]==label:
                m00 += 1 #zero order or area of the label object
                m01 += i #1st order (sums of y)
                m10 += j  # 1st order (sums of x)
                m11 += i*j #1st order x and y
                m02 += i**2 #2nd order (sums of y)
                m20 += j**2 #second order (sum of x)

    #calculate centroid
    xc,yc = (m10/m00,m01/m00)
    print('xc:', xc,'yc:',yc)


    #calculate central moments
    for i in range(height):
        for j in range(width):
            if label_img[i,j]==label:

                u00 = m00 #zero order or area
                u01 += i-yc #1st order (sums of y-yc)
                u10 += j-xc  # 1st order (sums of x-xc)
                u11 += m11-yc*m10 #1st order x and y
                u02 += m02-yc*m01 #2nd order (sums of y-yc)^2
                u20 += m20-xc*m10 #second order (sum of x-xc)^2

    moments = [m00,m01,m10,m11,m02,m20] #store moments in an array
    central_moments = [u00,u01,u10,u11,u02,u20] #store central moments in another array
    area = m00

    return moments,central_moments,area

#PCA

def pca(moments,central_m):
    #moments = [m00, m01, m10, m11, m02, m20]
    #central_moments = [u00, u01, u10, u11, u02, u20]
    central_m = np.array(central_m, dtype=float)
    #eigen value 1
    ev1=1/(2*central_m[0])*(central_m[-1]+central_m[-2]+                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            np.sqrt((central_m[-1]-central_m[-2])**2+4*central_m[-3]**2))
    #eigen value 2
    ev2=1/(2*central_m[0])*(central_m[-1]+central_m[-2]-np.sqrt((central_m[-1]-central_m[-2])**2+4*central_m[-3]**2))
    # angle
    theta = 0.5*np.atan2(2*central_m[-3],central_m[-1]-central_m[-2])
    # major axis length
    major_ax_length=2*np.sqrt(ev1)
    # minor axis length
    minor_ax_length = 2 * np.sqrt(ev2)
    #eccentricity
    eccentricity = np.sqrt(1-np.sqrt(ev2)/np.sqrt(ev1))
    # eccentricity = np.sqrt((2 * np.sqrt((central_m[-1]-central_m[-2])**2+4*central_m[-3]**2)) /
    #                       (central_m[-1] + central_m[-2] + np.sqrt((central_m[-1]-central_m[-2])**2+4*central_m[-3]**2)))
    return ev1,ev2,theta,major_ax_length,minor_ax_length,eccentricity

#### Object Detection ####
def turnright(direction):
    if direction == "N":
        direction = "E"
    elif direction == "E":
        direction = "S"
    elif direction == "S":
        direction ="W"
    else:
        direction = "N"
    return direction

def turnleft(direction):
    if direction == "N":
        direction = "W"
    elif direction == "E":
        direction = "N"
    elif direction == "S":
        direction ="E"
    else:
        direction = "S"
    return direction

def move(direction,seed): #seedpoint is (y,x)
    if direction == "N":
        return tuple([seed[0]-1,seed[1]]) #move up (y-1)
    elif direction == "E":
        return tuple([seed[0],seed[1]+1]) #move right (x+1)
    elif direction == "S":
        return tuple([seed[0]+1,seed[1]]) #move down (y+1)
    else:
        return tuple([seed[0],seed[1]-1]) #move right (x-1)

def isFrontOn(dir,in_img,seed,label):
    height,width = in_img.shape
    if dir == "N":
        if seed[0]-1>=0:
            if in_img[seed[0]-1,seed[1]] == label:
                return True
    elif dir == "E":
        if seed[1]+1  <= width:
            if in_img[seed[0],seed[1]+1] == label:
                return True
    elif dir == "S":
        if seed[0] +1  <= height:
            if in_img[seed[0]+1,seed[1]] == label:
                return True
    elif dir == "W":
        if seed[1]-1 >=0:
            if in_img[seed[0],seed[1]-1] == label:
                return True
    else:
        return False

def isLeftOn(dir,in_img,seed,label):
    height,width =in_img.shape
    if dir == "N":
        if seed[0] - 1 >= 0:
            if in_img[seed[0],seed[1]-1] == label: #check left of north
                return True
    elif dir == "E":
        if seed[1] + 1 <= width:
            if in_img[seed[0]-1,seed[1]] == label: # check left of east
                return True
    elif dir == "S":
        if seed[0] + 1 <= height:
            if in_img[seed[0],seed[1]+1] == label: #check left of south
                return True
    elif dir == "W":
        if seed[1] - 1 >= 0:
            if in_img[seed[0]+1,seed[1]] == label: #check left of west
                return True
    else:
        return False

def wallfollowing(in_image,label):
    height,width = in_image.shape

    # initialize direction pointing north
    dir="N"
    found = False
    path = []

    # iterate and find first label pixel
    for i in range(height):
        if found == True:
            break
        for j in range(width):
            if in_image[i,j]==label:
                seed = (i,j)
                path.append(seed)
                found = True
                break
    current_s = seed

    #print("start:", current_s)
    #if front in label, keep turning right to get starting position
    while isFrontOn(dir,in_image,current_s,label):
        dir = turnright(dir)
        #print("Location A:",current_s,"direction:",dir)
    #turn right to start following wall
    dir = turnright(dir)
    #print("Location B:", current_s, "direction:", dir)
    #print(len(path))

    #repeat as long as starting seed is different from ending seed.
    while True:
        #print(len(path))
        if isLeftOn(dir,in_image,current_s,label):
            dir = turnleft(dir)
            current_s = move(dir,current_s)
            path.append(current_s)
            #print("Location C:", current_s, "direction:", dir)

        elif not isFrontOn(dir,in_image,current_s,label):
            dir = turnright(dir)
            #print("Location D:", current_s, "direction:", dir)
        else:
            current_s = move(dir, current_s)
            path.append(current_s)
            #print("Location E:", current_s, "direction:", dir)
        if current_s == seed and len(path) > 1:
            break
    #print(path)

    return path

