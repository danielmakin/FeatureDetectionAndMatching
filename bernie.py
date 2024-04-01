'''This File will Have the Main Focus of Trying to find the Icon that is Bernie.'''
import cv2
from matplotlib import pyplot as plt
import numpy as np
import copy

print(cv2.__version__)

def ReadImage(path):
    '''Read the Image'''
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return im

def HarrisPointDetector(path):
    bernie = ReadImage(path)
    # Calculate the Gradients
    dx, dy = SobelOperators(bernie)

    # Implement the Functionality to Find R
    gaussian = cv2.getGaussianKernel(5, 0.5)
    Ixx = cv2.filter2D(dx * dx, -1, gaussian, borderType=cv2.BORDER_REFLECT)
    Iyy = cv2.filter2D(dy * dy, -1, gaussian, borderType=cv2.BORDER_REFLECT)
    Ixy = cv2.filter2D(dx * dy, -1, gaussian, borderType=cv2.BORDER_REFLECT)

    det = Ixx * Iyy - (Ixy * Ixy)
    trace = Ixx + Iyy

    R = det - (0.05 * (trace ** 2))

    # Non-Maxima Supression
    max_strength = cv2.dilate(R, np.ones((7, 7)))
    R *= (R == max_strength)

    # Step 5: Get orientation at every pixel
    orientation = np.arctan2(dy, dx) * 180 / np.pi

    return R, orientation

def SobelOperators(img):
    #Pad the Image
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)

    return sobel_x, sobel_y

def ThresholdPoints(points, threshold):

    max = np.max(points)
    points = points * (points > max * threshold)
    return points

def GetThresholdPoints(points):
    '''Get the Number of Items lt that Threshold value '''
    arr = []
    increments = [0.05 * i for i in range(1, 21)]  # Generate increments from 0.05 to 1.0 in steps of 0.05

    max = np.max(points)

    for inc in increments:
        binary_image = points/max > inc
    
        # Count the number of keypoints
        num_keypoints = np.sum(binary_image)

        arr.append(num_keypoints)

    return increments, arr

def PlotGraph(points, path):

    x, y = GetThresholdPoints(points)

    # Plot the Effect of the Threshold Value being Changed
    plt.plot(x, y)
    plt.xlabel('Threshold Value')
    plt.ylabel('Number of Keypoints')
    plt.title('Number of Keypoints vs. Threshold Value')
    plt.grid(True)
    plt.savefig("ThresholdGraphs/" + path)
    plt.clf()

def FeatureDescription(img, kp, path):
    # Initiate ORB detector
    orb = cv2.ORB_create()

    kps = ConvertToKeyPoints(kp)
    kps, my_des = orb.compute(img, kps)
    img2 = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)
    cv2.imwrite("MyKeyPoints/" + path, img2)


    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    cv2.imwrite("ORBKeyPoints/" + path, img2)

    return my_des, kps

def ConvertToKeyPoints(arr):
    kps = []

    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[x, y] != 0:
                kp = cv2.KeyPoint(y, x, size=1)
                kps.append(kp)

    return kps

def ProcessImage(path, thres):
    '''Return the Feature Descriptors and KPs'''
    points, orien = HarrisPointDetector("Bernies/" + path)
    PlotGraph(points, path)
    points = ThresholdPoints(points, thres)
    bernie = ReadImage("Bernies/" + path)
    des, points = FeatureDescription(bernie, points, path)
    return points, des

def SSD(p1, p2):
    # Get the Distance Between
    return np.linalg.norm(p1 - p2)

def BestMatch(p1s, p2s, d1s, d2s, ratio, path):
    '''Get the Best Matching Feature in Both'''

    arr = [[0 for _ in range(len(p2s))] for _ in range(len(p1s))]

    for i in range(len(d1s)):
        for j in range(len(d2s)):
            # Calculate the Distance Between Every Feature
            arr[i][j] = SSD(d1s[i], d2s[j])

    matches = []
    vals = []

    for i, p in enumerate(arr):
        # Get the Index and Value of the Highest Element
        p_temp = copy.deepcopy(p)
        j = np.argmin(p_temp)
        val = p_temp[j]
        # Get the Second Highest
        p_temp[j] = float('inf')
        second_val = p_temp[np.argmin(p_temp)]

        if val == 0 or second_val == 0:
            continue


        vals.append(val / second_val)

        # Check the Ratio
        if val / second_val < ratio:
            tmp = cv2.DMatch(i, j, val)
            matches.append(tmp)

        second_val = 0
        val = 0

    # Return the Points that Are Relevant to This
            
    PlotRatios(vals, path)
    return matches

def DrawLines(img1, img2, p1, p2, matches):
    ref = ReadImage(img1)
    curr = ReadImage(img2)

    matched_img = cv2.drawMatches(ref, p1, curr, p2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return matched_img

def GetRatioGraphValues(vals):
    '''Get the Number of Items lt that Threshold value '''
    vals = np.array(vals)

    arr = []
    increments = [0.05 * i for i in range(1, 21)]  # Generate increments from 0.05 to 1.0 in steps of 0.05

    max = np.max(vals)

    for inc in increments:
        arr.append(np.count_nonzero(vals > inc))

    return increments, arr

def PlotRatios(vals, path):
    x, y = GetRatioGraphValues(vals)

    # Plot the Effect of the Threshold Value being Changed
    plt.plot(x, y)
    plt.xlabel('Threshold Value')
    plt.ylabel('Number of Keypoints')
    plt.title('Number of Keypoints vs. Ratio Threshold Value')
    plt.grid(True)
    plt.savefig("RatioThresholdGraphs/" + path)
    plt.clf()

ims = ['bernie180.jpg', 'darkerBernie.jpg',
       'bernieBenefitBeautySalon.jpeg', 'BernieFriends.png', 
       'bernieMoreblurred.jpg', 'bernieNoisy2.png', 
       'berniePixelated2.png', 'bernieShoolLunch.jpeg', 
       'brighterBernie.jpg']

kpthres = [0.04, 0.6, 0.2, 0.1, 0.03, 0.38, 0.35, 0.04, 0.04]
rvals = [0.88, 0.88, 0.85, 0.85, 0.85, 0.88, 0.85, 0.85, 0.85]

kps, des = ProcessImage("bernieSanders.jpg", 0.04)

for i in range(len(ims)):
    kps_temp, des_temp = ProcessImage(ims[i], kpthres[i])

    # Now get the Matches
    matches = BestMatch(kps, kps_temp, des, des_temp, rvals[i], ims[i])

    mi = DrawLines("Bernies/bernieSanders.jpg", "Bernies/" + ims[i], kps, kps_temp, matches)

    # Save this Image
    cv2.imwrite("OutputFiles/" + ims[i], mi)

