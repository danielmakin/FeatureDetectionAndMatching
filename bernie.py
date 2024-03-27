'''This File will Have the Main Focus of Trying to find the Icon that is Bernie.'''
import cv2
from matplotlib import pyplot as plt
import numpy as np

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

    for threshold in range(int(np.max(points)) + 1):
        binary_image = points > threshold
    
        # Count the number of keypoints
        num_keypoints = np.sum(binary_image)

        arr.append(num_keypoints)

    return arr

def PlotGraph(points):

    vals = GetThresholdPoints(points)

    # Plot the Effect of the Threshold Value being Changed
    plt.plot(vals)
    plt.xlabel('Threshold Value')
    plt.ylabel('Number of Keypoints')
    plt.title('Number of Keypoints vs. Threshold Value')
    plt.grid(True)
    plt.savefig("KeyPointsvThreshold.png")

def FeatureDescription(img, kp):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # Compute the descriptors with ORB
    # Find the keypoints with ORB

    kps = ConvertToKeyPoints(kp)
    kp, my_des = orb.compute(img, kps)
    # img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    # plt.imshow(img2)
    # plt.show()


    kp = orb.detect(img,None)

    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    # img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    # plt.imshow(img2), plt.show()

    # print(my_des)

    return my_des

def ConvertToKeyPoints(arr):
    kps = []

    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[x, y] != 0:
                kp = cv2.KeyPoint(y, x, size=1)
                kps.append(kp)

    return kps

def ProcessImage(path):
    '''Return the Feature Descriptors and KPs'''
    points, orien = HarrisPointDetector(path)
    points = ThresholdPoints(points, 0.025)
    bernie = ReadImage(path)
    des = FeatureDescription(bernie, points)
    return points, des

def ssd(p1, p2):
    # Get the Distance Between
    return np.linalg.norm(p1 - p2)

def best_match(p1s, p2s, d1s, d2s, ratio):
    '''Get the Best Matching Feature in Both'''

    arr = [[0 for _ in range(len(p2s))] for _ in range(len(p1s))]

    for i in range(len(d1s)):
        for j in range(len(d2s)):
            # Calculate the Distance Between Every Feature
            arr[i][j] = ssd(d1s[i], d2s[j])
    print(arr)

    matches = []

    while True:
        # Find the Array that Has the Highest Index
        i = np.argmax(np.max(arr, axis=1))
        # Get that array located at the index
        sub_array = arr[i]
        # Reset that Index incase the Loop is ran again
        tmp = [0 for _ in range(len(p2s))]
        arr[i] = tmp
        # Now find the highest index of that array
        j = np.argmax(sub_array)

        # Store that Value in the Variable
        max_val = sub_array[j]

        print(max_val)

        if max_val == 0:
            # This Means no Values are Left
            print("No Match Found")
            break
        sub_array[j] = -1

        # Then get the second Highest
        k = np.argmax(sub_array)
        second_max_val = sub_array[k]

        # Calculate the Ratio
        r = second_max_val / max_val

        # print(r)

        # if r < ratio:
        tmp = cv2.DMatch(i, j, max_val)

        matches.append(tmp)

    # Return the Points that Are Relevant to This
    return matches

    # Now get the best Features


    # Get the Highest Value

def draw_lines(img1, img2, p1, p2, matches):
    ref = ReadImage(img1)
    curr = ReadImage(img2)

    matched_img = cv2.drawMatches(ref, p1, curr, p2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('Matches', matched_img)
    cv2.waitKey(0)



ims = ['bernie180.jpg', 'darkerBernie.jpg',
       'bernieBenefitBeautySalon.jpeg', 'BernieFriends.png', 
       'bernieMoreblurred.jpg', 'bernieNoisy2.png', 
       'berniePixelated2.png', 'bernieShoolLunch.jpeg', 
       'brighterBernie.jpg']

kps, des = ProcessImage("Bernies/bernieSanders.jpg")

for im in ims:
    kps_temp, des_temp = ProcessImage("Bernies/" + im)

    # Now get the Matches
    matches = best_match(kps, kps_temp, des, des_temp, 0.8)

    draw_lines("Bernies/bernieSanders.jpg", "Bernies/" + im, kps, kps_temp, matches)

    print("Done")


