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
    # compute the descriptors with ORB
    # find the keypoints with ORB

    kps = ConvertToKeyPoints(kp)
    kp, des = orb.compute(img, kps)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()


    kp = orb.detect(img,None)

    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()

    print(des.shape)

    return des

def ConvertToKeyPoints(arr):
    kps = []

    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[x, y] != 0:
                kp = cv2.KeyPoint(y, x, size=1)
                kps.append(kp)

    return kps

def ProcessImage(path):
    points, orien = HarrisPointDetector(path)

    points = ThresholdPoints(points, 0.025)

    bernie = ReadImage(path)

    FeatureDescription(bernie, points)

ims = ['bernieSanders.jpg', 'bernie180.jpg', 
       'bernieBenefitBeautySalon.jpeg', 'BernieFriends.png', 
       'bernieMoreblurred.jpg', 'bernieNoisy2.png', 
       'berniePixelated2.png', 'bernieShoolLunch.jpeg', 
       'brighterBernie.jpg','darkerBernie.jpg']

for im in ims:
    ProcessImage("Bernies/" + im)

# ProcessImage("bernieSanders.jpg")

# ProcessImage("Bernies/bernieBenefitBeautySalon.jpeg")

# ProcessImage("Bernies/bernieShoolLunch.jpeg")

# ProcessImage("Bernies/bernie180.jpg")

