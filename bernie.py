'''This File will Have the Main Focus of Trying to find the Icon that is Bernie.'''
import cv2

print(cv2.__version__)

def ReadImage(path):
    '''Read the Image'''
    im = cv2.imread(path)

    return im

def HarrisPointDetector():
    bernie = ReadImage("bernieSanders.jpg")
    # Calculate the Gradients
    dx, dy = SobelOperators(bernie)

    gaussian = cv2.getGaussianKernel(5, 0.5)
    Ixx = cv2.filter2D(dx * dx, -1, gaussian, borderType=cv2.BORDER_REFLECT)
    Iyy = cv2.filter2D(dy * dy, -1, gaussian, borderType=cv2.BORDER_REFLECT)
    Ixy = cv2.filter2D(dx * dy, -1, gaussian, borderType=cv2.BORDER_REFLECT)

    det = Ixx * Iyy - (2 * Ixy)
    trace = Ixx * Iyy

    R = det - 0.05 * (trace * trace)

    return R

def SobelOperators(img):
    #Pad the Image
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)

    return sobel_x, sobel_y
    


bernie = ReadImage("bernieSanders.jpg")
print(HarrisPointDetector())