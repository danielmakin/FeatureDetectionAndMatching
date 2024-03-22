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
    bernie = SobelOperator(bernie)

def SobelOperators(img):
    #Pad the Image
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)

    return sobel_x, sobel_y
    


bernie = ReadImage("bernieSanders.jpg")