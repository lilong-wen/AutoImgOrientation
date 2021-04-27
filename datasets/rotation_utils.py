import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    R_rand = random.randint(150,255)
    G_rand = random.randint(150,255)
    B_rand = random.randint(150,255)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int(math.ceil((h * sin) + (w * cos)))
    nH = int(math.ceil((h * cos) + (w * sin)))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_WRAP)

def rotate(image, angle, center = None, scale = 1.0):

    (h, w) = image.shape[:2]
    R_rand = random.randint(150,255)
    G_rand = random.randint(150,255)
    B_rand = random.randint(150,255)
    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_WRAP)

    return rotated

def correct_ratation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    # plt.imshow(binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    rect = cv2.minAreaRect(contours[0])  #获取蓝色矩形的中心点、宽高、角度

    '''
    retc=((202.82777404785156, 94.020751953125),
     (38.13406753540039, 276.02105712890625),
     -75.0685806274414)
    '''

    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = rect[2]
    #print(angle)

    if width < height:  #计算角度，为后续做准备
      angle = angle - 90
    print(angle)

    # if  angle < -45:
    #     angle += 90.0
    #        #保证旋转为水平
    # width,height = height,width
    src_pts = cv2.boxPoints(rect)

    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # img2= img.copy()
    # cv2.drawContours(img2, [box], 0, (0,255,0), 2)
    #
    plt.imshow(img2)
    dst_pts = np.array([[0, height],
                        [0, 0],
                        [width, 0],
                        [width, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    if angle<=-90:  #对-90度以上图片的竖直结果转正
        warped = cv2.transpose(warped)
        warped = cv2.flip(warped, 0)  # 逆时针转90度，如果想顺时针，则0改为1
        # warped=warped.transpose

    return warped

def rotate_bound_black(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    R_rand = random.randint(150,255)
    G_rand = random.randint(150,255)
    B_rand = random.randint(150,255)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int(math.ceil((h * sin) + (w * cos)))
    nH = int(math.ceil((h * cos) + (w * sin)))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
