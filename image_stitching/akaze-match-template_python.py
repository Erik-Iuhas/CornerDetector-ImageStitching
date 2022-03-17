import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('uttower_right.jpg',0)          # queryImage
img2 = cv2.imread('large2_uttower_left.jpg',0) # trainImage

detector = cv2.AKAZE_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
print(matches)

src_pts = np.float32([ kp1[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
print(src_pts)
print(dst_pts)

cv2.imshow('half', img1)
cv2.imshow('long', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
