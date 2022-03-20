from pickletools import uint8
import cv2
import numpy as np

def add_circles(image,cords_list):
    cur_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    for cord in cords_list:
        c_cord=(int(cord[0][0]),int(cord[0][1]))
        cur_image = cv2.circle(cur_image,center=c_cord,radius=3,color=(0,255,0),thickness=-1)
    return cur_image

def main():
    
    base = cv2.imread("large2_uttower_left.jpg", cv2.IMREAD_GRAYSCALE)
    trans = cv2.imread("uttower_right.jpg", cv2.IMREAD_GRAYSCALE)

    detector = cv2.AKAZE_create()

    # find the keypoints and descriptors with SIFT
    kp_base, desc_base = detector.detectAndCompute(base, None)
    kp_trans, desc_trans = detector.detectAndCompute(trans, None)

    # create BFMatcher object
    bf_match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf_match.match(desc_trans,desc_base)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    #print(matches)

    trans_pts = np.float32([ kp_trans[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
    base_pts = np.float32([ kp_base[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
    #print(src_pts)
    #print(dst_pts)

    # Draw matching points on both images.
    trans_circle = add_circles(trans,trans_pts)
    base_circle = add_circles(base,base_pts)

    cv2.imwrite("Transform_Image_Matching_Points.png",trans_circle)
    cv2.imwrite("Base_Image_Matching_Points.png",base_circle)
    cv2.imshow('Transform Image', trans_circle)
    cv2.imshow('Base Image', base_circle)
    
    # Compute Homeography by using the trans_pts and base_pts.
    trans_matrix, _ = cv2.findHomography(trans_pts,base_pts,cv2.RANSAC) 
    #print(trans_matrix)

    #Prefrom warped perspective using the transformation matrix on the trans image. 
    #Use the base.shape so that it has the same matrix size as the other.
    trans_warp = cv2.warpPerspective(trans, trans_matrix, (base.shape[1], base.shape[0]))
    cv2.imshow("Transform Warp", trans_warp)
    cv2.imwrite("Transform_Warp.png", trans_warp)
    
    #Merge the images using the or wise method. 
    merge_image = base | trans_warp
    cv2.imshow("Merged Image", merge_image)
    cv2.imwrite("Merged_Image.png", merge_image)

    cv2.imshow("Merged Image", merge_image)
    cv2.imwrite("Merged_Image.png", merge_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
