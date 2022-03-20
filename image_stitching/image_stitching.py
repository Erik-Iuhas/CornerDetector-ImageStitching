import cv2
import numpy as np

def add_circles(image,cords_list):
    """This method is responsible for drawing circles on the images from cordinates stored in an array

    Args:
        image (numpy array): the image which will have circles drawn on it.
        cords_list (array[array[]]): array containing an array of cordinates

    Returns:
        numpy array: returns an image with cicles drawn on them 
    """

    # Convert the grayscale image to colour for displaying purposes.
    cur_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    # Iterate through te cordinates and draw it onto the image.
    for cord in cords_list:
        c_cord=(int(cord[0][0]),int(cord[0][1]))
        cur_image = cv2.circle(cur_image,center=c_cord,radius=3,color=(0,255,0),thickness=-1)
    return cur_image

def main():
    
    # Load both images as grayscale.
    base = cv2.imread("large2_uttower_left.jpg", cv2.IMREAD_GRAYSCALE)
    trans = cv2.imread("uttower_right.jpg", cv2.IMREAD_GRAYSCALE)

    # Create a detector object.
    detector = cv2.AKAZE_create()

    # find the keypoints and descriptors for both images.
    kp_base, desc_base = detector.detectAndCompute(base, None)
    kp_trans, desc_trans = detector.detectAndCompute(trans, None)

    # create Matcher object.
    bf_match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match match descriptors between the base and trans images.
    matches = bf_match.match(desc_trans,desc_base)

    # Sort matches by distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Obtain matching points that will be used to draw on both images.
    trans_pts = np.float32([ kp_trans[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
    base_pts = np.float32([ kp_base[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)

     # Draw circles on both images illustraiting the matching points.
    trans_circle = add_circles(trans,trans_pts)
    base_circle = add_circles(base,base_pts)

    # Display and write images.
    cv2.imwrite("Transform_Image_Matching_Points.png",trans_circle)
    cv2.imwrite("Base_Image_Matching_Points.png",base_circle)
    cv2.imshow('Transform Image', trans_circle)
    cv2.imshow('Base Image', base_circle)
    
    # Compute Homeography by using the trans_pts and base_pts.
    trans_matrix, _ = cv2.findHomography(trans_pts,base_pts,cv2.RANSAC) 
    # print(trans_matrix)

    # Prefrom warped perspective using the transformation matrix on the trans image. 
    # Use the base.shape so that it has the same matrix size as the other.
    trans_warp = cv2.warpPerspective(trans, trans_matrix, (base.shape[1], base.shape[0]))
    cv2.imshow("Transform Warp", trans_warp)
    cv2.imwrite("Transform_Warp.png", trans_warp)
    
    # Merge the images using the python or wise method. 
    merge_image = base | trans_warp
    cv2.imshow("Merged Image", merge_image)
    cv2.imwrite("Merged_Image.png", merge_image)

    # Output a gray base image for example showing blending.
    cv2.imwrite("Base_Image_Grey.png",base)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main method for running all the functions.
if __name__ == "__main__":
    main()
