from tkinter.tix import WINDOW
import cv2
import numpy as np

class coner():
    def __init__(self) -> None:
        #Set Image Name 
        self.image_name = "box_in_scene.png"
        #Read Image
        self.orig_img = cv2.imread(self.image_name,cv2.IMREAD_GRAYSCALE)
        #Display Original Image
        cv2.imshow("Original Image", self.orig_img)
        
        self.INITIAL_SLIDER = 100
        self.MAX_SLIDER = 200
        self.WINDOW_SIZE = 5
    

    def binary_threshold(self,image,threshold):
        image[image < threshold] = 0
        image[image >= threshold] = 255
        return image

    

    def min_eigen_trackbar(self,threshold):
        # Apply cornerMinEigenVal
        min_eigen = cv2.cornerMinEigenVal(self.orig_img,self.WINDOW_SIZE)
        print(np.max(min_eigen))
        self.threshold_eigen = self.binary_threshold(min_eigen,threshold/10000)

        #Set all pixels that pass to 1 to find the passing corner values.
        passing_edges = self.threshold_eigen/255
        new_passed_photo = passing_edges * min_eigen

        cords_list = self.non_maximum_suppression(new_passed_photo,filter_size=5)
        circle_image = self.add_circles(cords_list)
        print(len(cords_list))

        cv2.imshow("Threshold Slider Window", self.threshold_eigen)
        cv2.imshow("Final Non Max Circles", circle_image)


    # Method is responsbile for doing a blank type padding where it adds blank pixels to the edges of the image.
    # Returns the padded image
    def pad_image(self,image,pad):
        padded_image = image
        for top in range(pad):
            top_bottom_pad = np.zeros(padded_image[0].shape)
            # Use the top row to stack on the top of the image.
            padded_image = np.vstack((top_bottom_pad,padded_image))
            # Uses bottom row to stack on the bottom of the image
            padded_image = np.vstack((padded_image,top_bottom_pad))
        for rows in range(pad):
            left_right_pad = np.zeros(padded_image[:,0].shape)
            # Use the left column to insert on the left.
            padded_image= np.insert(padded_image,0,left_right_pad,axis=1)
            # Use the right column to insert on the right.
            padded_image = np.insert(padded_image,padded_image.shape[1],left_right_pad,axis=1)
        return padded_image

    def add_circles(self,cords_list):
        cur_image = cv2.cvtColor(self.orig_img,cv2.COLOR_GRAY2RGB)
        for cord in cords_list:
            cur_image = cv2.circle(cur_image,center=cord,radius=3,color=(0,255,0),thickness=-1)

        return cur_image

    def non_maximum_suppression(self,image, filter_size):
        # Create the blank array for the convolution output.
        conv_image = np.zeros(image.shape)

        # Calculate the required amount of padding needed to keep the original dimensions of the image. 
        pad = filter_size//2

        # Doing so lets us optimize by not needing to check the pixel bounds every  
        padded_image = self.pad_image(image,pad)

        # Iterate through the image using the kernal 
        brightest_centers = []
        # Iterate through the padded image starting and stopping in the areas of the orignal image.
        for x in range(pad,  image.shape[1]+pad):
            for y in range(pad,  image.shape[0]+pad):
                non_max_filter = padded_image[y-pad :y + pad+1,x-pad :x + pad+1]
                max_index = np.unravel_index(np.argmax(non_max_filter, axis=None), non_max_filter.shape)
                if(max_index == (pad,pad)):
                    brightest_centers.append((x,y))
                            
        # Return the conv image
        return brightest_centers

    def main(self):
        cv2.namedWindow("Threshold Slider Window")
        cv2.createTrackbar("Current", "Threshold Slider Window" ,self.INITIAL_SLIDER, self.MAX_SLIDER, self.min_eigen_trackbar)
        cv2.waitKey(0)
        

if __name__ == "__main__":
    coner_class = coner()
    coner_class.main()
