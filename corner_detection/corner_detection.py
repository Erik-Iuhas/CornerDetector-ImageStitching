from tkinter.tix import WINDOW
import cv2
import numpy as np
from sympy import true
from copy import deepcopy

class coner():
    """This class is responsible for handling the global variables for the images so it can easily be accessed through class methods. 
    """
    def __init__(self) -> None:
        """This method is responsible for loading the image and setting initial variables.
        """
        # Set file name for image
        self.image_name = "box_in_scene.png"
        # Read in image as a grayscale
        self.orig_img = cv2.imread(self.image_name,cv2.IMREAD_GRAYSCALE)
        # Display the un altered image
        cv2.imshow("Original Image", self.orig_img)
        
        # Set initial value of slider to 100 which is then divided to equal 0.01
        self.INITIAL_SLIDER = 100
        # The max value of the slider can only go up to 0.02
        self.MAX_SLIDER = 200
        # Sets the window size for minimum eigen values and non max supression
        self.WINDOW_SIZE = 5
        
        self.save_image_once = True
    

    def binary_threshold(self,image,threshold):
        """Threshold pixel values to zero if they are less than the threshold
        and set pixels to 255 if they are greater. 

        Args:
            image (numpy array): A numpy array which stores the image
            threshold (float): A float value which dictates the threshold 

        Returns:
            (numpy array): Retruns an image which has the thresholded values.  
        """
        image[image < threshold] = 0
        image[image >= threshold] = 255
        return image

    

    def min_eigen_trackbar(self,threshold):
        """This method is handling the entire process of edge detection which is then called by the trackbar

        Args:
            threshold (int): threshold value which determines the intensity of the value for binary_threshold method. 
        """
        # Use cornerMinEigenVal to find the minimum eignen values.
        min_eigen = cv2.cornerMinEigenVal(self.orig_img,self.WINDOW_SIZE)
        #print(np.max(min_eigen))
    
        # Threshold the result to obtain pixels which are greater than threshold/10000 
        self.threshold_eigen = self.binary_threshold(deepcopy(min_eigen),threshold/10000)

        # Set all pixels that pass to 1 to find the passing corner values.
        passing_edges = self.threshold_eigen/255

        # Get cordinates of all passing edges to ignore all pixels which don't pass the conditions for non_maximum_supression
        non_max_list = np.argwhere(passing_edges > 0)

        # Multiply the values in passing_edges to have an array containing only the values of the passing edges.
        new_passed_photo = passing_edges *deepcopy(min_eigen) 

        # Apply non maximum supression which keeps only the brightest pixels in a 5 by 5 window.
        cords_list = self.non_maximum_suppression(new_passed_photo,non_max_list,filter_size=self.WINDOW_SIZE)

        # Add circles to the original image to show resulting corners
        circle_image = self.add_circles(cords_list)
        #print(len(cords_list))



        #Display the threshold result 
        cv2.imshow("Threshold Slider Window", self.threshold_eigen)
        #Display the final image with circles added
        cv2.imshow("Final Non Max Circles", circle_image)
        #Display the corner min eigen value window
        cv2.imshow("cornerMinEigenVal Window",min_eigen)

        if(self.save_image_once):
            self.save_image_once = False
            #Display the threshold result 
            cv2.imwrite("Threshold_Slider.png", self.threshold_eigen)
            #Display the final image with circles added
            cv2.imwrite("Final_Non_Max_Circles.png", circle_image)
            #Display the 
            cv2.imwrite("cornerMinEigenVal.png",min_eigen)


    def pad_image(self,image,pad):
        """Method is responsbile for doing a blank type padding where it adds blank pixels to the edges of the image.

        Args:
            image (numpy array): image that will be padded
            pad (int): how many pixels of padding that will be added.

        Returns:
            numpy array: padded image
        """
        padded_image = image
        for top in range(pad):
            top_bottom_pad = np.zeros(padded_image[0].shape)
            # Make blank array with length of the width to pad on top and bottom
            padded_image = np.vstack((top_bottom_pad,padded_image))
            padded_image = np.vstack((padded_image,top_bottom_pad))
        for rows in range(pad):
            left_right_pad = np.zeros(padded_image[:,0].shape)
            # Make blank array with height of image to pad on the left and right
            padded_image= np.insert(padded_image,0,left_right_pad,axis=1)
            padded_image = np.insert(padded_image,padded_image.shape[1],left_right_pad,axis=1)
        return padded_image

    def add_circles(self,cords_list):
        """Convert the image back into rgb and then adds circles to the image at the specified cords

        Args:
            cords_list (array[cords]): An array which stores cordinates for the passing corners.

        Returns:
            numpy array: returns a coloured image with circles drawn at corner points
        """
        # Change image to RGB
        cur_image = cv2.cvtColor(self.orig_img,cv2.COLOR_GRAY2RGB)
        for cord in cords_list:
            # Draw cordinate with cv2.Circle
            cur_image = cv2.circle(cur_image,center=cord,radius=3,color=(0,255,0),thickness=-1)

        return cur_image

    def non_maximum_suppression(self,image,non_max_list, filter_size):
        """This method iterates through a list of cordinates in the image and determines if the pixel in the center
        is the greatest value. If it is it is added to a list for passing corner points, if not it is ignored.

        Args:
            image (numpy array): contains the image from new_passed_photo which are pixels that passed thresholding
            non_max_list (array[cords[]]): contains the cordinates of the passing pixels
            filter_size (int): the filter size which determines how big the window must for the non_maximum_supression

        Returns:
            array[cords()]: returns a list of cordinates
        """

        # Calculate the required amount of padding needed to keep the original dimensions of the image
        pad = filter_size//2

        # Padding image the lets us optimize by not needing to check the image bounds.
        padded_image = self.pad_image(image,pad)

        # Set a blank array which will consist with the passing corner cords.
        brightest_centers = []

        # Iterate through the non_max_list of pixels which passed the threshold.
        for cords in non_max_list:
            # Obtain x and y value of non padded cordinates
            y = cords[0] + pad
            x = cords[1] + pad
            # Cut out a portion of the image with a pad+1 by pad+1 size.
            non_max_filter = padded_image[y-pad :y + pad+1,x-pad :x + pad+1]

            # Check the array for the index of the largest value
            max_index = np.unravel_index(np.argmax(non_max_filter, axis=None), non_max_filter.shape)

            # if the largest index is (2,2) that means the center pixel is the largest and can pass the non maximum supression test.
            if(max_index == (pad,pad)):
                # Append cordinates to brightest_centers array.
                brightest_centers.append((cords[1],cords[0]))
                            
        # Return the non maximum supression corners 
        return brightest_centers

    def main(self):
        """Main class responsible for calling the method which runs the entire process.
        """
        # Set a name for the trackbar slider window.
        cv2.namedWindow("Threshold Slider Window")

        # Set initial values for the trackback 
        cv2.createTrackbar("Current", "Threshold Slider Window" ,self.INITIAL_SLIDER, self.MAX_SLIDER, self.min_eigen_trackbar)
        cv2.waitKey(0)
        

if __name__ == "__main__":
    # Construct and run main class.
    coner_class = coner()
    coner_class.main()
