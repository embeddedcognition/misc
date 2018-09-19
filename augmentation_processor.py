#########################################################
## AUTHOR: James Beasley                               ##
## DATE: March 27, 2018                                ##
#########################################################

#############
## IMPORTS ##
#############
import cv2
import random
import numpy as np
import pickle
from sklearn.utils import shuffle
import matplotlib.image as mpimg

###############
## FUNCTIONS ##
###############

#reduce image resolution (spatial binning) while still preserving relevant features
def perform_spatial_resolution_reduction(image, new_size):
    #resize the image
    return cv2.resize(image, (new_size, new_size)) 

#perform noise reduction on image by masking off tube and background regions
def perform_noise_reduction(image):
    #make a copy to draw on
    masked_tube_img = image.copy()
    ### MASK TUBE ###
    x = 2500 #x pixel location
    y = 2265 #y pixel location
    circle_center = (x, y) #x/y center of circle
    circle_radius = 500 #radius of circle
    circle_color = (0, 0, 0) #color of circle
    #draw filled circle ("-1" parameter fills the circle) which will mask the tube
    cv2.circle(masked_tube_img, circle_center, circle_radius, circle_color, -1)
    ### MASK BACKGROUND ###
    x = 2500 #x pixel location
    y = 2230 #y pixel location
    circle_center = (x, y) #x/y center of circle
    circle_radius = 1600 #radius of circle
    circle_color = (255, 255, 255) #color of circle
    #create image of same size to draw roi (region of interest) on
    roi = np.zeros_like(masked_tube_img).astype(np.uint8)
    #draw filled circle ("-1" parameter fills the circle) which will be our region of interest
    cv2.circle(roi, circle_center, circle_radius, circle_color, -1)
    #combine masked tube image and roi mask (the yarn) to create and return the final masked image
    return cv2.bitwise_and(masked_tube_img, roi)

#translate (change position of) training example
#translation matrix found here: http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
def generate_translation_matrix(image):
    #randomly translate x
    translated_x = np.random.uniform(low=-15, high=15)
    #randomly translate y
    translated_y = np.random.uniform(low=-15, high=15)
    #return translation matrix based on above values
    return np.float32([[1, 0, translated_x],[0, 1, translated_y]])

#perform brightness adjustment (brighten or darken)
def perform_brightness_adjustment(image):
    #convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #randomly adjust V channel
    hsv[:, :, 2] = hsv[:, :, 2] * np.random.uniform(low=0.2, high=1.0)
    #convert back to RGB and return
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#perform y-axis flip
def perform_y_axis_flip(image):
    #return y-axis flipped image
    return np.fliplr(image)

#randomly translate object (within specified matrix bounds)
def perform_translation(image):
    #get tanslation matrix
    object_transform_matrix = generate_translation_matrix(image)
    #return randomly translated image
    return cv2.warpAffine(image, object_transform_matrix, (image.shape[1], image.shape[0]))

#generate a synthetic example from the supplied training example
def generate_synthetic_training_example(image):
    #list of transformation functions available
    transformation_functions = [perform_translation, perform_brightness_adjustment, perform_y_axis_flip]
    #choose the number of transformations to perform at random (between 1 and 3)
    num_transformations_to_perform = random.randint(1, len(transformation_functions))
    #perform the number of transformations chosen
    for _ in range(0, num_transformations_to_perform):
        #select a transformation function at random
        selected_transformation_function = random.choice(transformation_functions)           
        #execute the transformation function and return the result
        image = selected_transformation_function(image)
        #ensure each transformation can only be performed once by removing it from the list
        transformation_functions.remove(selected_transformation_function)
    #return transformed image
    return image

#on-the-fly synthetic data generator
def generate_synthetic_training_batch(X_train, y_train, batch_size):
    #loop forever
    while True:
        X_train_synthetic = []  #batch of images (after-processing)
        y_train_synthetic = []  #batch of labels (unmodified)
        #shuffle data
        X_train, y_train = shuffle(X_train, y_train)
        #create enough synthetic images to fill a batch
        for i in range(batch_size):
            #randomly select an index within X_train (zero indexed)
            random_index = np.random.randint(len(X_train))
            #load image
            image = mpimg.imread(X_train[random_index])
            #perform noise reduction on image
            nr_image = perform_noise_reduction(image)
            #performs spatial reduction on image
            sr_image = perform_spatial_resolution_reduction(nr_image, 256)
            #create a synthetic example based on that image
            synthetic_image = generate_synthetic_training_example(sr_image)
            #append synthetic image
            X_train_synthetic.append(synthetic_image)
            #append label
            y_train_synthetic.append(y_train[random_index])
        #yeild a new batch
        yield (np.array(X_train_synthetic), np.array(y_train_synthetic))

#generate batches of validation data
def generate_validation_batch(X_validation, y_validation, batch_size):
    #determine validation set length
    num_validation_examples = len(X_validation)
    #loop forever
    while True:
        #shuffle data
        X_validation, y_validation = shuffle(X_validation, y_validation)
        #walk through validation set loading image batches equal to batch size and yielding them
        for offset in range(0, num_validation_examples, batch_size):
            #list to store loaded images
            X_validation_image_batch = []
            #get the current batch of image paths
            cur_image_path_batch = X_validation[offset:offset+batch_size]
            #load images from paths contained in cur_image_path_batch 
            for image_path in cur_image_path_batch:
                #load image
                image = mpimg.imread(image_path)
                #perform noise reduction on image
                nr_image = perform_noise_reduction(image)
                #performs spatial reduction on image
                sr_image = perform_spatial_resolution_reduction(nr_image, 256)
                #append to validation image batch list
                X_validation_image_batch.append(image)
            #yeild batch
            yield (np.array(X_validation_image_batch), y_validation[offset:offset+batch_size])

######################################################
## DE-PICKLE, EXTRACT, SHUFFLE TRAINING & TEST SETS ##
######################################################

print()
print("De-pickling training, validation, and test data sets...")
print()

#pickled file names
training_set_file = "pickled_objects/paths_training_set.p"
validation_set_file = "pickled_objects/paths_validation_set.p"
test_set_file = "pickled_objects/paths_test_set.p"

#import pickled data files
with open(training_set_file, mode="rb") as f:
    training_set = pickle.load(f)
with open(validation_set_file, mode="rb") as f:
    validation_set = pickle.load(f)
with open(test_set_file, mode="rb") as f:
    test_set = pickle.load(f)

#unpack training, validation, and test sets    
X_train, y_train = training_set["features"], training_set["labels"]
X_validation, y_validation = validation_set["features"], validation_set["labels"]
X_test, y_test = test_set["features"], test_set["labels"]

#shuffle the data sets (image-paths/labels)
X_train, y_train = shuffle(X_train, y_train)
X_validation, y_validation = shuffle(X_validation, y_validation)
X_test, y_test = shuffle(X_test, y_test)

#init generators
#generates a synthetic batch based on the training data
training_generator = generate_synthetic_training_batch(X_train, y_train, batch_size=256)
#fetches a batch from the validation data we carved out earlier
validation_generator = generate_validation_batch(X_validation, y_validation, batch_size=256)