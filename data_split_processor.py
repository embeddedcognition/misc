#########################################################
## AUTHOR: James Beasley                               ##
## DATE: March 27, 2018                                ##
#########################################################

#############
## IMPORTS ##
#############
import cv2
import glob
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

###############
## FUNCTIONS ##
###############

#loads a set of training image file paths from a particular path, assigning the supplied label to each image 
def load_training_set(path_to_training_images, label_to_assign):
    X_train = []    #training examples (image file paths)
    y_train = []    #labels: this is binary classification so 0 or 1
    #enumerate image file path list, adding each file path to the X_train list 
    for cur_image_file_path in glob.iglob(path_to_training_images, recursive=True):
        #append image file path to X_train list
        X_train.append(cur_image_file_path)
        #append supplied label to y_train list
        y_train.append(label_to_assign)
    #return tuple of training image file paths and labels
    return (np.array(X_train), np.array(y_train))

####################################
## LOAD, STACK, SHUFFLE DATA SETS ##
####################################

print()
print("Building training data set from provided good and bad images...")
print()

#good
X_train_good, y_train_good = load_training_set("data_sets/good/*.bmp", label_to_assign=1)
#bad
X_train_bad, y_train_bad = load_training_set("data_sets/bad/*.bmp", label_to_assign=0)

print("Num good images:", X_train_good.shape)
print("Num good labels:", y_train_good.shape)
print("Num bad images:", X_train_bad.shape)
print("Num bad labels:", y_train_bad.shape)

#stack the two data sets into a single training set
X_train = np.concatenate((X_train_good, X_train_bad), axis=0) #vertical stack
y_train = np.concatenate((y_train_good, y_train_bad), axis=0) #vertical stack

#shuffle the training set before carving out the test set
X_train, y_train = shuffle(X_train, y_train)

########################
## CARVE OUT TEST SET ##
########################

print()
print("Carving out test data set...")
print()

#carve out a portion of the training set to use for final model validation
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

##############################
## CARVE OUT VALIDATION SET ##
##############################

print()
print("Carving out validation data set...")
print()

#carve out a portion of the training set to use for model validation
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

print("Num X_train images:", X_train.shape)
print("Num y_train labels:", y_train.shape)
print("Num X_validation images: ", X_validation.shape)
print("Num y_validation labels: ", y_validation.shape)
print("Num X_test images:", X_test.shape)
print("Num y_test labels:", y_test.shape)

##############################################
## PICKLE TRAINING, VALIDATION, & TEST SETS ##
##############################################

print()
print("Pickling training, validation, and test data sets...")
print()

#pickled file names
training_set_file = "pickled_objects/paths_training_set.p"
validation_set_file = "pickled_objects/paths_validation_set.p"
test_set_file = "pickled_objects/paths_test_set.p"

#embed data in dictionary
training_set_dict = {"features": X_train, "labels": y_train}
validation_set_dict = {"features": X_validation, "labels": y_validation}
test_set_dict = {"features": X_test, "labels": y_test}

#pickle training data
with open(training_set_file, mode="wb") as f:
    pickle.dump(training_set_dict, f)
#pickle validation data
with open(validation_set_file, mode="wb") as f:
    pickle.dump(validation_set_dict, f)
#pickle test data
with open(test_set_file, mode="wb") as f:
    pickle.dump(test_set_dict, f)
    
print("Done.")
print()