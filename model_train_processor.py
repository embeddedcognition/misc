#########################################################
## AUTHOR: James Beasley                               ##
## DATE: March 27, 2018                                ##
#########################################################

#############
## IMPORTS ##
#############
import pickle
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from augmentation_processor import generate_synthetic_training_batch, generate_validation_batch

######################################################
## DE-PICKLE, EXTRACT, SHUFFLE TRAINING & TEST SETS ##
######################################################

print()
print("De-pickling training and test data sets...")
print()

#pickled file names
training_set_file = "pickled_objects/training_set.p"
test_set_file = "pickled_objects/test_set.p"

#import pickled data files
with open(training_set_file, mode="rb") as f:
    training_set = pickle.load(f)
with open(test_set_file, mode="rb") as f:
    test_set = pickle.load(f)

#unpack training and test sets    
X_train, y_train = training_set["features"], training_set["labels"]
X_test, y_test = test_set["features"], test_set["labels"]

#shuffle the training set (image-paths/labels)
X_train, y_train = shuffle(X_train, y_train)

print("Num X_train images:", X_train.shape)
print("Num y_train labels:", y_train.shape)
print("Num X_test images:", X_test.shape)
print("Num y_test labels:", y_test.shape)

#get a dictionary with the tally for each class in the data sets (in descending order)
num_class_instances_by_class_train = Counter(y_train)
num_class_instances_by_class_test = Counter(y_test)
print()
print("Num class instances by class:")
print("X_train:", num_class_instances_by_class_train)
print("X_test:", num_class_instances_by_class_test)

#################
## TRAIN MODEL ##
#################

print("Begin training model...")
print()
print()
