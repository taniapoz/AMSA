#####################################################################################################
# title         : Machine learning exercise for Sentinel-2 data
# purpose       : Implementing a machine learning workflow in R 
# author        : Abdulhakim M. Abdi (Twitter: @HakimAbdi / www.hakimabdi.com)
# input         : A multi-temporal raster stack of Sentinel-2 data comprising scenes from four dates 
# output        : One classified land cover map from each of three machine learning algorithms  
# Note 1        : This brief tutorial assumes that you are already well-grounded in R concepts and are 
#               : familiar with image classification procedure and terminology
# Reference		  : Please cite Abdi (2020): "Land cover and land use classification performance of machine learning 
#				        : algorithms in a boreal landscape using Sentinel-2 data" in GIScience & Remote Sensing if you find this 
#               : tutorial useful in a publication. 
# Reference URL	: https://doi.org/10.1080/15481603.2019.1650447 
#####################################################################################################

rm(list = ls(all.names = TRUE)) # will clear all objects, including hidden objects
gc() # free up memory and report memory usage


# set working directory. This will be the directory where all the unzipped files are located


# load required libraries (Note: if these packages are not installed, then install them first and then load)
# rgdal: a comprehansive repository for handling spatial data
# raster: for the manipulation of raster data
# caret: for the machine learning algorithms
# sp: for the manipulation of spatial objects
# nnet: Artificial Neural Network
# randomForest: Random Forest 
# kernlab: Support Vector Machines
# e1071: provides miscellaneous functions requiered by the caret package

library(rgdal)
install.packages("raster")
library(raster)
library(caret)
library(sp)
library(nnet)
library(randomForest)
library(kernlab)
library(e1071)
library(tidyverse)
library(sf)

# Load the Sentinel-2 stack of the study area
s2data = stack("../AMSA/data/S2StackSmall.tif")

# Name the layers of the Sentinel-2 stack based on previously saved information
names(s2data) = as.character(read.csv("../AMSA/data/S2StackSmall_Names.csv")[,1])

# Load the sample data
# Alternatively, you can use the supplied orthophotos to generate a new set of training and validation data 
# Your samples layer must have a column for each image in the raster stack, a column for the land cover class that point represents, an X and Y column
# You can create such a sample file using QGIS or another GIS software
samples = read.csv("../AMSA/data/Samples.csv") %>%
  dplyr::select(c(x, y, class, B05N, B05M, B11M, B06M, B8AM, B06JA)) 

# Split the data frame into 70-30 by class
trainx = list(0)
evalx = list(0)
for (i in 1:8){ # loop through all eight classes
  cls = samples[samples$class == i,]
  smpl <- floor(0.70 * nrow(cls))
  tt <- sample(seq_len(nrow(cls)), size = smpl)
  trainx[[i]] <- cls[tt,]
  evalx[[i]] <- cls[-tt,]
}

# combine them all into training and evaluation data frames
trn = do.call(rbind, trainx) 
eva = do.call(rbind, evalx)

# Set up a resampling method in the model training process
tc <- trainControl(method = "repeatedcv", # repeated cross-validation of the training data
                   number = 10, # number of folds
                   repeats = 5, # number of repeats
                   allowParallel = TRUE, # allow use of multiple cores if specified in training
                   verboseIter = TRUE) # view the training iterations
                        
# Generate a grid search of candidate hyper-parameter values for inclusion into the models training
# These hyper-parameter values are examples. You will need a more complex tuning process to achieve high accuracies
# For example, you can play around with the parameters to see which combinations gives you the highest accuracy. 
nnet.grid = expand.grid(size = seq(from = 2, to = 10, by = 2), # number of neurons units in the hidden layer 
                        decay = seq(from = 0.1, to = 0.5, by = 0.1)) # regularization parameter to avoid over-fitting 

rf.grid <- expand.grid(mtry=1:20) # number of variables available for splitting at each tree node

svm.grid <- expand.grid(sigma=seq(from = 0.01, to = 0.10, by = 0.02), # controls for non-linearity in the hyperplane
                        C=seq(from = 2, to = 10, by = 2)) # controls the influence of each support vector

## Begin training the models. It took my laptop 8 minutes to train all three algorithms
# Train the neural network model
nnet_model <- caret::train(x = trn[,(5:ncol(trn)-1)], y = as.factor(as.integer(as.factor(trn$class))),
                    method = "nnet", metric="Accuracy", trainControl = tc, tuneGrid = nnet.grid)

# Train the random forest model
rf_model <- caret::train(x = trn[,(5:ncol(trn)-1)], y = as.factor(as.integer(as.factor(trn$class))),
                    method = "rf", metric="Accuracy", trainControl = tc, tuneGrid = rf.grid)

# Train the support vector machines model
svm_model <- caret::train(x = trn[,(5:ncol(trn)-1)], y = as.factor(as.integer(as.factor(trn$class))),
                    method = "svmRadialSigma", metric="Accuracy", trainControl = tc, tuneGrid = svm.grid)

## Apply the models to data. It took my laptop 2 minutes to apply all three models
# Apply the neural network model to the Sentinel-2 data. 
nnet_prediction = raster::predict(s2data, model=nnet_model)

# Apply the random forest model to the Sentinel-2 data
rf_prediction = raster::predict(s2data, model=rf_model)

# Convert the evaluation data into a spatial object using the X and Y coordinates and extract predicted values
coords <- cbind(eva$x, eva$y) # Replace with your actual data
eva.sp <- SpatialPointsDataFrame(coords = coords, data = eva, 
                                 proj4string = CRS("+proj=utm +zone=33 +datum=WGS84"))


## Superimpose evaluation points on the predicted classification and extract the values
# neural network
nnet_Eval <- raster::extract(nnet_prediction, eva.sp)
# random forest
rf_Eval <- raster::extract(rf_prediction, eva.sp)


# Create an error matrix for each of the classifiers
nnet_errorM = confusionMatrix(as.factor(nnet_Eval),as.factor(eva$class)) # nnet is a poor classifier, so it will not capture all the classes
rf_errorM = confusionMatrix(as.factor(rf_Eval),as.factor(eva$class))



# Plot the results next to one another along with the 2018 NMD dataset for comparison
nmd2018 = raster("../AMSA/data/NMD_S2Small.tif") # load NMD dataset (Nationella Marktaeckedata, Swedish National Land Cover Dataset)
crs(nmd2018) <- crs(nnet_prediction) # Correct the coordinate reference system so it matches with the rest
rstack = stack(nmd2018, nnet_prediction, rf_prediction) # combine the layers into one stack
names(rstack) = c("NMD 2018", "Single Layer Neural Network", "Random Forest") # name the stack
plot(rstack) # plot it! 

# Congratulations! You conducted your first machine learning classification in R. 
# Please cite the paper referred to at the beginning if you use any part of this script in a publication. Thank you! :-)