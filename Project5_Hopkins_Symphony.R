#############################################
#                                           #
# Author:     Symphony Hopkins              #
# Date:       04/10/2023                    #
# Subject:    Project 5                     #
# Class:      DSCI 512                      #
# Section:    01W                           #         
# Instructor: Juan David Munoz              #
# File Name:  Project5_Hopkins_Symphony.R   #
#                                           #
#############################################

#1. Load the dataset bike.csv Download bike.csv
#   into memory. Then split the data into a training 
#   set containing 2/3 of the original data (test set 
#   containing remaining 1/3 of the original data).
#   Answer: See code.

#importing the dataset
bike <- read.csv("~/Documents/Maryville_University/DSCI_512/Week_5/Bike.csv")
View(bike)

#setting seed for reproducibility
set.seed(1)

#performing train-test split
train = sample(1:nrow(bike), nrow(bike) * 2/3)

#2..Build a tree model using function tree().
#   Answer: See code.

#importing library to use tree() for following questions
library(tree)

#2a.The response is count and the predictors are season, 
#   holiday, workingday, temp, atemp, humidity, windspeed, 
#   casual, and registered.
#   Answer: As we can see, our final model only used two predictors
#   which were registered and casual.

bike_tree = tree(count ~ season + holiday + workingday + temp +
                   atemp + humidity + windspeed + casual + registered,
                 data = bike, subset = train)
summary(bike_tree)

#2b.Perform cross-validation to choose the best tree by calling 
#   cv.tree().
#   Answer: See code.

bike_tree_cv <- cv.tree(bike_tree)

#2c.Plot the model results of b) and determine the best size 
#   of the optimal tree.
#   Answer: Although we want to choose a size that leads to
#   the lowest test error, which would be size 8, we would
#   have a complex model. We want a simple model for inter-
#   pretability, generalization, efficiency, and robustness.
#   The difference in test error between size 8 and 5 is
#   small; and choosing size 5 would help us create a less 
#   complex model so we will determine that as the best size
#   for our model.

plot(bike_tree_cv$size, bike_tree_cv$dev, type = 'b')

#2d.Prune the tree by calling prune.tree() function with 
#   the best size found in c).
#   Answer: See code.

bike_tree_pruned <- prune.tree(bike_tree, best = 5) 

#2e.Plot the best tree model.
#   Answer: See code.

plot(bike_tree_pruned)
text(bike_tree_pruned, pretty=0)

#2f.Compute the test error using the test data set.
#   Answer: For this model, we have a MSE of 3411.454.

y_pred_tree <-predict(bike_tree_pruned, newdata = bike[-train, ])
bike_tree_test <- bike[-train,'count']
mean((y_pred_tree-bike_tree_test)^2)

#3..Build a random forest model using function randomForest().
#   Answer: See code.

#importing library to use randomForest() for following questions
library(randomForest)

#3a.The response is count and the predictors are season, 
#   holiday, workingday, temp, atemp, humidity, windspeed, 
#   casual, and registered.
#   Answer: See code.

bike_rf = randomForest(count ~ season + holiday + workingday + 
                           temp + atemp + humidity + windspeed + 
                           casual + registered, data = bike, 
                         subset = train, importance=TRUE)
summary(bike_rf)

#3b.Compute the test error using the test data set.
#   Answer: For this model, we have a MSE of 156.4885.

y_pred_rf = predict(bike_rf, newdata=bike[-train, ])
bike_rf_test <- bike[-train, 'count']
mean((y_pred_rf-bike_rf_test)^2)

#3c.Extract variable importance measure using importance() function.
#   Answer: See code.

importance(bike_rf)

#3d.Plot the variable importance using function varImpPlot(). 
#   Which are the top 2 important predictors in this model?
#   Answer: The top 2 important predictors in this model 
#   are registered and causal.

varImpPlot(bike_rf)


#End Assignment





