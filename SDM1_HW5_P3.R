library(ElemStatLearn)
library(dplyr)
library(caret)
library(neuralnet)
library(mlbench)
library(DataExplorer)
library(ISLR)
library(e1071)

df = ISLR::OJ
set.seed(420)
training_samples_cont = df$Purchase %>% createDataPartition(p = 0.75, list = FALSE)
train_data_cont = df[training_samples_cont, ]
test_data_cont = df[-training_samples_cont, ]

############# Linear ###############################

tune.model <- tune(svm, Purchase~., data = train_data_cont, kernel = "linear",
                   ranges = list(cost = c(0.001, 0.01, 0.1, 0.5, 0.75,1,3,5,7, 10)))
plot(tune.model)
summary(tune.model)
perf = tune.model$performances

bestmod <- tune.model$best.model
bestmod

# predict the test data
y_hat <- predict(bestmod, newdata = test_data_cont)
y_true <- test_data_cont$Purchase
accur_lin <- length(which(y_hat == y_true))/length(y_true)
accur_lin #  0.8352

table(predict = y_hat, truth = y_true)

############## Radial #######################

tune.model.rad <- tune(svm, Purchase~., data = train_data_cont, kernel = "radial",
                       ranges = list(cost = c(0.001, 0.01, 0.1, 0.5, 0.75,1,3,5,7 ,10)))

tune.model.rad
summary(tune.model.rad)

perf = tune.model.rad$performances
bestmod <- tune.model.rad$best.model
bestmod

# predict the test data
y_hat <- predict(bestmod, newdata = test_data_cont)
y_true <- test_data_cont$Purchase
accur_rad <- length(which(y_hat == y_true))/length(y_true)
accur_rad #  #.8614

table(predict = y_hat, truth = y_true)

############# Polynomial Degree 2 #####################

tune.model.2deg <- tune(svm, Purchase~., data = train_data_cont, degree = 2, kernel="polynomial",
                       ranges = list(cost = c(0.001, 0.01, 0.1, 0.5, 0.75,1, 3,5,7,10)))
tune.model.2deg
summary(tune.model.2deg)
perf = tune.model.2deg$performances

bestmod <- tune.model.2deg$best.model
bestmod

y_hat <- predict(bestmod, newdata = test_data_cont)
y_true <- test_data_cont$Purchase
accur_2deg <- length(which(y_hat == y_true))/length(y_true)
accur_2deg #  #.8576
