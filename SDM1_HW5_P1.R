################
library(ElemStatLearn)
library(dplyr)
library(caret)
library(neuralnet)

set.seed(420)
df_spam = ElemStatLearn::spam
training_samples_cont = df_spam$spam %>% createDataPartition(p = 0.75, list = FALSE)
train_data_cont = df_spam[training_samples_cont, ]
test_data_cont = df_spam[-training_samples_cont, ]
summary(train_data_cont$spam)
i =1
n = names(train_data_cont)

#### Finding significant variables ###########
logistic_fit <- glm(spam~., data = train_data_cont, family = "binomial")
log_sum = summary(logistic_fit)
x = as.data.frame(log_sum$coefficients[log_sum$coefficients[,4]<0.01,])

train_new = subset(train_data_cont,select = c(A.5,A.7,A.8,A.9,A.16,A.17,A.21,A.23,A.25,A.26,A.27,A.35,A.36,A.45,
                                              A.46,A.49,A.52,A.53,A.57,spam))
test_new = subset(test_data_cont,select = c(A.5,A.7,A.8,A.9,A.16,A.17,A.21,A.23,A.25,A.26,A.27,A.35,A.36,A.45,
                                             A.46,A.49,A.52,A.53,A.57,spam))

############# Logistic Fit ###########################################
logistic_fit <- glm(spam~., data = train_new, family = "binomial")
logistic_probs_train <- predict(logistic_fit, newdata = train_new, type = "response")
train_class = as.factor(as.numeric(train_data_cont$spam)-1)
output_predict_train <- as.factor(round(logistic_probs_train))
logistic_probs_test <- predict(logistic_fit, newdata = test_new, type = "response")
test_class = as.factor(as.numeric(test_data_cont$spam)-1)
output_predict_test <- as.factor(round(logistic_probs_test))
CM_train = confusionMatrix(train_class,output_predict_train)
accuracy_train = CM_train$overall[1]
CM_test = confusionMatrix(test_class,output_predict_test)
accuracy_test = CM_test$overall[1]


accuracy_df = c('logistic_accuracy',accuracy_train,accuracy_test)
for (i in 1:5)
{
  nn <- neuralnet(spam~. , data = train_new, hidden = i,err.fct = 'ce', threshold = 0.1,linear.output = FALSE)
  
  pred_class_train = round(nn$net.result[[1]][,1])
  train_class = as.numeric(train_new$spam)-1
  accuracy_train = sum(abs(train_class - pred_class_train))/length(pred_class_train)
  
  pred_class_test = as.numeric(round(neuralnet::compute(nn, test_new[,1:20])$net.result[,1]))
  test_Y = as.numeric(test_new$spam)-1
  pred_class_test = as.numeric(pred_class_test)
  accuracy = sum(abs(test_Y - pred_class_test))/length(pred_class_test)
  accuracy_df = rbind(accuracy_df,cbind(i,accuracy_train,accuracy))
}
