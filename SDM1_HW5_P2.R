library(ElemStatLearn)
library(dplyr)
library(caret)
library(neuralnet)
library(mlbench)
library(DataExplorer)

infert = datasets::infert
df = infert
summary(df)
infert[7,6] = 100.100
df_new = infert
summary(df$spontaneous)
summary(df_new$spontaneous)

set.seed(420)
training_samples_cont = df$case %>% createDataPartition(p = 0.75, list = FALSE)
train_data_cont = df[training_samples_cont, ]
test_data_cont = df[-training_samples_cont, ]

summary(train_data_cont)
error_df = c()
for (i in 1:5)
{
  nn <- neuralnet(case ~ age + parity + induced + spontaneous , data = train_data_cont, hidden = i,err.fct = 'ce',
                  threshold = 0.15,stepmax = 1e6,linear.output = FALSE)
  pred_class_train = round(nn$net.result[[1]][,1])
  train_class = as.numeric(train_data_cont$case)
  error_train = sum(abs(train_class - pred_class_train))/length(pred_class_train)
  
  pred_class_test = as.numeric(round(neuralnet::compute(nn, test_data_cont)$net.result[,1]))
  test_Y = as.numeric(test_data_cont$case)
  pred_class_test = as.numeric(pred_class_test)
  error_test = sum(abs(test_Y - pred_class_test))/length(pred_class_test)
  error_df = rbind(error_df,cbind(i,error_train,error_test))
}

nn <- neuralnet(case ~ age + parity + induced + spontaneous , data = train_data_cont, hidden = 4,err.fct = 'ce',
                threshold = 0.155,stepmax = 1e6,linear.output = FALSE)
nn_normal = nn
plot(nn_normal)

############## neural network with Outlier ###########

training_samples_cont = df_new$case %>% createDataPartition(p = 0.75, list = FALSE)
train_data_cont = df_new[training_samples_cont, ]
test_data_cont = df_new[-training_samples_cont, ]

error_df_new = c()
for (i in 1:5)
{
  nn <- neuralnet(case ~ age + parity + induced + spontaneous , data = train_data_cont, hidden = i,err.fct = 'ce',
                  threshold = 0.15,stepmax = 1e6,linear.output = FALSE)
  
  pred_class_train = round(nn$net.result[[1]][,1])
  train_class = as.numeric(train_data_cont$case)
  error_train = sum(abs(train_class - pred_class_train))/length(pred_class_train)
  
  pred_class_test = as.numeric(round(neuralnet::compute(nn, test_data_cont)$net.result[,1]))
  test_Y = as.numeric(test_data_cont$case)
  pred_class_test = as.numeric(pred_class_test)
  error_test = sum(abs(test_Y - pred_class_test))/length(pred_class_test)
  error_df_new = rbind(error_df_new,cbind(i,error_train,error_test))
}

nn <- neuralnet(case ~ age + parity + induced + spontaneous , data = train_data_cont, hidden = 3,err.fct = 'ce',
                threshold = 0.1,stepmax = 1e6,linear.output = FALSE)
plot(nn)

error_df_out = c()
for (i in 1:10)
{
  df = df_new
  df[7,6] = 100.100/(i*10)
  training_samples_cont = df_new$case %>% createDataPartition(p = 0.75, list = FALSE)
  train_data_cont = df_new[training_samples_cont, ]
  test_data_cont = df_new[-training_samples_cont, ]
  
  nn <- neuralnet(case ~ age + parity + induced + spontaneous , data = train_data_cont, hidden = 3,
                  err.fct = 'ce', threshold = 0.1,stepmax = 1e6,linear.output = FALSE)
  
  pred_class_train = round(nn$net.result[[1]][,1])
  train_class = as.numeric(train_data_cont$case)
  error_train = sum(abs(train_class - pred_class_train))/length(pred_class_train)
  
  pred_class_test = as.numeric(round(neuralnet::compute(nn, test_data_cont)$net.result[,1]))
  test_Y = as.numeric(test_data_cont$case)
  pred_class_test = as.numeric(pred_class_test)
  error_test = sum(abs(test_Y - pred_class_test))/length(pred_class_test)
  error_df_out = rbind(error_df_out,cbind(i,error_train,error_test))
}
