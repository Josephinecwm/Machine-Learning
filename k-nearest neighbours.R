###### Question 3 #####

######
#(1)
#libraries
library(caret)
library(class)
library(pROC)
library(MASS)

#####
#read data
patients= read.table(file = "newthyroid.txt", header=TRUE, sep=",", stringsAsFactors = TRUE)
summary(patients$class)

#data cleaning 
sum(is.na(patients)) #check missing value

######
#split the data into training and testing set 
set.seed(983)
trainIndex = createDataPartition(patients$class, times=10, p = 0.7, list = FALSE) #repeat random split 10 times

# for resample 1 
train.feature1=patients[trainIndex[1:130, 1],-1] # training features
train.label1=patients$class[trainIndex[1:130, 1]] # training labels
test.feature1=patients[-trainIndex[1:130, 1],-1] # test features
test.label1=patients$class[-trainIndex[1:130, 1]] #test labels

# for resample 2 
train.feature2=patients[trainIndex[1:130, 2],-1];train.label2=patients$class[trainIndex[1:130, 2]] 
test.feature2=patients[-trainIndex[1:130, 2],-1];test.label2=patients$class[-trainIndex[1:130, 2]] 

# for resample 3
train.feature3=patients[trainIndex[1:130, 3],-1];train.label3=patients$class[trainIndex[1:130, 3]] 
test.feature3=patients[-trainIndex[1:130, 3],-1];test.label3=patients$class[-trainIndex[1:130, 3]] 

# for resample 4 
train.feature4=patients[trainIndex[1:130, 4],-1];train.label4=patients$class[trainIndex[1:130, 4]] 
test.feature4=patients[-trainIndex[1:130, 4],-1];test.label4=patients$class[-trainIndex[1:130, 4]] 

# for resample 5 
train.feature5=patients[trainIndex[1:130, 5],-1];train.label5=patients$class[trainIndex[1:130, 5]] 
test.feature5=patients[-trainIndex[1:130, 5],-1];test.label5=patients$class[-trainIndex[1:130, 5]] 

# for resample 6 
train.feature6=patients[trainIndex[1:130, 6],-1];train.label6=patients$class[trainIndex[1:130, 6]] 
test.feature6=patients[-trainIndex[1:130, 6],-1];test.label6=patients$class[-trainIndex[1:130, 6]]

# for resample 7 
train.feature7=patients[trainIndex[1:130, 7],-1];train.label7=patients$class[trainIndex[1:130, 7]] 
test.feature7=patients[-trainIndex[1:130, 7],-1];test.label7=patients$class[-trainIndex[1:130, 7]]

# for resample 8 
train.feature8=patients[trainIndex[1:130, 8],-1];train.label8=patients$class[trainIndex[1:130, 8]] 
test.feature8=patients[-trainIndex[1:130, 8],-1];test.label8=patients$class[-trainIndex[1:130, 8]]

# for resample 9 
train.feature9=patients[trainIndex[1:130, 9],-1];train.label9=patients$class[trainIndex[1:130, 9]] 
test.feature9=patients[-trainIndex[1:130, 9],-1];test.label9=patients$class[-trainIndex[1:130, 9]]

# for resample 10 
train.feature10=patients[trainIndex[1:130, 10],-1];train.label10=patients$class[trainIndex[1:130, 10]] 
test.feature10=patients[-trainIndex[1:130, 10],-1];test.label10=patients$class[-trainIndex[1:130, 10]]

########################### kNN ##############################################
# set up train control
fitControl = trainControl( #5-fold Cross-validation
  method = "repeatedcv",
  number = 5, 
  repeats = 5,
  summaryFunction = twoClassSummary, 
  classProbs = TRUE)

kNNGrid=expand.grid(k=c(3, 5, 7, 9, 11, 13, 15)) #tuning grid

#####
#training process

#for resample 1
set.seed(5)
knnFit1=train(train.feature1,train.label1, method = "knn",
             trControl = fitControl,
             tuneGrid=kNNGrid,
             metric = "ROC",
             preProcess = c("center","scale"),
             tuneLength=10)
knnFit1

#for resample 2
set.seed(5)
knnFit2=train(train.feature2,train.label2, method = "knn",
             trControl = fitControl,
             tuneGrid=kNNGrid,
             metric = "ROC",
             preProcess = c("center","scale"),
             tuneLength=10)

#for resample 3
set.seed(5)
knnFit3=train(train.feature3,train.label3, method = "knn",
             trControl = fitControl,
             tuneGrid=kNNGrid,
             metric = "ROC",
             preProcess = c("center","scale"),
             tuneLength=10)

#for resample 4
set.seed(5)
knnFit4=train(train.feature4,train.label4, method = "knn",
             trControl = fitControl,
             tuneGrid=kNNGrid,
             metric = "ROC",
             preProcess = c("center","scale"),
             tuneLength=10)

#for resample 5
set.seed(5)
knnFit5=train(train.feature5,train.label5, method = "knn",
             trControl = fitControl,
             tuneGrid=kNNGrid,
             metric = "ROC",
             preProcess = c("center","scale"),
             tuneLength=10)

#for resample 6
set.seed(5)
knnFit6=train(train.feature6,train.label6, method = "knn",
              trControl = fitControl,
              tuneGrid=kNNGrid,
              metric = "ROC",
              preProcess = c("center","scale"),
              tuneLength=10)

#for resample 7
set.seed(5)
knnFit7=train(train.feature7,train.label7, method = "knn",
              trControl = fitControl,
              tuneGrid=kNNGrid,
              metric = "ROC",
              preProcess = c("center","scale"),
              tuneLength=10)

#for resample 8
set.seed(5)
knnFit8=train(train.feature8,train.label8, method = "knn",
              trControl = fitControl,
              tuneGrid=kNNGrid,
              metric = "ROC",
              preProcess = c("center","scale"),
              tuneLength=10)

#for resample 9
set.seed(5)
knnFit9=train(train.feature9,train.label9, method = "knn",
              trControl = fitControl,
              tuneGrid=kNNGrid,
              metric = "ROC",
              preProcess = c("center","scale"),
              tuneLength=10)

#for resample 10
set.seed(5)
knnFit10=train(train.feature10,train.label10, method = "knn",
              trControl = fitControl,
              tuneGrid=kNNGrid,
              metric = "ROC",
              preProcess = c("center","scale"),
              tuneLength=10)


######
# get prediction and ROC

#for resample 1
knn.pred1 = predict(knnFit1,test.feature1)
confusionMatrix(knn.pred1,test.label1) 
knn.probs1 = predict(knnFit1,test.feature1,type="prob")
knn.ROC1 = roc(predictor=knn.probs1$h,
              response=test.label1)
knn.ROC1 = as.numeric(knn.ROC1$auc)

#for resample 2
knn.probs2 = predict(knnFit2,test.feature2,type="prob")
knn.ROC2 = roc(predictor=knn.probs2$h,
               response=test.label2)
knn.ROC2 = as.numeric(knn.ROC2$auc)

#for resample 3
knn.probs3 = predict(knnFit3,test.feature3,type="prob")
knn.ROC3 = roc(predictor=knn.probs3$h,
               response=test.label3)
knn.ROC3 = as.numeric(knn.ROC3$auc)

#for resample 4
knn.probs4 = predict(knnFit4,test.feature4,type="prob")
knn.ROC4 = roc(predictor=knn.probs4$h,
               response=test.label4)
knn.ROC4 = as.numeric(knn.ROC4$auc)

#for resample 5
knn.probs5 = predict(knnFit5,test.feature5,type="prob")
knn.ROC5 = roc(predictor=knn.probs5$h,
               response=test.label5)
knn.ROC5 = as.numeric(knn.ROC5$auc)

#for resample 6
knn.probs6 = predict(knnFit6,test.feature6,type="prob")
knn.ROC6 = roc(predictor=knn.probs6$h,
               response=test.label6)
knn.ROC6 = as.numeric(knn.ROC6$auc)

#for resample 7
knn.probs7 = predict(knnFit7,test.feature7,type="prob")
knn.ROC7 = roc(predictor=knn.probs7$h,
               response=test.label7)
knn.ROC7 =as.numeric(knn.ROC7$auc)

#for resample 8
knn.probs8 = predict(knnFit8,test.feature8,type="prob")
knn.ROC8 = roc(predictor=knn.probs8$h,
               response=test.label8)
knn.ROC8 = as.numeric(knn.ROC8$auc)

#for resample 9
knn.probs9 = predict(knnFit9,test.feature9,type="prob")
knn.ROC9 = roc(predictor=knn.probs9$h,
               response=test.label9)
knn.ROC9 = as.numeric(knn.ROC9$auc)

#for resample 10
knn.probs10 = predict(knnFit10,test.feature10,type="prob")
knn.ROC10 = roc(predictor=knn.probs10$h,
               response=test.label10)
knn.ROC10 = as.numeric(knn.ROC10$auc)

#record the results in a vector
knn.ROC= c(knn.ROC1, knn.ROC2, knn.ROC3, knn.ROC4, knn.ROC5, knn.ROC6, knn.ROC7, 
           knn.ROC8, knn.ROC9, knn.ROC10)

###################### LDA ####################
# fit the LDA model
patient_lda1 = lda(train.feature1,train.label1)#fit the LDA model for resample 1
patient_lda1
patient_lda2 = lda(train.feature2,train.label2)
patient_lda3 = lda(train.feature3,train.label3)
patient_lda4 = lda(train.feature4,train.label4)
patient_lda5 = lda(train.feature5,train.label5)
patient_lda6 = lda(train.feature6,train.label6)
patient_lda7 = lda(train.feature7,train.label7)
patient_lda8 = lda(train.feature8,train.label8)
patient_lda9 = lda(train.feature9,train.label9)
patient_lda10 = lda(train.feature10,train.label10)

# get prediction and accuracy for test set for resample 1
pred1=predict(patient_lda1,test.feature1)
pred1
acc1 = mean(test.label1==pred1$class)
acc1

######
#AUC score for the 10 partitions of data
#resample 1
lda.pred1 = predict(patient_lda1,test.feature1)
confusionMatrix(lda.pred1$class,test.label1) 
lda.probs1 = predict(patient_lda1,test.feature1,type="prob")
lda.ROC1 = roc(predictor=lda.probs1$posterior[,2],
              response=test.label1)
lda.roc1 = as.numeric(lda.ROC1$auc)

#resample 2
lda.probs2 = predict(patient_lda2,test.feature2,type="prob")
lda.ROC2 = roc(predictor=lda.probs2$posterior[,2],
               response=test.label2)
lda.roc2 = as.numeric(lda.ROC2$auc)

#resample 3
lda.probs3 = predict(patient_lda3,test.feature3,type="prob")
lda.ROC3 = roc(predictor=lda.probs3$posterior[,2],
               response=test.label3)
lda.roc3 = as.numeric(lda.ROC3$auc)

#resample 4
lda.probs4 = predict(patient_lda4,test.feature4,type="prob")
lda.ROC4 = roc(predictor=lda.probs4$posterior[,2],
               response=test.label4)
lda.roc4 = as.numeric(lda.ROC4$auc)

#resample 5
lda.probs5 = predict(patient_lda5,test.feature5,type="prob")
lda.ROC5 = roc(predictor=lda.probs5$posterior[,2],
               response=test.label2)
lda.roc5 = as.numeric(lda.ROC5$auc)

#resample 6
lda.probs6 = predict(patient_lda6,test.feature6,type="prob")
lda.ROC6 = roc(predictor=lda.probs6$posterior[,2],
               response=test.label6)
lda.roc6 = as.numeric(lda.ROC6$auc)

#resample 7
lda.probs7 = predict(patient_lda7,test.feature7,type="prob")
lda.ROC7 = roc(predictor=lda.probs7$posterior[,2],
               response=test.label7)
lda.roc7 = as.numeric(lda.ROC7$auc)

#resample 8
lda.probs8 = predict(patient_lda8,test.feature8,type="prob")
lda.ROC8 = roc(predictor=lda.probs8$posterior[,2],
               response=test.label8)
lda.roc8 = as.numeric(lda.ROC8$auc)

#resample 9
lda.probs9 = predict(patient_lda9,test.feature9,type="prob")
lda.ROC9 = roc(predictor=lda.probs9$posterior[,2],
               response=test.label9)
lda.roc9 = as.numeric(lda.ROC9$auc)

#resample 10
lda.probs10 = predict(patient_lda10,test.feature10,type="prob")
lda.ROC10 = roc(predictor=lda.probs10$posterior[,2],
               response=test.label10)
lda.roc10 = as.numeric(lda.ROC10$auc)

#record the roc results in a vector
lda.roc = c(lda.roc1, lda.roc2, lda.roc3, lda.roc4, lda.roc5, lda.roc6, lda.roc7, lda.roc8, 
            lda.roc9, lda.roc10)


######
#(2)
#plot boxplots
boxplot(lda.roc, knn.ROC, ylab = "test AUC values", names=c("LDA", "kNN"))
 