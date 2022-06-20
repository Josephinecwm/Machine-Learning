######
# libraries
library(tree)
library(rpart)
library(randomForest)
library(gbm)
library(caret)
library(ISLR)

######
# import data
data("GermanCredit")
#clean the dataset
GermanCredit[,c("Purpose.Vacation","Personal.Female.Single")] = list(NULL) #delete two variables where all values are the same for both classes
sum(is.na(GermanCredit)) #check missing values
GermanCredit$Class=factor(GermanCredit$Class) #convert the dependent variable to factor

# scale the data 
GermanCredit$Duration = scale(GermanCredit$Duration)
GermanCredit$Amount = scale(GermanCredit$Amount)
GermanCredit$Age = scale(GermanCredit$Age)
GermanCredit$InstallmentRatePercentage = scale(GermanCredit$InstallmentRatePercentage)
GermanCredit$ResidenceDuration = scale(GermanCredit$ResidenceDuration)
GermanCredit$NumberExistingCredits = scale(GermanCredit$NumberExistingCredits)
GermanCredit$NumberPeopleMaintenance = scale(GermanCredit$NumberPeopleMaintenance)

######
# create training and testing sets - 70% for training and 30% for testing 
set.seed(12)
trainIndex = createDataPartition(GermanCredit$Class, p = 0.7, list = FALSE)
train=GermanCredit[trainIndex,]
test=GermanCredit[-trainIndex,]

######
# (1) create decision tree model with caret package
# fit control with 5 fold cross-validation
fitcontrol=trainControl(method = "repeatedcv", 
                        number = 5, #5 fold cross-validation
                        repeats = 3)
set.seed(1) 
GermanCredit.rpart=train(train[,-10],
                         train[,10], 
                         method = "rpart", 
                         tuneLength=5, 
                         trControl = fitcontrol) 

GermanCredit.rpart

# predict on the testing sets
pred.rpart=predict(GermanCredit.rpart,newdata=test[,-10])
mean(pred.rpart==test$Class)

# specify our cost complexity to update the model
set.seed(1)
cpGrid=expand.grid(cp=c(0.03,0.04,0.06))
GermanCredit.rparts=train(train[,-10],
                          train[,10],
                          method = "rpart",
                          tuneGrid=cpGrid,
                          trControl = fitcontrol)
GermanCredit.rparts

# predict the adjusted model on testing instances
pred.rparts=predict(GermanCredit.rparts,newdata=test[,-10])

# obtain accuracy
accuracy = mean(pred.rparts==test$Class)
CM = confusionMatrix(pred.rparts,test$Class) 

# test error rate
1- accuracy

# plot the pruned tree
par(mfrow= c(1,1) )
print(GermanCredit.rparts$finalModel)
plot(GermanCredit.rparts$finalModel)
text(GermanCredit.rparts$finalModel, cex=.8)
library(rattle)
fancyRpartPlot(GermanCredit.rparts$finalModel)

#####
# (2) create Random Forest model with caret package
set.seed(2)
GermanCredit.rf=train(Class~.,data=train,method="rf",metric="Accuracy", 
            trControl=fitcontrol,tuneLength=5, ntree=1000) #1000 number of trees

# model details
GermanCredit.rf
plot(GermanCredit.rf)
GermanCredit.rf$finalModel

# predict the random forest model on testing data
pred.rf=predict(GermanCredit.rf,newdata=test[,-10])

# get accuracy
accuracy_rf = mean(pred.rf==test[,10])
confusionMatrix(pred.rf,test[,10]) 

# test error rate
1- accuracy_rf

# variable importance
varImp(GermanCredit.rf)
plot(varImp(GermanCredit.rf), top=20)

######
# (3) ROC curves for decision tree and random forest 
library("ROCR")

# obtain the decision tree model predictions
pred_DT <-predict(GermanCredit.rparts, newdata = GermanCredit, type = "prob")[, 2]
pred_DT2 = prediction(pred_DT, GermanCredit$Class)

# obtain the random forest model predictions
pred_RF <-predict(GermanCredit.rf, newdata = GermanCredit, type = "prob")[, 2]
pred_RF2 = prediction(pred_RF, GermanCredit$Class)

# plot the ROC curves for both model
plot(performance(pred_DT2, "tpr", "fpr"), col = "red")
plot(performance(pred_RF2, "tpr", "fpr"), add = TRUE, col = "blue")
abline(0, 1, lty = 2) 
legend(0.6,0.2,c("Decision Tree",
               "Random Forest"), col=c("red","blue"), lty=1:1)
