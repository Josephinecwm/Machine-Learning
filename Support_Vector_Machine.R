###### Question 2 #####

#####
# simulate a three-class dataset
set.seed(16)
# simulate random numbers for the two features
features = matrix(rnorm(150*2), ncol=2)
# class labels
n=50
class= factor(c(rep("f",n), rep("m",n), rep("u",n)))
features[class=="m", ] = features[class=="m", ] + 1
features[class=="u", ] = features[class=="u", ] + 2
# create dataframe
data = data.frame(features, class)
data

######
# (1) scatter plot of the data 
#set color for the classes
cols<- c("steelblue1", "hotpink", "mediumpurple")  
# set symbols for the classes
pchs<-c(1,2,3) 
# plot the scatter plot
plot(data[,1:2], pch = pchs[data$class],cex = 1, 
     col=cols[data$class])
# create legend
par(xpd = TRUE)
legend("topright",legend=c("f","m","u"),
       col=cols,pch=pchs,cex=0.5,text.font=3)

######
# (2) split the data into 50% training and 50% testing data
set.seed(983)
trainIndex = createDataPartition(data$class, times=1, p = 0.5, list = FALSE)
train=data[trainIndex,] # training set
test=data[-trainIndex,-3] # test feature
test_label=data[-trainIndex, 3]# test label

######## SVM models ######
#import library
library(caret)

# fit control with 5 fold cross-validation
fitControl=trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3)

######
# RBF kernel
set.seed(2333)
svm.Radial=train(class ~., data = train, method = "svmRadial",
                 trControl=fitControl,
                 preProcess = c("center", "scale"),
                 tuneLength = 5)
svm.Radial 
plot(svm.Radial)

# test with testing data
pred_Radial = predict(svm.Radial, test)
# check accuracy
table(pred_Radial, test_label)
mean(pred_Radial==test_label)
confusionMatrix(pred_Radial,test_label) 

######
# Linear kernel
set.seed(2333)
svm.Linear=train(class ~., data = train, method = "svmLinear",
                 trControl=fitControl,
                 preProcess = c("center", "scale"),
                 tuneLength = 5)
svm.Linear 

# test with testing data
pred_Linear = predict(svm.Linear, test)
# Check accuracy
table(pred_Linear, test_label)
mean(pred_Linear==test_label)
confusionMatrix(pred_Linear,test_label) 

######
# Polynomial kernel
set.seed(2333)
svm.Poly=train(class ~., data = train, method = "svmPoly",
                 trControl=fitControl,
                 preProcess = c("center", "scale"),
                 tuneLength = 5)
svm.Poly 
plot(svm.Poly)

# test with testing data
pred_Poly = predict(svm.Poly, test)
# Check accuracy
table(pred_Poly, test_label)
mean(pred_Poly==test_label)
# confusion matrix
confusionMatrix(pred_Poly,test_label) 
table(pred_Poly,test_label)

