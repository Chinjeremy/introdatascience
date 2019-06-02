install.packages('quantmod')
library(quantmod)

date  <- as.Date(Sys.Date() - 1)
endDate <- date #as.Date("2016--01--01")
d <- as.POSIXlt(endDate)
d$year <- d$year - 2
#To take last 2 years of data
startDate <- as.Date(d)
STOCK <- getSymbols("3182.KL", env = NULL, src = "yahoo", from = startDate, to  =  endDate)

RSI <- RSI(Op(STOCK), n = 3)
#Calculate a 3--period relative strength index (RSI) off the open price
EMA <- EMA(Op(STOCK), n = 5)
#Calculate a 5--period exponential moving average (EMA)
EMAcross <- Op(STOCK) - EMA
#Let us explore the difference between the open price and our 5--period EMA
MACD <-  MACD(Op(STOCK),  fast = 12, slow =  26, signal = 9)

#Calculate a MACD with standard parameters
MACD <- MACD[, 2]

#Grab just the signal line to use as our indicator
SMI <- SMI(Op(STOCK), n = 13, slow = 25, fast = 2, signal = 9)

#Stochastic oscillator with standard parameters
SMI <- SMI[,1]

#Grab just the oscillator to use as our indicator
WPR <- WPR(Cl(STOCK), n = 14)
WPR <- WPR[, 1]

#Williams %R with standard parameters
ADX <- ADX(STOCK, n = 14)
ADX <- ADX[,1]

#Average Directional Index with standard parameters 
CCI <- CCI(Cl(STOCK), n = 14)
CCI <- CCI[, 1]
#Commodity Channel Index with standard parameters
CMO <- CMO(Cl(STOCK), n = 14)
CMO <- CMO[, 1]
#Collateralized Mortgage Obligation with standard parameters
ROC <- ROC(Cl(STOCK), n = 2)
ROC <- ROC[, 1]
#Price Rate Of Change with standard parameters
PriceChange <- Cl(STOCK) - Op(STOCK)
#Calculate the difference between the close price and open price
Class <- ifelse(PriceChange > 0, "UP", "DOWN")
#Create a binary classification variable, the variable we are trying to predict.
DataSet <- data.frame(Class, RSI, EMAcross, MACD, SMI, WPR, ADX, CCI, CMO, ROC)
#Create our data set
colnames(DataSet) <- c("Class", "RSI", "EMAcross", "MACD", "SMI", "WPR","ADX", "CCI", "CMO", "ROC")

TrainingSet <- DataSet[1:floor(nrow(DataSet) * trainPerc),]

DecisionTree <- rpart(Class ~ RSI + EMAcross + WPR + ADX + CMO + CCI + ROC, data = TrainingSet, 
                      na.action = na.omit, cp = .001)
#Specifying the indicators to we want to use to predict the class and controlling the growth of the tree
#by setting the minimum amount of information gained (cp) needed to justify a split.
prp(DecisionTree, type = 2, extra = 8)
#Plotting tool with a couple parameters to make it look good.
fit <- printcp(DecisionTree)
#Shows the minimal cp for each trees of each size.
mincp <- fit[which.min(fit[, 'xerror']), 'CP']
#Get the lowest cross-validated error (xerror)
plotcp(DecisionTree, upper = "splits")
#plots the average geometric mean for trees of each size.
PrunedDecisionTree <- prune(DecisionTree, cp = mincp)
#Selecting the complexity parameter (cp) that has the lowestcross-?-validated error (xerror)
t <- prp(PrunedDecisionTree, type  = 2, extra = 8)
confmat <- table(predict(PrunedDecisionTree, TestSet, type = "class"),
                 TestSet[,1], dnn = list('predicted', 'actual'))
#Building confusion matrix
print(confmat)
acc <- (confmat[1,"DOWN"] + confmat[2, "UP"]) * 100/(confmat[2, "DOWN"] + confmat[1, "UP"] +
                                                       confmat[1, "DOWN"] + confmat[2, "UP"])
#Calculating accuracy
xy <- paste('Decision Tree: Considering the output for', SYM, sep = ' ')
yz <- paste('Accuracy = ', acc, sep = ' ')
print(xy)
print(yz)
predout <- data.frame(predict(PrunedDecisionTree, TestSet))
predval <- predout['UP'] - predout['DOWN']
predclass <- ifelse(predout['UP'] >= predout['DOWN'], 1, 0)
predds <- data.frame(predclass, TestSet$Class)
colnames(predds) <-c("pred", "truth")

predds[, 2] <- ifelse(predds[, 2] == 'UP', 1, 0)
pred <- prediction(predds$pred, predds$truth)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.perf = performance(pred, measure = 'auc')
#Calculating the AUC
rmse.perf = performance(pred, measure = 'rmse')
#Calculating the RMSE
RMSE <- paste('RMSE = ', rmse.perf@y.values, sep = ' ')
AUC <- paste('AUC = ', auc.perf@y.values, sep = ' ')
print(AUC)
print(RMSE)

clean_df <- na.omit(DataSet)
write.csv(clean_df, file = "GAMU.csv")
