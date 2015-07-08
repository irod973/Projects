#########
##File: 3FG_Analysis
##Author: Irving Rodriguez
##Description: 
## This file predicts the 3FG% for every NBA player for the "next" season (2014-2015) based on observed changes
## in 3FG% from season to season in the 3-point era. 
#########

##Libraries
library(gdata)

### Constants ###

#Filenames
SEASON_INDEX_FILE <- "/home/irod973/Documents/Projects/3FG%_Forecast/IAR_seasonIndex3FGEra.csv"

#Dataframe Column Strings
FG3P <- "X3P.1"
AGE <- "Age"
PLAYER_NAME <- "Player"
SEASON <- "Season"
POS <- "Pos"
FG3A <- "X3PA"
YEAR <- "Year"
DELTA3FGP <- "Delta3FGP"

currentSeason <- "2013-2014"
nextSeason<-"2014-2015"

THRESHOLD_3FGA <- 50


######### Body #########

### Load dataframe from file
seasonIndexFull <- read.csv(SEASON_INDEX_FILE)
#Take out data from season we are predicting, 2014-2015
seasonIndex <- seasonIndexFull[seasonIndexFull[[SEASON]]!=nextSeason,]

##########
## 3FG% Analysis

## Can optionally look at the change in players' 3FG% for every season in 3-Point era, by year of experience and age
#plot(seasonIndex[[YEAR]][seasonIndex[[FG3A]] > THRESHOLD_3FGA], seasonIndex[[DELTA3FGP]][seasonIndex[[FG3A]] > THRESHOLD_3FGA], type='p',
#      xlab="Years of Experience",ylab="Change in 3FG%")
#plot(seasonIndex[[AGE]][seasonIndex[[FG3A]] > THRESHOLD_3FGA], seasonIndex[[DELTA3FGP]][seasonIndex[[FG3A]] > THRESHOLD_3FGA ], type='p',
#      xlab="Age",ylab="Change in 3FG%")


## Functions ## 


#####
#Function: regressionAnalysis
#Usage: regressions <- regressionAnalysis(seasonDataFrame, timeList, timeString, positionString)
#Output: weightedRegressions lm object
#This function runs a weighted and unweighted l-s regression on change in 3FG% through a given timeList,
#like a list of years of experience or age. The function plots both regressions.
regressionAnalysis <- function(seasonIndex, timeList, timeString, position){
  
  ## Plot unweighted mean change in 3FG% for a given time "slice"
  #To avoid problems with ill-defined variances, only return slices with >1 player
  meanDelta <- sapply(timeList, function(timeSlice) 
  {deltas<-seasonIndex[[DELTA3FGP]][seasonIndex[[timeString]] == timeSlice & 
                                      seasonIndex[[POS]]==position & seasonIndex[[FG3A]] > THRESHOLD_3FGA]
   if(length(deltas[!is.na(deltas)])== 1) return(NaN)
   else return(mean(deltas, na.rm=TRUE))
  })
  
  plot(timeList, meanDelta, type='p', col='blue', pch=16, cex=1.5,
       ylab="Change in 3FG%", xlab=timeString, bty='L', main=position)
  abline(h=0) #x-axis, for reference
  
  ## First: Regression on unweighted data
  regressionDelta <- lm(meanDelta ~ timeList,
                        weights=sapply(timeList, function(timeSlice){
                          variance<-1/var(seasonIndex[[DELTA3FGP]][seasonIndex[[timeString]] == timeSlice &
                                                                     seasonIndex[[POS]] == position & seasonIndex[[FG3A]] > THRESHOLD_3FGA], na.rm=TRUE)
                          if(is.infinite(variance)) return(1/.000000001) ### just something really big instead
                          else return(variance)
                        })
                        )
    
  #####
  #Function: extractFittedValues
  #Usage: fittedValues <- extractFittedValues(meanDeltaList, regressionLM)
  #Output: list of equal length as meanDeltaList
  #This function takes an lm object and matches non-na values from the object's $fitted.values to
  #each index of meanDeltaList. This helps for plotting line in given time range.
  extractFittedValues <- function(meanDelta, regression){
    return(
      sapply(meanDelta, function(elem){
        if(identical(elem, NaN)) return(NA)
        else return(regression$fitted.values[[ match(elem, meanDelta[!is.na(meanDelta)]) ]])
      })
    )
  }
  
  #Add regression line to plot
  regressionUnweighted <- extractFittedValues(meanDelta, regressionDelta)
  lines(timeList, regressionUnweighted, col='blue', lwd=4)#plot regression line
  
  ## Next: Regression on weighted data
  #First, weight meanDelta by 3PA
  weightedMeanDelta <- sapply(timeList, function(timeSlice){
    deltas<-seasonIndex[[DELTA3FGP]][seasonIndex[[timeString]] == timeSlice & seasonIndex[[POS]] == position & seasonIndex[[FG3A]] > THRESHOLD_3FGA]
    if(length(deltas[!is.na(deltas)])== 1) return(NaN)
    else return(weighted.mean(deltas, seasonIndex[[FG3A]][seasonIndex[[timeString]] == timeSlice & seasonIndex[[POS]] == position & seasonIndex[[FG3A]] > THRESHOLD_3FGA],na.rm=TRUE)            
    )})
  points(timeList, weightedMeanDelta, type='p', col='black', pch=16, cex=1.5)
  
  #Weighted least-squares regression using variance from each time "slice"
  #Need to ensure that we approximate variances of 0 to avoid multiplying by infinity
  regressionDeltaWeight <- lm(weightedMeanDelta ~ timeList, 
                              weights=sapply(timeList, function(timeSlice){
                                variance<-1/var(seasonIndex[[DELTA3FGP]][seasonIndex[[timeString]] == timeSlice &
                                                                           seasonIndex[[POS]] == position & seasonIndex[[FG3A]] > THRESHOLD_3FGA], na.rm=TRUE)
                                if(is.infinite(variance)) return(1/.000000001) ### just something really big instead
                                else return(variance)
                              }
                              ))
  
  regressionWeighted <- extractFittedValues(weightedMeanDelta, regressionDeltaWeight)
  lines(timeList,regressionWeighted, col="black", lwd=4)
  
  #Display legend to plot
  #legend("topleft", legend=c("Unweighted Mean", "Unweighted Regression", "Weighted Mean", "Weighted Regression"),
  #       lty=c(1,0,0,1),
  #       lwd=c(4, 1, 1, 4),
  #       pch=c(-1, 16, 16, -1),
  #       cex=1.5,
  #       col=c("blue","blue","black","black"),
  #       bg = "gray90")
  
  ##Return weighted regression lm object
  return(regressionDeltaWeight)
}


#####
#Function: Mode
#Find mode of list.
Mode <- function(x){
  uniqList <- unique(x)
  uniqList[which.max(tabulate(match(x, uniqList)))]
}

#####
#Function: predictPlayer3FGP
#Usage: playerFuture3FGP  <- predictPlayer3FGP(seasonIndex, regressionsYear, "Year", player)
#Output: List containing (predictedChange, meanPredictedFG%, - 2-stDev from predicted%, + 2-stDev from predicted%)
#This function predicts a player's 3FG% for the upcoming NBA season.
#It uses the weighted difference between their career changes in 3FG% and the changes for the average player at their position
#to predict the next change. It models this change as a Normal RV to produce a lower and upper 2-standard deviation bound
#on 3FG% for the upcoming season.
predictPlayer3FGP<-function(seasonIndex, regression, timeString, player){
  nSamples = 100000
  
  #Get the regression line corresponding to player's position.
  positionRegression<-regression[[1,toString(Mode(seasonIndex[[POS]][seasonIndex[[PLAYER_NAME]] == player]))]]
  
  #Get player's career changes in 3FG%
  playerDelta<-seasonIndex[[DELTA3FGP]][seasonIndex[[PLAYER_NAME]] == player]
  #Get player's timeslice for next season
  playerNextTimeslice<-tail(
    seasonIndex[[timeString]][seasonIndex[[PLAYER_NAME]]==player & seasonIndex[[SEASON]] == currentSeason], n=1) + 1
  #Get the value of the regression for player's next timeslice. [[1]] dereferences list, [[1]] indicates coeff; [1] is intercept, [2] is slope
  nextRegressionPoint <- positionRegression[[1]][2] * playerNextTimeslice + positionRegression[[1]][1]
  
  ##Use average career difference (weighted by 3PA) to predict 3FG% for next season from previous season's 3FG%
  #For rookies, use point on regression after rookie year as predictedDelta. Use year since rookie could be traded mid-season
  if(tail(seasonIndex[[YEAR]][seasonIndex[[PLAYER_NAME]] == player & seasonIndex[[SEASON]]==currentSeason], n=1) == 0) {
    predictedDelta <- signif(nextRegressionPoint, digits=5)
  } else{ 
    #Non rookies. For regression, [[3]] selects fitted values
    averageDifference<-weighted.mean(sapply(1:length(playerDelta), function(index){
      playerDelta[[index]] - positionRegression[[3]][index] 
    }), seasonIndex[[FG3A]][seasonIndex[[PLAYER_NAME]] == player],
    na.rm=TRUE)
    
    predictedDelta <- signif(averageDifference - nextRegressionPoint, digits=5)
  }
  
  #Model predictedDelta as Normal random variable. More details in write-up.
  predictedDeltaDistribution <- rnorm(n=nSamples, predictedDelta, abs(predictedDelta))
  predictedDistMean <- mean(predictedDeltaDistribution)
  predictedDistVar <- var(predictedDeltaDistribution)
  
  #For 95% probability estimate, use mean +/- 2*st.dev.
  predicted3FGUpper<-round(tail(seasonIndex[[FG3P]][seasonIndex[[PLAYER_NAME]] == player], n=1)
                           +(predictedDistMean + 2*predictedDistVar^(0.5)),
                           digits=3)
  predicted3FGLower<-round(tail(seasonIndex[[FG3P]][seasonIndex[[PLAYER_NAME]] == player], n=1)
                           +(predictedDistMean - 2*predictedDistVar^(0.5)),
                           digits=3)
  
  return(c(predictedDelta, predicted3FGLower, predicted3FGUpper, round(mean(c(predicted3FGLower, predicted3FGUpper), na.rm=TRUE), digits=3)))
}


#####
#Function: predictSeason3FG
#Usage: playerFuture3FGP  <- predictPlayer3FGP(seasonIndex, seasonIndexFull, regression, "Year", playerList)
#Output: Dataframe containing the following for each player: 3FG% for current season, predicted change for next season,
# predicted 3FG% for next season, a 95% confidence interval, the 3FG% for next season (if available), and difference between actual and predicted 3FG%
#
predictSeason3FGP <- function(seasonIndex, seasonIndexFull, regression, timeString, playerList){  
  #Column names for dataframe that will hold results
  resultNames<-c("Player", "Current3FGP", "PredictedChange","95PLowerBound", "95PUpperBound","MeanPredicted3FGP","Actual3FGP","Actual-MeanPrediction")
  predicted3FGPResults <- as.data.frame(matrix(ncol=length(resultNames)))
  colnames(predicted3FGPResults)<-resultNames
  
  ##Add each player in current season to results dataFrame.
  for(player in currentPlayers){
    #Need following for each player: current3FG%, 4 values returned from predict3FGP, "next season" 3FG%, and the difference from "next3FG%" and meanPredicted
    #Next season 3FG%
    next3FGP <- tail(seasonIndexFull[[FG3P]][seasonIndexFull[[PLAYER_NAME]] == player 
                                             & seasonIndexFull[[SEASON]] == nextSeason],n=1)
    #List of values returned by predict3FGP
    predictions <- predictPlayer3FGP(seasonIndex, regression, timeString, player)
    differenceActualPredicted <- next3FGP - predictions[4]
    
    ##Checks whether there is a 3FG% for predicted season. If not, actual and difference columns are NA
    if(identical(numeric(0), next3FGP)){
      next3FGP <- "DNA"
      differenceActualPredicted <- NA
    }
    
    #Put all values into one vector
    results<-c(player, tail(seasonIndex[[FG3P]][seasonIndex[[PLAYER_NAME]] == player & seasonIndex[[SEASON]] == currentSeason], n=1),
               predictions, 
               next3FGP,
               differenceActualPredicted
    )
    
    #Append vector to dataframe
    predicted3FGPResults<-rbind(predicted3FGPResults, results)
  }
  predicted3FGPResults <- predicted3FGPResults[-1,] ##to get rid of first row of NAs from matrix construction
  return(predicted3FGPResults)
}

#####
#Function: checkSuccessfulPredictions
#Usage: checkSuccessfulPredictions(predictedResultsYear)
#Output: None (print statement)
#This function reports the % of players in the results dataframe whose 3FG% for the predicted season (if available)
#fell within the predicted 95% confidence interval.
checkSuccessfulPredictions<-function(predictionResults){
  successfulPredicts <- 0
  predicts <- 0
  
  for(player in predictionResults$Player[!is.na(predictionResults[["Actual-MeanPrediction"]])]){
    actual3FGP <- as.numeric(predictionResults[predictionResults$Player==player,][["Actual3FGP"]])
    lowerBound <- as.numeric(predictionResults[predictionResults$Player==player,][["95PLowerBound"]])
    upperBound <- as.numeric(predictionResults[predictionResults$Player==player,][["95PUpperBound"]])
    if(!is.nan(lowerBound) & actual3FGP >= lowerBound & actual3FGP <= upperBound) successfulPredicts <- successfulPredicts + 1 
    predicts <- predicts + 1
  }
  
  cat("Percentage of Actual 3FG% within 2 standard deviations: ", signif(100*successfulPredicts/predicts, digits=4))
}



### Regression analysis ###


#Lists of unique ages, years of experience, and positions.
ageList <- min(unique(unlist(seasonIndex[[AGE]], use.names = FALSE))):max(unique(unlist(seasonIndex[[AGE]], use.names = FALSE)))
yearsList <- 0:max(unique(unlist(seasonIndex[[YEAR]], use.names=FALSE)))
positionList <- c("PG", "SG", "SF", "PF", "C")

##Make 1xlength(positionList) matrix of regression coefficients per position
regressionsYear<-rbind(lapply(positionList, function(position){regressionAnalysis(seasonIndex, yearsList, YEAR, position)}))
regressionsAge <-rbind(lapply(positionList, function(position){regressionAnalysis(seasonIndex, ageList, AGE, position)}))
#Make column names of regression matrices match to positions for easy, generalized reference
colnames(regressionsYear) <- positionList
colnames(regressionsAge) <- positionList

#Print weighted regression results
print("Regression Coefficients (int, slope): ")
for(i in 1:length(positionList)){
  cat(positionList[i], "- Year: ", regressionsYear[[i]][[1]], "\n")
  cat(positionList[i], "- Age: ", regressionsAge[[i]][[1]], "\n")
}


### Predictions ###

#List of players in current season. Want to use model on players with thresholdFGA. 
currentPlayers <- unique(unlist(seasonIndex[[PLAYER_NAME]][seasonIndex[[SEASON]] == currentSeason 
                                                           & seasonIndex[[FG3A]] > THRESHOLD_3FGA]), use.names=FALSE)

predictedResultsYear <- predictSeason3FGP(seasonIndex, seasonIndexFull, regressionsYear, YEAR, currentPlayers)
predictedResultsAge <- predictSeason3FGP(seasonIndex, seasonIndexFull, regressionsAge, AGE, currentPlayers)

## Write dataframes to file
write.csv(predictedResultsYear, "IAR_3FG%Prediction_YearsExp.csv")
write.csv(predictedResultsAge, "IAR_3FG%Prediction_Age.csv")

#Prints percentage of predictions within 95% confidence interval
print("Results from Years of Experience: ")
checkSuccessfulPredictions(predictedResultsYear)
print("Results from Age: ")
checkSuccessfulPredictions(predictedResultsAge)