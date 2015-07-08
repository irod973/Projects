#########
##File: seasonIndexBuilder
##Author: Irving Rodriguez
##Description: 
## This file builds an index containing player stats, including change in 3FG% from the previous season, from 
## a database and writes it as a csv. 
#########

##Libraries
library(gdata)

### Constants ###

#Filenames
SEASON_FILE_STRING <- "/home/irod973/Documents/Projects/3FG%_Forecast/seasonStats_3FGEra.csv"

#Dataframe Column Strings
FG3P <- "X3P.1"
AGE <- "Age"
PLAYER_NAME <- "Player"
SEASON <- "Season"
POS <- "Pos"
X3PA <- "X3PA"

######### Body #########

### Load season stats for 3FG Era Into Data Frame

seasonIndex <- read.csv(SEASON_FILE_STRING)
seasonIndex <- seasonIndex[order(seasonIndex[[PLAYER_NAME]], seasonIndex[[SEASON]]),]

### Calculate d3FG% and Years of Experience for each season and append to dataframe

##Special cases to consider:
#Players with the same name
#Players traded in-season (multiple listings for one season)

#As a workaround to pigeonhole, start with first row's values.
prevPlayer <- seasonIndex[[PLAYER_NAME]][1]
prev3FGP <- seasonIndex[[FG3P]][1]
#In calculating rookie years, need to make sure we're not cutting off rookie years prior to 79-80. Can solve this later by using playerIndex
rookieAge <- seasonIndex[[AGE]][1]
delta <- c(NA)
years <- c(0) #First year in data is always rookie year, since secondarily sorted by season

## Loop through dataframe
for(i in 2:nrow(seasonIndex)){
  #Set values for current row
  currPlayer <- seasonIndex[[PLAYER_NAME]][i]
  currAge <- seasonIndex[[AGE]][i]
  curr3FGP <- seasonIndex[[FG3P]][i]
  
  #we can get a new player when (1) there is a name difference or (2) Name stays the same but age change is negative (different players have identical names)
  #we want to change the value of rookie year and prevPlayer to match the ones for the new player. 
  #by definition, d3FG% for rookie year is NA
  if((!identical(prevPlayer, currPlayer)) || (identical(prevPlayer, currPlayer) && (seasonIndex[[AGE]][i] - seasonIndex[[AGE]][i-1] < 0))){
    rookieAge <- currAge
    prevPlayer <- currPlayer
    prev3FGP <- NA
  }
  
  ##this is the case when a player was traded mid season
  #we only want a delta value for the season total, which is the first row where currAge != nextAge. Note that we do not change prevYear or prev3FGP
  if(i != nrow(seasonIndex) && identical(currPlayer, seasonIndex[[PLAYER_NAME]][i+1]) && seasonIndex[[AGE]][i] == seasonIndex[[AGE]][i+1]) {
    delta <- c(delta, NA)
    years <- c(years, currAge - rookieAge)
    next
  }

  #Concatenate to vectors
  years <- c(years, currAge - rookieAge)
  delta <- c(delta, curr3FGP - prev3FGP)
  
  #Change previous values now that we have recorded the differences
  prevPlayer <- currPlayer
  prev3FGP <- curr3FGP
}

#Add vectors to dataframe
seasonIndex$Year <- c(years)
seasonIndex$Delta3FGP <- c(delta)

#Write dataframe to file
write.csv(seasonIndex, file="IAR_season
          Index3FGEra.csv")