########################################################################################################
#                                                                                                      #
#                                         Virat Kohli Runs Stats                                       #
#                                                                                                      #
########################################################################################################

## PURPOSE      Average Analysis of Virat Kohli
##              
## AUTHOR       Pulkit Bajpai

## LAST UPDATE  February 4, 2022


## INPUTS

# - Cricinfo scraped data


## OUTPUTS

# - Trend analysis


## CONTENTS

### 1. Preparation

### 2. Creating Summary Stats

### 3. High Resolution GGPlot 

### 4. Runs/Balls Faced Analysis 




## NOTES

# - Run project_main.R prior to running this.




#########################################################################################################
#########################################################################################################

##############################
#####   1. Preparation   #####
##############################


### Packages and globals ----------
cricket_output <-  "/Users/pbajpai/Dropbox/Home Office/Personal Projects/Cricket/" # for output

if(!require("pacman")) install.packages("cricketr")

pacman::p_load(
  
  cricketr, dplyr, foreign, rvest,
  
  lubridate, ggplot2, scatterplot3d
  
)



### Import data ----------
virat_test <-  getPlayerData(253802, dir=cricket_output,  type="batting", homeOrAway=c(1,2,3),result=c(1,2,3,4))
virat_test$Match <- "Test"
virat_odi <-  getPlayerDataOD(253802, dir=cricket_output,  type="batting", homeOrAway=c(1,2,3),result=c(1,2,3,4))
virat_odi$Match <- "ODI"
virat_tt <- getPlayerDataTT(253802, dir=cricket_output,  type="batting", homeOrAway=c(1,2,3),result=c(1,2,3,4)) 
virat_tt$Match <- "T20"

### Tidying data ----------

#Combining Test, ODI and T20 together
virat_df <- rbind(virat_test, virat_odi, virat_tt) 

#Removing did not bat
virat_df <- virat_df %>% filter(Runs != "DNB" & Runs != "TDNB")

# Creating not out cases
virat_df$Notout <- ifelse(grepl("\\*", virat_df$Runs, ignore.case = T), 1, 0)

# Cleaning the runs variables column and de-stringing it
virat_df <- virat_df %>%  mutate(Runs = gsub("\\*", "", Runs))
virat_df$Runs <- as.numeric(virat_df$Runs) 

# Ordering by format and not-out
virat_df <- virat_df %>% arrange(Match, Notout)

# Converting dates and adding year variables
virat_df$`Start Date` <- as.Date(virat_df$`Start Date`,format='%d %B  %Y')
virat_df$Year <- year(virat_df$`Start Date`)

# Cleaning opposition variable
virat_df <- virat_df %>%  mutate(Opposition = gsub("v ", "", Opposition))

# Exporting cleaned test match for predictive model in CSV
virat_df_test <- virat_df %>% filter(Match == "Test")
write.csv(virat_df_test, paste0(cricket_output,"kohli_test.csv"))

##############################
##### 2. Creating Stats   ####
##############################



# Adding total runs across each format
virat_df <- virat_df %>% group_by(Opposition, Year) %>% 
  
  summarise(Total_runs = sum(Runs),
         
         Total_not_outs = sum(Notout),
         
         Total_games = n())

# Removing Not Outs from Total Games for correcting average
virat_df$Adjusted_games <- virat_df$Total_games - virat_df$Total_not_outs

# Computing averages by Team and Year
virat_df$Average <- virat_df$Total_runs/virat_df$Adjusted_games

# Selecting relevant variables to keep
virat_df <- virat_df %>% select(Opposition, Year, Average)

# Seplacing Infinity averages with 100 (as is done theoretically?)
virat_df$Average <- ifelse(virat_df$Average == Inf, 100, virat_df$Average)


# Filtering out countries
virat_df <- virat_df %>% filter(Opposition == "Australia" | Opposition == "England" | Opposition == "New Zealand" |
                                  
                                  Opposition == "Pakistan" | Opposition == "South Africa" | Opposition == "Sri Lanka" |
                                  
                                  Opposition == "West Indies")

########################################
###### 3. High Resolution GGPlot   #####
########################################  

virat_plot <- ggplot(data=virat_df, aes(x=Year, y=Average, group = Opposition,
                                            colour = Opposition)) + geom_line() + labs(y= "Average across formats", x = "Year") 
  

virat_plot + geom_point() + theme_classic()  + scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) 
       

########################################
##### 4. Runs/Balls Faced Analysis   ####
########################################  


batsmanrunslikelihood <- function(file, name="A Squarecut") {
  
  Runs <- BF <-Mins <- Ground <-Wkts <- NULL
  batsman <- clean(file)
  data <- select(batsman,Runs,BF,Mins)
  
  # Use K Means with 3 clusters
  fit <- kmeans(data, 3)
  str <- paste(name,"'s Runs likelihood vs BF, Mins")
  # Create a 3D scatterplot
  s <-scatterplot3d(data$BF,data$Mins,data$Runs,color="lightblue",main=str,pch=20,
                    xlab="Balls Faced",ylab="Minutes at crease",zlab="Runs scored")
  
  # Plot the centroids
  s$points3d(fit$centers[1,2],fit$centers[1,3],fit$centers[1,1],col="blue",type="h", pch=15,lwd=4)
  s$points3d(fit$centers[2,2],fit$centers[2,3],fit$centers[2,1],col="red",type="h", pch=15,lwd=4)
  s$points3d(fit$centers[3,2],fit$centers[3,3],fit$centers[3,1],col="black",type="h", pch=15,lwd=4)
  mtext("Data source-Courtesy:ESPN Cricinfo", side=1, line=4, adj=1.0, cex=0.8, col="blue")
  
  p1 <- (sum(fit$cluster==1)/length(fit$cluster)) * 100
  p2 <- (sum(fit$cluster==2)/length(fit$cluster)) * 100
  p3 <- (sum(fit$cluster==3)/length(fit$cluster)) * 100
  
  # Print the summary of the centroids
  
  cat("Summary of ",name,"'s runs scoring likelihood\n")
  cat("**************************************************\n\n")
  cat("There is a",round(p1,2), "% likelihood that",name," will make ",round(fit$centers[1,1]),
      "Runs in ",round(fit$centers[1,2]),"balls over",round(fit$centers[1,3])," Minutes \n")
  
  cat("There is a",round(p2,2), "% likelihood that",name," will make ",round(fit$centers[2,1]),
      "Runs in ",round(fit$centers[2,2]),"balls over ",round(fit$centers[2,3])," Minutes \n")
  
  cat("There is a",round(p3,2), "% likelihood that",name," will make ",round(fit$centers[3,1]),
      "Runs in ",round(fit$centers[3,2]),"balls over",round(fit$centers[3,3])," Minutes \n")
  
}


batsmanrunslikelihood(paste0(cricket_output,"kohli_test.csv"),"Virat Kohli")

