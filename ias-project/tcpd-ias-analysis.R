### 1. Packages
library(tidyverse)
library(readxl)
library(lubridate)
library(haven)
library(httr)
library(zoo)
library(knitr)
library(summarytools)
library(reshape)
library(grepl)
library(geojsonio)

### 2. Importing Files
impath<-"/Users/pbajpai/Dropbox/Personal/Economics/Research/ias_analysis/"
profile_path<-"ias_profile.csv"
experience_path<-"ias_experience.csv"
education_path<-"ias_education.csv"

profile_data<-read_csv(paste0(impath,profile_path))
experience_data<-read_csv(paste0(impath,experience_path))
education_data<-read_csv(paste0(impath,education_path))

### 3. Merging and cleaning datasets
ias_full_data <- left_join(profile_data, education_data,experience_data, by = c("ID", "Source", "Service","Name", "Cadre"))
ias_full_data <- distinct(ias_full_data, ID, .keep_all = TRUE)

### 4. Aggregate By State
ias_state_data <- ias_full_data %>% 
  group_by(Place_of_Domicile, Allotment_Year) %>%
  summarise(Count = n_distinct(Place_of_Domicile, Allotment_Year)) %>% group_by(Place_of_Domicile) %>% 
  summarise(Frequency = sum(Count)) %>% subset(Place_of_Domicile != "N.A." & Place_of_Domicile != "Not found")
 
### 5. State wise Chloropeth
spdf <- geojson_read("https://github.com/Subhash9325/GeoJson-Data-of-Indian-States/blob/master/Indian_States")


ias_state_data <- ias_state_data %>%
  ggplot(aes(x=Frequency))

ias_state_data <- ias_state_data + geom_line(aes(y=Place_of_Domicile), colour="#7570B3")
epi_mobility <- epi_mobility + geom_line(aes(y=case_sum.1), colour="#69b3a4")
epi_mobility <- epi_mobility + ylab("7-day moving average of new cases") + scale_x_date(breaks = "4 month")  + theme_classic()

epi_mobility <- epi_mobility + theme(axis.text=element_text(size=18), axis.title=element_text(size=20,face="bold"))

ias_state_data
