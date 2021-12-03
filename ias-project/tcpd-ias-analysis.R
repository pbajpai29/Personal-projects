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

### 4. Filtering dataset
surname <- c("Bajpai")
filter(ias_full_data, str_detect(surname))
bajpai <- filter(ias_full_data, grepl("Lalit", Name))
ias_1972 <- ias_full_data %>% filter(Allotment_Year==1972) 
xwtable(ias_1972$Name)
