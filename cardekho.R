library(data.table)
library(tinytex)
library(tidyverse)

#
library(rpart)
library(rpart.plot)

#

library(caret)

# The dataset resides at kaggle and can be downloaded using following link
# https://www.kaggle.com/saisaathvik/used-cars-dataset-from-cardekhocom/download
# However to simplify, i have uploaded the csv file to google drive and is open
# and accessible to all, can be downloaded by R using the following link
# https://drive.google.com/u/0/uc?id=1KlGCf0r2E56WhCqzJ-9_v5KOzPn3T6UJ&export=download


# Temp file location
dl <- tempfile()
# Download the dataset file from google drive/ open access to all
download.file("https://drive.google.com/u/0/uc?id=1KlGCf0r2E56WhCqzJ-9_v5KOzPn3T6UJ&export=download", dl)

# read the contents of file from temp location and populate to dataframe src
src <- fread(text = gsub("::", ",", readLines(dl)))

# view the contents of the dataset
src

# Check to see path of temporary file location where the file is downloaded
dl

# glance at the columns that the file has
names(src)

# next level take a look at the 1st,2nd, 3rd, max quartiles of measures data and groups of attributes
summary(src)

# filter outlier that have a huge selling price, very very far from the mean/median
src<-src %>% filter(is.na(selling_price)==FALSE & selling_price<10000000)


# correct the data in min cost price , create col mcp that holds correct values of min cost price
# the data in min_cost_price column is wrong for cars with range as XXLakh to X crore
# eg for a car with range 70L to 1.1 Crore the min_cost_price was 700,000,000 to 11,000,000
# the min cost price was wrongly calculated with additional 2 zeros for such cases, hence correcting data
# the latest min cost price is stored in mcp column
src<-src %>% mutate(mcp=case_when (min_cost_price>100000000 ~ min_cost_price/100 , TRUE ~ min_cost_price))

# filter out incomplete data rows, lots of missing data, hence need to cleanup before we use it
src<-src %>%  filter(is.na(min_cost_price)==FALSE & is.na(vehicle_age)==FALSE &
                         is.na(engine)==FALSE & is.na(km_driven)==FALSE & is.na(transmission_type)==FALSE)

# plot km_driven, we can see outliers
plot(src$km_driven)

# plot vehicle_age, we can see outliers
plot(src$vehicle_age)

# filter out the outlier in terms of km_driven, looks like error records of km_driven more than 1Million km :)
src<- src %>% filter(km_driven<600000)

# filter outliers in terms of vehicle age, considering the scrappage policy, cannot drive more than 20yrs old
src<- src %>% filter(vehicle_age<=20)


# take a look at data again
src %>% arrange(desc(km_driven)) %>% head(10) %>% select(km_driven)
src %>% arrange(desc(vehicle_age))

# a glance at summary again to see if the max, min and quartile values looks reliable
summary(src)


# set seed to 1
set.seed(1,sample.kind = NULL)

# partitioning the dataset src into train and test datasets, 20% into testing
index<-createDataPartition(src$selling_price,times = 1,p=0.2,list = FALSE)

# populate train_set and test_set: train_set=~80% of src, test_set=~20% of src
train_set<-src[-index,]
test_set<-src[index,]

# check row counts to see its per expectations 80/20 split
NROW(train_set)
NROW(test_set)


# normalize the cost price range from min to max - using avg of min & max cost price
train_set<- train_set %>% mutate(avg_cost_price=(mcp+max_cost_price)/2)
test_set<- test_set %>% mutate(avg_cost_price=(mcp+max_cost_price)/2)



# Linear regression model to predict the selling price
fit1<- train(selling_price ~ vehicle_age + avg_cost_price + km_driven + max_power + mileage, 
             data=train_set, method="lm", na.action = na.omit)

fit1
# RMSE      Rsquared   MAE     
# 482269.5  0.6944868  252031.4

# we got the RMSE of 482,269 and R^2 of .69, lets see if we can improve this

# check the importance of factors used in model
varImp(fit1)

# predict the outcome/selling price of cars in test_set using the above model (linear regression)
p2<-predict(fit1,newdata=test_set)

# calculate the Root mean square error
RMSE(p2,test_set$selling_price)

# Calculate and plot the residual value, histogram
test_set<- test_set %>% mutate(p=p2,diff=selling_price-p2)
test_set %>% arrange(desc(abs(diff))) %>%  select(selling_price, p, diff) %>% head(5)
test_set %>% ggplot(aes(diff)) + geom_histogram() 





# using decision tree - rpart ml algorithm to see if we can get better RMSE values
fit3<-train(selling_price ~ vehicle_age + engine + avg_cost_price + km_driven + max_power + mileage, 
            data=train_set, method="rpart", na.action = na.omit, tuneGrid = data.frame(cp= seq(0, 0.05, 0.002)))

# list down all cp's and corresponding R^2 & RMSE values
fit3

# final model / best fit from all the predicted ones
fit3$finalModel

# prune the tree using cp=0.006, making a little easy to apprehend
new_tree<-prune(fit3$finalModel,cp=0.006)

# plot the decision tree
prp(new_tree)


# predict the selling price using about decision tree model and test data set
p2<-predict(fit3,newdata=test_set)

# Evaluate the Root mean square error values 
RMSE(p2,test_set$selling_price)

# calculate and plot the residual values in histogram
test_set<- test_set %>% mutate(p=p2,diff=selling_price-p2)
test_set %>% arrange(desc(abs(diff)))
test_set %>% ggplot(aes(diff)) + geom_histogram() 



# plot(fit3$finalModel)
# text(fit3$finalModel)

# cp     RMSE      Rsquared   MAE     
# 0.000  262238.8  0.9075905  116887.9
# 0.002  307165.5  0.8726727  166339.4
# 0.004  332194.0  0.8511685  181058.1






# Let us see if we can use random forest algorithm and further improve the RMSE

# Setting up the control parameters and tuning grid parameter for random forest ml algorithm
# using repeated control validation, with 3 repeats (random)
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")

# setting up mtry / no of attributes to try before split as a sequence from 1 to 10
tunegrid <- expand.grid(.mtry = (1:5)) 


# Train the random forest ml algorithm to predict the selling price of used car in train set
fit4<-train(selling_price ~ vehicle_age + avg_cost_price + km_driven + max_power + mileage, 
            data=train_set, method="rf", na.action = na.omit, tuneGrid = tunegrid)

# display the importance of attibutes
varImp(fit4)

# check to see what mtry value is the best fit and R^2 values
fit4

# predict the selling price in test data set using above random forest trained model
p2<-predict(fit4,newdata=test_set)

# 162784.9
# 143159.1
# 140404.8
# 141422.8

# calculate the Root Mean square value using test data set/ 
# comparing predicted selling price with actual selling price
RMSE(p2,test_set$selling_price)


# Calculate the residual value and plot the same in histogram
test_set<- test_set %>% mutate(p=p2,diff=selling_price-p2)
test_set %>% arrange(desc(abs(diff)))
test_set %>% ggplot(aes(diff)) + geom_histogram() 



