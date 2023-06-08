# Simple Linear Regression

# Importing the dataset
dataset = read.csv("Salary_Data.csv")

# Split into test and train set
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting linear regression to training set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

# Predicting test results
y_pred = predict(regressor, newdata = test_set)

# Building the graph
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  
  ggtitle('Salary vs. Experience (Training Set)') +
  xlab('Years of Experience') + 
  ylab('Salary')
