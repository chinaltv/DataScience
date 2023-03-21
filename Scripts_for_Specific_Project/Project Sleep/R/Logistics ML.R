## This is a project-based code for Classic Logistics Model in Machine Learning field.
## Xi Chen 2022.08.11

## This R file would not be instructive but still referable considering the multicollinearity issue.

# Considering we do not need inference in this case, each line of the dataframe means "participants * task(open & close eyes) * task slice (4 times) * cut-times (3s each)". All the features/variables would enter machine learning model.
# No missing values.

# library ----
library(tidyverse) # data preprocessing
library(caret) # machine learning
library(ggplot2) # visualization

# data import and preprocessing ----
df_regul_total <- read.table("./data/Regularization_ML/features_for_3s_v4_0808.txt") %>%
  rename(act = V64)
df_regul_total$act <- as.factor(df_regul_total$act)

# set train and test group ----
split <- createDataPartition(y=df_regul_total$act, p = 0.75, list = FALSE)
df_regul_train <- df_regul_total[split,] 
df_regul_test <- df_regul_total[-split,]

# resampling
train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5,
  savePredictions = "final" # best prediction
)

# Logistics Reg using glm package ----
mdl_log <- glm(
  act ~ V1 + V3 + V4 + V10 + V16 + V60,
  data = df_regul_train,
  family = binomial,
  na.action = na.omit)


# Logistics Reg using caret package ----
mdl_MLlog <- train(
  act ~ V1 + V3 + V4 + V10 + V16 + V60, # set formula
  data = df_regul_train,
  method = 'glm',
  family = 'binomial',
)

mdl_MLlog2 <- train(
  act ~ V3 + V60, # set formula
  data = df_regul_train,
  method = 'glm',
  family = 'binomial',
)

# Cross-validation ----
pr_MLlog <- postResample(pred = predict(mdl_MLlog, newdata = df_regul_test), obs = df_regul_test$act)
pr_MLlog
pr_MLlog2 <- postResample(pred = predict(mdl_MLlog2, newdata = df_regul_test), obs = df_regul_test$act)
pr_MLlog2