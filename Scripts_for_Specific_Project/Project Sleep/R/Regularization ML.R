## This is a project-based code for Regularization Logistics Model in Machine Learning field.
## Xi Chen 2022.08.08

## Someone said Ridge, Lasso and Elnet is only suitable for continuous y regularization. Before figuring out this issue, this R file would not be instructive but still referable.

# Considering we do not need inference in this case, each line of the dataframe means "participants * task(open & close eyes) * task slice (4 times) * cut-times (3s each)". All the features/variables would enter machine learning model.
# No missing values.

# library ----
library(tidyverse) # data preprocessing
library(caret) # machine learning basic package
library(glmnet) # glmnet/lasso/ridge package
library(ggplot2) # visualization

# data import and preprocessing ----
df_regul_total <- read.table("./data/Regularization_ML/features_for_3s_v3_0808.txt") %>%
  rename(act = V11)
# df_regul_total <- read.table("./data/Regularization_ML/features_for_3s_v4_0808.txt") %>%
#   rename(act = V64)
df_regul_total$act <- as.factor(df_regul_total$act)

# set train and test group ----
split <- createDataPartition(y=df_regul_total$act, p = 0.75, list = FALSE)
df_regul_train <- df_regul_total[split,] 
df_regul_test <- df_regul_total[-split,]

# use K-fold for a common object
train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5,
  savePredictions = "final" # best prediction
)

# ridge reg ----
mdl_ridge <- train(
  act ~ ., # set formula
  data = df_regul_train,
  method = "glmnet",
  metric = "Accuracy",  # for Logistics Reg/Classification task, it should be Accuracy/Kappa/ROC... For continuous, you should choose from RMSE, RSquared, AIC, BIC, ...
  preProcess = c("center", "scale"), # standardization
  tuneGrid = expand.grid(
    .alpha = 0,  # optimize a ridge regression
    .lambda = seq(0, 5, length.out = 101)
  ),
  trControl = train_control # for a common model
)
mdl_ridge

# lasso reg ----
mdl_lasso <- train(
  act ~ .,
  data = df_regul_train,
  method = "glmnet",
  metric = "Accuracy",
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    .alpha = 1,  # optimize a lasso regression
    .lambda = seq(0, 5, length.out = 101)
  ),
  trControl = train_control
)
mdl_lasso

# Elastic net regularization ----
mdl_Elastic <- train(
  act ~ .,
  data = df_regul_train,
  method = "glmnet", 
  metric = "Accuracy",
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    .alpha = seq(0, 1, length.out = 10),
    .lambda = seq(0, 5, length.out = 101) # Elastic net regularization
  ),
  trControl = train_control
)
mdl_Elastic

# Routinely visualization ----
fig_ridge <- ggplot(mdl_ridge) + 
  labs(title = "Ridge Reg Parameter Tuning", x = "lambda")
fig_ridge
fig_important_ridge <- mdl_ridge %>% varImp() %>% plot()
fig_important_ridge

fig_lasso <- ggplot(mdl_lasso) + 
  labs(title = " Lasso Reg Parameter Tuning", x = "lambda")
fig_lasso
fig_important_lasso <- mdl_lasso %>% varImp() %>% plot()
fig_important_lasso

fig_Elastic <- ggplot(mdl_Elastic) + 
  labs(title = "Elastic Net Reg Parameter Tuning", x = "lambda")
fig_Elastic
fig_important_Elastic <- mdl_Elastic %>% varImp() %>% plot()
fig_important_Elastic

# Cross-validation ----

pr_ridge <- postResample(pred = predict(mdl_ridge, newdata = df_regul_test), obs = df_regul_test$act)
pr_lasso <- postResample(pred = predict(mdl_lasso, newdata = df_regul_test), obs = df_regul_test$act)
pr_Elastic <- postResample(pred = predict(mdl_Elastic, newdata = df_regul_test), obs = df_regul_test$act)
pr_ridge
pr_lasso
pr_Elastic