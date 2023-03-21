# This is a lavaan example code for latent growth cruves model
# Xi Chen 2022.08.01

install.packages("lavaan", dependencies = TRUE)
library(lavvan)
formula <- '
i <- ~ 1*y1 + 1*y2 + 1*y3 + 1*y4 # fixed intercept
s <- ~ 0*y1 + 1*y2 + 2*y3 + 3*y4 # random slope, y1 as baseline
i+s ~ interest IV # this is as my interest group manifest vari flow up on the model. do not write as i+s=~IV, this would treat IV as an indicator(manifest variable) of intercept and slope.
'

# For any interested variable, you should decide its intercept and slope to be fixed or random by adjusting the coeffecient. For example, 1,1,1,1 or 0,1,2,3.

model <- growth(formula, data = df)
summary(model, fit.measures = TRUE)

# result summary
## model test user model = chi-square fit test, p > 0.05 means good (notice: highly dependent on sample size!)
## CFI/TLI >0.9
## AIC/BIC smaller better
## RMSEA: <0.05 closed <0.08 acceptable
## SRMR: <0.05

# latent variable
## intercept:
## .intercept estimate: average interested group variable means
## .slope estimate: average interested group variable slope
## variance:
## y1, y2, y3...: residual of each time point
## .intercept: 
## .slope:

# regression: make a reg about each variables you are interested in
## coefficient could be simply add/minus into the avg estimate I mention above, it means the difference.

# covariance
## intercept with slope