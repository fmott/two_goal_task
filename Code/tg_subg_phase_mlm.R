rm(list = ls())
library(data.table)
library(arm)
library(scales)
library(lme4)
library(lmerTest)

dt <- fread("../Results/preprocessed_results.csv")

### Fitting model using glmer():
optimg      <- dt[(   !is.nan(dt[,suboptimal_goal_decision])    &   valid == 1  &  phase > 1    ),suboptimal_decision]
subg <- optimg == 0
phase    <- dt[(   !is.nan(dt[,suboptimal_goal_decision])    &   valid == 1  &  phase > 1    ),phase] 
subject <- dt[(   !is.nan(dt[,suboptimal_goal_decision])    &   valid == 1  &  phase > 1    ), subject] 

# Fitting different models with glmer().
MLM1 <- glmer(formula = subg ~ phase + (1 + phase | subject), family = binomial(link = "logit"))

mu_a <- fixef (MLM1)[1]
mu_b <- fixef (MLM1)[2]
se_mu_a <- sqrt(diag(vcov(MLM1)))[1]
se_mu_b <- sqrt(diag(vcov(MLM1))) [2]
std_a <- as.data.frame(VarCorr(MLM1))$sdcor[1]
std_b <- as.data.frame(VarCorr(MLM1))$sdcor[2]
corr_ab <- as.data.frame(VarCorr(MLM1))$sdcor[3]
a_j <- coef(MLM1)$subject$`(Intercept)`
b_j <- coef(MLM1)$subject$phase
estimated_subject_errors <- ranef((MLM1))$subject$`(Intercept)`

# Predicted data
invlogit(mu_a + mu_b*1)
invlogit(mu_a + mu_b*2)
invlogit(mu_a + mu_b*3)

summary(MLM1)


