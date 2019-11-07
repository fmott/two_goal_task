rm(list = ls())
library(data.table)
library(scales)
library(lme4)
library(lmerTest)

dt <- fread("../Results/preprocessed_results.csv")
threshold <- 11

accum <- rep(NA,89*3)
subject <- rep(NA,89*3)
phase <- rep(NA,89*3)

idx = 1
for (s in 1:89) {
  for(p in 1:3){
    
    A <- dt[ trial == 15   &   phase == 1+p     &  subject == s   ,score_A_after]
    B <- dt[ trial == 15   &   phase == 1+p     &  subject == s   ,score_B_after]
    G2 <- sum(A >= threshold &  B>=threshold)
    G1 <- sum((A >= threshold &  B < threshold) | (A < threshold &  B >= threshold)) 
    fail <- sum(A < threshold &  B < threshold)
    accum[idx] <- G2*10 + G1*5
    subject[idx] <-s 
    phase[idx] <- p
    idx = idx +1 
  }
}

## Fit the model 
MLM1 <- lmer(formula = accum ~ phase + (1 + phase | subject))


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

summary(MLM1)


