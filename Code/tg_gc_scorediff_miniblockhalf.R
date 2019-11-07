rm(list = ls())
library(data.table)
library(arm)

dt <- fread("../Results/preprocessed_results.csv")

gc      <- dt[(   !is.nan(dt[,goal_decision])    &   valid == 1  &  phase > 1    ),goal_decision] - 1 
phase     <- dt[(   !is.nan(dt[,goal_decision])    &   valid == 1  &  phase > 1    ),phase] 
trial    <- dt[(   !is.nan(dt[,goal_decision])    &   valid == 1  &  phase > 1    ),trial] 
score_diff    <- abs(dt[(   !is.nan(dt[,goal_decision])    &   valid == 1  &  phase > 1    ),score_difference]) 

miniblock_half = rep(0,length(gc))
miniblock_half[trial<= 7] <- 1
miniblock_half[trial >= 8] <- 0

subject <- dt[(   !is.nan(dt[,goal_decision])    &   valid == 1  &  phase > 1    ), subject] 
res_glm <- glm(gc ~ score_diff + miniblock_half + score_diff*miniblock_half, family = binomial(link = "logit" ))

b0 <- coef(res_glm)[1]
b1 <- coef(res_glm)[2]
b2 <- coef(res_glm)[3]
b3 <- coef(res_glm)[4]

# score_diff_unique <- seq(0,15)
# first <- invlogit(b0+ b1*score_diff_unique + b2*1 + b3*1*score_diff_unique)
# second <- invlogit(b0+ b1*score_diff_unique + b2*0 + b3*0*score_diff_unique)
# plot(score_diff_unique,first,col = 'black',ylim=c(0,1),type = 'line',ylab = 'P(g2)',xlab = 'absolute score diff')
# lines(score_diff_unique,second,col='red',ylim=c(0,1),type = 'line',ylab = 'P(g2)',xlab = 'absolute score diff')

summary(res_glm)