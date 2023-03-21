# This is a practical code for dealing with Extract Logistics Regression (ELR, for small sample and non-normal distribution data) and verifying variance.
# Xi Chen 2022.08.01

# data clean ----
library(tidyverse) # 数据整理
library(openxlsx) # 数据导入
library(psych) # 因子分析（本例中无用）
library(sandwich) # 稳健标准误（本例中无用）
library(lmtest) # 标准误内的函数（本例中无用）
library(haven) # 导入/导出sav
library(elrm) # 精确逻辑回归
options(scipen = 200) # 设置位数
options(max.print = 1000000) # 设置打印

# elrm部分 ----
# data import, cor test and duplicate removal
df1 <- read.table("./data/Sta_data/07.22-07.25/elrm/sta2.txt") %>% # elrm，人*任务（1/0）
  rename(act = V62, ID = V63)
cor1 <- cor(df1, use = 'na.or.complete')
write.csv(cor1, file = "./data/Sta_data/07.22-07.25/elrm/cortable2.csv") # 提供相关性矩阵给其他人看，手动决定清除哪些变量


# 去除人和任务1/0的影响，每一行现在仅代表一个被试。考虑到研究者并不知道这些被试的具体情况，直接简单抽样即可
newdf1 <- df1 %>% 
  slice(c(1,3,5,7,9,11,13,15,17,20,22,24,26,28,30,32,34,36)) %>% 
  mutate(trials = 1)

# elrm
elrm.V1 <- elrm(act/trials ~ V1, interest = ~V1, r = 4, iter = 1000, dataset = newdf1, burnIn = 0, alpha = 0.05)
elrm.V3 <- elrm(act/trials ~ V3, interest = ~V3, r = 4, iter = 1000, dataset = newdf1, burnIn = 0, alpha = 0.05)
elrm.V4 <- elrm(act/trials ~ V4, interest = ~V4, r = 4, iter = 1000, dataset = newdf1, burnIn = 0, alpha = 0.05)
elrm.V10 <- elrm(act/trials ~ V10, interest = ~V10, r = 4, iter = 1000, dataset = newdf1, burnIn = 0, alpha = 0.05)
elrm.V16 <- elrm(act/trials ~ V16, interest = ~V16, r = 4, iter = 1000, dataset = newdf1, burnIn = 0, alpha = 0.05)
elrm.V60 <- elrm(act/trials ~ V60, interest = ~V60, r = 4, iter = 1000, dataset = newdf1, burnIn = 0, alpha = 0.05)
elrm.V1$p.values
elrm.V3$p.values
elrm.V4$p.values
elrm.V10$p.values
elrm.V16$p.values
elrm.V60$p.values

# classic logistics
# model1 <- glm(act ~ V1+V3+V4+V10+V16+V60, data = newdf1, family = binomial, na.action = na.omit)
# summary(model1)

# 验证方差部分 ----
df2origin <- read.table("./data/Sta_data/07.22-07.25/方差验证/data_for_3s_s6close.txt") # 验证方差数据
df2origin
df2 <- df2origin %>% slice(1:10) %>% scale(scale = FALSE) %>% data.frame # 手动观察数据后发现数据差别太大，手动切片观察
variance <- c(var(df2$V1), var(df2$V2), var(df2$V3), var(df2$V4), var(df2$V5), var(df2$V6), var(df2$V7), var(df2$V8), var(df2$V9), var(df2$V10), var(df2$V11), var(df2$V12), var(df2$V13), var(df2$V14), var(df2$V15), var(df2$V16), var(df2$V17), var(df2$V18), var(df2$V19), var(df2$V20), var(df2$V21), var(df2$V22), var(df2$V23), var(df2$V24), var(df2$V25), var(df2$V26), var(df2$V27), var(df2$V28), var(df2$V29), var(df2$V30), var(df2$V31), var(df2$V32), var(df2$V33), var(df2$V34), var(df2$V35), var(df2$V36), var(df2$V37), var(df2$V38), var(df2$V39), var(df2$V40), var(df2$V41), var(df2$V42), var(df2$V43), var(df2$V44), var(df2$V45), var(df2$V46), var(df2$V47), var(df2$V48), var(df2$V49), var(df2$V50), var(df2$V51), var(df2$V52), var(df2$V53), var(df2$V54), var(df2$V55), var(df2$V56), var(df2$V57), var(df2$V58), var(df2$V59), var(df2$V60), var(df2$V61))
variance