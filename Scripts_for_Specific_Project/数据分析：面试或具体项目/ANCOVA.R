## 使用方差分析而非回归的原因（虽然本质就是回归），more peer-acceptable
## 使用ANCOVA而非重复测量方差分析的原因：more understandable
# 如果使用重复测量方差分析，则由时间*组的交互项测得。其重点在于三组患者从治疗前到治疗后结果的平均变化是否不同。
# 如果使用ANCOVA，则将前测视为控制变量（调整使得前测一致），三组后测均值是否不同。其重点是某组在治疗后是否具有较高的平均值。当研究问题不是关于收益、增长或变化时较为合适，并在医学研究中常见。主要原因在于研究的重点是治疗效果的大小。
## 依赖包
install.packages("openxlsx")
install.packages("psych")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("car")
install.packages("ggpubr")
install.packages("interactions")
library(openxlsx)
library(psych)
library(dplyr)
library(ggplot2)
library(car)
library(ggpubr)
library(interactions)
## 在外加载和清理数据
ANOVAdata <- read.xlsx("G:\\个人文件\\心理学工作\\Data analysis work\\Data analysis temp folder\\数据分析材料\\Data for ANOVA(N=109).xlsx")
## 进行ANCOVA的假设准备
# test of normality: Shapiro–Wilk
shapiro.test(ANOVAdata$Pre_Anx)
shapiro.test(ANOVAdata$Pre_Dep)
shapiro.test(ANOVAdata$Pre_Sleep_issues)
shapiro.test(ANOVAdata$Pre_Stress)
shapiro.test(ANOVAdata$Post_Anx)
shapiro.test(ANOVAdata$Post_Dep)
shapiro.test(ANOVAdata$Post_Sleep_issues)
shapiro.test(ANOVAdata$Post_Stress)
# Levene's test
leveneTest(Post_Anx ~ Group_info, ANOVAdata, center=median)
leveneTest(Post_Dep ~ Group_info, ANOVAdata, center=median)
leveneTest(Post_Sleep_issues ~ Group_info, ANOVAdata, center=median)
leveneTest(Post_Stress ~ Group_info, ANOVAdata, center=median)
## 线性假设与另一个基于Johnson-Neyman procedure包的分析图（需要Homogen检验）
# interact_plot(Homog_Anx_2, pred = Pre_Anx, modx = Group_info, plot.points = TRUE)
# interact_plot(Homog_Dep_2, pred = Pre_Dep, modx = Group_info, plot.points = TRUE)
ggscatter(
  ANOVAdata, x = "Pre_Anx", y = "Post_Anx",
  color = "Group_info", add = "reg.line"
)+
  stat_regline_equation(
    aes(label =  paste(..eq.label.., ..rr.label.., sep = "~~~~"), color = Group_info)
  )
ggscatter(
  ANOVAdata, x = "Pre_Dep", y = "Post_Dep",
  color = "Group_info", add = "reg.line"
)+
  stat_regline_equation(
    aes(label =  paste(..eq.label.., ..rr.label.., sep = "~~~~"), color = Group_info)
  )
ggscatter(
  ANOVAdata, x = "Pre_Sleep_issues", y = "Post_Sleep_issues",
  color = "Group_info", add = "reg.line"
)+
  stat_regline_equation(
    aes(label =  paste(..eq.label.., ..rr.label.., sep = "~~~~"), color = Group_info)
  )
ggscatter(
  ANOVAdata, x = "Pre_Stress", y = "Post_Stress",
  color = "Group_info", add = "reg.line"
)+
  stat_regline_equation(
    aes(label =  paste(..eq.label.., ..rr.label.., sep = "~~~~"), color = Group_info)
  )
# 回归斜率齐性假设：补充检验
Homog_Anx <- aov(ANOVAdata$Post_Anx ~ ANOVAdata$Group_info * ANOVAdata$Pre_Anx)
Homog_Dep <- aov(ANOVAdata$Post_Dep ~ ANOVAdata$Group_info * ANOVAdata$Pre_Dep)
Homog_Sleep <- aov(ANOVAdata$Post_Sleep_issues ~ ANOVAdata$Group_info * ANOVAdata$Pre_Sleep_issues)
Homog_Stress <- aov(ANOVAdata$Post_Stress ~ ANOVAdata$Group_info * ANOVAdata$Pre_Stress)
Anova(Homog_Anx, type = "II")
Anova(Homog_Dep, type = "II")
Anova(Homog_Sleep, type = "II")
Anova(Homog_Stress, type = "II")
# 进一步检测
Homog_Anx_1 <- lm(Post_Anx ~ Group_info + Pre_Anx, data = ANOVAdata)
Homog_Anx_2 <- lm(Post_Anx ~ Group_info * Pre_Anx, data = ANOVAdata)
anova(Homog_Anx_2, Homog_Anx_1)
Homog_Dep_1 <- lm(Post_Dep ~ Group_info + Pre_Dep, data = ANOVAdata)
Homog_Dep_2 <- lm(Post_Dep ~ Group_info * Pre_Dep, data = ANOVAdata)
anova(Homog_Dep_2, Homog_Dep_1)
# 尽管Johnson-Neyman procedure可以用于进一步解释违反回归斜率齐性假设情况下的ANCOVA，R上似乎没有已经集成的方便函数用来进一步分析在multigroups情况下的Johnson-Neyman inverval。根据线性假设的作图先验认定两个实验组由于抽样误差导致Anx与Dep的数据交汇，实际可以近似认定是平行。因此，主要针对控制组与实验组进行Johnson-Neyman procedure。将组别重新分为if.Exp的二分变量，对数据重新进行分析。并使用sim_slopes()对数据进行解释。
jnp_Anx <- lm(Post_Anx ~ if.Exp * Pre_Anx, data = ANOVAdata)
jnp_Dep <- lm(Post_Dep ~ if.Exp * Pre_Dep, data = ANOVAdata)
interact_plot(jnp_Anx, pred = Pre_Anx, modx = if.Exp, plot.points = TRUE)
interact_plot(jnp_Dep, pred = Pre_Dep, modx = if.Exp, plot.points = TRUE)
sim_slopes(jnp_Anx, pred = if.Exp, modx = Pre_Anx, jnplot = TRUE)
sim_slopes(jnp_Dep, pred = if.Exp, modx = Pre_Dep, jnplot = TRUE)
# 协变量和处理效果之间的独立性
summary(aov(ANOVAdata$Pre_Sleep_issues ~ ANOVAdata$Group_info))
summary(aov(ANOVAdata$Pre_Stress ~ ANOVAdata$Group_info))
# ANCOVA to Sleep & Stress
sleep.ancova <- lm(Post_Sleep_issues ~ Pre_Sleep_issues + Group_info, data = ANOVAdata)
stress.ancova <- lm(Post_Stress ~ Pre_Stress + Group_info, data = ANOVAdata)
Anova(sleep.ancova, type = "III")
Anova(stress.ancova, type = "III")





## 假定需要查看均值或标准误
# install.package("effects")
# library(effects)
# Stress_object<-effect(Post_Stress, stress.ancova, se=TRUE)
# summary(Stress_object)
# object$se

## Another not-used function of Johnson-Neyman procedure
# jnt = function(.lm, predictor, moderator, alpha=.05) {
#   require(stringi)
#   b1 = coef(.lm)[predictor]
#   b3 = coef(.lm)[stri_startswith_fixed(names(coef(.lm)), paste0(predictor,":")) | stri_endswith_fixed(names(coef(.lm)), paste0(":",predictor))]
#   se_b1 = coef(summary(.lm))[predictor, 2]
#   se_b3 = coef(summary(.lm))[stri_startswith_fixed(names(coef(.lm)), paste0(predictor,":")) | stri_endswith_fixed(names(coef(.lm)), paste0(":",predictor)), 2]
#   COV_b1b3 = vcov(.lm)[predictor, stri_startswith_fixed(names(coef(.lm)), paste0(predictor,":")) | stri_endswith_fixed(names(coef(.lm)), paste0(":",predictor))]
#   t_crit = qt(1-alpha/2, .lm$df.residual)
#   # see Bauer & Curran, 2005
#   a = t_crit^2 * se_b3^2 - b3^2
#   b = 2 * (t_crit^2 * COV_b1b3 - b1 * b3)
#   c = t_crit^2 * se_b1^2 - b1^2
#   jn = c(
#     (-b - sqrt(b^2 - 4 * a * c)) / (2 * a),
#     (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
#   )
#   JN = sort(unname(jn))
#   JN = JN[JN>=min(.lm$model[,moderator]) & JN<=max(.lm$model[,moderator])]
#   JN
# }