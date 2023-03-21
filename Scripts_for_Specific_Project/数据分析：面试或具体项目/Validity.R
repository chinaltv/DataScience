## 依赖包
install.packages("openxlsx")
install.packages("psy")
install.packages("ggplot2")
library(openxlsx)
library(psy)
library(ggplot2)
## 在外加载和清理数据与基础设置
Vdata <- read.xlsx("G:\\个人文件\\心理学工作\\Data analysis work\\Data analysis temp folder\\数据分析材料\\Validity.xlsx")
options(scipen = 200)
## Predictive validity
cor.test(Vdata$sumscore, Vdata$Anxiety)
cor.test(Vdata$sumscore, Vdata$Depression)
cor.test(Vdata$sumscore, Vdata$Sleep_issues)
cor.test(Vdata$sumscore, Vdata$Stress)
## Convergent and divergent validity
MTMMresult <- mtmm(Vdata, list(
  c("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"), 
  c("d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"), 
  c("o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8"), 
  c("j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8"), 
  c("r1", "r2", "r3", "r4", "r5", "r6", "r7")
  ), graphItem = TRUE)
MTMMresult.m <- melt(MTMMresult, id="Item", variable.name="ScaleI", value.name="Correlation")
MTMMresult.m <- MTMMresult.m[-c(1:39),]
ggplot(MTMMresult.m, aes(ScaleI, Item, fill=abs(Correlation))) + 
  geom_tile() + 
  geom_text(aes(label = round(Correlation, 2)), size=2.5) + 
  theme_bw(base_size=10) + 
  theme(axis.text.x = element_text(angle = 90), 
        axis.title.x=element_blank(), 
        axis.title.y=element_blank(), 
        plot.margin = unit(c(3, 1, 0, 0), "mm")) +
  scale_fill_gradient(low="white", high="red") + 
  guides(fill="none") 