## 依赖包 ---------------------------------
install.packages("openxlsx")
install.packages("psych")
install.packages("ggplot2")
install.packages("plotly")
install.packages("reshape2")
library(openxlsx)
library(psych)
library(ggplot2)
library(plotly)
library(reshape2)
## 加载和清理数据 ---------------------------------
Mindfulness4EFA <- read.xlsx("G:\\个人文件\\心理学工作\\Data analysis work\\Data analysis temp folder\\数据分析材料\\Mindfulness(N=482)reset&cleaned.xlsx")
Mindfulness4EFAreset <- Mindfulness4EFA[,-1]
Mindfulness4EFAreset <- Mindfulness4EFAreset[,-1]
## 设定格式并进行KMO/Bartlett检验 ---------------------------------------------
options(scipen = 200)
KMO(Mindfulness4EFAreset)
bartlett.test(Mindfulness4EFAreset)
## 进行因子分析并创建碎石图所需要向量，试图重画scree plot -----------
FAresult <- fa(Mindfulness4EFAreset, fm = "pa", rotate = "varimax", nfactors = 5)
scree(Mindfulness4EFAreset)
indices <- c(10:39)
scree.fv.9 <- scree(Mindfulness4EFAreset)[[1]][-indices]
scree.pcv.9 <- scree(Mindfulness4EFAreset)[[2]][-indices]
x <- c(1:9)
data <- data.frame(x, scree.fv.9, scree.pcv.9)
## 一个图 ----------------------------------------------------------------------
# qplot(c(1:8), scree.fv.9) +
#   geom_line() +
#   geom_point(size=9)+
#   xlab("Factor or component number") +
#   ylab("Eigen values of factors and components") +
#   ggtitle("Scree Plot") +
#   ylim(0, 10)
## 另一个图 ---------------------------------
# scree(Mindfulness4EFAreset)
## plotly的好看点，虽然感觉和Excel没什么差别
hline <- function(y = 0, color = "black") {
  list(type = "line",
       x0 = 0,
       x1 = 1,
       xref = "paper",
       y0 = y,
       y1 = y,
       line = list(color = color))
  }
fig <- plot_ly(data,
               x = ~x,
               y = ~scree.fv.9,
               name = 'FA',
               type = 'scatter',
               mode = 'lines+markers') %>%
  layout(
    title ='Scree Plot',
    font = list(size = 20),
    margin = list(l=50, r=50, b=100, t=100, pad=4),
    xaxis = list(title = 'Factor or component number'),
    yaxis = list(title = 'Eigen values of factors and components'),
    legend = list(title=list(text='<b> Methods </b>')),
    shapes = list(hline(1))
  )
fig <- fig %>% add_trace(y = ~scree.pcv.9, name = 'PC', mode = 'lines+markers')
fig_2 <- style(fig, marker = list(size = 12))
## 查看EFA数据 ---------------------------------
FAresult
FAcorrs <- FAresult[["r"]]
FAloadings <- FAresult[["loadings"]]
## 分离表格1,2 ---------------------------------
Lambda <- unclass(FAloadings)
p <- nrow(Lambda)
factors <- ncol(Lambda)

vx <- colSums(FAloadings^2)
varex <- rbind(`SS loadings` = vx)

if (is.null(attr(FAloadings, "covariance"))) {
  varex <- rbind(varex, `Proportion Var` = vx/p)
  if (factors > 1)
    varex <- rbind(varex, `Cumulative Var` = cumsum(vx/p))
}

tibble::rownames_to_column(as.data.frame(varex), "FAloadings")
## 在Excel根据结果重命名因子 ---------------------------------
Lambda <- data.frame(Lambda)
write.xlsx(Lambda, "G:\\个人文件\\心理学工作\\Data analysis work\\Data analysis temp folder\\数据分析材料\\Lambda.xlsx")
Lambda <- read.xlsx("G:\\个人文件\\心理学工作\\Data analysis work\\Data analysis temp folder\\数据分析材料\\Lambda.xlsx")
## 作factor loading图 ---------------------------------
Lambda.m <- melt(Lambda, id="ItemName",
                 measure=c("ActAware", "Describing",
                           "Observing", "NonJudging", "NonReacting"),
                 variable.name="Factor", value.name="Loading")

ggplot(Lambda.m, aes(ItemName, abs(Loading), fill=Loading)) +
  facet_wrap(~ Factor, nrow=1) +
  geom_bar(stat="identity") +
  coord_flip() +
  scale_fill_gradient2(name = "Loading",
                       high = "blue", mid = "white", low = "red",
                       midpoint=0, guide=F) +
  ylab("Loading Strength") +
  theme_bw(base_size=10)
## 作相关矩阵图 ---------------------------------
FAcorrs <- data.frame(FAcorrs)
write.xlsx(FAcorrs, "G:\\个人文件\\心理学工作\\Data analysis work\\Data analysis temp folder\\数据分析材料\\FAcorrs.xlsx")
FAcorrs <- read.xlsx("G:\\个人文件\\心理学工作\\Data analysis work\\Data analysis temp folder\\数据分析材料\\FAcorrs.xlsx")
FAcorrs.m <- melt(FAcorrs, id="ItemName", variable.name="Test", value.name="Correlation")
library(grid)
ggplot(FAcorrs.m, aes(Test, ItemName, fill=abs(Correlation))) +
  geom_tile() +
  geom_text(aes(label = round(Correlation, 2)), size=2.5) +
  theme_bw(base_size=10) +
  theme(axis.text.x = element_text(angle = 90),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        plot.margin = unit(c(3, 1, 0, 0), "mm")) +
  scale_fill_gradient(low="white", high="red") +
  guides(fill=F)
## 输出varex table ---------------------------------
varex <- data.frame(varex)
write.xlsx(varex, "G:\\个人文件\\心理学工作\\Data analysis work\\Data analysis temp folder\\数据分析材料\\varex.xlsx")
