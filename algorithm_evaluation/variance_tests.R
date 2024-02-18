setwd('C:/Users/pedro/Desktop/test2')
library(ggplot2)

res = read.csv('cv_results.csv')

# Boxplots
ggplot()+
  geom_boxplot(data=res, aes(y=MSE, fill=GROUP)) +
  theme_light()

# Densities
ggplot() +
  geom_density(data=res, aes(x=MSE, fill=GROUP), alpha=0.5) +
  theme_light()

# ANOVA
summary(aov(MSE~GROUP, res))

# Tukey HSD
TukeyHSD(aov(MSE~GROUP, res))

# T-test - SVR vs TREE
t.test(MSE~GROUP, res[which(res$GROUP != 'rn'),])

# T-test - SVR vs RN
t.test(MSE~GROUP, res[which(res$GROUP != 'tree'),])

# T-test - RN vs TRE
t.test(MSE~GROUP, res[which(res$GROUP != 'svr'),])
