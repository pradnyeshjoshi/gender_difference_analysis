setwd("~/Desktop/reddit/src/")
source("utils.R")
library(randomForest)
library(ROCit)
library(caret)
library(dplyr)
library(lme4)
library(ggplot2)

reddit_path <- "../data/reddit_data.csv"
input <- load_reddit(reddit_path)
reddit_data <- input[[1]]
features <- input[[2]]
comment_sentiments <- input[[3]]
rm(input)

# Now, let's plot histograms of all variables.
# Will do log transforms where distribution is skewed.
plot_histograms(reddit_data, features)

# Linear Mixed Models
# Let's subset the data for one comment sentiment.
c <- 'trust_comment'
d <- cbind(reddit_data[c(features, "link_id")], response = reddit_data[[c]])
d <- transform_features(d, features)

# Making sure our features are scaled!
d <- scale_features(d, features)

# plot_interactions(d, 'sadness')
# plot_interactions(d, 'positive')
# plot_interactions(d, 'negative')
# plot_interactions(d, 'trust')

# Finally, building an LMER model.
mixed.lmer <- lmer(response ~ . + predicted_gender_comment:comment_wordcount +
                           sadness_post:predicted_gender_post +
                           sadness_post:predicted_gender_comment +
                           # fairness_post:predicted_gender_post +
                           # fairness_post:predicted_gender_comment +
                           predicted_gender_post:predicted_gender_comment -
                           link_id +
                           (1|link_id), data = d)

# Diagnostics
summary(mixed.lmer)

# Plot residuals vs fitted values.
plot(mixed.lmer)

# QQ plot
qqnorm(resid(mixed.lmer))
qqline(resid(mixed.lmer))