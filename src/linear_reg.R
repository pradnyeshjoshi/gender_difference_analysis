setwd("~/Desktop/reddit/src/")
source("utils.R")
library(ROCit)
library(caret)
library(dplyr)
library(ggplot2)

# For each comment sentiment, this function builds a linear regression model,
# and saves the summary to an output file.
# comment_sentiment ~ gender_post + gender_comment + post_sentiments +
#                       subreddit_type + post/title/comment wordcounts +
#                       ups + author_premium + is_self + no_follow + num_comments
fit_linear_reg <- function(
        reddit_data,
        features,
        comment_sentiments
){
        output_path <- "../output/linear_reg_summary.txt"
        for(c in comment_sentiments){
                print(c)
                d <- cbind(reddit_data[features], response = reddit_data[[c]])
                d <- transform_features(d, features)
                d <- scale_features(d, features)
                lm_fit <- lm(response~. + predicted_gender_post : predicted_gender_comment, data = d)
                cat(paste0("\n",c,"\n"), file = output_path, append = TRUE)
                capture.output(summary(lm_fit), file = output_path, append = TRUE)
        }
}

reddit_path <- "../data/reddit_data.csv"
input <- load_reddit(reddit_path)
reddit_data <- input[[1]]
features <- input[[2]]
comment_sentiments <- input[[3]]
rm(input)

fit_linear_reg(reddit_data, features, comment_sentiments)
