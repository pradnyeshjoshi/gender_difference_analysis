setwd("~/Desktop/gender_difference_analysis/src/")
source("utils.R")
library(ROCit)
library(caret)
library(dplyr)
library(ggplot2)

# This function dichotomizes each comment sentiment using
# a median threshold, builds a logistic regression model, and plots ROC curve.
# Formula:
# comment_sentiment ~ gender_post + gender_comment + post_sentiments +
#                       subreddit_type + post/title/comment wordcounts +
#                       ups + author_premium + is_self + no_follow + num_comments
fit_logreg <- function(
        reddit_data,
        features,
        comment_sentiments,
        output_path = '../output/logreg_summary.txt'
){
        for(c in comment_sentiments){
                print(c)
                start_time <- proc.time()
                d <- cbind(reddit_data[features], response = reddit_data[[c]])
                d <- transform_features(d, features)
                d <- scale_features(d, features)
                d$response <- as.factor(ifelse(d$response >= median(d$response), 1, 0))
        
                lr_model <- glm(response~., family = binomial, data = d)
                predictions_prob <- predict(lr_model, type="response")
                predictions_class <- as.factor(ifelse(predictions_prob>0.5, 1, 0))
                
                print(proc.time()-start_time)
                cat(paste0("\n",c,"\n"), file = output_path, append = TRUE)
                capture.output(summary(lr_model), file = output_path, append = TRUE)
                capture.output(confusionMatrix(predictions_class, d$response), file = output_path, append = TRUE)
                
                ROCit_obj <- rocit(score = predictions_prob, class = d$response)
                png(filename=paste0("../plots/logreg_plots/roc_logreg_",c,".png"))
                plot(ROCit_obj)
                dev.off()
                rm(lr_model, predictions_prob, predictions_class)
        }
}

reddit_path <- "../data/reddit_data.csv"
input <- load_reddit(reddit_path)
reddit_data <- input[[1]]
features <- input[[2]]
comment_sentiments <- input[[3]]
rm(input)

fit_logreg(reddit_data, features, comment_sentiments)
