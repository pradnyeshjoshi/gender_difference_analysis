setwd("~/Desktop/gender_difference_analysis/src/")
source("utils.R")
library(randomForest)
library(ROCit)
library(caret)
library(dplyr)
library(lme4)
library(ggplot2)

# This function dichotomizes each comment sentiment using
# a median threshold, builds a random forest classifier, and plots ROC curve.
# Formula:
# comment_sentiment ~ gender_post + gender_comment + post_sentiments +
#                       subreddit_type + post/title/comment wordcounts +
#                       ups + author_premium + is_self + no_follow + num_comments
fit_random_forest <- function(
        reddit_data,
        features,
        comment_sentiments
){
        output_path <- '../output/rf50_scaled_summary.txt'
        for(c in comment_sentiments){
                print(c)
                start_time <- proc.time()
                d <- cbind(reddit_data[features], response = reddit_data[[c]])
                d <- transform_features(d, features)
                d <- scale_features(d, features)
                d$response <- as.factor(ifelse(d$response >= median(d$response), 1, 0))
                rf_model <- randomForest(
                        response ~ .,
                        data = d,
                        importance = TRUE,
                        ntree = 50,
                        mtry = 5
                )
                print(proc.time()-start_time)
                cat(paste0("\n",c,"\n"), file = output_path, append = TRUE)
                capture.output(rf_model, file = output_path, append = TRUE)
                capture.output(importance(rf_model), file = output_path, append = TRUE)
                # varImpPlot(rf_model)
                predictions <- predict(rf_model, d, type = 'prob')
                ROCit_obj <- rocit(score=predictions[,2],class=d$response)
                png(filename=paste0("../plots/rf_plots/roc_rf50_scaled_",c,".png"))
                plot(ROCit_obj)
                dev.off()
                rm(rf_model, predictions)
        }
}

reddit_path <- "../data/reddit_data.csv"
input <- load_reddit(reddit_path)
reddit_data <- input[[1]]
features <- input[[2]]
comment_sentiments <- input[[3]]
rm(input)

fit_random_forest(reddit_data, features, comment_sentiments)