setwd("~/Desktop/entrepreneurship_research/sentiment_analysis/src/")
library(randomForest)
library(ROCit)
library(caret)
library(dplyr)
library(lme4)
library(ggplot2)

# Load the merged reddit post and comment data.
load_reddit <- function(reddit_path){
        reddit_data <- read.csv(reddit_path)
        reddit_data <- reddit_data[complete.cases(reddit_data),]
        cols <- colnames(reddit_data)
        features <- cols[grepl("_post", cols) | grepl("gender_", cols) | cols == "subreddit"]
        features <- c(features, 'post_wordcount', 'title_wordcount', 'comment_wordcount', 'ups', 'author_premium',
                      'is_self', 'no_follow', 'num_comments')
        comment_sentiments <- cols[grepl("_comment", cols) & !grepl("gender", cols)]
        return(list(reddit_data, features, comment_sentiments))
}

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
        output_path <- "../output/model_summary/model_summary_agg.txt"
        for(c in comment_sentiments){
                print(c)
                d <- cbind(reddit_data[features], response = reddit_data[[c]])
                lm_fit <- lm(response~. + gender_post : gender_comment, data = d)
                cat(paste0("\n",c,"\n"), file = output_path, append = TRUE)
                capture.output(summary(lm_fit), file = output_path, append = TRUE)
        }
}

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
        output_path <- '../output/model_summary/rf50_scaled_summary.txt'
        for(c in comment_sentiments){
                print(c)
                start_time <- proc.time()
                d <- cbind(reddit_data[features], response = reddit_data[[c]])
                d$response <- as.factor(ifelse(d$response >= median(d$response), 1, 0))
                scale_features <- features[!features %in% c("gender_post","gender_comment", "subreddit", "author_premium", "is_self","no_follow")]
                d <- d %>% mutate_at(scale_features, ~(scale(.) %>% as.vector))
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
                png(filename=paste0("../output/plots/roc_rf50_scaled_",c,".png"))
                plot(ROCit_obj)
                dev.off()
                rm(rf_model, predictions)
        }
}

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
        output_path = '../output/model_summary/logreg_summary.txt'
){
        for(c in comment_sentiments){
                print(c)
                start_time <- proc.time()
                d <- cbind(reddit_data[features], response = reddit_data[[c]])
                d$response <- as.factor(ifelse(d$response >= median(d$response), 1, 0))
                scale_features <- features[!features %in% c("gender_post","gender_comment", "subreddit", "author_premium", "is_self","no_follow")]
                d <- d %>% mutate_at(scale_features, ~(scale(.) %>% as.vector))
                
                lr_model <- glm(response~., family = binomial, data = d)
                predictions_prob <- predict(lr_model, type="response")
                predictions_class <- as.factor(ifelse(predictions_prob>0.5, 1, 0))
                
                print(proc.time()-start_time)
                cat(paste0("\n",c,"\n"), file = output_path, append = TRUE)
                capture.output(summary(lr_model), file = output_path, append = TRUE)
                capture.output(confusionMatrix(predictions_class, d$response), file = output_path, append = TRUE)
                
                ROCit_obj <- rocit(score = predictions_prob, class = d$response)
                png(filename=paste0("../output/plots/roc_logreg_",c,".png"))
                plot(ROCit_obj)
                dev.off()
                rm(lr_model, predictions_prob, predictions_class)
        }
}

reddit_path <- "../output/reddit_data.csv"
input <- load_reddit(reddit_path)
reddit_data <- input[[1]]
features <- input[[2]]
comment_sentiments <- input[[3]]
rm(input)

# Linear Mixed Models
# Let's subset the data for one comment sentiment.
c <- 'trust_comment'
d <- cbind(reddit_data[c(features, "parent_id")], response = reddit_data[[c]])

# Data type conversion.
d$gender_comment <- as.factor(d$gender_comment)
d$gender_post <- as.factor(d$gender_post)

# histogram of wordcount
# p <- ggplot(d, aes(x=comment_wordcount)) +
#         ggtitle("histogram of comment_wordcount") +
#         geom_histogram(fill="white", position="dodge", color="orange", bins = 50) +
#         geom_vline(aes(xintercept=mean(comment_wordcount, na.rm = T)), linetype="dashed", size=0.5)
# p
# d <- d[d$comment_wordcount>20,]
# d <- d[-outlier_ids,]


# Now, let's plot histograms of all variables.
# Will do log transforms where distribution is skewed.
# is_factor <- sapply(features, FUN = function(c) {return(is.factor(d[[c]]))})
# hist_features <- features[!is_factor]
# for(h in hist_features){
#         print(paste0("../output/hist_plots/", h, ".png"))
#         png(filename = paste0("../output/hist_plots/", h, ".png"))
#         print(ggplot(d, aes_(x=as.name(h))) + geom_histogram(color="darkblue", fill="lightblue", bins = 50))
#         dev.off()
# }

# Let's take log transforms of skewed features.
skewed_features <- c("authority_post", "care_post", "comment_wordcount",
                     "fairness_post", "loyalty_post", "non_moral_post",
                     "num_comments", "positive_post", "post_wordcount",
                     "purity_post", "sentistrength_neg_post", "sentistrength_neutral_post",
                     "sentistrength_pos_post", "title_wordcount", "ups")
for(s in skewed_features){
        d[[s]] <- log(1+d[[s]] - min(d[[s]]))
}

# d$gender_wordcnt_interaction <- as.numeric(d$gender_comment)*d$comment_wordcount
# d$comment_sq <- as.numeric(d$gender_comment)*d$comment_wordcount**2

# Making sure our features are scaled!
scale_features <- features[!features %in% c("gender_post","gender_comment", "subreddit",
                                            "author_premium", "is_self","no_follow")]
d <- d %>% mutate_at(scale_features, ~(scale(.) %>% as.vector))

# Finally, building an LMER model.
mixed.lmer <- lmer(response ~ . + gender_comment:comment_wordcount +
                           trust_post:gender_post +
                           trust_post:gender_comment +
                           fairness_post:gender_post +
                           fairness_post:gender_comment +
                           gender_post:gender_comment -
                           parent_id +
                           (1|parent_id), data = d)

# Diagnostics
summary(mixed.lmer)

# Plot residuals vs fitted values.
plot(mixed.lmer)

# Plot trust_comment vs comment_wordcount
ggplot(d, aes(x=comment_wordcount, y=response, color=gender_comment)) + geom_point(aes(alpha=0.9)) + geom_smooth(aes(group=gender_comment), method = "nls", formula = y ~ a * x + b, se = F, method.args = list(start = list(a = 0.1, b = 0.1))) + ylab(c) + scale_color_manual(labels = c("male", "female"), values = c("lightblue", "pink"))
ggsave("../plots/sadness_plot1.png")

ggplot(d, aes(x=sadness_post, y=response, color=gender_post)) + geom_point(aes(alpha=0.9)) + geom_smooth(aes(group=gender_post), method = "nls", formula = y ~ a * x + b, se = F, method.args = list(start = list(a = 0.1, b = 0.1))) + ylab(c) + scale_color_manual(labels = c("male", "female"), values = c("lightblue", "pink"))
ggsave("../plots/sadness_plot2.png")

ggplot(d, aes(x=sadness_post, y=response, color=gender_comment)) + geom_point(aes(alpha=0.9)) + geom_smooth(aes(group=gender_comment), method = "nls", formula = y ~ a * x + b, se = F, method.args = list(start = list(a = 0.1, b = 0.1))) + ylab(c) + scale_color_manual(labels = c("male", "female"), values = c("lightblue", "pink"))
ggsave("../plots/sadness_plot3.png")

# Remove outliers
outlier_ids <- which(resid(mixed.lmer)>=0.8)
# cd <- cooks.distance(mixed.lmer)
# outlier_ids <- which(cd > 10*mean(cd))

# QQ plot
qqnorm(resid(mixed.lmer))
qqline(resid(mixed.lmer))

reddit_full <- read.csv('../output/reddit_data_full.csv')
data_of_interest <- reddit_full[resid(mixed.lmer)>0.8,]
View(data_of_interest[,c("gender_comment", "full_text_comment")])
data_of_interest <- reddit_full[reddit_full$comment_wordcount>30, c("gender_comment", "full_text_comment")]
removed_residuals <- resid(mixed.lmer)[resid(mixed.lmer)>0.8]
removed_fitted <- fitted.values(mixed.lmer)[resid(mixed.lmer)>0.8]
plot(removed_fitted, removed_residuals)


# Why does removing short comments change the trend?
# Let's plot trust_comment against gender_comment
ggplot(d[d$comment_wordcount<20,], aes(x=gender_comment, y=response, fill=gender_comment)) +
        geom_boxplot()
ggplot(d[d$comment_wordcount<20,], aes(response, fill = gender_comment)) + geom_density(alpha = 0.2)
