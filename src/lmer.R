setwd("~/Desktop/reddit/src/")
library(randomForest)
library(ROCit)
library(caret)
library(dplyr)
library(lme4)
library(ggplot2)

# Load the merged reddit post and comment data.
load_reddit <- function(reddit_path="../data/reddit_data.csv"){
        reddit_data <- read.csv(reddit_path)
        cols <- colnames(reddit_data)
        features <- cols[grepl("_post", cols) | grepl("gender_", cols)]
        features <- c(features, 'subreddit', 'post_wordcount', 'title_wordcount', 'comment_wordcount', 'ups', 'author_premium',
                      'is_self', 'no_follow', 'num_comments')
        comment_sentiments <- cols[grepl("_comment", cols) & !grepl("gender", cols) & !(cols == 'num_comments')]
        return(list(reddit_data, features, comment_sentiments))
}

plot_histograms <- function(d, features){
        is_numeric <- sapply(features, FUN = function(c) {return(is.numeric(d[[c]]))})
        hist_features <- features[is_numeric]
        for(h in hist_features){
                print(paste0("../plots/hist_plots/", h, ".png"))
                png(filename = paste0("../plots/hist_plots/", h, ".png"))
                print(ggplot(d, aes_(x=as.name(h))) + geom_histogram(color="darkblue", fill="lightblue", bins = 50))
                dev.off()
        }
}

transform_features <- function(d, features){
        is_numeric <- sapply(features, FUN = function(c) {return(is.numeric(d[[c]]))})
        skewed_features <- features[is_numeric]
        for(s in skewed_features){
                d[[s]] <- log(1+d[[s]] - min(d[[s]]))
        }
        return(d)
}

scale_features <- function(d, features){
        is_numeric <- sapply(features, FUN = function(c) {return(is.numeric(d[[c]]))})
        scale_f <- features[is_numeric]
        d <- d %>% mutate_at(scale_f, ~(scale(.) %>% as.vector))
        return(d)
}

plot_interactions <- function(d, sentiment){
        # Plot trust_comment vs comment_wordcount
        ggplot(d, aes(x=comment_wordcount, y=response, color=predicted_gender_comment)) + geom_point(aes(alpha=0.9)) + geom_smooth(aes(group=predicted_gender_comment), method = "nls", formula = y ~ a * x + b, se = F, method.args = list(start = list(a = 0.1, b = 0.1))) + ylab(c) + scale_color_manual(labels = c("male", "female"), values = c("lightblue", "pink"))
        ggsave(paste0("../plots/", sentiment, "_plot1.png"))
        
        ggplot(d, aes_string(x=paste0(sentiment, "_post"), y="response", color="predicted_gender_post")) + geom_point(aes(alpha=0.9)) + geom_smooth(aes(group=predicted_gender_post), method = "nls", formula = y ~ a * x + b, se = F, method.args = list(start = list(a = 0.1, b = 0.1))) + ylab(c) + scale_color_manual(labels = c("male", "female"), values = c("lightblue", "pink"))
        ggsave(paste0("../plots/", sentiment, "_plot2.png"))
        
        ggplot(d, aes_string(x=paste0(sentiment, "_post"), y="response", color="predicted_gender_comment")) + geom_point(aes(alpha=0.9)) + geom_smooth(aes(group=predicted_gender_comment), method = "nls", formula = y ~ a * x + b, se = F, method.args = list(start = list(a = 0.1, b = 0.1))) + ylab(c) + scale_color_manual(labels = c("male", "female"), values = c("lightblue", "pink"))
        ggsave(paste0("../plots/", sentiment, "_plot3.png"))
}

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