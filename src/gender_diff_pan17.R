setwd("~/Desktop/entrepreneurship_research/sentiment_analysis/src/")

library(readxl)
library(stringr)
library(ggplot2)

# read PAN 17 data
pan <- read_excel("../../gender_prediction/data/pan17/pan17_df.xlsx", sheet = "Sheet1")

# get wordcount
pan[["word_count"]] <- sapply(pan[["text"]], str_count, '\\w+')

# filter english texts
pan <- pan[pan[,"lang"] == "en",]

# histogram of wordcount
p <- ggplot(pan, aes(x=word_count)) +
        geom_histogram(fill="white", position="dodge", color="orange", bins = 30) +
        geom_vline(aes(xintercept=mean(word_count, na.rm = T)), linetype="dashed", size=0.5)
p

pan[["gender"]] <- as.factor(ifelse(pan[["gender"]] == "male", 0, 1))

tags_list <- c("#entrepreneur", "#business", "#entrepreneurship", "#entrepreneurlife", "#smallbusiness", "#businessowner", "#startup", "#businesswoman", "#entrepreneurs", "#womeninbusiness", "#business", "#entrepreneurlifestyle", "#entrepreneurmindset", "#entrepreneurquotes", "#entrepreneurial", "#entrepreneurslife", "#entrepreneurgoals", "#entrepreneurspirit", "#entrepreneurtips", "#entrepreneurmind", "#entrepreneurship101", "#entrepreneurstyle", "#entrepreneurtip", "#entrepreneurwoman")
is_enterpreneur <- rep(F, nrow(pan))
for(tag in tags_list){
        is_enterpreneur <- is_enterpreneur | grepl(tag, pan$text)
}
pan <- pan[is_enterpreneur,]

lm_fit <- lm(fairness ~ gender, data = pan)
summary(lm_fit)

lm_fit <- lm(positive ~ gender, data = pan)
summary(lm_fit)

lm_fit <- lm(negative ~ gender, data = pan)
summary(lm_fit)

lm_fit <- lm(trust ~ gender, data = pan)
summary(lm_fit)

lm_fit <- lm(sadness ~ gender, data = pan)
summary(lm_fit)

table(pan[["gender"]])

