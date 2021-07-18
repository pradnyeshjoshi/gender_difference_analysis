df <- read.csv("~/Desktop/gender_difference_analysis/data/posts.csv")

sent <- 'sentistrength_neutral'
m <- median(df[[sent]])
View(df[df[[sent]] == m & df$post_wordcount>10,])

View(df[df[[sent]]== m,])

