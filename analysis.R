## R CODE FOR GRAPHICS FOR
## FINAL DELIVERABLE FOR CS143 PROJECT 2B

# If you already have R installed on your home machine, transfer the resulting files
# to your shared directory and do the visualizations in R on your home system rather than
# in the VM because R is not installed there.

# R is actually the simplest software for making plots.

##################################################
# PLOT 1: SENTIMENT OVER TIME (TIME SERIES PLOT)
###################################################

# This assumes you have a CSV file called "time_data.csv" with the columns:
# date (like 2018-08-01), Positive, Negative
# You should use the FULL PATH to the file, just in case.

time.data <- read.csv("part2.csv", stringsAsFactors=FALSE, header=FALSE)
time.data$V1 <- as.Date(time.data$V1)
# Fix a small data integrity issue from the data developer.
time.data <- time.data[time.data$V1 != "2018-12-31", ]

# Get start and end of time series.
start <- min(time.data$V1)
end <- max(time.data$V1)

# Turn the data into a time series.
positive.ts <- ts(time.data$V2, start=start, end=end)
negative.ts <- ts(time.data$V3, start=start, end=end)

jpeg('plot1.jpeg')
# Plot it
ts.plot(positive.ts, negative.ts, col=c("darkgreen", "red"),
gpars=list(xlab="Day", ylab="Sentiment", main="President Trump Sentiment on /r/politics Over Time"))
dev.off()

################################################################
# PLOT 2: SENTIMENT BY STATE (POSITIVE AND NEGATIVE SEPARATELY)
################################################################

# May need to do 
# install.packages("ggplot2")
# install.packages("dplyr")

# This assumes you have a CSV file called "state_data.csv" with the columns:
# state, Positive, Negative
# You should use the FULL PATH to the file, just in case.

library(ggplot2)
library(dplyr)

state.data <- read.csv("part3.csv", header=FALSE)
# rename it due to the format of the state data
state.data$region <- state.data$V1
chloro <- state.data %>% mutate(region=tolower(region)) %>%
  right_join(map_data("state"))

jpeg('plot2_pos.jpeg')
ggplot(chloro, aes(long, lat)) +
  geom_polygon(aes(group=group, fill=V2)) + 
  coord_quickmap() +
  scale_fill_gradient(low="#FFFFFF",high="#006400") +
  ggtitle("Positive Trump Sentiment Across the US")
dev.off()

jpeg('plot2_pos.jpeg')
ggplot(chloro, aes(long, lat)) +
  geom_polygon(aes(group=group, fill=V3)) + 
  coord_quickmap() +
  scale_fill_gradient(low="#FFFFFF",high="#FF0000") +
  ggtitle("Negative Trump Sentiment Across the US")
dev.off()

################################################################
# PLOT 3: SENTIMENT DIFF BY STATE
################################################################
jpeg('plot3.jpeg')
chloro$Difference <- chloro$V2 - chloro$V3
ggplot(chloro, aes(long, lat)) +
  geom_polygon(aes(group=group, fill=Difference)) + 
  coord_quickmap() +
  scale_fill_gradient(low="#FFFFFF",high="#000000") +
  ggtitle("Difference in Sentiment Across the US")
dev.off()

########################################
# PLOT 5A: SENTIMENT BY STORY SCORE
########################################
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called submission_score.csv with the following coluns
# submission_score, Positive, Negative

jpeg('plot5a.jpeg')
submission.data <- read.csv("by_s_score.csv", quote="", header=FALSE)
plot(V2~V1, data=story.data, col='darkgreen', pch='.',
     main="Sentiment By Score on Submission")
points(V3~V1, data=story.data, col='red', pch='.')
dev.off()

########################################
# PLOT 5B: SENTIMENT BY COMMENT SCORE
########################################
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called comment_score.csv with the following columns
# comment_score, Positive, Negative

jpeg('plot5b.jpeg')
comment.data <- read.csv("by_c_score.csv", quote="", header=FALSE)
plot(V2~V1, data=comment.data, col='darkgreen', pch='.',
     main="Sentiment By Score on Comments")
points(V3~V1, data=comment.data, col='red', pch='.')
dev.off()

########################################
# PLOT 6: SENTIMENT BY MONTH
########################################

# This assumes you have a CSV file called "time_data.csv" with the columns:
# date (like 2018-08-01), Positive, Negative
# You should use the FULL PATH to the file, just in case.

month <- read.csv("part5.csv", stringsAsFactors=FALSE, header=FALSE)

#jpeg('plot1.jpeg')
# Plot it
matplot(month[1], month[3], col=c("blue"), type='p', pch=0, main="Sentiment By Month", xlab="Month", ylab="Sentiment")
dev.off()
