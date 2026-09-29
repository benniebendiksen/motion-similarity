getwd()

library(FSA)
library(DescTools)
library(rcompanion)
library(multcompView)
library(tidyverse)
library(plotrix)
library(gridExtra)
library(ggplot2)
library(reshape2)
library(purrr)
library(data.table)
library(GPArotation)
library(psych)
library(car)
library("Hmisc")
library("xtable")
library("writexl")
library("readxl")
library("rstatix")
library('ggpubr')
library(stringr)
library(broom)
library(cowplot)
library(dplyr)

combineRowPermutations <- function(df) {
  
  getEffortsTuplesCountsDF <- function(df) {
    #generate counts df by efforts_tuples
    df <- df %>% 
      count(efforts_tuples, selected0, selected1, name= "count")
    #create unique string identifier
    df$id <- row.names(df)
    print("df one")
    print(df)
    return (df)
  }
  
  delete_duplicate_subsequent_row <- function(df) {
    df = df[order(df[,'duplicate_generator'],-df[,'count']),]
    df = df[!duplicated(df$duplicate_generator),]
    return (df)
  }
  
  df <- getEffortsTuplesCountsDF(df)
  
  for (i_row in 1:nrow(df)) {
    for (j_row in i_row:nrow(df)) {
      if (df[i_row, "efforts_tuples"] == df[j_row, "efforts_tuples"]) {
        if (df[i_row, "selected0"] == df[j_row, "selected1"] & df[i_row, "selected1"] == df[j_row, "selected0"]) {
          #of matched rows, give min valued select0 column precedence 
          if (df[i_row, "selected0"] < df[j_row, "selected0"]) {
            df[i_row, "count"] <- df[i_row, "count"] + df[j_row, "count"]
          } else {
            df[j_row, "count"] <- df[i_row, "count"] + df[j_row, "count"]
          }
          df[i_row, "duplicate_generator"] <- paste(as.character(i_row), as.character(j_row))
          df[j_row, "duplicate_generator"] <- paste(as.character(i_row), as.character(j_row))
        } 
        
        else {
          df[i_row, "duplicate_generator"] <- NA
          df[j_row, "duplicate_generator"] <- NA
          # of unmatched row, give min valued select0 column precedence
          if (df[i_row, "selected0"] > df[i_row, "selected1"]) {
            df[i_row, c("selected0", "selected1")] <- df[i_row, c("selected1", "selected0")]
          }
        }
        
      }
    }
  }
  print("df two")
  print(df)
  # replace Nas in duplicate_generator with the corresponding id value
  df <- df %>% 
    mutate(duplicate_generator = coalesce(duplicate_generator,id))
  # delete duplicate subsequent permutation's row and eliminate extraneous columns, while sorting & resetting row idx numbers
  print("df three")
  print(df)
  df <- delete_duplicate_subsequent_row(df)
  df$index <- as.numeric(row.names(df))
  df <- df[order(df$index), ]
  rownames(df) <- NULL  
  print("df four")
  print(df)
  df <- df[, !names(df) %in% c("id", "duplicate_generator", "index")]
  return (df)
}

# De-identified release: input is the pooled de-identified responses shipped alongside this script.
# (Original ran on the raw, identifier-bearing validAnswers.csv; identifiers were removed for release
#  and are provably inert to the aggregate — see README round-trip verification.)
INPUT <- Sys.getenv("LMA_INPUT", unset = "../validAnswersWalking_deidentified.csv")
lma_pilot <- read.csv(file = INPUT, stringsAsFactors = FALSE, header = TRUE)

lma_pilot <- as.data.frame(lma_pilot)
str(lma_pilot)
effort_classes <- unique(c(lma_pilot$effortsLeft, lma_pilot$effortsRight))
# 49
length(effort_classes)
# mydf <- lma_pilot %>%
#   rowwise() %>% 
#   mutate(combined = paste(sort(c_across(starts_with("efforts"))),collapse = "_"))
# 
# length(unique(mydf$combined))
# View(mydf)

# Create a new column combining effortsLeft and effortsRight by a policy (to eliminate potential permutations)
lma_pilot$efforts_tuples <- paste(pmin(lma_pilot$effortsLeft, lma_pilot$effortsRight), pmax(lma_pilot$effortsLeft, lma_pilot$effortsRight), sep = "_")

### Correct selected index if effortRight was moved to the left (and effortLeft to the right)
#make mask for relevant rows
bools_efforts_right_first <- pmin(lma_pilot$effortsLeft, lma_pilot$effortsRight) == lma_pilot$effortsRight
print(bools_efforts_right_first)
#grab indices for where mask is true and selected values were either 2 or 0
selected0_indices_with_2 <- bools_efforts_right_first & lma_pilot$selected0 == 2
selected1_indices_with_2 <- bools_efforts_right_first & lma_pilot$selected1 == 2
selected0_indices_with_0 <- bools_efforts_right_first & lma_pilot$selected0 == 0
selected1_indices_with_0 <- bools_efforts_right_first & lma_pilot$selected1 == 0
# change such values from 2 to 0 and 0 to 2
lma_pilot$selected0[selected0_indices_with_2] <- 0
lma_pilot$selected1[selected1_indices_with_2] <- 0
lma_pilot$selected0[selected0_indices_with_0] <- 2
lma_pilot$selected1[selected1_indices_with_0] <- 2

# NOTE (de-identified release): the original swapped selected0/selected1 alongside identifier
# columns (hitId); those identifiers were dropped for release and are inert to the aggregate, so
# this reorder now operates on the non-identifying columns only.
lma_pilot[bools_efforts_right_first, c("qInd", "effortsLeft", "effortsRight", "selected0", "selected1", "efforts_tuples")] <- lma_pilot[bools_efforts_right_first, c("qInd", "effortsLeft", "effortsRight", "selected1", "selected0", "efforts_tuples")]


# NOTE (de-identified release): original subset began with the raw "id" column (regenerated
# internally as row.names in combineRowPermutations); dropped here. motionType absent in some
# per-action files, so select only guaranteed columns.
lma_pilot <- lma_pilot[, intersect(c("efforts_tuples", "effortsLeft", "effortsRight", "selected0", "selected1", "qInd"), names(lma_pilot))]
# NOTE (de-identified release): original renamed motionType->motion_type here; motionType is dropped
# above and unused by the aggregate transform, so the rename is removed.

# Generate counts of distinct effort_tuples values
counts <- sort(table(lma_pilot$efforts_tuples), decreasing = TRUE)
# Server write out errors resulted in < C(56, 2) = 1540 comparisons
# Only 49 comparisons captured C(49, 2) = 1176
# Plus self-to-self "comparisons" were written out: num comparisons = 1,120
length(counts)
# hist, exclude very first entry of 973 counts
hist(counts[-1], main = "Effort Pair Frequency Histogram", xlab = "Count of Comparison Sample Size", ylab = "Frequency",
     breaks = seq(0, max(counts[-1]) + 3, by = 3))


# get counts by efforts_tuplesXselection as well as by motion
lma_pilot_test <- lma_pilot[, !names(lma_pilot) %in% c("motion_type")]
lma_pilot_counts <- combineRowPermutations(lma_pilot_test)
grouped_lma_pilot_counts <- lma_pilot_counts %>% group_by(efforts_tuples)
# Identify groups lacking specific select column values, populate such, and assign 0 to the corresponding count column
grouped_lma_pilot_counts <- grouped_lma_pilot_counts %>%
  complete(selected0=c(0,1), selected1 = c(1,2)) %>%
  mutate(count = replace_na(count, 0))
# Filter out the (1, 1) combination
grouped_lma_pilot_counts <- grouped_lma_pilot_counts %>%
  filter(!(selected0 == 1 & selected1 == 1))
# Normalize counts within each group to sum to 1
grouped_lma_pilot_counts <- grouped_lma_pilot_counts %>%
  mutate(count_normalized = count / sum(count))
str(grouped_lma_pilot_counts)

#combine selected0 and selected1 into one column for all three dfs (orig and motion 0 and motion1) and eliminate permutations
grouped_lma_pilot_counts$selected_motions <- paste(grouped_lma_pilot_counts$selected0, grouped_lma_pilot_counts$selected1, sep="_")
# write out df as csv
write.csv(grouped_lma_pilot_counts, "similarity_comparisons_ratios_11_17_23.csv", row.names = FALSE)


