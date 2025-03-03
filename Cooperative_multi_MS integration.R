#cooperative learning 

#part1-exploration analysis
library(ggplot2)
library(reshape2)
library(corrplot)

HDF <- read.csv("/Users/xiruihe/Downloads/dataset/HDF.csv", row.names=1)
BM <- read.csv("/Users/xiruihe/Downloads/dataset/BM.csv",row.names=1)
LD <- read.csv("/Users/xiruihe/Downloads/dataset/LD.csv",row.names=1)
PM <- read.csv("/Users/xiruihe/Downloads/dataset/PM.csv",row.names=1)
Y <- read.csv("/Users/xiruihe/Downloads/dataset/Y.csv",row.names=1)
y<- Y[,1]
#PCA
perform_pca <- function(data, scale = TRUE) {
  pca <- prcomp(data, scale. = scale)
  var_exp <- summary(pca)$importance[2, 1:5]
  return(list(pca = pca, var_exp = var_exp))
}

select_top_pcs <- function(var_exp, n = 2) {
  top_pcs <- order(var_exp, decreasing = TRUE)[1:n]
  return(top_pcs)
}

extract_top_pcs <- function(pca, top_pcs) {
  pc1 <- pca$x[, top_pcs[1]]
  pc2 <- pca$x[, top_pcs[2]]
  return(list(pc1 = pc1, pc2 = pc2))
}

pca_results <- list(
  HDF = perform_pca(HDF),
  BM = perform_pca(BM),
  LD = perform_pca(LD),
  PM = perform_pca(PM)
)

for (dataset in names(pca_results)) {
  cat(dataset, "Principal Component Variance Explained:\n")
  print(pca_results[[dataset]]$var_exp)
}

top_pcs <- lapply(pca_results, function(x) select_top_pcs(x$var_exp))

for (dataset in names(top_pcs)) {
  cat("Top PCs for", dataset, ":", top_pcs[[dataset]], "\n")
}

top_pcs_data <- lapply(names(pca_results), function(dataset) {
  extract_top_pcs(pca_results[[dataset]]$pca, top_pcs[[dataset]])
})
names(top_pcs_data) <- names(pca_results)

# Combine the data
df_pca <- data.frame(
  PC1 = unlist(lapply(top_pcs_data, function(x) x$pc1)),
  PC2 = unlist(lapply(top_pcs_data, function(x) x$pc2)),
  Dataset = rep(names(top_pcs_data), times = sapply(top_pcs_data, function(x) length(x$pc1)))
)
#correlation matrix
df_pca <- data.frame(
  PC1_HDF = pc1_HDF,
  PC2_HDF = pc2_HDF,
  PC1_BM = pc1_BM,
  PC2_BM = pc2_BM,
  PC1_LD = pc1_LD,
  PC2_LD = pc2_LD,
  PC1_PM = pc1_PM,
  PC2_PM = pc2_PM
)
correlation_matrix <- cor(df_pca)
corrplot(correlation_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")


df_pca_HDF <- data.frame(PC1 = pca_HDF$x[, 1], PC2 = pca_HDF$x[, 2], Group = factor(y, labels = c("Non-DIAB", "DIAB")))
df_pca_BM <- data.frame(PC1 = pca_BM$x[, 1], PC2 = pca_BM$x[, 2], Group = factor(y, labels = c("Non-DIAB", "DIAB")))
df_pca_LD <- data.frame(PC1 = pca_LD$x[, 1], PC2 = pca_LD$x[, 2], Group = factor(y, labels = c("Non-DIAB", "DIAB")))
df_pca_PM <- data.frame(PC1 = pca_PM$x[, 1], PC2 = pca_PM$x[, 2], Group = factor(y, labels = c("Non-DIAB", "DIAB")))

# HDF
ggplot(df_pca_HDF, aes(x = PC1, y = PC2, color = Group)) +
  geom_point(alpha = 0.7, size = 3) +
  stat_ellipse(aes(fill = Group), geom = "polygon", alpha = 0.3) +  # 添加置信椭圆
  scale_color_manual(values = c("blue", "red")) +  # 设置颜色
  scale_fill_manual(values = c("blue", "red")) +  # 设置椭圆填充颜色
  labs(title = "HDF Dataset: PC1 vs PC2 by Diabetes Status",
       x = "PC1", y = "PC2", color = "Group") +
  theme_minimal() +
  theme(legend.position = "top")

# BM
ggplot(df_pca_BM, aes(x = PC1, y = PC2, color = Group)) +
  geom_point(alpha = 0.7, size = 3) +
  stat_ellipse(aes(fill = Group), geom = "polygon", alpha = 0.3) +  # 添加置信椭圆
  scale_color_manual(values = c("blue", "red")) +  # 设置颜色
  scale_fill_manual(values = c("blue", "red")) +  # 设置椭圆填充颜色
  labs(title = "BM Dataset: PC1 vs PC2 by Diabetes Status",
       x = "PC1", y = "PC2", color = "Group") +
  theme_minimal() +
  theme(legend.position = "top")


# LD
ggplot(df_pca_LD, aes(x = PC1, y = PC2, color = Group)) +
  geom_point(alpha = 0.7, size = 3) +
  stat_ellipse(aes(fill = Group), geom = "polygon", alpha = 0.3) +  # 添加置信椭圆
  scale_color_manual(values = c("blue", "red")) +  # 设置颜色
  scale_fill_manual(values = c("blue", "red")) +  # 设置椭圆填充颜色
  labs(title = "LD Dataset: PC1 vs PC2 by Diabetes Status",
       x = "PC1", y = "PC2", color = "Group") +
  theme_minimal() +
  theme(legend.position = "top")

# PM
ggplot(df_pca_PM, aes(x = PC1, y = PC2, color = Group)) +
  geom_point(alpha = 0.7, size = 3) +
  stat_ellipse(aes(fill = Group), geom = "polygon", alpha = 0.3) +  # 添加置信椭圆
  scale_color_manual(values = c("blue", "red")) +  # 设置颜色
  scale_fill_manual(values = c("blue", "red")) +  # 设置椭圆填充颜色
  labs(title = "PM Dataset: PC1 vs PC2 by Diabetes Status",
       x = "PC1", y = "PC2", color = "Group") +
  theme_minimal() +
  theme(legend.position = "top")

# Print the combined data frame
print(df_pca)

# plot
library(ggplot2)

ggplot(df_pca, aes(x = PC1, y = PC2, color = Dataset)) +
  geom_point(alpha = 0.6, size = 3) +  
  scale_color_manual(values = c("HDF" = "blue", "BM" = "red", "LD" = "green", "PM" = "purple")) +  # 设置颜色
  labs(title = "PCA - First Two Principal Components of four datasets", x = "PC1", y = "PC2") +
  theme_minimal() +
  theme(legend.title = element_blank())  

#clustering 
#elbow method to identify the number of k
wss <- sapply(1:10, function(k) {
  kmeans(df_pca[, 1:2], centers = k, nstart = 25)$tot.withinss
})

ggplot(data.frame(k = 1:10, WSS = wss), aes(x = k, y = WSS)) +
  geom_line() + geom_point() +
  labs(title = "Elbow Method for Optimal K", x = "Number of Clusters (K)", y = "Total Within-Cluster Sum of Squares") +
  theme_minimal()

k_optimal <- 5

# K-means clustering
set.seed(123)  
kmeans_result <- kmeans(df_pca[, 1:2], centers = k_optimal, nstart = 25)
df_pca$Cluster <- as.factor(kmeans_result$cluster)

# plot
ggplot(df_pca, aes(x = PC1, y = PC2, color = Cluster, shape = Dataset)) +
  geom_point(alpha = 0.6, size = 3) +
  scale_color_manual(values = c("red", "blue", "green", "purple", "orange", "yellow", "pink")) +
  scale_shape_manual(values = c(16, 17, 18, 19)) +
  labs(title = "PCA with K-means Clustering Results", x = "PC1", y = "PC2") +
  theme_minimal() +
  theme(legend.title = element_blank())

#part2-exploration of multiview package
install.packages("multiview")
library(multiview)
library(pROC)
help(package = "multiview")

#check the data
hist(HDF[,1])
col_means<-colMeans(HDF)

#split the data
train_frac <- 0.8
smp_size_train <- floor(train_frac * nrow(HDF)) 
train_ind <- sort(sample(seq_len(nrow(HDF)), size = smp_size_train))
test_ind <- setdiff(seq_len(nrow(HDF)), train_ind)

train_HDF <- HDF[train_ind, ]
test_HDF <- HDF[test_ind, ]
train_BM <- BM[train_ind, ]
test_BM <- BM[test_ind, ]
train_LD <- LD[train_ind, ]
test_LD <- LD[test_ind, ]
train_PM <- PM[train_ind, ]
test_PM <- PM[test_ind, ]

train_y <- y[train_ind]
test_y <- y[test_ind]

#first exploration: cross-validation to choose lambda 
cvfit <- cv.multiview(
  list(train_HDF,train_BM,train_LD,train_PM), 
  train_y,               
  family = binomial(),           
  type.measure = "deviance",     
  rho = 0.1,
  nfolds=5,
  trace.it=TRUE
)

print(cvfit$lambda.min)
plot(cvfit)

#Performance 
predictions <- predict(cvfit, 
                       newx = list(test_HDF, test_BM, test_LD, test_PM), 
                       type = "response",        
                       s = "lambda.min")         

predicted_class <- ifelse(predictions > 0.5, 1, 0)  
accuracy <- mean(predicted_class == test_y)
cat("Accuracy: ", accuracy, "\n")
roc_curve <- roc(test_y, predictions)  # the propability of predictions 
cat("AUC: ", roc_curve$auc, "\n")

# part3--MCCV:see the stability of CL model
n_repeats <- 100
mccv_auc_scores <- c()

# loop for MCCV
for (i in 1:n_repeats) {
  
  set.seed(123 + i) 
  
  # random split spices into training set and testing set 
  test_indices <- sample(1:nrow(HDF), size = floor(0.2 * nrow(HDF)))  # 20% testing
  train_indices <- setdiff(1:nrow(HDF), test_indices)  # rest were training
  
  train_data <- list(HDF[train_indices,], BM[train_indices,], LD[train_indices,], PM[train_indices,])
  test_data <- list(HDF[test_indices,], BM[test_indices,], LD[test_indices,], PM[test_indices,])
  train_labels <- y[train_indices]
  test_labels <- y[test_indices]
  
  # fit 
  fit <- multiview(
    train_data,
    train_labels,
    family = binomial(),
    rho = best_rho,  
    lambda = best_lambda,  
  )
  
  predictions <- predict(fit, newx = test_data, type = "response")
  roc_curve <- roc(test_labels, predictions)
  auc <- roc_curve$auc
  mccv_auc_scores <- c(mccv_auc_scores, auc)
}
# the average of AUC
cat("MCCV AUC Scores: ", mccv_auc_scores, "\n")
cat("Average AUC across all repeats: ", mean(mccv_auc_scores), "\n")

##early integration
n_repeats <- 100
mccv_auc_early <- c()
for (i in 1:n_repeats) {
  set.seed(123 + i) 
  test_indices <- sample(1:nrow(HDF), size = floor(0.2 * nrow(HDF)))  # 20% testing
  train_indices <- setdiff(1:nrow(HDF), test_indices)  # rest were training
  
  train_data <- list(HDF[train_indices,], BM[train_indices,], LD[train_indices,], PM[train_indices,])
  test_data <- list(HDF[test_indices,], BM[test_indices,], LD[test_indices,], PM[test_indices,])
  train_labels <- y[train_indices]
  test_labels <- y[test_indices]
  fit <- multiview(
    train_data,
    train_labels,
    family = binomial(),
    rho = 0,  
    lambda = 0.0337248089,  
  )
  predictions <- predict(fit, newx = test_data, type = "response")
  roc_curve <- roc(test_labels, predictions)
  auc1 <- roc_curve$auc
  mccv_auc_early <- c(mccv_auc_early, auc1)
}

cat("MCCV_early_AUC Scores: ", mccv_auc_early, "\n")

##late integration
n_repeats <- 100
mccv_auc_late <- c()
for (i in 1:n_repeats) {
  set.seed(123 + i) 
  test_indices <- sample(1:nrow(HDF), size = floor(0.2 * nrow(HDF)))  # 20% testing
  train_indices <- setdiff(1:nrow(HDF), test_indices)  # rest were training
  
  train_data <- list(HDF[train_indices,], BM[train_indices,], LD[train_indices,], PM[train_indices,])
  test_data <- list(HDF[test_indices,], BM[test_indices,], LD[test_indices,], PM[test_indices,])
  train_labels <- y[train_indices]
  test_labels <- y[test_indices]
  fit <- multiview(
    train_data,
    train_labels,
    family = binomial(),
    rho = 1,  
    lambda = 0.0011891208,  
  )
  predictions <- predict(fit, newx = test_data, type = "response")
  roc_curve <- roc(test_labels, predictions)
  auc2 <- roc_curve$auc
  mccv_auc_late <- c(mccv_auc_late, auc2)
}

cat("MCCV_late_AUC Scores: ", mccv_auc_late, "\n")

#part4--model performance evaluation(error bar)

# multiview 
calculate_ci <- function(data, ci_level=0.90) {
  results <- data.frame(mean=numeric(), margin_of_error=numeric(), CI_lower=numeric(), CI_upper=numeric())
  for (col in colnames(data)) {
    col_data <- data[[col]]
    
    if (length(col_data) < 2) {
      warning(paste("Column", col, "has less than 2 valid values. Cannot calculate CI."))
      results[col,] <- c(NA, NA, NA, NA)
      next
    }
    
    mean_val <- mean(col_data, na.rm = TRUE)
    sem <- sd(col_data, na.rm = TRUE) / sqrt(length(col_data))
    ci <- qt(c((1 - ci_level) / 2, 1 - (1 - ci_level) / 2), df=length(col_data)-1) * sem + mean_val
    error <- (ci[2] - ci[1]) / 2
    
    results <- rbind(results, c(mean_val, error, ci[1], ci[2]))
  }
  
  rownames(results) <- colnames(data)
  return(results)
}
str(multiview_data)
multiview_data <- as.data.frame(multiview_data)

# 90%CI
auc_ci_multiview <- calculate_ci(multiview_data, ci_level=0.90)
print(auc_ci_multiview)

#plot
color <- c("skyblue", "white", "white") 
par(mar=c(8, 6, 6, 3))  
p1 <- boxplot(multiview_data, 
              col = color, 
              names = c("Cooperative", "Early integration", "Late integration"),  
              xlab = "", ylab = "AUC", xaxt = "n", 
              cex.axis = 1,  
              cex.lab = 1.5) 

tick <- seq_along(p1$names)
for (i in 1:length(tick)) {
  mean_val <- auc_ci_multiview[i, 1]  # meain
  ci_lower <- auc_ci_multiview[i, 3]  # ci_lower
  ci_upper <- auc_ci_multiview[i, 4]  # ci_upper
  arrows(x0 = i, y0 = ci_lower, y1 = ci_upper, 
         angle = 90, code = 3, length = 0.1, col = "red", lwd = 2)  # 误差条的颜色为红色
}

text(x = tick, y = par("usr")[3] - 0.2 * (par("usr")[4] - par("usr")[3]), 
     labels = p1$names, xpd = TRUE, cex = 0.8)  # 自定义横坐标标签字体大小
title(main = "AUC with 90% CI", line = 1.8, cex.lab = 2.75)

##MAMSI
file_path <- "/Users/xiruihe/Desktop/auc_mamsi.csv"
mamsi_auc <- read.csv(file_path)
auc_ci_mamsi <- calculate_ci(mamsi_auc, ci_level=0.90)

color <- c("white", "white", "skyblue","white") 

par(mar=c(8, 6, 6, 3))  
p1 <- boxplot(mamsi_auc, 
              col = color, 
              names = c("PM", "HDF_PM", "HDF_BM_PM","HDF_BM_LD_PM"),  
              xlab = "", ylab = "AUC", xaxt = "n", 
              cex.axis = 1,  
              cex.lab = 1.5) 

tick <- seq_along(p1$names)
for (i in 1:length(tick)) {
  mean_val <- auc_ci_mamsi[i, 1]  
  ci_lower <- auc_ci_mamsi[i, 3]
  ci_upper <- auc_ci_mamsi[i, 4]  
  
  arrows(x0 = i, y0 = ci_lower, y1 = ci_upper, 
         angle = 90, code = 3, length = 0.1, col = "red", lwd = 2)  # 误差条的颜色为红色
}
text(x = tick, y = par("usr")[3] - 0.2 * (par("usr")[4] - par("usr")[3]), 
     labels = p1$names,srt=45, xpd = TRUE, cex = 0.8)  # 自定义横坐标标签字体大小

title(main = "AUC with 90% CI", line = 1.8, cex.lab = 2.75)


#part5--see the contribution of each block(cooperative learning)
library(ggplot2)
library(gridExtra)

help("view.contribution", package = "multiview")
contribution_result <- view.contribution(x_list, y, rho = 0.9, family = binomial(), eval_data = "train",trace.it=TRUE)
print(contribution_result)

block_importance <- contribution_result
block_importance$Data_blocks <- factor(block_importance$Data_blocks, 
                                       levels = c("NULL", "HDF", "BM", "LD", "PM", "Cooperative(all)"))

p1 <- ggplot(block_importance, aes(x = Data_blocks, y = metric, fill = Data_blocks)) +
  geom_bar(stat = "identity") +
  labs(title = "LogLoss by Data Blocks", x = "Data Blocks", y = "LogLoss") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle =40 , hjust = 0.5))

# Percentage_improvement 
p2 <- ggplot(block_importance, aes(x = Data_blocks, y = Percentage_improvement, fill = Data_blocks)) +
  geom_bar(stat = "identity") +
  labs(title = "Percentage Improvement by Data Blocks", x = "Data Blocks", y = "Percentage Improvement (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 40, hjust = 0.5))

grid.arrange(p1, p2, ncol = 2)


#part6-find the best rho and lambda 
set.seed(333)
# rho
rho_values <- seq(0.1, 0.9, by = 0.1)

# initialize AUC，rho、lambda
best_auc <- -Inf
best_rho <- NULL
best_lambda <- NULL
aucs <- c()

# loop：rho
for (rho in rho_values) {
  # 5-fold cv : lambda
  cvfit <- cv.multiview(
    list(train_HDF, train_BM, train_LD, train_PM),
    train_y,
    family = binomial(),
    type.measure = "deviance",  
    rho = rho,
    nfolds = 5,
    trace.it = TRUE  
  )
  
  best_lambda_for_rho <- cvfit$lambda.min  # best lambda
  plot(cvfit)
  # predictions
  predictions <- predict(cvfit, 
                         newx = list(test_HDF, test_BM, test_LD, test_PM), 
                         type = "response", 
                         s = "lambda.min")
  # auc
  roc_curve <- roc(test_y, predictions)
  auc <- roc_curve$auc
  aucs <- c(aucs, auc)
  # update parameters
  if (auc > best_auc) {
    best_auc <- auc
    best_rho <- rho
    best_lambda <- best_lambda_for_rho
  }
}
cat("Best rho:", best_rho, "Best lambda:", best_lambda, "Best AUC:", best_auc, "\n")
cat("AUC from repeated experiments: ", aucs, "\n")
cat("Average AUC: ", mean(aucs), "\n")


#final model 
multiview.control(mxitnr = 1000)  # improve mxitnr
fit1 <- multiview(list(HDF,BM,LD,PM), y, family = binomial(), rho = best_rho, lambda = best_lambda)
plot(fit1)
print(fit1)
coef(fit1,s=0.1)
coef_ordered(fit1,s=0.1)
str(fit1)

#select features
coefficients <- coef(fit1)
print(coefficients)
str(coefficients)

#non_zeri_coeffs 
non_zero_coeffs <- coefficients@x 
non_zero_variables <- coefficients@Dimnames[[1]][coefficients@i+1 ]  # 获取非零系数对应的特征名（加1是因为索引从0开始）
important_features <- data.frame(
  Variable = non_zero_variables,
  Coefficient = non_zero_coeffs
)

# data frame
important_features <- data.frame(
  Variable = non_zero_variables,
  Coefficient = non_zero_coeffs
)
write.csv(important_features, file = "important_features_fit1.csv", row.names = FALSE)

#part7-visualize if the feature is stable
selected_columns <- c("rho_0.3","rho_0.6", "rho_0.9")
plot_list <- list()

for (col in selected_columns) {
  p <- ggplot(upset_data, aes(x = factor(1:nrow(upset_data)), y = upset_data[[col]])) +
    geom_bar(stat = "identity", fill = ifelse(upset_data[[col]] == 1, "blue", "red")) +
    labs(title = col, x = "Sample Index", y = "non_zero features") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    scale_x_discrete(breaks = seq(1, nrow(upset_data), by = 100))  # visualize a label by 100 features
  plot_list[[col]] <- p
}

grid.arrange(grobs = plot_list, ncol = 1)  # one plot a row

#part8-see how the number of features change with rho
# Set the range of rho values
rho_values <- seq(0, 1, by = 0.1)

# Initialize variables to store results
aucs <- c()  # Store AUCs
non_zero_coeffs_list <- list()  # To store non-zero coefficients for each rho
selected_features_list <- list()  # To store selected features for each rho
lambda_values <- c()  # To store lambda values corresponding to each rho

# Loop over different rho values
for (rho in rho_values) {
  # 5-fold CV to choose best lambda for the current rho
  cvfit <- cv.multiview(
    list(train_HDF, train_BM, train_LD, train_PM),
    train_y,
    family = binomial(),
    type.measure = "deviance",  
    rho = rho,
    nfolds = 5,
    trace.it = TRUE  
  )
  
  best_lambda_for_rho <- cvfit$lambda.min  # Optimal lambda for current rho
  lambda_values <- c(lambda_values, best_lambda_for_rho)  # Store lambda value

  predictions <- predict(cvfit, 
                         newx = list(test_HDF, test_BM, test_LD, test_PM), 
                         type = "response", 
                         s = "lambda.min")

  roc_curve <- roc(test_y, predictions)
  auc <- roc_curve$auc
  aucs <- c(aucs, auc)  # Store AUCs for each rho
  
  # Save coefficients for the current rho
  coefficients <- coef(cvfit, s = "lambda.min")
  
  non_zero_coeffs <- coefficients@x
  non_zero_variables <- coefficients@Dimnames[[1]][coefficients@i + 1]  # Get feature names for non-zero coefficients (indexing starts from 0)
  
  # Store non-zero coefficients and their variables
  important_features <- data.frame(
    Variable = non_zero_variables,
    Coefficient = non_zero_coeffs
  )
  selected_features_list[[paste0("rho_", rho)]] <- important_features
  
  num_selected_features <- nrow(important_features)
  cat("For rho =", rho, "Number of selected features:", num_selected_features, "\n")
}

cat("AUCs from different rho values: ", aucs, "\n")
cat("Average AUC: ", mean(aucs), "\n")

all_selected_features <- do.call(rbind, selected_features_list)
write.csv(all_selected_features, file = "all_selected_features.csv", row.names = FALSE)


# features of each rho
for (rho in rho_values) {
  cat("Features selected for rho =", rho, ":\n")
  features <- selected_features_list[[paste0("rho_", rho)]]
  print(features)
  cat("\n")
}


#see the mapping of features
install.packages("UpSetR")
library(UpSetR)

all_features <- unique(unlist(lapply(selected_features_list, function(x) x$Variable)))

upset_data <- data.frame(matrix(0, nrow = length(all_features), ncol = length(rho_values)))
colnames(upset_data) <- paste0("rho_", rho_values)
rownames(upset_data) <- all_features

for (rho in rho_values) {
  selected_features <- selected_features_list[[paste0("rho_", rho)]]$Variable
  upset_data[selected_features, paste0("rho_", rho)] <- 1
}

upset(upset_data, 
      main.bar.color = "steelblue", 
      sets.bar.color = "darkred", 
      order.by = "freq",
      nsets = 11,  # the first 11
      nintersects = 20) 

#part9-choose the top 33 features(in order to compare with MAMSI)
all_selected_features <- read.csv("all_selected_features.csv")
feature_counts <- table(all_selected_features$Variable)
feature_counts_df <- as.data.frame(feature_counts)
colnames(feature_counts_df) <- c("Feature", "Frequency")

feature_counts_df <- feature_counts_df[order(-feature_counts_df$Frequency), ]

top_384_features <- head(feature_counts_df, 384)

rho_0.9_data <- selected_features_list[["rho_0.9"]]

# coefficeint from rho_0.9
top_384_coefficients <- rho_0.9_data[rho_0.9_data$Variable %in% top_384_features$Feature, ]
head(top_384_coefficients$Coefficient)
str(top_384_coefficients$Coefficient)

top_384_coefficients <- top_384_coefficients[order(-abs(top_384_coefficients$Coefficient)), ]
class(top_384_coefficients$Coefficient)
top_384_coefficients <- top_384_coefficients[top_384_coefficients$Variable != "(Intercept)", ]

top_33_coefficients <- head(top_384_coefficients, 33)

ggplot(top_33_coefficients, aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = ifelse(top_33_coefficients$Coefficient > 0, "steelblue", "darkred")) +
  labs(
    title = "Top 33 Features by Coefficient at rho_0.9 (Absolute Value)",
    x = "Feature",
    y = "Coefficient"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),  
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold") 
  ) +
  coord_flip()

# visualize coefficient of rho_0.9
all_coefficients <- numeric(coefficients@Dim[1])  
all_coefficients[coefficients@i + 1] <- abs(coefficients@x) 

plot_data <- data.frame(
  FeatureIndex = 1:length(all_coefficients),  
  Coefficient = all_coefficients  
)

x_breaks <- seq(0, length(all_coefficients), by = 500)  
x_labels <- x_breaks 

ggplot(plot_data, aes(x = FeatureIndex, y = Coefficient)) +
  geom_line(color = "limegreen", linewidth = 0.8) +  
  labs(
    title = "Feature Coefficients Visualization",
    x = "Feature Index",
    y = "Absolute Coefficient Value"
  ) +
  theme_minimal() +  
  theme(
    plot.title = element_text(hjust = 0.5),  
    axis.text.x = element_text(hjust = 1) 
  ) +
  scale_x_continuous(breaks = x_breaks, labels = x_labels) +  
  scale_y_continuous(limits = c(0, max(all_coefficients) * 1.1))  


