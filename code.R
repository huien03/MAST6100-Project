
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(xgboost)
library(keras3)

library(pROC)
library(gridExtra)
library(corrplot)

set.seed(123)

# Data loading and preprocessing
telco <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", stringsAsFactors = FALSE)
dim(telco)

# Remove customerID
telco <- telco %>% select(-customerID)

# Handle missing values
colSums(is.na(telco))
telco <- telco %>% filter(!is.na(TotalCharges))
dim(telco)

# Convert to factors
telco$SeniorCitizen <- as.factor(telco$SeniorCitizen)
char_vars <- sapply(telco, is.character)
telco[char_vars] <- lapply(telco[char_vars], as.factor)
telco$Churn_binary <- ifelse(telco$Churn == "Yes", 1, 0)

str(telco)
table(telco$Churn)
prop.table(table(telco$Churn))

# Exploratory visualizations
png("plot_01_churn_distribution.png", width=800, height=600)
ggplot(telco, aes(x=Churn, fill=Churn)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title="Customer Churn Distribution", x="Churn Status", y="Count") +
  theme_minimal() +
  scale_fill_manual(values=c("No"="#2ecc71", "Yes"="#e74c3c"))
dev.off()

png("plot_02_churn_by_contract.png", width=900, height=600)
ggplot(telco, aes(x=Contract, fill=Churn)) +
  geom_bar(position="fill") +
  geom_text(aes(label=scales::percent(..count../tapply(..count.., ..x.., sum)[..x..])),
            stat="count", position=position_fill(vjust=0.5)) +
  labs(title="Churn Rate by Contract Type", x="Contract Type", y="Proportion") +
  theme_minimal() +
  scale_fill_manual(values=c("No"="#2ecc71", "Yes"="#e74c3c"))
dev.off()

png("plot_03_churn_by_tenure.png", width=900, height=600)
ggplot(telco, aes(x=tenure, fill=Churn)) +
  geom_density(alpha=0.6) +
  labs(title="Churn Distribution by Tenure", x="Tenure (months)", y="Density") +
  theme_minimal() +
  scale_fill_manual(values=c("No"="#2ecc71", "Yes"="#e74c3c"))
dev.off()

png("plot_04_monthly_charges_churn.png", width=900, height=600)
ggplot(telco, aes(x=Churn, y=MonthlyCharges, fill=Churn)) +
  geom_boxplot() +
  labs(title="Monthly Charges by Churn Status", x="Churn Status", y="Monthly Charges ($)") +
  theme_minimal() +
  scale_fill_manual(values=c("No"="#2ecc71", "Yes"="#e74c3c"))
dev.off()

png("plot_05_churn_by_internet.png", width=900, height=600)
ggplot(telco, aes(x=InternetService, fill=Churn)) +
  geom_bar(position="fill") +
  labs(title="Churn Rate by Internet Service Type", x="Internet Service", y="Proportion") +
  theme_minimal() +
  scale_fill_manual(values=c("No"="#2ecc71", "Yes"="#e74c3c"))
dev.off()

png("plot_06_churn_multiple_factors.png", width=1200, height=800)
p1 <- ggplot(telco, aes(x=PaymentMethod, fill=Churn)) +
  geom_bar(position="fill") +
  labs(title="Payment Method", x="", y="Proportion") +
  theme_minimal() +
  scale_fill_manual(values=c("No"="#2ecc71", "Yes"="#e74c3c")) +
  theme(axis.text.x = element_text(angle=45, hjust=1))
p2 <- ggplot(telco, aes(x=PaperlessBilling, fill=Churn)) +
  geom_bar(position="fill") +
  labs(title="Paperless Billing", x="", y="Proportion") +
  theme_minimal() +
  scale_fill_manual(values=c("No"="#2ecc71", "Yes"="#e74c3c"))
p3 <- ggplot(telco, aes(x=SeniorCitizen, fill=Churn)) +
  geom_bar(position="fill") +
  labs(title="Senior Citizen", x="", y="Proportion") +
  theme_minimal() +
  scale_fill_manual(values=c("No"="#2ecc71", "Yes"="#e74c3c"))
p4 <- ggplot(telco, aes(x=Partner, fill=Churn)) +
  geom_bar(position="fill") +
  labs(title="Partner", x="", y="Proportion") +
  theme_minimal() +
  scale_fill_manual(values=c("No"="#2ecc71", "Yes"="#e74c3c"))
grid.arrange(p1, p2, p3, p4, ncol=2)
dev.off()

png("plot_07_correlation_matrix.png", width=800, height=800)
telco_num <- telco %>% select(tenure, MonthlyCharges, TotalCharges, Churn_binary)
cor_matrix <- cor(telco_num)
corrplot(cor_matrix, method="color", type="upper",
         addCoef.col="black", number.cex=0.9,
         tl.col="black", tl.srt=45,
         title="Correlation Matrix", mar=c(0,0,2,0))
dev.off()

# Data preparation for modeling
telco_model <- telco %>% select(-Churn, -Churn_binary)
target <- telco$Churn_binary

# Train-test split (80-20)
train_index <- createDataPartition(target, p=0.8, list=FALSE)
train_data <- telco_model[train_index, ]
test_data <- telco_model[-train_index, ]
train_target <- target[train_index]
test_target <- target[-train_index]

nrow(train_data)
nrow(test_data)
mean(train_target)
mean(test_target)

# Model 1: Logistic Regression (GLM)
train_glm <- train_data
train_glm$Churn <- train_target
glm_model <- glm(Churn ~ ., data=train_glm, family=binomial(link="logit"))
summary(glm_model)

glm_pred_prob <- predict(glm_model, newdata=test_data, type="response")
glm_pred_class <- ifelse(glm_pred_prob > 0.5, 1, 0)
glm_cm <- confusionMatrix(as.factor(glm_pred_class), as.factor(test_target), positive="1")
glm_cm
glm_auc <- auc(roc(test_target, glm_pred_prob, quiet=TRUE))
glm_auc

# Model 2: Random Forest
train_rf <- train_data
train_rf$Churn <- as.factor(train_target)
rf_model <- randomForest(Churn ~ ., data=train_rf, ntree=500, importance=TRUE)
rf_model

rf_pred_prob <- predict(rf_model, newdata=test_data, type="prob")[,2]
rf_pred_class <- predict(rf_model, newdata=test_data, type="class")
rf_cm <- confusionMatrix(rf_pred_class, as.factor(test_target), positive="1")
rf_cm
rf_auc <- auc(roc(test_target, rf_pred_prob, quiet=TRUE))
rf_auc
importance(rf_model)

# Model 3: XGBoost
train_matrix <- model.matrix(~.-1, data=train_data)
test_matrix <- model.matrix(~.-1, data=test_data)
dtrain <- xgb.DMatrix(data=train_matrix, label=train_target)
dtest <- xgb.DMatrix(data=test_matrix, label=test_target)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.3,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train=dtrain, test=dtest),
  verbose = 0
)
xgb_model

xgb_pred_prob <- predict(xgb_model, dtest)
xgb_pred_class <- ifelse(xgb_pred_prob > 0.5, 1, 0)
xgb_cm <- confusionMatrix(as.factor(xgb_pred_class), as.factor(test_target), positive="1")
xgb_cm
xgb_auc <- auc(roc(test_target, xgb_pred_prob, quiet=TRUE))
xgb_auc

# Model 4: Deep Learning (Neural Network)
train_matrix_scaled <- scale(train_matrix)
test_matrix_scaled <- scale(
  test_matrix,
  center = attr(train_matrix_scaled, "scaled:center"),
  scale  = attr(train_matrix_scaled, "scaled:scale")
)

input_dim <- ncol(train_matrix_scaled)

dl_model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(input_dim)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

dl_model %>% compile(
  optimizer = "adam",
  loss      = "binary_crossentropy",
  metrics   = c("accuracy")
)

summary(dl_model)

# Train model
history <- fit(
  dl_model,
  x = train_matrix_scaled, 
  y = train_target,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

history

# Predictions
dl_pred_prob <- predict(dl_model, test_matrix_scaled, verbose=0)
dl_pred_class <- ifelse(dl_pred_prob > 0.5, 1, 0)
dl_cm <- confusionMatrix(as.factor(dl_pred_class), as.factor(test_target), positive="1")
dl_cm
dl_auc <- auc(roc(test_target, as.vector(dl_pred_prob), quiet=TRUE))
dl_auc

# Model comparison
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost", "Deep Learning"),
  Accuracy = c(glm_cm$overall['Accuracy'],
               rf_cm$overall['Accuracy'],
               xgb_cm$overall['Accuracy'],
               dl_cm$overall['Accuracy']),
  Sensitivity = c(glm_cm$byClass['Sensitivity'],
                  rf_cm$byClass['Sensitivity'],
                  xgb_cm$byClass['Sensitivity'],
                  dl_cm$byClass['Sensitivity']),
  Specificity = c(glm_cm$byClass['Specificity'],
                  rf_cm$byClass['Specificity'],
                  xgb_cm$byClass['Specificity'],
                  dl_cm$byClass['Specificity']),
  AUC = c(as.numeric(glm_auc),
          as.numeric(rf_auc),
          as.numeric(xgb_auc),
          as.numeric(dl_auc))
)
model_comparison

# Model comparison visualization
png("plot_08_model_comparison.png", width=1000, height=600)
comparison_long <- reshape2::melt(model_comparison, id.vars="Model")
ggplot(comparison_long, aes(x=Model, y=value, fill=variable)) +
  geom_bar(stat="identity", position="dodge") +
  labs(title="Model Performance Comparison", x="Model", y="Score", fill="Metric") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1)) +
  scale_fill_brewer(palette="Set2")
dev.off()

# ROC curves comparison
png("plot_09_roc_curves.png", width=900, height=700)
roc_glm <- roc(test_target, glm_pred_prob, quiet=TRUE)
roc_rf <- roc(test_target, rf_pred_prob, quiet=TRUE)
roc_xgb <- roc(test_target, xgb_pred_prob, quiet=TRUE)
roc_dl <- roc(test_target, as.vector(dl_pred_prob), quiet=TRUE)

plot(roc_glm, col="blue", main="ROC Curves Comparison", lwd=2)
plot(roc_rf, col="red", add=TRUE, lwd=2)
plot(roc_xgb, col="green", add=TRUE, lwd=2)
plot(roc_dl, col="purple", add=TRUE, lwd=2)
legend("bottomright", 
       legend=c(paste("GLM (AUC =", round(glm_auc, 3), ")"),
                paste("RF (AUC =", round(rf_auc, 3), ")"),
                paste("XGBoost (AUC =", round(xgb_auc, 3), ")"),
                paste("DL (AUC =", round(dl_auc, 3), ")")),
       col=c("blue", "red", "green", "purple"), lwd=2)
dev.off()