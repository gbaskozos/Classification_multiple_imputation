#This is a function that calculates performance statistics and predictions from models applied on multiply imputed datasets. Predict then aggregate.

#Column bind (cbind), concatenate columns of dataframes. If they have different sizes, fill empty cells with NAs. A function that you will find usefull in a lot of projects
cbind.fill <- function(...){
    nm <- list(...) 
    nm <- lapply(nm, as.matrix)
    n <- max(sapply(nm, nrow)) 
    do.call(cbind, lapply(nm, function (x) 
        rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}

#Define my classMI function. It take as arguments: A list of models trained on completed datasets from multiple imputation. A list of completed training and test sets. The outcome name, the predictor's names, a binary variable that indicates if we have performed RFE (default is FALSE)
classMI <- function(model, train_list, test_list, outcomeName, predictors, rfe=FALSE) {
############### Variable Importance ###############
# Libraries needed for the function to work
require(caret)
library(gbm)
library(ranger)
require(foreach)
require(doParallel)
require(ggplot2)
require(reshape2)
require(forcats)
require(CORElearn)
require(lattice)


#If RFE is true then do whatever is in the {} block 
if(rfe == TRUE) {
#Variable Inclusion Rate
InclusionRate <- table(unlist(lapply(model, function(x) x$optVariable))) / length(model) * 100
InclusionRate <- InclusionRate[order(-InclusionRate)]

#Variable AUC profiling
varAUC <- do.call(rbind, lapply(model, function(x) x$results))
varProfile <- aggregate(varAUC, list(varAUC$Variables), mean)[,-1]
within_Imps_var <- varProfile$AUCSD^2
between_Imps_var <- (aggregate(varAUC, list(varAUC$Variables), var)[,-1])
varProfile$total_AUC_var <- sqrt(within_Imps_var + between_Imps_var$AUC + between_Imps_var$AUC/imps) 
#Variable Importance
varImportance <- data.frame(row.names= names(InclusionRate))

for (i in 1:length(model)) {
varImportance <- cbind.fill(varImportance, varImp(model[[i]],model[[i]]$optsize)[rownames(varImportance),])
}

varImp_agg <- data.frame(Importance = rowMeans(varImportance, na.rm = TRUE), SD = apply(varImportance,1, sd, na.rm = TRUE))
varImp_agg <- varImp_agg[order(-varImp_agg$Importance),]
varImp_agg$CI_low <- varImp_agg$Importance - qnorm(0.975)*varImp_agg$SD/sqrt(imps)
varImp_agg$CI_high <- varImp_agg$Importance + qnorm(0.975)*varImp_agg$SD/sqrt(imps)
varImp_agg$Variable <- rownames(varImp_agg)

#Plots
rate_plot <- dotplot(as.matrix(InclusionRate)[order(InclusionRate),], main = paste(model[[1]]$fit$method, "Inclusion Rate (rfe)", sep=" "), xlab="% percentage")

profile_plot <- ggplot(varProfile, aes(x=Variables)) + geom_line(aes(y = AUC, colour="AUC")) + geom_point(aes(y = AUC, colour="AUC")) + geom_line(aes(y = F, colour="F")) + geom_point(aes(y = F, colour="F")) + scale_y_continuous(sec.axis = sec_axis(~.*1, name = "F")) + scale_x_continuous(breaks=c(varProfile$Variables)) + scale_colour_manual(values = c("blue", "red")) + labs(y = "Area Under the Curve", x = "Variables", colour = "Metric") + ggtitle(paste(model[[1]]$fit$method, "Variable selection profile", sep=" ")) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_text(size=10, face="bold"), axis.text.x = element_text(size= 8, face="bold"), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))

varImportance_tp <- melt(varImportance)
names(varImportance_tp) <- c("Variable", "Imputation", "Importance")

varImp_plot <- ggplot(varImportance_tp, aes(x = fct_reorder(Variable, Importance, .fun = median, .desc =TRUE), y = Importance, color = Variable, fill = Variable)) +
  geom_bar(data = varImp_agg, stat = "identity", alpha = .3) + geom_point() +
  guides(color = "none", fill = "none") + geom_errorbar(data = varImp_agg, aes(ymin=Importance-SD, ymax=Importance+SD), width=.3, position=position_dodge(.9), color="black") +
  xlab("Variable") + ggtitle(paste(model[[1]]$method, "Variable Importance", sep=" ")) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_text(size=10, face="bold"), axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size =10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))

################# Resample Performance ###############
MI_index <- rep(1:imps, each=5*10)

resamples_MI <- do.call(rbind, lapply(lapply(model, function(x) x$fit$resample), function(x) x[!duplicated(x$Resample),]))
resamples_MI$MI_index <- MI_index
#within_imps_var <- apply(aggregate(resamples_MI, list(resamples_MI$MI_index), var)[,c(2:5)],2,mean)
within_imps_var <- aggregate(resamples_MI, list(resamples_MI$MI_index), var)[,c(2:5)]
within_imps_sd <- aggregate(resamples_MI, list(resamples_MI$MI_index), sd)[,c(2:5)]
names(within_imps_sd) <- paste0(names(within_imps_sd), "SD") 
resampleProfile <- aggregate(resamples_MI, list(resamples_MI$MI_index), mean)[,-c(1,6)]
resampleProfile <- cbind(resampleProfile, within_imps_sd)
between_imps_var <- apply(resampleProfile[c(1:4)],2,var)
total_sd <- sqrt(colMeans(within_imps_var) + between_imps_var + between_imps_var/imps)

pooledResampleProfile <- data.frame(variable = names(colMeans(resampleProfile)[c(1,4)]), value = colMeans(resampleProfile)[c(1,4)], sd = total_sd[c(1,4)]) 

df <- melt(resampleProfile[,c(1,4)])

bestPerf_accross_MI <- max(unlist(lapply(lapply(model, function(x) x$fit$resample$AUC), function(x) median(x))))
bestModel_accross_MI <- model[[which(unlist(lapply(lapply(model, function(x) x$fit$resample$AUC), function(x) median(x))) == bestPerf_accross_MI)]] 
bestModel_MI_index <- which(unlist(lapply(lapply(model, function(x) x$fit$resample$AUC), function(x) median(x))) == bestPerf_accross_MI)

#Plots
resample_plot <- ggplot(resampleProfile, aes(x="", y=AUC)) + geom_boxplot(colour="blue", fill = "white", outlier.shape=16, outlier.size=2) + ggtitle(paste(model[[1]]$fit$method, "Precision/Recall AUC", sep=" ")) + geom_jitter(width = 0.2) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_blank(), axis.text.x = element_blank(), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))


pooled_resample_plot <- ggplot(pooledResampleProfile, aes(x=variable, y=value, fill=variable)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) +
  geom_errorbar(aes(ymin=value - sd, ymax=value + sd), width=.2,
                 position=position_dodge(.9), data = pooledResampleProfile) +
	geom_hline(yintercept=max(prop.table(table(train_list[[1]]$Outcome))), color='black', linetype = "dashed") + ggtitle(paste(model[[1]]$fit$method, "Precision/Recall AUC, F score", sep=" ")) + geom_jitter(width = 0.2, data = df) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_blank(), axis.text.x = element_blank(), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))


################# Predictions Resamples ###############
pred_OOF <- list()

for (x in 1:length(model))
{
    bestSubset_pred <- model[[x]]$pred[model[[x]]$pred$Variables==model[[x]]$bestSubset ,]

	pred_OOF[[x]] <- bestSubset_pred[order(bestSubset_pred$rowIndex), model[[1]]$obsLevels[[1]]] 
	}

trainSet_OOF_imps <- as.data.frame(do.call(cbind, lapply(pred_OOF, function(x) '['(x))))

trainSet_OOF <- rowMeans(trainSet_OOF_imps)

################# Predictions test set ###############
pred <- list()

for (x in 1:length(model))
{
    for (y in 1:length(model))
    { pred[[y+((x-1)*length(model))]] <- predict(model[[x]], newdata =test_list[[y]][,predictors])$pred
	
	}
	}

predAgg <- as.data.frame(do.call(cbind, lapply(pred, function(x) '['(x))))

predAgg <- as.data.frame(ifelse(predAgg == 1, model[[1]]$obsLevels[[1]], model[[1]]$obsLevels[[2]]))

majority <- apply(predAgg, 1, function(x) ifelse(sum(x == model[[1]]$obsLevels[[1]], na.rm=TRUE) > sum(x == model[[1]]$obsLevels[[2]], na.rm=TRUE),  model[[1]]$obsLevels[[1]], model[[1]]$obsLevels[[2]]))

testSet_pred <- factor(majority, levels=c(model[[1]]$obsLevels[[1]], model[[1]]$obsLevels[[2]]))

pred_probs <- list()

for (x in 1:length(model))
{
    for (y in 1:length(model))
    { pred_probs[[y+((x-1)*length(model))]] <- predict(model[[x]], newdata =test_list[[y]][,predictors])[,model[[1]]$obsLevels[[1]]]
	
	}
	}

predAgg_probs <- as.data.frame(do.call(cbind, lapply(pred_probs, function(x) '['(x))))

predAgg_probs <- rowMeans(predAgg_probs)
#predAgg_probs <- apply(predAgg_probs,1, median)

predAgg_probs_majority <- factor(ifelse(predAgg_probs >= 0.5, model[[1]]$obsLevels[[1]], model[[1]]$obsLevels[[2]]), levels = c(model[[1]]$obsLevels[[1]], model[[1]]$obsLevels[[2]]))

cMatrix <- confusionMatrix(predAgg_probs_majority, test_list[[1]][,outcomeName], positive=model[[1]]$obsLevels[[1]])

testProbs <- data.frame(obs = test_list[[1]][,outcomeName], model = predAgg_probs)

require(lattice)
cal_obj <- calibration(obs ~ model, data = testProbs, class = model[[1]]$obsLevels[[1]])

calibration_plot <- ggplot(cal_obj) + ggtitle(paste(model[[1]]$fit$method, "calibration plot", sep=" ")) + geom_jitter(width = 0.2) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_text(size=10, face="bold"), axis.text.x = element_text(size= 8, face="bold"), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))

calibration <- calibrate(testProbs$obs, testProbs$model, class1=1, method = "binIsoReg", noBins = 15, weight=NULL, assumeProbabilities=TRUE)
          
calibrated_probs <- applyCalibration(testProbs$model, calibration)

predAgg_probs_majority_calibrated <- factor(ifelse(calibrated_probs >= 0.5, model[[1]]$obsLevels[[1]], model[[1]]$obsLevels[[2]]), levels = c(model[[1]]$obsLevels[[1]], model[[1]]$obsLevels[[2]]))

predsTestSet <- data.frame(pred_probs = predAgg_probs, predA_class = predAgg_probs_majority, calibrated_probs = calibrated_probs, calibrated_class = calibrated_probs, real_class = test_list[[1]][,outcomeName])

calibrated_cMatrix <- confusionMatrix(predAgg_probs_majority_calibrated, test_list[[1]][,outcomeName], positive=model[[1]]$obsLevels[[1]])
###############################################

return_list <- list(InclusionRate = InclusionRate, varProfile = varProfile, varImpROC = varImp_agg, varImportance = varImportance_tp, resamplePerf = resampleProfile, pooledResampleProfile = pooledResampleProfile, OOF_pred = trainSet_OOF, testSet_pred = testSet_pred, testSet_probs = predAgg_probs, cMatrix = cMatrix, calibrated_cMatrix = calibrated_cMatrix, predsTestSet = predsTestSet , bestModel = bestModel_accross_MI, bestModel_MI_index = bestModel_MI_index, resample_plot = pooled_resample_plot, rate_plot = rate_plot, profile_plot = profile_plot, varImp_plot = varImp_plot)

return(return_list)

}
else {
#Variable Importance

calculate_var_imp <- TRUE

tryCatch(varImportance <- do.call(cbind.fill, lapply(model, function(x) varImp(x)$importance)), error = function(e) { calculate_var_imp <<- FALSE})
if (calculate_var_imp) {
colnames(varImportance) <- 1:ncol(varImportance)

varImp_agg <- data.frame(Importance = rowMeans(varImportance, na.rm = TRUE), SD = apply(varImportance,1, sd, na.rm = TRUE))
varImp_agg <- varImp_agg[order(-varImp_agg$Importance),]
varImp_agg$CI_low <- varImp_agg$Importance - qnorm(0.975)*varImp_agg$SD/sqrt(imps)
varImp_agg$CI_high <- varImp_agg$Importance + qnorm(0.975)*varImp_agg$SD/sqrt(imps)
varImp_agg$Variable <- rownames(varImp_agg)

#Plots
varImportance_tp <- melt(varImportance)
names(varImportance_tp) <- c("Variable", "Imputation", "Importance")

varImp_plot <- ggplot(varImportance_tp, aes(x = fct_reorder(Variable, Importance, .fun = median, .desc =TRUE), y = Importance, color = Variable, fill = Variable)) +
  geom_bar(data = varImp_agg, stat = "identity", alpha = .3) + geom_point() +
  guides(color = "none", fill = "none") + geom_errorbar(data = varImp_agg, aes(ymin=Importance-SD, ymax=Importance+SD), width=.3, position=position_dodge(.9), color="black") +
  xlab("Variable") + ggtitle(paste(model[[1]]$method, "Variable Importance", sep=" ")) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_text(size=10, face="bold"), axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size =10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5)) }

#varImp_plot <- ggplot(varImportance_tp, aes(x=fct_reorder(Variable, Importance, .fun = median, .desc =TRUE), y=Importance)) + geom_violin(trim=TRUE, scale = "width", aes(fill = fct_reorder(Variable, Importance, .fun = median, .desc =TRUE))) + stat_summary(fun.y=median, geom="point", size=2, color="black") + scale_fill_discrete(guide = guide_legend(title = "Variable")) + xlab("Variable") + ggtitle(paste(model[[1]]$method, "Variable Importance", sep=" ")) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_text(size=10, face="bold"), axis.text.x = element_blank(), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5)) }
else {
varImp_agg <- NA
varImportance_tp <- NA
varImp_plot <- NA
}

################# Resample Performance ###############

MI_index <- rep(1:imps, each=5*10)

resamples_MI <- do.call(rbind, lapply(model, function(x) x$resample))
resamples_MI$MI_index <- MI_index

within_imps_var <- aggregate(resamples_MI, list(resamples_MI$MI_index), var)[,c(2:5)]
within_imps_sd <- aggregate(resamples_MI, list(resamples_MI$MI_index), sd)[,c(2:5)]
names(within_imps_sd) <- paste0(names(within_imps_sd), "SD") 

resampleProfile <- aggregate(resamples_MI, list(resamples_MI$MI_index), mean)[,-c(1,6)]
resampleProfile <- cbind(resampleProfile, within_imps_sd)
between_imps_var <- apply(resampleProfile[c(1:4)],2,var)
total_sd <- sqrt(colMeans(within_imps_var) + between_imps_var + between_imps_var/imps)

bestPerf_accross_MI <- max(unlist(lapply(lapply(model, function(x) x$resample$AUC), function(x) median(x))))
bestModel_accross_MI <- model[[which(unlist(lapply(lapply(model, function(x) x$resample$AUC), function(x) median(x))) == bestPerf_accross_MI)]] 
bestModel_MI_index <- which(unlist(lapply(lapply(model, function(x) x$resample$AUC), function(x) median(x))) == bestPerf_accross_MI)

pooledResampleProfile <- data.frame(variable = names(colMeans(resampleProfile)[c(1,4)]), value = colMeans(resampleProfile)[c(1,4)], sd = total_sd[c(1,4)]) 

df <- melt(resampleProfile[,c(1,4)])

#Plots
resample_plot <- ggplot(resampleProfile, aes(x="", y=AUC)) + geom_boxplot(colour="blue", fill = "white", outlier.shape=16, outlier.size=2) + ggtitle(paste(model[[1]]$fit$method, "Precision/Recall AUC", sep=" ")) + geom_jitter(width = 0.2) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_blank(), axis.text.x = element_blank(), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))


pooled_resample_plot <- ggplot(pooledResampleProfile, aes(x=variable, y=value, fill=variable)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) +
  geom_errorbar(aes(ymin=value - sd, ymax=value + sd), width=.2,
                 position=position_dodge(.9), data = pooledResampleProfile) +
	geom_hline(yintercept=max(prop.table(table(train_list[[1]]$Outcome))), color='black', linetype = "dashed") + ggtitle(paste(model[[1]]$method, "Precision/Recall AUC, F score", sep=" ")) + geom_jitter(width = 0.2, data = df) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_blank(), axis.text.x = element_blank(), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))

################# Predictions Resamples ###############
trainSet_OOF <- lapply(model, function(y) aggregate(y$pred[,eval(model[[1]]$levels[[1]])], list(y$pred$rowIndex), mean)$x)

trainSet_OOF_imps <- as.data.frame(do.call(cbind, lapply(trainSet_OOF, function(x) '['(x))))

trainSet_OOF <- rowMeans(trainSet_OOF_imps)

################# Predictions test set ###############
pred <- list()

for (x in 1:length(model))
{
    for (y in 1:length(model))
    { pred[[y+((x-1)*length(model))]] <- predict(model[[x]], newdata =test_list[[y]][,predictors])
	
	}
	}

predAgg <- as.data.frame(do.call(cbind, lapply(pred, function(x) '['(x))))

#predAgg <- as.data.frame(ifelse(predAgg == 1, model[[1]]$levels[[1]], model[[1]]$levels[[2]]))

majority <- apply(predAgg, 1, function(x) ifelse(sum(x == 1, na.rm=TRUE) > sum(x == 2, na.rm=TRUE),  model[[1]]$levels[[1]], model[[1]]$levels[[2]]))
#majority <- apply(predAgg, 1, function(x) ifelse(sum(x == model[[1]]$levels[[1]], na.rm=TRUE) > sum(x == model[[1]]$levels[[2]], na.rm=TRUE),  model[[1]]$levels[[1]], model[[1]]$levels[[2]]))

testSet_pred <- factor(majority, levels = c(model[[1]]$levels[[1]], model[[1]]$levels[[2]]))

pred_probs <- list()

for (x in 1:length(model))
{
    for (y in 1:length(model))
    { pred_probs[[y+((x-1)*length(model))]] <- predict(model[[x]], newdata =test_list[[y]][,predictors], type="prob")[,model[[1]]$levels[[1]]]
	
	}
	}

predAgg_probs <- as.data.frame(do.call(cbind, lapply(pred_probs, function(x) '['(x))))

predAgg_probs <- rowMeans(predAgg_probs)
#predAgg_probs <- apply(predAgg_probs,1, median)

predAgg_probs_majority <- factor(ifelse(predAgg_probs > 0.5, model[[1]]$levels[[1]], model[[1]]$levels[[2]]), levels = c(model[[1]]$levels[[1]], model[[1]]$levels[[2]]))

#cMatrix <- confusionMatrix(predAgg_probs_majority, test_list[[1]][,outcomeName], positive=model[[1]]$levels[[1]])
cMatrix <- confusionMatrix(testSet_pred, test_list[[1]][,outcomeName], positive=model[[1]]$levels[[1]])

testProbs <- data.frame(obs = test_list[[1]][,outcomeName], model = predAgg_probs)

#### Calibrate
cal_obj <- calibration(obs ~ model, data = testProbs, class = model[[1]]$levels[[1]], cuts=20)

calibration_plot <- ggplot(cal_obj) + ggtitle(paste(model[[1]]$method, "calibration plot", sep=" ")) + geom_jitter(width = 0.2) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_text(size=10, face="bold"), axis.text.x = element_text(size= 8, face="bold"), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))

calibration <- calibrate(testProbs$obs, testProbs$model, class1=1, method = "isoReg", weight=NULL, assumeProbabilities=TRUE)

         
calibrated_probs <- applyCalibration(testProbs$model, calibration)

predAgg_probs_majority_calibrated <- factor(ifelse(calibrated_probs > 0.5, model[[1]]$levels[[1]], model[[1]]$levels[[2]]), levels = c(model[[1]]$levels[[1]], model[[1]]$levels[[2]]))

predsTestSet <- data.frame(pred_probs = predAgg_probs, predA_class = predAgg_probs_majority, calibrated_probs = calibrated_probs, calibrated_class = calibrated_probs, real_class = test_list[[1]][,outcomeName])

calibrated_cMatrix <- confusionMatrix(predAgg_probs_majority_calibrated, test_list[[1]][,outcomeName], positive=model[[1]]$levels[[1]])


###############################################
#Put values in alist with convenient names and return it to the outside world
return_list <- list(varImpROC = varImp_agg, varImportance = varImportance_tp, resamplePerf = resampleProfile, pooledResampleProfile = pooledResampleProfile, OOF_pred = trainSet_OOF, testSet_pred = testSet_pred, testSet_probs = predAgg_probs, cMatrix = cMatrix, calibrated_cMatrix = calibrated_cMatrix, predsTestSet = predsTestSet, bestModel = bestModel_accross_MI, bestModel_MI_index = bestModel_MI_index, resample_plot = pooled_resample_plot, varImp_plot = varImp_plot)

return(return_list)
}

}

