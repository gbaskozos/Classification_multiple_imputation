#It is a function that calculates perfromance statistics and predictions from models applied on multiply imputed datasets.

#Column bind (cbind), concatenate columns of dataframes. If they have different sizes, fill empty cells with NAs. A function that you will find usefull in a lot of projects
cbind.fill <- function(...){
    nm <- list(...) 
    nm <- lapply(nm, as.matrix)
    n <- max(sapply(nm, nrow)) 
    do.call(cbind, lapply(nm, function (x) 
        rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}

#Define my classMI function. It take as arguments: A list of models trained on completed datasets from multiple imputation. A list of completed training and test sets. The outcome name, the predictor's names, a binary variable that indecates if we have performed RFE (default is FALSE)
classMI <- function(model, train_list, test_list, outcomeName, predictors, rfe=FALSE) {
############### Variable Importance ###############
# Libraries needed for the function to work
require(caret)
require(foreach)
require(doParallel)
require(ggplot2)
require(reshape2)
require(forcats)

#If RFE is true the  do whatever is in the {} block 
if(rfe == TRUE) {
#Variable Inclusion Rate
InclusionRate <- table(unlist(lapply(model, function(x) x$optVariable))) / length(model) * 100
InclusionRate <- InclusionRate[order(-InclusionRate)]

#Variable AUC profiling
varAUC <- do.call(rbind, lapply(model, function(x) x$results))
varProfile <- aggregate(varAUC, list(varAUC$Variables), mean)[,-1]

#Variable Importance
varImportance <- data.frame(row.names= names(InclusionRate))

for (i in 1:length(model)) {
varImportance <- cbind.fill(varImportance, varImp(model[[i]],model[[i]]$optsize)[rownames(varImportance),])
}

varImp_agg <- data.frame(Importance = rowMeans(varImportance, na.rm = TRUE), SD = apply(varImportance,1, sd, na.rm = TRUE))
varImp_agg <- varImp_agg[order(-varImp_agg$Importance),]

#Plots
rate_plot <- dotplot(as.matrix(InclusionRate)[order(InclusionRate),], main = paste(model[[1]]$fit$method, "Inclusion Rate (rfe)", sep=" "), xlab="% percentage")

profile_plot <- ggplot(varProfile, aes(x=Variables)) + geom_line(aes(y = AUC, colour="AUC")) + geom_point(aes(y = AUC, colour="AUC")) + geom_line(aes(y = F, colour="F")) + geom_point(aes(y = F, colour="F")) + scale_y_continuous(sec.axis = sec_axis(~.*1, name = "F")) + scale_x_continuous(breaks=c(varProfile$Variables)) + scale_colour_manual(values = c("blue", "red")) + labs(y = "Area Under the Curve", x = "Variables", colour = "Metric") + ggtitle(paste(model[[1]]$fit$method, "Variable selection profile", sep=" ")) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_text(size=10, face="bold"), axis.text.x = element_text(size= 8, face="bold"), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))

varImportance_tp <- melt(varImportance)
names(varImportance_tp) <- c("Variable", "Imputation", "Importance")

varImp_plot <- ggplot(varImportance_tp, aes(x=fct_reorder(Variable, Importance, fun = median, .desc =TRUE), y=Importance)) + geom_violin(trim=TRUE, scale = "width", aes(fill = fct_reorder(Variable, Importance, fun = median, .desc =TRUE))) + stat_summary(fun.y=median, geom="point", size=2, color="black") + scale_fill_discrete(guide = guide_legend(title = "Variable")) + xlab("Variable") + ggtitle(paste(model[[1]]$fit$method, "Variable Importance", sep=" ")) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_text(size=10, face="bold"), axis.text.x = element_blank(), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))

################# Resample Performance ###############
MI_index <- rep(1:imps, each=5*10)

resamples_MI <- do.call(rbind, lapply(lapply(model, function(x) x$fit$resample), function(x) x[!duplicated(x$Resample),]))
resamples_MI$MI_index <- MI_index

resampleProfile <- aggregate(resamples_MI, list(resamples_MI$MI_index), median)[,-c(1,6)]

bestPerf_accross_MI <- max(unlist(lapply(lapply(model, function(x) x$fit$resample$AUC), function(x) median(x))))
bestModel_accross_MI <- model[[which(unlist(lapply(lapply(model, function(x) x$fit$resample$AUC), function(x) median(x))) == bestPerf_accross_MI)]] 
bestModel_MI_index <- which(unlist(lapply(lapply(model, function(x) x$fit$resample$AUC), function(x) median(x))) == bestPerf_accross_MI)

#Plots
resample_plot <- ggplot(resampleProfile, aes(x="", y=AUC)) + geom_boxplot(colour="blue", fill = "white", outlier.shape=16, outlier.size=2) + ggtitle(paste(model[[1]]$fit$method, "Precision/Recall AUC", sep=" ")) + geom_jitter(width = 0.2) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_blank(), axis.text.x = element_blank(), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))



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
    { pred_probs[[y+((x-1)*length(model))]] <- predict(model[[x]], newdata =test_list[[y]][,predictors])[,2]
	
	}
	}

predAgg_probs <- as.data.frame(do.call(cbind, lapply(pred_probs, function(x) '['(x))))

predAgg_probs <- rowMeans(predAgg_probs)

predAgg_probs_majority <- as.factor(ifelse(predAgg_probs > 0.5, model[[1]]$obsLevels[[1]], model[[1]]$obsLevels[[2]]))


cMatrix <- confusionMatrix(testSet_pred, test_list[[1]][,outcomeName])
###############################################

return_list <- list(InclusionRate = InclusionRate, varProfile = varProfile, varImpROC = varImp_agg, varImportance = varImportance_tp, resamplePerf = resampleProfile, OOF_pred = trainSet_OOF, testSet_pred = testSet_pred, testSet_probs = predAgg_probs, cMatrix = cMatrix, bestModel = bestModel_accross_MI, bestModel_MI_index = bestModel_MI_index, resample_plot = resample_plot, rate_plot = rate_plot, profile_plot = profile_plot, varImp_plot = varImp_plot)

return(return_list)

}
else {
#Variable Importance
 
varImportance <- do.call(cbind.fill, lapply(model, function(x) varImp(x)$importance))

varImp_agg <- data.frame(Importance = rowMeans(varImportance, na.rm = TRUE), SD = apply(varImportance,1, sd, na.rm = TRUE))
varImp_agg <- varImp_agg[order(-varImp_agg$Importance),]

#Plots
varImportance_tp <- melt(varImportance)
names(varImportance_tp) <- c("Variable", "Imputation", "Importance")

varImp_plot <- ggplot(varImportance_tp, aes(x=fct_reorder(Variable, Importance, fun = median, .desc =TRUE), y=Importance)) + geom_violin(trim=TRUE, scale = "width", aes(fill = fct_reorder(Variable, Importance, fun = median, .desc =TRUE))) + stat_summary(fun.y=median, geom="point", size=2, color="black") + scale_fill_discrete(guide = guide_legend(title = "Variable")) + xlab("Variable") + ggtitle(paste(model[[1]]$method, "Variable Importance", sep=" ")) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_text(size=10, face="bold"), axis.text.x = element_blank(), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))

################# Resample Performance ###############
MI_index <- rep(1:imps, each=5*10)

resamples_MI <- do.call(rbind, lapply(model, function(x) x$resample))
resamples_MI$MI_index <- MI_index

resampleProfile <- aggregate(resamples_MI, list(resamples_MI$MI_index), median)[,-c(1,6)]

bestPerf_accross_MI <- max(unlist(lapply(lapply(model, function(x) x$resample$AUC), function(x) median(x))))
bestModel_accross_MI <- model[[which(unlist(lapply(lapply(model, function(x) x$resample$AUC), function(x) median(x))) == bestPerf_accross_MI)]] 
bestModel_MI_index <- which(unlist(lapply(lapply(model, function(x) x$resample$AUC), function(x) median(x))) == bestPerf_accross_MI)

#Plots
resample_plot <- ggplot(resampleProfile, aes(x="", y=AUC)) + geom_boxplot(colour="blue", fill = "white", outlier.shape=16, outlier.size=2) + ggtitle(paste(model[[1]]$method, "Precision/Recall AUC", sep=" ")) + geom_jitter(width = 0.2) + theme(axis.text.y = element_text(size= 8, face="bold"), axis.title.y = element_text(size=10, face="bold"), axis.title.x = element_blank(), axis.text.x = element_blank(), legend.text=element_text(size=10, face="bold"), plot.title=element_text(size=14, face="bold", hjust = 0.5))

################# Predictions Resamples ###############
trainSet_OOF <- lapply(model, function(y) aggregate(y$pred$painful_neuropathy, list(y$pred$rowIndex), mean)$x)

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

predAgg <- as.data.frame(ifelse(predAgg == 1, model[[1]]$levels[[1]], model[[1]]$levels[[2]]))

majority <- apply(predAgg, 1, function(x) ifelse(sum(x == model[[1]]$levels[[1]], na.rm=TRUE) > sum(x == model[[1]]$levels[[2]], na.rm=TRUE),  model[[1]]$levels[[1]], model[[1]]$levels[[2]]))

testSet_pred <- factor(majority, levels=c(model[[1]]$levels[[1]], model[[1]]$levels[[2]]))

pred_probs <- list()

for (x in 1:length(model))
{
    for (y in 1:length(model))
    { pred_probs[[y+((x-1)*length(model))]] <- predict(model[[x]], newdata =test_list[[y]][,predictors], type="prob")[,1]
	
	}
	}

predAgg_probs <- as.data.frame(do.call(cbind, lapply(pred_probs, function(x) '['(x))))

predAgg_probs <- rowMeans(predAgg_probs)

predAgg_probs_majority <- as.factor(ifelse(predAgg_probs > 0.5, model[[1]]$levels[[1]], model[[1]]$levels[[2]]))


cMatrix <- confusionMatrix(testSet_pred, test_list[[1]][,outcomeName])
###############################################
#Put values in alist with convenient names and return it to the outside world
return_list <- list(varImpROC = varImp_agg, varImportance = varImportance_tp, resamplePerf = resampleProfile, OOF_pred = trainSet_OOF, testSet_pred = testSet_pred, testSet_probs = predAgg_probs, cMatrix = cMatrix, bestModel = bestModel_accross_MI, bestModel_MI_index = bestModel_MI_index, resample_plot = resample_plot, varImp_plot = varImp_plot)

return(return_list)
}

}



