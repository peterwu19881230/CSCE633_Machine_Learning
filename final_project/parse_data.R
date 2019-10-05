# render the .RData obj(s) into .txt(s)
currentDir=dirname(rstudioapi::getActiveDocumentContext()$path)


## phenotpye data
load(paste(currentDir,"Nich_Price_quantitative.RData",sep="/"))
str(Nich_Price_quantitative)
Nich_Price_quantitative=Nich_Price_quantitative[order(as.numeric(rownames(Nich_Price_quantitative))),]
write.table(Nich_Price_quantitative,file=paste(currentDir,"Nich_Price_quantitative.csv",sep="/"),sep=",",quote=F,row.names = T,col.names = F)


## annotations
load(paste(currentDir,"annot_dummies.RData",sep="/"))
str(annot_dummies)
annot_dummies=ifelse(annot_dummies==T,1,0)
write.csv(annot_dummies,file=paste(currentDir,"annot_dummies.csv",sep="/"),quote=F,row.names = T)


