#render the .RData obj into a .txt
currentDir=dirname(rstudioapi::getActiveDocumentContext()$path)

load(paste(currentDir,"Nich_Price_quantitative.RData",sep="/"))

str(Nich_Price_quantitative)

Nich_Price_quantitative=Nich_Price_quantitative[order(as.numeric(rownames(Nich_Price_quantitative))),]

write.table(Nich_Price_quantitative,file=paste(currentDir,"Nich_Price_quantitative.txt",sep="/"),sep="\t",quote=F)
