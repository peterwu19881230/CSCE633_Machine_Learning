# render the .RData obj(s) into .txt(s)
currentDir=dirname(rstudioapi::getActiveDocumentContext()$path)


## phenotpye data
load(paste(currentDir,"Nich_Price_quantitative.RData",sep="/"))
str(Nich_Price_quantitative)
Nich_Price_quantitative=Nich_Price_quantitative[order(as.numeric(rownames(Nich_Price_quantitative))),]
write.table(Nich_Price_quantitative,file=paste(currentDir,"Nich_Price_quantitative.csv",sep="/"),sep=",",quote=F,row.names = T,col.names = F)

load("Data/fuhrer_quantitative.RData")

#I want to upload to github so I have to partition if I don't want to use git large file system
fuhrer_quantitative_part1=fuhrer_quantitative[,1:1506]
fuhrer_quantitative_part2=fuhrer_quantitative[,1507:3012]
fuhrer_quantitative_part3=fuhrer_quantitative[,3013:4518]
fuhrer_quantitative_part4=fuhrer_quantitative[,4519:6024]
fuhrer_quantitative_part5=fuhrer_quantitative[,6025:7534]

file_names=paste0("fuhrer_quantitative_part",1:5,".csv")
files=list(fuhrer_quantitative_part1,fuhrer_quantitative_part2,fuhrer_quantitative_part3,fuhrer_quantitative_part4,fuhrer_quantitative_part5)

for (i in seq(file_names)){
  dat=files[[i]]
  write.table(dat,file=paste(currentDir,file_names[i],sep="/"),sep=",",quote=F,row.names = F,col.names = F)
}




## annotations
load(paste(currentDir,"annot_dummies.RData",sep="/"))
str(annot_dummies)
annot_dummies=ifelse(annot_dummies==T,1,0)
write.csv(annot_dummies,file=paste(currentDir,"annot_dummies.csv",sep="/"),quote=F,row.names = T)


