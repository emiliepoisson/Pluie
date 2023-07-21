library(class)
library("DTWUMI")
#files2=list.files("Douala/CML_Douala",full.names = TRUE)
load(file = "Clust_F1.rdata")
d=read.csv("Douala/CML_Douala/L1_172.22.204.1P4.1-172.22.204.6P4.1_29A-B.csv")
d2=read.csv("Douala/CML_Douala/L1_172.22.206.16P4.4-172.22.206.30P3.1_35A-B.csv")
data=data.frame(d$TSL_mean,d$RSL_mean)
data2=data.frame(d2$TSL_mean,d2$RSL_mean)
Clust_F3=list()
dataimp=DTWUMI_imputation(data,gap_size_threshold=1000)
dataimp3=DTWUMI_imputation(data2,gap_size_threshold=1000)
data=dataimp$output
data2=dataimp3$output

for (i in 1:6) {
  focus=(6576*(i-1)+1):(6576*i)#cet intevalle doit etre conforme avec la taille du fichier
  dapp=data.frame(t=focus,s=data$d.TSL_mean[focus],r=data$d.RSL_mean[focus])
  labelAPP=Clust[focus]
  dtest=data.frame(t=focus,s=data2$d2.TSL_mean[focus],r=data2$d2.RSL_mean[focus])
  predictions=knn(train =dapp,test=dtest,cl=labelAPP,k = 1,prob = TRUE)
  plot(focus,dtest$s,col=predictions,ylim=c(0,16))
  lines(d2$Rgage,col="forestgreen")
  Clust_F3=c(Clust_F3,list(predictions))
}
final_clust=unlist(Clust_F3)






#interval=4001:8000

#lines(d2$Rgage,col="yellow")
#plot(d2$RSL_mean,col=predictions,ylim = c(-70,16))

#plot(focus,dapp$s,col=labelAPP,pch=labelAPP)
#predictionsAPP=knn(train=dapp,test=dapp,cl=labelAPP,k = 1,prob = TRUE)
#table(predictionsAPP,labelAPP)
#plot(focus,dapp$s/max(dapp$s,na.rm=T),col=predictionsAPP,ylim=c(0,1.5))
#lines(focus,0.2+d$Rgage[focus]/max(d$Rgage[focus],na.rm=T),col="forestgreen")
#title("kppv - data train")
