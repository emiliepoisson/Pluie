library("RclusTool")
library("DTWUMI")
Clust_F1=list()
d=read.csv("Douala/CML_Douala/L1_172.22.204.1P4.1-172.22.204.6P4.1_29A-B.csv")

trapeze <- function(x1,x2,taille){
  q <- matrix(ncol=taille,nrow =taille)
  for (i in 1:taille) {
    for (j in 1:taille) {
      if (abs(i-j)<=x1)
        q[i,j]=1
      else if (abs(i-j)<x1+x2)
        q[i,j]=1+x1/x2-abs(j-i)/x2
      else 
        q[i,j]=0
    }
  }
  return(q)
}

Q=trapeze(1000,1000,6576)

alpha=0.4

for (i in 1:6) {
  data=d[6576*(i-1)+1:6576*i,c(6,7,8,9,10,11)]
  dataimp=DTWUMI_imputation(data,gap_size_threshold=100)
  data2=dataimp$output
  SIMZP <- computeGaussianSimilarityZP(data2)
  SIM2=(SIMZP^alpha)*(Q^(1-alpha))
  RES2= spectralClusteringNg(SIM2,K=12)
  plot(data$TSL_mean[6576*(i-1)+1:6576*i],col=RES2$label,ylim=c(0,8),pch=RES2$label)
  lines(d$Rgage[6576*(i-1)+1:6576*i],col="green")
  Clust_F1=c(Clust_F1,list(RES2$label))
}






