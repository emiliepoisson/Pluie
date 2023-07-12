library("RclusTool")
library(scatterplot3d)
library("DTWUMI")

data=rio::import("Pluie.csv")
plot(x=data$times,y=data$TSL_mean,type = 'l',col='red',ylim = c(-60,100))
lines(x=data$times,y=data$RSL_mean,col='blue')
lines(x=data$times,y=data$TSL_mean-data$RSL_mean,col='yellow')
lines(x=data$times,y=data$Rgage,col='green')
query=data$Rgage

index=c(1:4,6:14)
data2=data[,index]

data3= DTWUMI_imputation(data2,gap_size_threshold=381)
data4=data3$output

SIMZP <- computeGaussianSimilarityZP(data4$TSL_mean)

RESZP <- spectralClusteringNg(SIMZP, K=10)

plot(data$Rgage,col=RESZP$label,main = "Rgage SC")
plot(data$TSL_mean,col=RESZP$label,main = "TSL_mean SC")
plot(data$RSL_mean,col=RESZP$label,main = "RSL_mean SC")

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
triangle <- function(x,taille){
  q <- matrix(ncol=taille,nrow =taille)
  for (i in 1:taille) {
    for (j in 1:taille) {
      if (i==j)
        q[i,j]=1
      else if (abs(i-j)<x)
        q[i,j]=1-abs(j-i)/x
      else 
        q[i,j]=0
    }
  }
  return(q)
}

Q=triangle(100,576)

alpha=0.4

SIM2=SIMZP^(alpha)*Q^(1-alpha)

RES2 <- spectralClusteringNg(SIM2, K=12)

plot(x=data$times,y=data$TSL_mean,col=RES2$label,pch=RES2$label,main = "TSL_mean")
legend("topleft", legend=c(1:12),bty="o",col=c(1:12),pch=c(1:12), cex=0.8)
#plot(x = data$times,y=data$RSL_mean,col=RES2$label,pch=RES2$label,main = "RSL_mean")