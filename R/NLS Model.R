library(tuneR)
library(stats)

wave <- readWave("~/HumPi/Audio/2016-01-07_08-45-40.wav")
a = 0.94
b= 50*2*pi 
c = 4

subsamplingRate = 1
audioLength = length(wave@left)/wave@samp.rate #seconds
fittingDuration = 2 #seconds of data the sin gets fit. Needs to be >= 1
#holds the measurments. Every row is a individual second.
frequencys = matrix(nrow = length(wave@left)/wave@samp.rate-fittingDuration, ncol=5)
colnames(frequencys) <- c("Frequency","amplitude","Loss", "Iterations", "Converged_To")
j=0
for (j in 1:(nrow(frequencys)-1)) {
  y = wave@left[(j*wave@samp.rate/subsamplingRate+1):(j*wave@samp.rate/subsamplingRate+wave@samp.rate/subsamplingRate*fittingDuration)]
  df =data.frame(y/max(abs(y)),(1:length(y))/(wave@samp.rate/subsamplingRate))
  colnames(df) = c("y","x")

  #build the model with the NLS package. 
  model<-nls(y ~ I(a*sin(b*x+c)), data = df, start = list(a=a, b=b, c= c))
  
  a=coef(model)[1]
  b=coef(model)[2]
  c=coef(model)[3]
  L=sum(residuals(model)^2)
  frequencys[j+1,1] <- b/(2*pi)
  frequencys[j+1,2] <- a
  frequencys[j+1,3] <- L
  frequencys[j+1,4] <- model$convInfo$finIter
  frequencys[j+1,5] <- model$convInfo$finTol
  print(paste(j,paste(frequencys[j+1,], collapse = ", ")))
}

plot(frequencys[,1], type="l", col="grey", main="Frequency of audio", ylab="Frequency")


hist(residuals(model), breaks=150)
