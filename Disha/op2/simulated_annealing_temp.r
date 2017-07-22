#initial.guess<-c(100,5,0.5,1.0,1.5,2.0,-0.5,-1.0,-2.0,-5,-100)
MIN<-c()
#for (j in 1:length(initial.guess)){
			x<-c()
			x[1]<-5;x[2]<-4
			
			func<-function(x){ (1-x[1])^2 + 100*(x[2]-x[1]^2)^2}

			MIN[1]<-func(x)
			optimum<-MIN[1]
			N<-10000
			h<-0.05

			#choose initial temp
			accepted<-c()
			loop=1
			repeat{
			temp=1000
			for(j in 1:100){
					r<-runif(1)
					x.n<-c()
					   if(r<0.5){x.n[1]<-x[1]-(h*runif(1))}
					   if(r>=0.5){x.n[1]<-x[1]+(h*runif(1))}
					   if(r<0.5){x.n[2]<-x[2]-(h*runif(1))}
					   if(r>=0.5){x.n[2]<-x[2]+(h*runif(1))}
					   MIN[j+1]<-MIN.n<-func(x.n)
						  if(MIN.n<=optimum){x<-x.n;optimum<-MIN.n;accepted[loop]<-1;loop=loop+1}
						  if(MIN.n>optimum){     if(r<exp(-(func(x.n) - func(x))/temp ) ){x<-x.n;optimum<-MIN.n;accepted[loop]<-1;loop=loop+1}     }
			
			}
			if (sum(accepted)/100 <0.95){temp=temp+(0.5*temp)}
			else{break}
			}

			rm(MIN);MIN<-c()
			#temp<-c(1000,100,10,0.1)
			for(i in 1:100)
			{
			 		for(sims in 1:N){
					r<-runif(1)
					x.n<-c()
					   if(r<0.5){x.n[1]<-x[1]-(h*runif(1))}
					   if(r>=0.5){x.n[1]<-x[1]+(h*runif(1))}
					   if(r<0.5){x.n[2]<-x[2]-(h*runif(1))}
					   if(r>=0.5){x.n[2]<-x[2]+(h*runif(1))}
					   MIN[sims+1]<-MIN.n<-func(x.n)
						  if(MIN.n<=optimum){x<-x.n;optimum<-MIN.n}
						  if(MIN.n>optimum){     if(r<exp(-(func(x.n) - func(x))/temp ) ){x<-x.n;optimum<-MIN.n}     }
					}
			
			    
			}
#}

print(MIN)



