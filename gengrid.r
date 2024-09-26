library(statmod)
nx =30
nyw = 30
sigmU = 0.023
grid=gauss.quad(nyw, kind = "hermite", alpha = 0, beta = 0)
#xx=seq(-sqrt(3) -2, sqrt(3) -2, length.out = 15)
xxo = rbeta(10000, 2, 2)#rnorm(10000, -1, 1)###
U =1.732* sigmU * runif(10000, -1, 1) #rnorm(10000, 0, sigmU)# 
w = xxo + U
data=read.csv('CSFPET.csv', header = F)
data = data[, -1]
w = data[, 2]# - min(data[, 2])
mx= mean(w)
sigmx = sqrt((var(w)- sigmU**2))
xx =seq(max(mx-2* sigmx), min(mx + 6* sigmx, max(w)), length.out = nx)#seq(min(w), max(w), length.out = nx)#
#xxo=quantile(xxo, seq(0, 1, length.out = nx))#seq(0, 1, length.out = nx)#
#xx = (xxo) *2 * 1.732 -1.732
wx =dnorm(xx, mx, sigmx)/sum(dnorm(xx, mx, sigmx ))#rep(1/nx, nx) #dbeta(xxo, 2, 2)/sum(dbeta(xxo, 2, 2))#rep(1/nx, nx)###rep(1/nx, nx)##
gridnodes=expand.grid(xx, grid$nodes, grid$nodes)
gridw=expand.grid(wx, grid$weights, grid$weights)
library(reticulate)
np=import('numpy')
np$savez('gridbigroot3', gridnodes = gridnodes, gridw = gridw)