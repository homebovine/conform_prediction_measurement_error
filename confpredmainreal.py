from confpredreal import *    
import numpy as np
import torch
import os
import seaborn
import pandas as pd
os.system('R CMD BATCH gengrid.r')
def regfun(x, z, b, a):
    xx = torch.cat((x, x**2), dim = 1)
    m = 1
    if m == 1:
        Lres = xx@b + z@a
    elif m ==2:
        Lres = torch.sin(xx@b+ z@a)
    elif m ==3:
        Lres =torch.exp(-(xx@b+ z@a)**2)
    else:
        Lres =torch.exp(-(xx@b+ z@a))
    return Lres
def dregfun(x,z, b, a):
    xx = torch.cat((x, x**2), dim = 1)
    m = 1
    if m == 1:
        Lres =1
    elif m ==2:
        Lres = torch.cos(xx@b+ z@a)
    elif m ==3:
        Lres =-torch.exp(-(xx@b+ z@a)**2) * 2*(xx@b+ z@a)
    else:
        Lres =-torch.exp(-(xx@b+ z@a))
    return Lres





nw = 30
ny =30
nyw = nw* ny
nx = 30
nywx = nw* ny * nx
q = 5
temp=np.load('gridbigroot3.npz')
gridweight=temp['gridw']
gridnodes=temp['gridnodes']
device=torch.device('cuda')
gridweight = torch.tensor(gridweight)
gridnodes = torch.tensor(gridnodes)
gridy = gridnodes[:, 1].reshape((-1, 1)).to(device)    
gridw = gridnodes[:, 2].reshape((-1, 1)).to(device)    
gridx = gridnodes[:, 0].reshape((-1, 1)).to(device)  
#gridx =(( gridx)+2) * 1.5 -2
#gridx = gridx - 2
xx = (gridx[:nx]).to(device)
wygrid = gridweight[:, 1].reshape((-1, 1)).to(device)      
wwgrid = gridweight[:, 2].reshape((-1, 1)).to(device)      
wwygrid = wygrid * wwgrid
wxgrid = gridweight[:, 0].reshape((-1, 1)).to(device)      
wx = gridweight[:nx, 0].to(device)
gridz1 = torch.ones((nywx, 2)).double().to(device)
gridz0 = gridz1.clone()
gridz0[:, 0] = 0
gridz0 = gridz0.to(device)
sig2eps = 0.198**2#0.011 example 1 ##0.235 example 2
sig2U =0.023**2 ##0.33 example 1 ##0.023 example 2
trueb = torch.tensor((2.504138e-03, 1)).to(device) 
trueb = trueb.reshape((2, 1)).double().to(device) 
truea = torch.tensor((-1.399803e-01, -7.159659e-05, 1.611410)).to(device)
truea = truea.reshape((3, 1)).double().to(device)
nsim = 100
upperm = torch.zeros(nsim)
lowerm = torch.zeros(nsim)
alpha = 0.1
tol =1e-12
xatol = 1e-12
maxitr = 500
sitr = 0 
factor =1
torch.cuda.empty_cache() 
zetam =  np.zeros((nsim))
xx1 = torch.tensor(np.linspace(start=0, stop=1, num=30000))
xx1 = xx#((1.732 * 2 *xx1 -1.732)).reshape([-1, 1]).double().to(device)
nx1 = nx#len(xx1)
wx1 = wx#np.ones(len(xx1)) * 1/nx1
Wx1 =np.linspace(wx.min().cpu().numpy(), wx.max().cpu().numpy(), nx1)#
wx1 =wx# torch.tensor(wx1).double().to(device)
covXm = torch.zeros((q, q, nsim))

data = np.genfromtxt("CSFPET1.csv",delimiter=",")
data= data[:, 1:]
data[:, 3]= np.log(data[:, 3])
n3 = 233
n = data.shape[0]-n3
sitr = 0
filenm = 'realdataCP2v2'
res = torch.load(filenm)
cpmcon = res['cpmcon']
uppermcon = res['uppermcon']
lowermcon = res['lowermcon']
rescon = res['rescon']
cpmnon = res['cpmnon']
uppermnon = res['uppermnon']
lowermnon = res['lowermnon']
resnon = res['resnon']
cpm2 = res['cpm2']
upperm2 = res['upperm2']
lowerm2 = res['lowerm2']
res2 = res['res2']
cpmnonker = res['cpmnonker']
uppermnonker = res['uppermnonker']
lowermnonker = res['lowermnonker']
cpmkersemi = res['cpmkersemi']
uppermkersemi = res['uppermkersemi']
lowermkersemi = res['lowermkersemi']
cpm4 = res['cpm4']
upperm4 = res['upperm4']
lowerm4 = res['lowerm4']
res4 = res['res4']
npvnon = res['npvnon']
npvnonker = res['npvnonker']
npvkersemi = res['npvkersemi']
npv4 = res['npv4']
npv2 = res['npv2']
npvcon = res['npvcon']
estppvnon = res['estppvcon']
estppvnonker = res['estppvnonker']
estppvkersemi = res['estppvkersemi']
estppv4 = res['estppv4']
estppv2 = res['estppv2']
estppvcon = res['estppvnon']


sitr = 0
uppermnon = torch.zeros(nsim)
zetamnon = torch.zeros(nsim)
cpmnon = torch.zeros(nsim)
lowermnon = torch.zeros(nsim)
ppvnon = np.zeros(nsim)
npvnon= np.zeros(nsim)
estppvnon= np.zeros(nsim)
estnpvnon= np.zeros(nsim)
nbetnon = np.zeros(nsim)
resnon = np.zeros((nsim, q))
while sitr <nsim:
    #try:
    torch.manual_seed(sitr)
    np.random.seed(sitr)
    cpmnon[sitr], uppermnon[sitr], lowermnon[sitr], resnon[sitr, :], zetamnon[sitr], ppvnon[sitr], npvnon[sitr], estppvnon[sitr], estnpvnon[sitr], nbetnon[sitr]=simufunoerror(data,trueb* factor, truea * factor, n, 2, n3, xx, xx1, wx, wx1,  nx, nx1, ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol, xatol,  alpha, regfun, dregfun, device, 1e-6, 0.99, 0,1,  factor, 0.1,eygwwrg,  True)
    print(cpmnon[sitr])
    print(npvnon[sitr])
    sitr = sitr + 1
    #except:
        #continue
trueb = torch.from_numpy(resnon.mean(0)[:2].reshape([-1, 1])).to(device)
truea = torch.from_numpy(resnon.mean(0)[2:].reshape([-1, 1])).to(device)
sitr = 0
cpmnonker = torch.zeros(nsim).to(device)
uppermnonker = torch.zeros(nsim).to(device)
lowermnonker = torch.zeros(nsim).to(device)
resnonker = np.zeros((nsim, q))
zetamnonker = np.zeros((nsim))
ppvnonker = np.zeros(nsim)
npvnonker= np.zeros(nsim)
estppvnonker= np.zeros(nsim)
estnpvnonker= np.zeros(nsim)
nbetnonker = np.zeros(nsim)

while sitr <nsim:
    #try:
    torch.manual_seed(sitr)
    np.random.seed(sitr)
    cpmnonker[sitr], uppermnonker[sitr], lowermnonker[sitr], resnonker[sitr, :], zetamnonker[sitr], ppvnonker[sitr], npvnonker[sitr], estppvnonker[sitr], estnpvnonker[sitr], nbetnonker[sitr]=simufunoerror(data,trueb* factor, truea * factor,n, 2, n3, xx, xx1, wx, wx1,  nx, nx1, ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol, xatol,  alpha, regfun, dregfun, device, 1e-6, 0.99, 0,1,  factor, 0.1,eygwker,  True)
    print(cpmnonker[sitr])
    sitr = sitr + 1
    #except:
        #continue
init = ((uppermnonker-lowermnonker)/2).cpu().numpy()
sitr = 0
cpmkersemi = torch.zeros(nsim).to(device)
uppermkersemi = torch.zeros(nsim).to(device)
lowermkersemi = torch.zeros(nsim).to(device)
reskersemi = np.zeros((nsim, q))
zetamkersemi = np.zeros((nsim))
ppvkersemi = np.zeros(nsim)
npvkersemi= np.zeros(nsim)
estppvkersemi= np.zeros(nsim)
estnpvkersemi= np.zeros(nsim)
nbetkersemi = np.zeros(nsim)
while sitr <nsim:
    #try:
    torch.manual_seed(sitr)
    np.random.seed(sitr)
    cpmkersemi[sitr], uppermkersemi[sitr], lowermkersemi[sitr],  zetamkersemi[sitr], ppvkersemi[sitr], npvkersemi[sitr], estppvkersemi[sitr], estnpvkersemi[sitr], nbetkersemi[sitr]=simufunker(data, trueb* factor, truea * factor, n, 2, n3, xx, xx1, wx, wx1,  nx, nx1, ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol, xatol,  alpha, regfun, dregfun, device, 1e-6, 0.99,np.mean(init),   factor, 0.1,eygwker,  True)#np.median(init)
    print(cpmkersemi[sitr])
    print(zetamkersemi[sitr])
    sitr = sitr + 1
    #except:
        #continue

init = ((uppermnon-lowermnon)/2).cpu().numpy()
sitr = 0
cpm4 = torch.zeros(nsim).to(device)
upperm4 = torch.zeros(nsim).to(device)
lowerm4 = torch.zeros(nsim).to(device)
res4 = np.zeros((nsim, q))
zetam4 = np.zeros((nsim))
ppv4 = np.zeros(nsim)
npv4= np.zeros(nsim)
estppv4= np.zeros(nsim)
estnpv4= np.zeros(nsim)
nbet4 = np.zeros(nsim)
while sitr <nsim:
    #try:
    torch.manual_seed(sitr)
    np.random.seed(sitr)
    cpm4[sitr], upperm4[sitr], lowerm4[sitr], res4[sitr, :], zetam4[sitr],  ppv4[sitr], npv4[sitr], estppv4[sitr], estnpv4[sitr], nbet4[sitr]=simusemiwrongcon(data, trueb* factor, truea * factor,  n,2, n3, xx, xx1, wx, wx1,  nx, nx1, ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol, xatol,  alpha, regfun, dregfun, device, 1e-6, 0.99, np.median(init),  factor,0.1,eygwwrg,  True)
    print(cpm4[sitr])
    print(sitr)
    print(estppv4[sitr])
    print(npv4[sitr])
    sitr = sitr + 1
    #except:
     #   continuewwwxe
trueb = torch.from_numpy(res4.mean(0)[:2].reshape([-1, 1])).to(device)
truea = torch.from_numpy(res4.mean(0)[2:].reshape([-1, 1])).to(device)
sitr = 0
cpm2 = torch.zeros(nsim).to(device)
upperm2 = torch.zeros(nsim).to(device)
lowerm2 = torch.zeros(nsim).to(device)
res2 = np.zeros((nsim, q))
zetam2 = torch.zeros(nsim).to(device)
ppv2 = np.zeros(nsim)
npv2= np.zeros(nsim)
estppv2= np.zeros(nsim)
estnpv2= np.zeros(nsim)
nbet2 = np.zeros(nsim)
while sitr <nsim:
    torch.manual_seed(sitr)
    np.random.seed(sitr)
    cpm2[sitr], upperm2[sitr], lowerm2[sitr], res2[sitr, :], zetam2[sitr], ppv2[sitr], npv2[sitr], estppv2[sitr], estnpv2[sitr], nbet2[sitr]=simusemiestonly(data, trueb* factor, truea * factor, n, 2, n3, xx, xx1, wx, wx1,  nx, nx1, ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol, xatol,  alpha, regfun, dregfun, device, 1e-6, 0.99, 0,1,  factor, 0.1,eygw,  True)
    print(cpm2[sitr])
    print(npv2[sitr])
    print(estppv2[sitr])
    sitr = sitr + 1
    #except:
     #   continueww
initcon = ((upperm4-lowerm4)/2).cpu().numpy()
sitr = 0
cpmcon = torch.zeros(nsim).to(device)
uppermcon = torch.zeros(nsim).to(device)
lowermcon = torch.zeros(nsim).to(device)
rescon = np.zeros((nsim, q))
zetamcon = np.zeros((nsim))
ppvcon = np.zeros(nsim)
npvcon= np.zeros(nsim)
estppvcon= np.zeros(nsim)
estnpvcon= np.zeros(nsim)
nbetcon = np.zeros(nsim)
while sitr <nsim:
    torch.manual_seed(sitr)
    np.random.seed(sitr)
    init=np.random.uniform(size = 1)
    cpmcon[sitr], uppermcon[sitr], lowermcon[sitr], rescon[sitr, :], zetamcon[sitr], ppvcon[sitr], npvcon[sitr], estppvcon[sitr], estnpvcon[sitr], nbetcon[sitr]=simusemifuncon(data, trueb * factor, truea* factor, n, 2, n3, xx, xx1, wx, wx1,  nx, nx1, ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol, xatol,  alpha, regfun, dregfun, device, 1e-6, 0.99,initcon[sitr],  factor, 0.1, eygw, 1,   True)#np.mean(initcon) * 1initcon[sitr]
    print(cpmcon[sitr])
    print(sitr)
    print(zetamcon[sitr])
    print(initcon[sitr])
    print(npvcon[sitr])
    print(npv2[sitr])
    sitr = sitr + 1
    #except:
     #   continueww

##note sin 100 for 5sigmx need to redo. w
def mad(x):
    res = np.abs((x-np.mean(x))).mean()
    return res

M1cp= cpmcon.mean().cpu().numpy().round(3)
M1cpstd= cpmcon.std().cpu().numpy().round(3)
M1L = (uppermcon-lowermcon).mean().cpu().numpy().round(3)
M1Lstd= (uppermcon-lowermcon).std().cpu().numpy().round(3)

M1kercp= cpmkersemi.mean().cpu().numpy().round(3)
M1kercpstd= cpmkersemi.std().cpu().numpy().round(3)
M1kerL=(uppermkersemi-lowermkersemi).mean().cpu().numpy().round(3)
M1kerLstd =  (uppermkersemi-lowermkersemi).std().cpu().numpy().round(3)



M1Cfmcp= cpm2.mean().cpu().numpy().round(3)
M1Cfmcpstd= cpm2.std().cpu().numpy().round(3)
M1CfmL=(upperm2-lowerm2).mean().cpu().numpy().round(3)
M1CfmLstd =  (upperm2-lowerm2).std().cpu().numpy().round(3)

M2cp= cpm4.mean().cpu().numpy().round(3)
M2cpstd= cpm4.std().cpu().numpy().round(3)
M2L=(upperm4-lowerm4).mean().cpu().numpy().round(3)
M2Lstd =  (upperm4-lowerm4).std().cpu().numpy().round(3)


M2Cfmcp= cpmnon.mean().cpu().numpy().round(3)
M2Cfmcpstd= cpmnon.std().cpu().numpy().round(3)
M2CfmL=(uppermnon-lowermnon).mean().cpu().numpy().round(3)
M2CfmLstd =  (uppermnon-lowermnon).std().cpu().numpy().round(3)

M3kercp= cpmnonker.mean().cpu().numpy().round(3)
M3kercpstd= cpmnonker.std().cpu().numpy().round(3)
M3kerL=(uppermnonker-lowermnonker).mean().cpu().numpy().round(3)
M3kerLstd =  (uppermnonker-lowermnonker).std().cpu().numpy().round(3)

str(M1cp) + ' (' + str(M1cpstd) + ')'+ '&'+ str(M1Cfmcp) + ' (' + str(M1Cfmcpstd) + ')' + '&'+ str(M1kercp) + ' (' + str(M1kercpstd) + ')'  + '&'+ str(M3kercp) + ' (' + str(M3kercpstd) + ')' + '&'+ str(M2cp) + ' (' + str(M2cpstd) + ')' + '&'+ str(M2Cfmcp) + ' (' + str(M2Cfmcpstd) + ')' +'\\'

print(str(M1L) + ' (' + str(M1Lstd) + ')' + '&'+ str(M1CfmL) + ' (' + str(M1CfmLstd) + ')'+ '&'+ str(M1kerL) + ' (' + str(M1kerLstd) + ')' + '&'+ str(M3kerL) + ' (' + str(M3kerLstd) + ')' + '&'+ str(M2L) + ' (' + str(M2Lstd) + ')' + '&'+ str(M2CfmL) + ' (' + str(M2CfmLstd) + ')'  +'\\\\')
filenm = 'realdataCP2v2'

torch.save({"cpmcon": cpmcon, "uppermcon": uppermcon,  "lowermcon": lowermcon, "cpmnon": cpmnon, "uppermnon": uppermnon,  "lowermnon": lowermnon,"rescon": rescon, "resnon": resnon,  "cpm4": cpm4, "upperm4": upperm4, "lowerm4": lowerm4, "res4": res4, "cpm2": cpm2, "upperm2": upperm2, "lowerm2": lowerm2, "res2": res2,  "sig2eps": sig2eps, "sig2U": sig2U, "trueb": trueb, "truea": truea, "cpmnonker": cpmnonker, "uppermnonker": uppermnonker,  "lowermnonker": lowermnonker, "cpmkersemi": cpmkersemi, "uppermkersemi": uppermkersemi,  "lowermkersemi": lowermkersemi, "npvcon": npvcon, "estppvcon": estppvcon, "npvnon": npvnon, "estppvnon": estppvnon, "npv2": npv2, "estppv2": estppv2, "npv4": npv4, "estppv4": estppv4, "npvnonker": npvnonker, "estppvnonker":estppvnonker, "npvkersemi":npvkersemi, "estppvkersemi": estppvkersemi},  f = filenm)    

#, 'estscorenon': estscorenon, 'upperscorenon':upperscorenon, 'lowerscorenon':lowerscorenon, 'estscorenonker': estscorenonker, 'upperscorenonker':upperscorenonker, 'lowerscorenonker':lowerscorenonker, 'estscorekersemi': estscorekersemi, 'upperscorekersemi':upperscorekersemi, 'lowerscorekersemi':lowerscorekersemi, 'estscore4': estscore4, 'upperscore4':upperscore4, 'lowerscore4':lowerscore4, 'estscore2': estscore2, 'upperscore2':upperscore2, 'lowerscore2':lowerscore2, 'estscorecon': estscorecon, 'upperscorecon':upperscorecon, 'lowerscorecon':lowerscorecon
#mfun = 'sin'
#model = 'm1'
#n = 500


res = torch.load(filenm)
CP =pd.DataFrame({'m1s': res['cpmcon'].cpu().numpy(), 'm1c': res['cpm2'].cpu().numpy(), 'm2s': res['cpmkersemi'].cpu().numpy(), 'm2c': res['cpmnonker'].cpu().numpy(), 'm3s': res['cpm4'].cpu().numpy(), 'm3c': res['cpmnon'].cpu().numpy()})
PI =pd.DataFrame({'m1s': (res['uppermcon'] - res['lowermcon']).cpu().numpy(), 'm1c': (res['upperm2']-res['lowerm2']).cpu().numpy(), 'm2s': (res['uppermkersemi']-res['lowermkersemi']).cpu().numpy(), 'm2c': (res['uppermnonker'] - res['lowermnonker']).cpu().numpy(), 'm3s': (res['upperm4'] - res['lowerm4']).cpu().numpy(), 'm3c': (res['uppermnon']- res['lowermnon']).cpu().numpy()})


AUClower =pd.DataFrame({'m1s': (res['npvcon']), 'm1c': (res['npv2']), 'm2s': (res['npvkersemi']), 'm2c': (res['npvnonker']), 'm3s': (res['npv4']), 'm3c': (res['npvnon'])})

AUCest =pd.DataFrame({'m1s': (res['estppvcon']), 'm1c': (res['estppv2']), 'm2s': (res['estppvkersemi']), 'm2c': (res['estppvnonker']), 'm3s': (res['estppv4']), 'm3c': (res['estppvnon'])})


CP.to_csv('CPrealexp2v2' + '.csv')
PI.to_csv('PIrealexp2v2'+ '.csv')

AUClower.to_csv('AUCnpvexp2v2' + '.csv')
AUCest.to_csv('AUCestexp2v2'+ '.csv')



fig=CPbox.boxplot(x= ['m1s', 'm1c', 'm2s', 'm2c', 'm3s', 'm3c'])
CPbox=seaborn.boxplot(data = CP.loc[:, ['m1s', 'm1c', 'm2s', 'm2c', 'm3s', 'm3c']])
fig =CPbox.get_figure()
fig.savefig("CPboxm1_100.png") 

PIbox=seaborn.boxplot(PI)
fig =PIbox.get_figure()
fig.savefig("PIboxm1_100.png") 

while sitr <nsim:
    #try:
    cpm[sitr], upperm[sitr], lowerm[sitr], res[sitr, :], zetam[sitr], covXm[:, :, sitr]=simusemifun(truea, trueb,100, 100, 1000, xx, xx1, wx, wx1,  nx, nx1, ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, s5Big2U, maxitr,tol, xatol,  alpha, regfun, dregfun, device, 1e-6, 0.99, 0,1,  factor, 100, True)
    print(cpm[sitr])
    print(sitr)
    sitr = sitr + 1
    #except:
     #   continueww


cpm1 = torch.zeros(nsim).to(device)
upperm1 = torch.zeros(nsim).to(device)
lowerm1 = torch.zeros(nsim).to(device)
res1 = np.zeros((nsim, q))
zetam1 = np.zeros((nsim))
while sitr <nsim:
    #try:
    cpm1[sitr], upperm1[sitr], lowerm1[sitr], res1[sitr, :], zetam1[sitr]=simusemiwrong(truea, trueb,100, 100, 1000, xx, xx1, wx, wx1,  nx, nx1, ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol, xatol,  alpha, regfun, dregfun, device, 1e-6, 0.99, 0,1,  factor, 2, True)
    print(cpm1[sitr])
    print(sitr)
    sitr = sitr + 1
    #except:
     #   continueww

res = torch.load('expcp')
res['cpm'].mean()
res['cpm'].std()
(res['upperm'] - res['lowerm']).mean()
(res['upperm'] - res['lowerm']).std()


nnx = n * nx
stemp = torch.distributions.Beta(torch.ones(n) * 2, torch.ones(n) * 2)
stp =stemp.sample()
x = ((1.732 * 2 *stp -1.732)).reshape([-1, 1]).double().to(device)
U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
w = x + U
z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
z =z.reshape((n, (2)))
cres = KMeans(m).fit(z)
clusterm = torch.from_numpy(cres.labels_).to(device)
clusterc = torch.from_numpy(cres.cluster_centers_)
z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
clusterc = torch.cat((clusterc, torch.ones([clusterc.shape[0], 1])), 1).to(device)
y = regfun(x, z, trueb, truea) + sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
inib = trueb  * factor
inia = truea *factor
thisb = inib.clone()
bdim = thisb.shape[0]
thisa = inia.clone() 
theta0 = torch.cat((thisb, thisa))
theta0 = theta0.cpu().numpy()
res = optimize.fsolve(rootfuncon,theta0.copy(), args = (bdim, gridy, gridw, gridx, clusterm, clusterc, xx, wx, wwygrid, ny, nw, nx, nyw, nywx, q, sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun, device),xtol = tol, maxfev = maxitr, factor = 0.2)
thisb = torch.tensor(res[:bdim]).double().to(device).reshape([-1, 1])
thisa = torch.tensor(res[bdim:]).double().to(device).reshape([-1, 1])
theta = torch.cat((thisb, thisa))
zetal=np.arange(1, 2.5, 0.01)
fres= np.zeros(len(zetal))
for i in range(len(zetal)):
    fres[i]=rootfunzetacon(zetal[i], theta, bdim, gridy, gridw, gridx, clusterm, clusterc, xx, wx, wwygrid, ny, nw, nx, nyw, nywx,  sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun,eygw, alpha,  device)
import matplotlib.pyplot as plt
plt.plot(zetal[0:], fres[0:])
plt.savefig('temp2.pdf')
plt.close()


plt.scatter(cpm, zetam)
plt.savefig('temp.pdf')
plt.close()
