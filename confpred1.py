import numpy as np
import torch
from scipy import optimize
import functools
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
def sbetafm(y, x, z, b, a, sig2eps, regfun, dregfun):
    score = ((y -regfun(x, z, b, a))/sig2eps) *dregfun(x, z, b, a)*  torch.cat((x, x**2,    z), dim = 1)
    return score
def simU(n, sig2U, device):
    #U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    U = 1.732*  (sig2U)**(1/2) * ((torch.rand((n, 1)) -1/2) * 2).double().to(device)
    return U
def simX(n, device):
    #stemp = torch.distributions.Beta(torch.ones(n) * 2, torch.ones(n) * 2)
    #stp =stemp.sample()
    #x = ((1.732 * 2 *stp -1.732)).reshape([-1, 1]).double().to(device)
    x =(torch.randn((n, 1))-1).double().to(device)
    return x
def simeps(n, sig2eps, device):
    #eps = sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    eps = torch.from_numpy(np.random.standard_t(3, (n, 1))).to(device)* sig2eps**(1/2) * 0.57735
    return eps
def sbetafnon(y, x, z, b, a, sig2eps, regfun,dregfun,  device):
    score = ((y -regfun(x, z, b, a))/sig2eps) 
    return score

def qywz(y, x, w, z, b, a, sig2eps, sig2U, regfun):
    pyz = torch.exp(-(y - regfun(x, z, b, a))**2/(2 * sig2eps)) * ((2 * sig2eps * 3.1415)**(-1/2))
    pwx = torch.exp(-(w - x)**2/(2 * sig2U)) *  ((2 * sig2U * 3.1415) **(-1/2))
    qywzres = pyz * pwx
    return qywzres

def qwz(x, w, z, b, a, sig2eps, sig2U, regfun):
    pwx = torch.exp(-(w - x)**2/(2 * sig2U)) *  ((2 * sig2U * 3.1415) **(-1/2))
    qywzres = pwx
    return qywzres

def Afun(gridy, gridw, gridx, gridz, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,b, a,   sig2eps, sig2U, regfun, dregfun, device):
    A = torch.zeros((nx, nx)).to(device)
    AAA=torch.ones(nywx, 1).to(device)
    for i in range(nx):
        xi = (AAA *xx[i]).double()
        iy = gridy *1.414 * sig2eps**(1/2) + regfun(xi, gridz, b, a)
        iw = gridw * 1.414 * sig2U**(1/2)+ xi
        qm =qywz(iy, gridx, iw, gridz, b, a, sig2eps, sig2U, regfun)
        qmc =torch.reshape(qm, (nyw, nx))
        qc = (qmc@wx).reshape((nyw, 1))
        iix =torch.arange(i, nywx, nx).to(device)
        for j in range(nx):
            jix =torch.arange(j, nywx, nx).to(device)
            A[i, j] = torch.sum(( qm[jix, 0] * wx[j]* wwygrid[iix, 0])/qc[:, 0])/torch.sum((  wwygrid[iix, :])[:, 0])
    return A

def mrAbfunzeta(i, zc, gridy, gridw, gridx, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,b, a,   sig2eps, sig2U, regfun, dregfun, zeta,eygw, y, w, z,  device):
    gridz=zc[i, :].repeat((nywx)).reshape((nywx, zc.shape[1]))
    AA=Afun(gridy, gridw, gridx, gridz, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,b, a,   sig2eps, sig2U, regfun, dregfun, device)
    bb=bfzeta(gridy, gridw, gridx, gridz, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,b, a,   sig2eps, sig2U, regfun, dregfun, zeta, eygw,y,w, z,  device)
    bbt=bb[:, 0].double()
    mbbt = bbt@wx
    abbt=mbbt -bbt
    az = torch.inverse(AA).double()@abbt
    zz = mbbt
    return az, zz

def mrAbfun(i, z, gridy, gridw, gridx, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,q, b, a,   sig2eps, sig2U, regfun, dregfun,  device):
    gridz=z[i, :].repeat((nywx)).reshape((nywx, z.shape[1]))
    AA=Afun(gridy, gridw, gridx, gridz, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,b, a,   sig2eps, sig2U, regfun, dregfun, device)
    bb= bfun(gridy, gridw, gridx, gridz, xx, wx, wwygrid, ny, nw, nx, nyw,nywx, q, b, a,  sig2eps, sig2U, regfun, dregfun, device)   
    aa = torch.inverse(AA)@bb
    return aa



def bfunnon(gridy, gridw, gridx, gridz, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx, q, b, a,  sig2eps, sig2U,regfun, dregfun,  device):
    bb = torch.zeros((nx, q)).to(device)
    bbb = torch.ones(nywx, 1).to(device)
    for i in range(nx):
        xi = (bbb *xx[i]).double()
        iy = gridy *1.414 * sig2eps**(1/2) + regfun(xi, gridz, b, a)
        iw = gridw * 1.414 * sig2U**(1/2)+ xi
        qm =qywz(iy, gridx, iw, gridz, b, a, sig2eps, sig2U, regfun)
        qmc =torch.reshape(qm, (nyw, nx))
        qc = (qmc@wx).reshape((nyw, 1))
        scoreres =sbetafnon(iy, gridx, gridz, b, a, sig2eps, regfun, dregfun, device)
        scoreresm=scoreres.t().reshape([-1, 1])
        scoreresm = scoreresm.reshape([-1, nx])
        sb = torch.zeros((nyw, q)).double().to(device)
        for col in range(q):
            colin =torch.arange(col*nyw, (col+1) * nyw)
            sb[:, col] = ((scoreresm[colin, :] * qmc) @wx)
        iix =torch.arange(i, ny * nw * nx, nx)
        bb[i, :]= torch.sum((torch.diag((wwygrid[iix, 0]/qc[:, 0])) @sb) , 0)/torch.sum((wwygrid[iix, :])[:, 0])
    return bb

def bfzeta(gridy, gridw, gridx, gridz, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx, b, a,  sig2eps, sig2U,regfun, dregfun, zeta,eygw,y, w, z,  device):
    q=1
    bb = torch.zeros((nx, 1)).to(device)
    bbb = torch.ones(nywx, 1).to(device)
    for i in range(nx):
        xi = (bbb *xx[i]).double()
        iy = gridy *1.414 * sig2eps**(1/2) + regfun(xi, gridz, b, a)
        iw = gridw * 1.414 * sig2U**(1/2)+ xi
        qm =qywz(iy, gridx, iw, gridz, b, a, sig2eps, sig2U, regfun)
        qmc =torch.reshape(qm, (nyw, nx))
        qc = (qmc@wx).reshape((nyw, 1))
        eywzres=eygw(iw, gridx, gridz,  wx, nyw,nx, q, b, a,y, w, z,   sig2eps, sig2U, regfun, dregfun, device)
        iix =torch.arange(i, ny * nw * nx, nx)
        rscore=(torch.abs(iy[iix] - eywzres) <=zeta) * 1.0
        #rscores=vseff(iy, iw, gridx,gridz,az,  xx, wx, n,nx, thisb, thisa, 1,  sig2eps, sig2U, regfun, dregfun, device)[iix, :]
        bb[i, :]= torch.sum((wwygrid[iix, :] * rscore)[:, 0])/torch.sum((wwygrid[iix, :])[:, 0])
    return bb




def bfun(gridy, gridw, gridx, gridz, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx, q,  b, a,  sig2eps, sig2U,regfun, dregfun,  device):
    bb = torch.zeros((nx, q)).to(device)
    bbb = torch.ones(nywx, 1).to(device)
    for i in range(nx):
        xi = (bbb *xx[i]).double()
        iy = gridy *1.414 * sig2eps**(1/2) + regfun(xi, gridz, b, a)
        iw = gridw * 1.414 * sig2U**(1/2)+ xi
        qm =qywz(iy, gridx, iw, gridz, b, a, sig2eps, sig2U, regfun)
        qmc =torch.reshape(qm, (nyw, nx))
        qc = (qmc@wx).reshape((nyw, 1))
        scoreres =sbetafm(iy, gridx, gridz, b, a, sig2eps, regfun, dregfun)
        scoreresm=scoreres.t().reshape([-1, 1])
        scoreresm = scoreresm.reshape([-1, nx])
        sb = torch.zeros((nyw, q)).double().to(device)
        for col in range(q):
            colin =torch.arange(col*nyw, (col+1) * nyw)
            sb[:, col] = ((scoreresm[colin, :] * qmc) @wx)
        iix =torch.arange(i, ny * nw * nx, nx)
        bb[i, :]= torch.sum((torch.diag((wwygrid[iix, :]/qc)[:, 0]) @sb) , 0)/torch.sum((wwygrid[iix, :])[:, 0])
    return bb

def seff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, q, b, a,  sig2eps, sig2U, regfun, dregfun, device):
    qm =qywz(gridyr, gridxr, gridwr, gridzr, b, a, sig2eps, sig2U, regfun)
    qmc =torch.reshape(qm, (n, nx))
    qc = (qmc@wx).reshape((n, 1))
    scoreres =sbetafm(gridyr, gridxr, gridzr, b, a, sig2eps, regfun, dregfun)
    scoreresm=scoreres.t().reshape([-1, 1])
    scoreresm = scoreresm.reshape([-1, nx])
    sb = torch.zeros((n, q)).double().to(device)
    for col in range(q):
        colin =torch.arange(col*n, (col+1) * n)
        temp =az[:, :, col]
        sb[:, col] = (((scoreresm[colin, :]-temp) * qmc) @wx)
    score=torch.mean(sb/qc, 0)
    return score        




def sTseff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, q, b, a,  sig2eps, sig2U, regfun, dregfun, device):
    qm =qywz(gridyr, gridxr, gridwr, gridzr, b, a, sig2eps, sig2U, regfun)
    qmc =torch.reshape(qm, (n, nx))
    qc = (qmc@wx).reshape((n, 1))
    scoreres =sbetafm(gridyr, gridxr, gridzr, b, a, sig2eps, regfun, dregfun)
    scoreresm=scoreres.t().reshape([-1, 1])
    scoreresm = scoreresm.reshape([-1, nx])
    sb = torch.zeros((n, q)).double().to(device)
    for col in range(q):
        colin =torch.arange(col*n, (col+1) * n)
        temp =az[:, :, col]
        sb[:, col] = (((scoreresm[colin, :]-temp) * qmc) @wx)
    score=sb/qc
    sT=score.T@score/n
    return sT

def seffnoerror(y, w, z, n, b, a,  sig2eps, sig2U, regfun, dregfun, device):
    scoreres =sbetafm(y, w, z, b, a, sig2eps, regfun, dregfun)
    score=torch.sum(scoreres, 0)
    return score        



def vseff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, b, a, q,  sig2eps, sig2U, regfun, dregfun, device):
    qm =qywz(gridyr, gridxr, gridwr, gridzr, b, a, sig2eps, sig2U, regfun)
    qmc =torch.reshape(qm, (n, nx))
    qc = (qmc@wx).reshape((n, 1))
    scoreres =sbetafnon(gridyr, gridxr, gridzr, b, a, sig2eps, regfun, dregfun, device)
    scoreresm=scoreres.t().reshape([-1, 1])
    scoreresm = scoreresm.reshape([-1, nx])
    sb = torch.zeros((n, q)).double().to(device)
    for col in range(q):
        colin =torch.arange(col*n, (col+1) * n)
        temp =az[:, :, col]
        sb[:, col] = ((scoreresm[colin, :]-temp) * qmc) @wx
    score=torch.sqrt(torch.sum((sb/qc)**2, 1))
    return score        


def vseffnoerror(y, w, z, n,  b, a,  sig2eps, sig2U, regfun, dregfun, device):
    scoreres =sbetafnon(y, w, z, b, a, sig2eps, regfun, dregfun, device)
    score=torch.sum(scoreres**2, 1)
    return score        


def eygw(gridwr, gridxr,gridzr, wx, n,nx, q, b, a, y, w, z,  sig2eps, sig2U, regfun, dregfun, device):
    EY=regfun(gridxr, gridzr, b, a)
    qm =qwz(gridxr, gridwr, gridzr, b, a, sig2eps, sig2U, regfun)
    qmc =torch.reshape(qm, (n, nx))
    qc = (qmc@wx).reshape((n, 1))
    yqm =EY * qm
    yqmc =torch.reshape(yqm, (n, nx))
    yqc = (yqmc@wx).reshape((n, 1))
    score = yqc/qc
    return score        

def eygwwrg(gridwr, gridxr,gridzr, wx, n,nx, q, b, a, y, w, z,   sig2eps, sig2U, regfun, dregfun, device):
    EY=regfun(gridwr, gridzr, b, a)
    yqm =EY 
    yqmc =torch.reshape(yqm, (n, nx))
    yqc = (yqmc[:, 0]).reshape((n, 1))
    return yqc        
def gker(x, h):
    res= torch.exp(-(x/h)**2/2)#0.75 * (1 - (x/h)**2) * ((x/h).abs() <=1)
    return res
def eygwker(gridwr, gridxr,gridzr, wx, n,nx, q, b, a,y, w, z,   sig2eps, sig2U, regfun, dregfun, device):
    n1 = gridwr.shape[0]
    ns = y.shape[0]
    wz1= torch.cat((gridwr, gridzr[:, :2].T.reshape([-1, 1])), 0).reshape([1, -1, 1])
    wz2 = torch.cat((w, z[:, :2].T.reshape([-1, 1])), 0).reshape([1, -1, 1])
    dism = torch.cdist(wz1, wz2)[0, :, :]
    disw= dism[:n1, :ns]
    disz1= dism[n1:(2* n1), ns:(2*ns)]
    disz2 = dism[(2* n1):(3*n1), (2*ns):(3* ns)]
    kw=gker(disw, w.std()* ns**(-1/5) * 1.06)
    kz1 = gker(disz1, z[:, 0].std()* ns**(-1/5) * 1.06)
    kz2 = gker(disz2, z[:, 1].std()* ns**(-1/5) * 1.06)
    kk=kw * kz1 * kz2
    ygw=(kk @ y)/kk.sum(1).reshape([-1, 1])
    ygw =torch.reshape(ygw, (n, nx))
    yqc = (ygw[:, 0]).reshape((n, 1))
    return yqc        

def psieff(gridyr, gridwr, gridxr,gridzr,az, zz,  xx, wx, n,nx, b, a,  sig2eps, sig2U, regfun, dregfun,alpha,  device):
    q=1
    qm =qywz(gridyr, gridxr, gridwr, gridzr, b, a, sig2eps, sig2U, regfun)
    qmc =torch.reshape(qm, (n, nx))
    qc = (qmc@wx).reshape((n, 1))
    num = ((az[:, :, 0] * qmc)@wx).reshape((n, 1))
    sb =num[:, 0]/qc[:, 0]-zz + 1 - alpha 
    score=torch.mean(sb)
    return score            
        
def simufun(truea, trueb, n1, n2, n3,xx, wx,  nx, ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol,   alpha, regfun, dregfun, device, eta, lr, factor,fct,  disp):
    n = n1
    nnx = n * nx
    stemp = torch.distributions.Beta(torch.ones(n) * 2, torch.ones(n) * 2)
    stp =stemp.sample()
    x = ((1.732 * 2 *stp -1.732)).reshape([-1, 1]).double().to(device)
    U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.bernoulli(torch.ones(n) * 0.8).double()
    z =z.reshape((n, 1))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    y = regfun(x, z, trueb, truea) + sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    inib = trueb  * factor
    inia = truea *factor
    thisb = inib.clone()
    thisa = inia.clone()
    theta0 = torch.cat((thisb, thisa))
    theta0 = theta0.cpu().numpy()
    res = optimize.fsolve(rootfun, theta0[:, 0].copy(), args = (gridy, gridw, gridx, gridz0, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw, nywx, q, sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun, device),xtol = tol, maxfev = maxitr, factor = fct)
    if disp:
       print(res)
    thisb = torch.tensor(res[:1]).double().to(device).reshape([-1, 1])
    thisa = torch.tensor(res[1:]).double().to(device).reshape([-1, 1])
    n =n2
    nnx = n * nx
    stemp = torch.distributions.Beta(torch.ones(n) * 2, torch.ones(n) * 2)
    stp =stemp.sample()
    x = ((1.732 *2 * stp-1.732)).reshape([-1, 1]).double().to(device)
    U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.bernoulli(torch.ones(n) * 0.8).double()
    z =z.reshape((n, 1))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    y = regfun(x, z, trueb, truea) + sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    sampley = y.clone()
    bb0= bfunnon(gridy, gridw, gridx, gridz0, xx, wx, wwygrid, ny, nw, nx, nyw,nywx, 1, thisb, thisa,  sig2eps, sig2U, regfun, dregfun, device )   
    bb1= bfunnon(gridy, gridw, gridx, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw,nywx, 1, thisb, thisa,  sig2eps, sig2U, regfun, dregfun, device)  
    AA0=Afun(gridy, gridw, gridx, gridz0, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,thisb, thisa,   sig2eps, sig2U, regfun, dregfun, device)
    AA1=Afun(gridy, gridw, gridx, gridz1, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,thisb, thisa,   sig2eps, sig2U, regfun, dregfun, device)
    aa0 = torch.inverse(AA0)@bb0
    aa1 = torch.inverse(AA1)@bb1
    gridyr, gridxr= torch.meshgrid( y[:, 0], xx[:, 0])
    gridwr, gridxr= torch.meshgrid( w[:, 0], xx[:, 0])
    gridzr, gridxr= torch.meshgrid( z[:, 0], xx[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridyr = gridyr.reshape([-1, 1])
    gridwr = gridwr.reshape([-1, 1])
    gridzr = torch.cat((gridzr.reshape([-1, 1]), torch.ones((nnx, 1)).double().to(device)), 1)
    az = torch.zeros((n, nx, 1)).double().to(device)
    az[z[:, 0]==1, :, :] = aa1.double() 
    az[z[:, 0]==0, :, :] = aa0.double()
    nscores=vseff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, thisb, thisa, 1,  sig2eps, sig2U, regfun, dregfun, device)
    sscores=torch.sort(nscores)[0]
    index = int((n+ 1) * (1-alpha))
    sm = torch.quantile(sscores, 1-alpha)#sscores[index]
    n = n3
    nnx = n * nx
    stemp = torch.distributions.Beta(torch.ones(1) * 2, torch.ones(1) * 2)
    stp =stemp.sample()
    x = ((1.732 * 2* stp-1.732) * torch.ones(n)).reshape([-1, 1]).double().to(device)
    U = ((torch.randn((1, 1)) * (sig2U)**(1/2)).reshape([-1, 1]) * torch.ones((n, 1))).to(device)
    w = x + U
    z = (torch.bernoulli(torch.ones(1) * 0.8).double()* torch.ones((n, 1)))
    z =z.reshape((n, 1))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    fl = sampley.min()
    cl = sampley.max() 
    y = (torch.arange(fl, cl, (cl-fl)/n).double().reshape([-1, 1])).to(device)
    y = y[0:n3, :]
    gridyr, gridxr= torch.meshgrid( y[:, 0], xx[:, 0])
    gridwr, gridxr= torch.meshgrid( w[:, 0], xx[:, 0])
    gridzr, gridxr= torch.meshgrid( z[:, 0], xx[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridyr = gridyr.reshape([-1, 1])
    gridwr = gridwr.reshape([-1, 1])
    gridzr = torch.cat((gridzr.reshape([-1, 1]), torch.ones((nnx, 1)).double().to(device)), 1)
    az = (torch.zeros((n, nx, q)).double()).to(device)
    az[z[:, 0]==1, :, :] = aa1.double() 
    az[z[:, 0]==0, :, :] = aa0.double()
    nscores=vseff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, thisb, thisa, 1,  sig2eps, sig2U, regfun, dregfun, device)
    lower = torch.min(y[torch.where(nscores<=sm)]).double()
    upper = torch.max(y[torch.where(nscores<=sm)]).double()
    y = regfun(x, z, trueb, truea) + (sig2eps**(1/2) * torch.randn((n, 1))).to(device)
    cpm = torch.mean(((y[:, 0]<=upper) & (y[:, 0]>=lower)).double())
    return cpm, upper, lower, res




def simufunoerror(truea, trueb, inib, inia, n1, m, n3,xx,xx1,  wx, wx1,  nx,nx1,  ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol,xatol,   alpha, regfun, dregfun, device, eta, lr,lw,up,  factor,fct, eygw,  disp):
    n = n1
    x =simX(n, device)#(torch.randn((n, 1)) -1).double().to(device)# ((1.732 *2 * stp-1.732)).reshape([-1, 1]).double().to(device)
    U = simU(n, sig2U, device)#(torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps
    trainy = y.clone()
    trainw = w.clone()
    trainz = z.clone()
    bdim = trueb.shape[0]
    thisb = inib.clone()
    thisa = inia.clone()
    theta0 = torch.cat((thisb, thisa))
    theta0 = theta0.cpu().numpy()
    res = optimize.fsolve(rootfunnon, theta0[:, 0], args = (bdim, y, w, z, n, sig2eps, sig2U, regfun, dregfun, device), maxfev=maxitr, xtol = tol, factor = fct)
    if disp:
        print(res)
    thisb = torch.tensor(res[:bdim]).double().to(device).reshape([-1, 1])
    thisa = torch.tensor(res[bdim:]).double().to(device).reshape([-1, 1])
    n =n1
    nnx = n * nx
    x =simX(n, device)#(torch.randn((n, 1)) -1).double().to(device)# ((1.732 * 2 * stp-1.732)).reshape([-1, 1]).double().to(device)
    U =simU(n, sig2U, device)#U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps#sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    #y = regfun(x, z, trueb, truea) + sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    gridyr, _= torch.meshgrid(y[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    EY=eygw(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa, trainy, trainw, trainz,  sig2eps, sig2U, regfun, dregfun, device)
    nscores=torch.abs(y-EY)
    sscores=torch.sort(nscores[:, 0])[0]
    index = int((n+ 1) * (1-alpha))
    zeta = sscores[index].clone()
    n = n3
    nnx = n * nx1
    sampley = y.clone()
    cl = sampley.max()+ 3
    fl = sampley.min()-3
    x =simX(n, device)#(torch.randn((n, 1)) -1).double().to(device)#((1.732 * 2* stp-1.732) * torch.ones(n)).reshape([-1, 1]).double().to(device)
    U =simU(n, sig2U, device)#U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps#sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    #y = regfun(x, z, trueb, truea) + (sig2eps**(1/2) * torch.randn((n, 1))).to(device)
    sampley = (torch.arange(fl, cl, (cl-fl)/n).double().reshape([-1, 1])).to(device)
    sampley = sampley[0:n, :]
    gridyr, _= torch.meshgrid(sampley[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    lower = torch.ones((n, 1)).to(device)
    upper = torch.zeros((n, 1)).to(device)
    EY=eygw(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa, trainy, trainw, trainz,  sig2eps, sig2U, regfun, dregfun, device)
    lower= EY-zeta
    upper = EY + zeta
    print(y.mean())
    cpm = torch.mean((((y-upper)<=0) & ((y-lower)>=0)).double())
    return cpm, upper.mean(), lower.mean(), res, zeta



def simufunker(truea, trueb, inib, inia, n1, m, n3,xx,xx1,  wx, wx1,  nx,nx1,  ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol,xatol,   alpha, regfun, dregfun, device, eta, lr,x0,   factor,fct, eygw,  disp):
    n = n1
    nnx = n * nx
    x =simX(n, device)
    U = simU(n, sig2U, device)#(torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps
    trainy = y.clone()
    trainw = w.clone()
    trainz = z.clone()
    bdim = trueb.shape[0]
    thisb = inib.clone()
    thisa = inia.clone()
    theta0 = torch.cat((thisb, thisa))
    theta0 = theta0.cpu().numpy()
    gridyr, _= torch.meshgrid(y[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    EY=eygw(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa, trainy, trainw, trainz,  sig2eps, sig2U, regfun, dregfun, device)
    res1 = optimize.minimize(rootfunzetaker, x0,  method = 'nelder-mead',  args = (y, w, z, EY, gridwr, gridzr, n, nx, alpha, device), options={"xatol": xatol})
    zeta = torch.tensor(res1.x).double().to(device)
    n = n3
    nnx = n * nx1
    x =simX(n, device)
    U =simU(n, sig2U, device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    lower = torch.ones((n, 1)).to(device)
    upper = torch.zeros((n, 1)).to(device)
    EY=eygw(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa, trainy, trainw, trainz,  sig2eps, sig2U, regfun, dregfun, device)
    lower= EY-zeta
    upper = EY + zeta
    print(y.mean())
    cpm = torch.mean((((y-upper)<=0) & ((y-lower)>=0)).double())
    return cpm, upper.mean(), lower.mean(), zeta


def simusemifun(truea, trueb, n1, n2, n3,xx,xx1,  wx, wx1,  nx,nx1,  ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol,xatol,   alpha, regfun, dregfun, device, eta, lr,lw,up,  factor,fct,  disp):
    n = n1
    nnx = n * nx
    stemp = torch.distributions.Beta(torch.ones(n) * 2, torch.ones(n) * 2)
    stp =stemp.sample()
    x = ((1.732 * 2 *stp -1.732)).reshape([-1, 1]).double().to(device)
    U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.bernoulli(torch.ones(n) * 0.8).double()
    z =z.reshape((n, 1))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps = torch.from_numpy(np.random.standard_t(3, (n, 1))).to(device)* sig2eps**(1/2) * 0.57735
    y = regfun(x, z, trueb, truea) +  eps#sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    inib = trueb  * factor
    inia = truea *factor
    thisb = inib.clone()
    bdim = thisb.shape[0]
    thisa = inia.clone() 
    theta0 = torch.cat((thisb, thisa))
    theta0 = theta0.cpu().numpy()
    #res = optimize.fsolve(rootfunnon, theta0[:, 0].copy(), args = (bdim, y, w, z, n, sig2eps, sig2U, regfun, dregfun, device), maxfev=maxitr, xtol = tol, factor = fct)
    res = optimize.fsolve(rootfun,theta0.copy(), args = (bdim, gridy, gridw, gridx, gridz0, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw, nywx, q, sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun, device),xtol = tol, maxfev = maxitr, factor = fct)
    if disp:
       print(res)
    thisb = torch.tensor(res[:bdim]).double().to(device).reshape([-1, 1])
    thisa = torch.tensor(res[bdim:]).double().to(device).reshape([-1, 1])
    theta = torch.cat((thisb, thisa))
    bb0= bfun(gridy, gridw, gridx, gridz0, xx, wx, wwygrid, ny, nw, nx, nyw,nywx, q, thisb, thisa,  sig2eps, sig2U, regfun, dregfun, device)   
    bb1= bfun(gridy, gridw, gridx, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw,nywx, q, thisb, thisa,  sig2eps, sig2U, regfun, dregfun, device)   
    AA0=Afun(gridy, gridw, gridx, gridz0, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,thisb, thisa,   sig2eps, sig2U, regfun, dregfun, device)
    AA1=Afun(gridy, gridw, gridx, gridz1, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,thisb, thisa,   sig2eps, sig2U, regfun, dregfun, device)
    aa0 = torch.inverse(AA0)@bb0
    aa1 = torch.inverse(AA1)@bb1
    gridyr, gridxr= torch.meshgrid( y[:, 0], xx[:, 0])
    gridwr, gridxr= torch.meshgrid( w[:, 0], xx[:, 0])
    gridzr, gridxr= torch.meshgrid( z[:, 0], xx[:, 0])
    gridxr = gridxr.reshape([-1, 1]).to(device)
    gridyr = gridyr.reshape([-1, 1]).to(device)
    gridwr = gridwr.reshape([-1, 1]).to(device)
    gridzr = torch.cat((gridzr.reshape([-1, 1]), torch.ones((nnx, 1)).double().to(device)), 1)
    az = torch.zeros((n, nx, q)).double().to(device)
    az[z[:, 0]==1, :, :] = aa1.double() 
    az[z[:, 0]==0, :, :] = aa0.double()
    def jSeff(theta):
        b = theta[:bdim]
        a = theta[bdim:]
        qm =qywz(gridyr, gridxr, gridwr, gridzr, b, a, sig2eps, sig2U, regfun)
        qmc =torch.reshape(qm, (n, nx))
        qc = (qmc@wx).reshape((n, 1))
        scoreres =sbetafm(gridyr, gridxr, gridzr, b, a, sig2eps, regfun, dregfun)
        scoreresm=scoreres.t().reshape([-1, 1])
        scoreresm = scoreresm.reshape([-1, nx])
        sb = torch.zeros((n, q)).double().to(device)
        for col in range(q):
            colin =torch.arange(col*n, (col+1) * n)
            temp =az[:, :, col]
            sb[:, col] = (((scoreresm[colin, :]-temp) * qmc) @wx)
        score=torch.mean(sb/qc, 0)
        return score        
    AA=torch.inverse(torch.autograd.functional.jacobian(jSeff, theta)[:, :, 0])
    sTs=sTseff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, q, thisb, thisa,  sig2eps, sig2U, regfun, dregfun, device)
    covX = AA@sTs@AA.T/n
    res1 = optimize.minimize_scalar(rootfunzeta,  method = 'bounded', bounds = (lw, up),  args = (theta, bdim, gridy, gridw, gridx, gridz0, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw, nywx,  sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun,eygw, alpha,  device), options={"xatol": xatol})
    zeta = torch.tensor(res1.x).double().to(device)
    n = n3
    nnx = n * nx1
    sampley = y.clone()
    cl = sampley.max()+ 3
    fl = sampley.min()-3
    stemp = torch.distributions.Beta(torch.ones(n) * 2, torch.ones(n) * 2)
    stp =stemp.sample()
    x = ((1.732 * 2* stp-1.732) * torch.ones(n)).reshape([-1, 1]).double().to(device)
    #U = ((torch.randn((1, 1)) * (sig2U)**(1/2)).reshape([-1, 1]) * torch.ones((n, 1))).to(device)
    U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = (torch.bernoulli(torch.ones((n, 1)) * 0.8).double()* torch.ones((n, 1)))
    z =z.reshape((n, 1))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    y = regfun(x, z, trueb, truea) + (sig2eps**(1/2) * torch.randn((n, 1))).to(device)
    sampley = (torch.arange(fl, cl, (cl-fl)/n).double().reshape([-1, 1])).to(device)
    sampley = sampley[0:n, :]
    gridyr, _= torch.meshgrid(sampley[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr.reshape([-1, 1]), torch.ones((nnx, 1)).double().to(device)), 1)
    lower = torch.ones((n, 1)).to(device)
    upper = torch.zeros((n, 1)).to(device)
    EY=eygw(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa,  sig2eps, sig2U, regfun, dregfun, device)
    # for it in range(n):
    #     diff=torch.abs(sampley-EY[it, 0])[:, 0]
    #     ix = diff <=zeta
    #     upper[it, 0] = sampley[ix].max()
    #     lower[it, 0] = sampley[ix].min()
    print(EY.mean())
    lower= EY-zeta
    upper = EY + zeta
    #upper = torch.cat((upper, upper1), 1).max(1).values.reshape((n, 1))
    #lower = torch.cat((lower, lower1), 1).min(1).values.reshape((n, 1))
    print(y.mean())
    cpm = torch.mean((((y-upper)<=0) & ((y-lower)>=0)).double())
    return cpm, upper.mean(), lower.mean(), res, zeta




def simusemifuncon(truea, trueb, inib, inia, n1, m, n3,xx,xx1,  wx, wx1,  nx,nx1,  ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol,xatol,   alpha, regfun, dregfun, device, eta, lr,x0,   factor,fct, eygw, estp,  disp):
    n = n1
    nnx = n * nx
    x =simX(n, device)
    U =simU(n, sig2U, device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    cres = KMeans(m, random_state=0).fit(z)
    clusterm = torch.from_numpy(cres.labels_).to(device)
    clusterc = torch.from_numpy(cres.cluster_centers_)
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    clusterc = torch.cat((clusterc, torch.ones([clusterc.shape[0], 1])), 1).to(device)
    eps =simeps(n, sig2eps, device) 
    y = regfun(x, z, trueb, truea) +eps
    trainy = y.clone()
    trainw = w.clone()
    trainz = z.clone()
    thisb = inib.clone()
    bdim = thisb.shape[0]
    thisa = inia.clone() 
    theta0 = torch.cat((thisb, thisa))
    theta0 = theta0.cpu().numpy()
    if(estp == 1):
        res = optimize.fsolve(rootfuncon,theta0[:, 0].copy(), args = (bdim, gridy, gridw, gridx, clusterm, clusterc, xx, wx, wwygrid, ny, nw, nx, nyw, nywx, q, sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun, device),xtol = tol, maxfev = maxitr, factor = fct)
    else:
        res = theta0.copy()
    if disp:
       print(res)
    thisb = torch.tensor(res[:bdim]).double().to(device).reshape([-1, 1])
    thisa = torch.tensor(res[bdim:]).double().to(device).reshape([-1, 1])
    theta = torch.cat((thisb, thisa))
    #res1 = optimize.minimize_scalar(rootfunzetacon,  method = 'bound', bounds = (lw, up),  args = (theta, bdim, gridy, gridw, gridx, clusterm, clusterc, xx, wx, wwygrid, ny, nw, nx, nyw, nywx,  sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun,eygw, alpha,  device), options={"xatol": xatol})
    res1 = optimize.minimize(rootfunzetacon, x0,  method = 'nelder-mead',  args = (theta, bdim, gridy, gridw, gridx, clusterm, clusterc, xx, wx, wwygrid, ny, nw, nx, nyw, nywx,  sig2eps, sig2U, trainy, trainw, trainz, n, nnx, regfun, dregfun,eygw, alpha,  device), options={"xatol": xatol})
    zeta = torch.tensor(res1.x).double().to(device)
    n = n3
    nnx = n * nx1
    sampley = y.clone()
    cl = sampley.max()+ 3
    fl = sampley.min()-3
    x =simX(n, device)#(torch.randn((n, 1)) -1).double().to(device)#((1.732 * 2* stp-1.732) * torch.ones(n)).reshape([-1, 1]).double().to(device)
    #U = ((torch.randn((1, 1)) * (sig2U)**(1/2)).reshape([-1, 1]) * torch.ones((n, 1))).to(device)
    U =simU(n, sig2U, device)#U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps#sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    #y = regfun(x, z, trueb, truea) + (sig2eps**(1/2) * torch.randn((n, 1))).to(device)
    sampley = (torch.arange(fl, cl, (cl-fl)/n).double().reshape([-1, 1])).to(device)
    sampley = sampley[0:n, :]
    gridyr, _= torch.meshgrid(sampley[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    lower = torch.ones((n, 1)).to(device)
    upper = torch.zeros((n, 1)).to(device)
    EY=eygw(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa, trainy, trainw, trainz,  sig2eps, sig2U, regfun, dregfun, device)
    print(EY.mean())
    lower= EY-zeta
    upper = EY + zeta
    print(y.mean())
    cpm = torch.mean((((y-upper)<=0) & ((y-lower)>=0)).double())
    return cpm, upper.mean(), lower.mean(), res, zeta




def simusemiwrong(truea, trueb, n1, n2, n3,xx,xx1,  wx, wx1,  nx,nx1,  ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol,xatol,   alpha, regfun, dregfun, device, eta, lr,lw,up,  factor,fct,  disp):
    n = n1
    nnx = n * nx
    stemp = torch.distributions.Beta(torch.ones(n) * 2, torch.ones(n) * 2)
    stp =stemp.sample()
    x = ((1.732 * 2 *stp -1.732)).reshape([-1, 1]).double().to(device)
    U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.bernoulli(torch.ones(n) * 0.8).double()
    z =z.reshape((n, 1))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    y = regfun(x, z, trueb, truea) + sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    inib = trueb  * factor
    inia = truea *factor
    thisb = inib.clone()
    bdim = thisb.shape[0]
    thisa = inia.clone() 
    theta0 = torch.cat((thisb, thisa))
    theta0 = theta0.cpu().numpy()
    res = optimize.fsolve(rootfunnon, theta0[:, 0].copy(), args = (bdim, y, w, z, n, sig2eps, sig2U, regfun, dregfun, device), maxfev=maxitr, xtol = tol, factor = fct)
    #res = optimize.fsolve(rootfun,theta0.copy(), args = (bdim, gridy, gridw, gridx, gridz0, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw, nywx, q, sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun, device),xtol = tol, maxfev = maxitr, factor = fct)
    if disp:
       print(res)
    thisb = torch.tensor(res[:bdim]).double().to(device).reshape([-1, 1])
    thisa = torch.tensor(res[bdim:]).double().to(device).reshape([-1, 1])
    theta = torch.cat((thisb, thisa))
    res1 = optimize.minimize_scalar(rootfunzeta,  method = 'bounded', bounds = (lw, up),  args = (theta, bdim, gridy, gridw, gridx, gridz0, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw, nywx,  sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun,eygwwrg, alpha,  device), options={"xatol": xatol})
    zeta = torch.tensor(res1.x).double().to(device)
    n = n3
    nnx = n * nx1
    sampley = y.clone()
    cl = sampley.max()+ 3
    fl = sampley.min()-3
    stemp = torch.distributions.Beta(torch.ones(n) * 2, torch.ones(n) * 2)
    stp =stemp.sample()
    x = ((1.732 * 2* stp-1.732) * torch.ones(n)).reshape([-1, 1]).double().to(device)
    #U = ((torch.randn((1, 1)) * (sig2U)**(1/2)).reshape([-1, 1]) * torch.ones((n, 1))).to(device)
    U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = (torch.bernoulli(torch.ones((n, 1)) * 0.8).double()* torch.ones((n, 1)))
    z =z.reshape((n, 1))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    y = regfun(x, z, trueb, truea) + (sig2eps**(1/2) * torch.randn((n, 1))).to(device)
    sampley = (torch.arange(fl, cl, (cl-fl)/n).double().reshape([-1, 1])).to(device)
    sampley = sampley[0:n, :]
    gridyr, _= torch.meshgrid(sampley[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr.reshape([-1, 1]), torch.ones((nnx, 1)).double().to(device)), 1)
    lower = torch.ones((n, 1)).to(device)
    upper = torch.zeros((n, 1)).to(device)
    EY=eygwwrg(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa,  sig2eps, sig2U, regfun, dregfun, device)
    # for it in range(n):
    #     diff=torch.abs(sampley-EY[it, 0])[:, 0]
    #     ix = diff <=zeta
    #     upper[it, 0] = sampley[ix].max()
    #     lower[it, 0] = sampley[ix].min()
    print(EY.mean())
    lower= EY-zeta
    upper = EY + zeta
    #upper = torch.cat((upper, upper1), 1).max(1).values.reshape((n, 1))
    #lower = torch.cat((lower, lower1), 1).min(1).values.reshape((n, 1))
    print(y.mean())
    cpm = torch.mean((((y-upper)<=0) & ((y-lower)>=0)).double())
    return cpm, upper.mean(), lower.mean(), res, zeta



    
def simusemiwrongcon(truea, trueb, inib, inia, n1, m, n3,xx,xx1,  wx, wx1,  nx,nx1,  ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol,xatol,   alpha, regfun, dregfun, device, eta, lr,x0,  factor,fct, eygwwrg,  disp):
    n = n1
    nnx = n * nx
    x =simX(n, device)
    U =simU(n, sig2U, device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    cres = KMeans(m, random_state=0).fit(z)
    clusterm = torch.from_numpy(cres.labels_).to(device)
    clusterc = torch.from_numpy(cres.cluster_centers_)
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    clusterc = torch.cat((clusterc, torch.ones([clusterc.shape[0], 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps
    trainy = y.clone()
    trainw = w.clone()
    trainz = z.clone()
    thisb = inib.clone()
    bdim = thisb.shape[0]
    thisa = inia.clone() 
    theta0 = torch.cat((thisb, thisa))
    theta0 = theta0.cpu().numpy()
    res = optimize.fsolve(rootfunnon, theta0[:, 0].copy(), args = (bdim, y, w, z, n, sig2eps, sig2U, regfun, dregfun, device), maxfev=maxitr, xtol = tol, factor = fct)
    if disp:
       print(res)
    thisb = torch.tensor(res[:bdim]).double().to(device).reshape([-1, 1])
    thisa = torch.tensor(res[bdim:]).double().to(device).reshape([-1, 1])
    theta = torch.cat((thisb, thisa))
    gridyr, _= torch.meshgrid(y[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    EY=eygwwrg(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa, trainy, trainw, trainz,  sig2eps, sig2U, regfun, dregfun, device)
    res1 = optimize.minimize(rootfunzetawrong, x0,  method = 'nelder-mead',  args = (y, w, z, EY, gridwr, gridzr, n, nx,  alpha, device), options={"xatol": xatol})
    zeta = torch.tensor(res1.x).double().to(device)
    n = n3
    nnx = n * nx1
    sampley = y.clone()
    cl = sampley.max()+ 3
    fl = sampley.min()-3
    stemp = torch.distributions.Beta(torch.ones(n) * 2, torch.ones(n) * 2)
    stp =stemp.sample()
    x =simX(n, device)
    U =simU(n, sig2U, device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps
    sampley = (torch.arange(fl, cl, (cl-fl)/n).double().reshape([-1, 1])).to(device)
    sampley = sampley[0:n, :]
    gridyr, _= torch.meshgrid(sampley[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    lower = torch.ones((n, 1)).to(device)
    upper = torch.zeros((n, 1)).to(device)
    EY=eygwwrg(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa, trainy, w, z,   sig2eps, sig2U, regfun, dregfun, device)
    print(EY.mean())
    lower= EY-zeta
    upper = EY + zeta
    print(y.mean())
    cpm = torch.mean((((y-upper)<=0) & ((y-lower)>=0)).double())
    return cpm, upper.mean(), lower.mean(), res, zeta





def simusemiestonly(truea, trueb, inib, inia, n1, m, n3,xx,xx1,  wx, wx1,  nx,nx1,  ny, nw, nyw, nywx,q,  gridx, gridy, gridw, gridz0, gridz1, wwygrid, sig2eps, sig2U, maxitr,tol,xatol,   alpha, regfun, dregfun, device, eta, lr,lw,up,  factor,fct, eygw,  disp):
    n = n1
    nnx = n * nx
    x =simX(n, device)#(torch.randn((n, 1)) -1).double().to(device)# ((1.732 * 2 *stp -1.732)).reshape([-1, 1]).double().to(device)
    U =simU(n, sig2U, device)#U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    cres = KMeans(m, random_state=0).fit(z)
    clusterm = torch.from_numpy(cres.labels_).to(device)
    clusterc = torch.from_numpy(cres.cluster_centers_)
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    clusterc = torch.cat((clusterc, torch.ones([clusterc.shape[0], 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps#sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    #y = regfun(x, z, trueb, truea) + sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    trainy= y.clone()
    trainw= w.clone()
    trainz = z.clone()
    thisb = inib.clone()
    bdim = thisb.shape[0]
    thisa = inia.clone() 
    theta0 = torch.cat((thisb, thisa))
    theta0 = theta0.cpu().numpy()
    res = optimize.fsolve(rootfuncon,theta0.copy(), args = (bdim, gridy, gridw, gridx, clusterm, clusterc, xx, wx, wwygrid, ny, nw, nx, nyw, nywx, q, sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun, device),xtol = tol, maxfev = maxitr, factor = fct)
    if disp:
       print(res)
    thisb = torch.tensor(res[:bdim]).double().to(device).reshape([-1, 1])
    thisa = torch.tensor(res[bdim:]).double().to(device).reshape([-1, 1])
    theta = torch.cat((thisb, thisa))
    n =n1
    nnx = n * nx
    x =simX(n, device)#(torch.randn((n, 1)) -1).double().to(device)# ((1.732 * 2 * stp-1.732)).reshape([-1, 1]).double().to(device)
    U =simU(n, sig2U, device)#U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps#sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    #y = regfun(x, z, trueb, truea) + sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    gridyr, _= torch.meshgrid(y[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    EY=eygw(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa, trainy, trainw, trainz,  sig2eps, sig2U, regfun, dregfun, device)
    nscores=torch.abs(y-EY)
    sscores=torch.sort(nscores[:, 0])[0]
    index = int((n+ 1) * (1-alpha))
    zeta = sscores[index].clone()
    n = n3
    nnx = n * nx1
    sampley = y.clone()
    cl = sampley.max()+ 3
    fl = sampley.min()-3
    x =simX(n, device)#(torch.randn((n, 1)) -1).double().to(device)#((1.732 * 2* stp-1.732) * torch.ones(n)).reshape([-1, 1]).double().to(device)
    U =simU(n, sig2U, device)#U = (torch.randn((n, 1)) * (sig2U)**(1/2)).reshape([-1, 1]).to(device)
    w = x + U
    z = torch.cat((torch.rand((n, 1)), torch.bernoulli(torch.ones(n) * 0.8).reshape([-1, 1])), 1).double()
    z =z.reshape((n, (2)))
    z = torch.cat((z, torch.ones([n, 1])), 1).to(device)
    eps =simeps(n, sig2eps, device)
    y = regfun(x, z, trueb, truea) +  eps#sig2eps**(1/2) * (torch.randn((n, 1))).to(device)
    #y = regfun(x, z, trueb, truea) + (sig2eps**(1/2) * torch.randn((n, 1))).to(device)
    sampley = (torch.arange(fl, cl, (cl-fl)/n).double().reshape([-1, 1])).to(device)
    sampley = sampley[0:n, :]
    gridyr, _= torch.meshgrid(sampley[:, 0], xx1[:, 0])
    gridwr, _= torch.meshgrid( w[:, 0], xx1[:, 0])
    gridwr = gridwr.reshape([-1, 1])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx1[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx1[:, 0])
    gridxr = gridxr.reshape([-1, 1])
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    lower = torch.ones((n, 1)).to(device)
    upper = torch.zeros((n, 1)).to(device)
    EY=eygw(gridwr, gridxr,gridzr, wx1, n,nx1, q, thisb,thisa, trainy, trainw, trainz,  sig2eps, sig2U, regfun, dregfun, device)
    lower= EY-zeta
    upper = EY + zeta
    print(y.mean())
    cpm = torch.mean((((y-upper)<=0) & ((y-lower)>=0)).double())
    return cpm, upper.mean(), lower.mean(), res, zeta




# import matplotlib.pyplot as plt
# plt.scatter(range(len(y[torch.where(nscores>=sm)])), y[torch.where(nscores>=sm)])
# plt.savefig('temp.png')
# plt.close()


def rootfunnon(theta,bdim,  y, w, z, n, sig2eps, sig2U, regfun, dregfun, device):
    theta = torch.tensor(theta).to(device)
    thisb=theta[:bdim].double().reshape([-1, 1])
    thisa=theta[bdim:].double().reshape([-1, 1])
    score= seffnoerror(y, w, z, n, thisb, thisa,  sig2eps, sig2U, regfun, dregfun, device)
    return score.cpu().numpy()

def rootfun(theta, bdim, gridy, gridw, gridx, gridz0, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw, nywx, q, sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun, device):
    theta = torch.tensor(theta).to(device)
    thisb=theta[:bdim].double().reshape([-1, 1])
    thisa=theta[bdim:].double().reshape([-1, 1])
    bb0= bfun(gridy, gridw, gridx, gridz0, xx, wx, wwygrid, ny, nw, nx, nyw,nywx, q, thisb, thisa,  sig2eps, sig2U, regfun, dregfun, device)   
    bb1= bfun(gridy, gridw, gridx, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw,nywx, q, thisb, thisa,  sig2eps, sig2U, regfun, dregfun, device)   
    AA0=Afun(gridy, gridw, gridx, gridz0, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,thisb, thisa,   sig2eps, sig2U, regfun, dregfun, device)
    AA1=Afun(gridy, gridw, gridx, gridz1, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,thisb, thisa,   sig2eps, sig2U, regfun, dregfun, device)
    aa0 = torch.inverse(AA0)@bb0
    aa1 = torch.inverse(AA1)@bb1
    gridyr, gridxr= torch.meshgrid( y[:, 0], xx[:, 0])
    gridwr, gridxr= torch.meshgrid( w[:, 0], xx[:, 0])
    gridzr, gridxr= torch.meshgrid( z[:, 0], xx[:, 0])
    gridxr = gridxr.reshape([-1, 1]).to(device)
    gridyr = gridyr.reshape([-1, 1]).to(device)
    gridwr = gridwr.reshape([-1, 1]).to(device)
    gridzr = torch.cat((gridzr.reshape([-1, 1]), torch.ones((nnx, 1)).double().to(device)), 1)
    az = torch.zeros((n, nx, q)).double().to(device)
    az[z[:, 0]==1, :, :] = aa1.double() 
    az[z[:, 0]==0, :, :] = aa0.double()
    score= seff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx,q,  thisb,  thisa,  sig2eps, sig2U, regfun, dregfun, device)
    return score.cpu().numpy()
    #score= seff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, thisb+0.1, thisa+0.1,  sig2eps, sig2U)
    #score= seff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, thisb+0.1, thisa+0.1,  sig2eps, sig2U)


def rootfuncon(theta, bdim, gridy, gridw, gridx, clusterm, clusterc, xx, wx, wwygrid, ny, nw, nx, nyw, nywx, q, sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun, device):
    theta = torch.tensor(theta).to(device)
    thisb=theta[:bdim].double().reshape([-1, 1])
    thisa=theta[bdim:].double().reshape([-1, 1])
    nc = clusterc.shape[0]
    mrAfun1 = functools.partial(mrAbfun, z=clusterc, gridy=gridy, gridw=gridw, gridx=gridx,  xx=xx, wx=wx,  wwygrid=wwygrid, ny=ny, nw=nw, nx=nx, nyw=nyw,nywx=nywx,q = q, b=thisb, a=thisa,   sig2eps=sig2eps, sig2U=sig2U, regfun=regfun, dregfun=dregfun,  device=device)
    Alist = list(map(mrAfun1, range(nc)))
    az = torch.zeros((n, nx, q)).double().to(device)
    for i in range(nc):
        az[clusterm==i, :, :] = Alist[i].double()
    gridyr, gridxr= torch.meshgrid( y[:, 0], xx[:, 0])
    gridwr, gridxr= torch.meshgrid( w[:, 0], xx[:, 0])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx[:, 0])
    gridxr = gridxr.reshape([-1, 1]).to(device)
    gridyr = gridyr.reshape([-1, 1]).to(device)
    gridwr = gridwr.reshape([-1, 1]).to(device)
    gridzr = torch.cat((gridzr0.reshape([-1, 1]), gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    score= seff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx,q,  thisb,  thisa,  sig2eps, sig2U, regfun, dregfun, device)
    return score.cpu().numpy()
    


def rootfunzeta(zeta, theta,bdim,  gridy, gridw, gridx, gridz0, gridz1, xx, wx, wwygrid, ny, nw, nx, nyw, nywx,  sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun,eygw, alpha,  device):
    q= 1
    zeta = torch.tensor(zeta).to(device)
    thisb=theta[:bdim].double().reshape([-1, 1])
    thisa=theta[bdim:].double().reshape([-1, 1])
    bb0= bfzeta(gridy, gridw, gridx, gridz0, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx, thisb, thisa,  sig2eps, sig2U,regfun, dregfun, zeta,  device)
    bb1= bfzeta(gridy, gridw, gridx, gridz1, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx, thisb, thisa,  sig2eps, sig2U,regfun, dregfun, zeta,  device)
    AA0=Afun(gridy, gridw, gridx, gridz0, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,thisb, thisa,   sig2eps, sig2U, regfun, dregfun, device)
    AA1=Afun(gridy, gridw, gridx, gridz1, xx, wx,  wwygrid, ny, nw, nx, nyw,nywx,thisb, thisa,   sig2eps, sig2U, regfun, dregfun, device)
    mbb0 = bb0[:, 0].double()@wx
    abb0=mbb0 -bb0
    mbb1 = bb1[:, 0].double()@wx
    abb1=mbb1 -bb1
    aa0 = torch.inverse(AA0)@abb0
    aa1 = torch.inverse(AA1)@abb1
    gridyr, gridxr= torch.meshgrid( y[:, 0], xx[:, 0])
    gridwr, gridxr= torch.meshgrid( w[:, 0], xx[:, 0])
    gridzr, gridxr= torch.meshgrid( z[:, 0], xx[:, 0])
    gridxr = gridxr.reshape([-1, 1]).to(device)
    gridyr = gridyr.reshape([-1, 1]).to(device)
    gridwr = gridwr.reshape([-1, 1]).to(device)
    gridzr = torch.cat((gridzr.reshape([-1, 1]), torch.ones((nnx, 1)).double().to(device)), 1)
    az = torch.zeros((n, nx, q)).double().to(device)
    az[z[:, 0]==1, :, :] = aa1.double() 
    az[z[:, 0]==0, :, :] = aa0.double()
    zz = torch.zeros((n)).double().to(device)
    zz[z[:, 0]==1] = mbb1.double() 
    zz[z[:, 0]==0] = mbb0.double()
    score = psieff(gridyr, gridwr, gridxr,gridzr,az,zz,   xx, wx, n,nx, thisb, thisa,  sig2eps, sig2U, regfun, dregfun,alpha,  device)**2
    return score.cpu().numpy()
    #score= seff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, thisb+0.1, thisa+0.1,  sig2eps, sig2U)
    #score= seff(gridyr, gridwr, gridxr,gridzr,az,  xx, wx, n,nx, thisb+0.1, thisa+0.1,  sig2eps, sig2U)


def rootfunzetacon(zeta, theta,bdim,  gridy, gridw, gridx, clusterm, clusterc,  xx, wx, wwygrid, ny, nw, nx, nyw, nywx,  sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun,eygw, alpha,  device):
    q= 1
    nc = clusterc.shape[0]
    zeta = torch.tensor(zeta).to(device)
    thisb=theta[:bdim].double().reshape([-1, 1])
    thisa=theta[bdim:].double().reshape([-1, 1])
    mrAfun1 = functools.partial(mrAbfunzeta, zc=clusterc, gridy=gridy, gridw=gridw, gridx=gridx,  xx=xx, wx=wx,  wwygrid=wwygrid, ny=ny, nw=nw, nx=nx, nyw=nyw,nywx=nywx,b=thisb, a=thisa,   sig2eps=sig2eps, sig2U=sig2U, regfun=regfun, dregfun=dregfun, zeta = zeta,eygw = eygw, y = y, w=w, z = z,  device=device)
    Alist = list(map(mrAfun1, range(nc)))
    az = torch.zeros((n, nx, 1)).double().to(device)
    zz = torch.zeros((n)).double().to(device)
    for i in range(nc):
        az[clusterm==i, :, 0] = Alist[i][0]
        zz[clusterm==i] = Alist[i][1]
    gridyr, gridxr= torch.meshgrid( y[:, 0], xx[:, 0])
    gridwr, gridxr= torch.meshgrid( w[:, 0], xx[:, 0])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx[:, 0])
    gridxr = gridxr.reshape([-1, 1]).to(device)
    gridyr = gridyr.reshape([-1, 1]).to(device)
    gridwr = gridwr.reshape([-1, 1]).to(device)
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    score = psieff(gridyr, gridwr, gridxr,gridzr,az,zz,   xx, wx, n,nx, thisb, thisa,  sig2eps, sig2U, regfun, dregfun,alpha,  device)**2
    return score.cpu().numpy()

def rootfunzetawrongcon(zeta, theta,bdim,  gridy, gridw, gridx, clusterm, clusterc,  xx, wx, wwygrid, ny, nw, nx, nyw, nywx,  sig2eps, sig2U, y, w, z, n, nnx, regfun, dregfun,eygw, alpha,  device):
    q= 1
    nc = clusterc.shape[0]
    zeta = torch.tensor(zeta).to(device)
    thisb=theta[:bdim].double().reshape([-1, 1])
    thisa=theta[bdim:].double().reshape([-1, 1])
    mrAfun1 = functools.partial(mrAbfunzeta, zc=clusterc, gridy=gridy, gridw=gridw, gridx=gridx,  xx=xx, wx=wx,  wwygrid=wwygrid, ny=ny, nw=nw, nx=nx, nyw=nyw,nywx=nywx,b=thisb, a=thisa,   sig2eps=sig2eps, sig2U=sig2U, regfun=regfun, dregfun=dregfun, zeta = zeta,eygw = eygw, y = y, w=w, z = z,  device=device)
    Alist = list(map(mrAfun1, range(nc)))
    az = torch.zeros((n, nx, 1)).double().to(device)
    zz = torch.zeros((n)).double().to(device)
    for i in range(nc):
        zz[clusterm==i] = Alist[i][1]
    gridyr, gridxr= torch.meshgrid( y[:, 0], xx[:, 0])
    gridwr, gridxr= torch.meshgrid( w[:, 0], xx[:, 0])
    gridzr0, gridxr= torch.meshgrid( z[:, 0], xx[:, 0])
    gridzr1, gridxr= torch.meshgrid( z[:, 1], xx[:, 0])
    gridxr = gridxr.reshape([-1, 1]).to(device)
    gridyr = gridyr.reshape([-1, 1]).to(device)
    gridwr = gridwr.reshape([-1, 1]).to(device)
    gridzr = torch.cat((gridzr0.reshape([-1, 1]),gridzr1.reshape([-1, 1]),  torch.ones((nnx, 1)).double().to(device)), 1)
    score = psieff(gridyr, gridwr, gridxr,gridzr,az,zz,   xx, wx, n,nx, thisb, thisa,  sig2eps, sig2U, regfun, dregfun,alpha,  device)**2
    return score.cpu().numpy()

def rootfunzetaker(zeta, y, w, z, EY, gridwr, gridzr, n, nx,  alpha, device):
    zeta = torch.tensor(zeta).to(device)
    n1 = gridwr.shape[0]
    ns = y.shape[0]
    wz1= torch.cat((gridwr, gridzr[:, :2].T.reshape([-1, 1])), 0).reshape([1, -1, 1])
    wz2 = torch.cat((w, z[:, :2].T.reshape([-1, 1])), 0).reshape([1, -1, 1])
    dism = torch.cdist(wz1, wz2)[0, :, :]
    disw= dism[:n1, :ns]
    disz1= dism[n1:(2* n1), ns:(2*ns)]
    disz2 = dism[(2* n1):(3*n1), (2*ns):(3* ns)]
    kw=gker(disw, w.std()* ns**(-1/5) * 1.06)
    kz1 = gker(disz1, z[:, 0].std()* ns**(-1/5) * 1.06)
    kz2 = gker(disz2, z[:, 1].std()* ns**(-1/5) * 1.06)
    kk=kw * kz1 * kz2
    yEY = y - EY
    r = yEY.abs()
    Ezetadis=r-zeta
    kh = r.std()* n**(-1/5)* 0.01
    khI=gker(Ezetadis, kh) * 0.39894/kh#0.75 * (1 - (Ezetadis/kh)**2) * ((Ezetadis/kh).abs() <=1) /kh#
    khI = khI#/khI.sum()
    ygw=(kk @ khI )/kk.sum(1).reshape([-1, 1])
    ygw =torch.reshape(ygw, (n, nx))
    yqc = (ygw[:, 0]).reshape((n, 1))
    score = (1- alpha - (r<=zeta).double().mean() - (yEY * yqc).mean())**2
    return score.cpu().numpy()


def rootfunzetawrong(zeta, y, w, z, EY, gridwr, gridzr, n, nx,  alpha, device):
    zeta = torch.tensor(zeta).to(device)
    n1 = gridwr.shape[0]
    ns = y.shape[0]
    wz1= torch.cat((gridwr, gridzr[:, :2].T.reshape([-1, 1])), 0).reshape([1, -1, 1])
    wz2 = torch.cat((w, z[:, :2].T.reshape([-1, 1])), 0).reshape([1, -1, 1])
    dism = torch.cdist(wz1, wz2)[0, :, :]
    disw= dism[:n1, :ns]
    disz1= dism[n1:(2* n1), ns:(2*ns)]
    disz2 = dism[(2* n1):(3*n1), (2*ns):(3* ns)]
    kw=gker(disw, w.std()* ns**(-1/7) * 3)
    kz1 = gker(disz1, z[:, 0].std()* ns**(-1/7) * 3)
    kz2 = gker(disz2, z[:, 1].std()* ns**(-1/7) * 3)
    kk=kw * kz1 * kz2
    yEY = y - EY
    r = yEY.abs()
    Ezetadis=r-zeta
    khI=(r<=zeta).double() * yEY
    ygw=(kk @ khI )/kk.sum(1).reshape([-1, 1])
    ygw =torch.reshape(ygw, (n, nx))
    yqc = (ygw[:, 0]).reshape((n, 1))
    khI= yEY **2
    ygw1=(kk @ khI )/kk.sum(1).reshape([-1, 1])
    ygw1 =torch.reshape(ygw1, (n, nx))
    yqcvar = (ygw1[:, 0]).reshape((n, 1))
    score = (1- alpha - (r<=zeta).double().mean() + (yqc*yEY/yqcvar).mean())**2
    return score.cpu().numpy()
    
