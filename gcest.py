import time
import numpy as np
import scipy.io
from scipy.stats import pearsonr
from gcmi import gccmi_ccc
import scipy.stats

def rdet(A):
	if not np.shape(A):
		return A
	else:
		return np.linalg.det(A) 


def bindata1(X,nbins):

        if(len(np.shape(X))> 1):
                X=np.reshape(X,(np.shape(X)[0]*np.shape(X)[1],1))

        X=np.squeeze(X)

	q = np.ndarray.tolist(np.arange(1,nbins)/float(nbins)*100)	
	bins = np.percentile(X,q)
	X1 = np.digitize(X,bins)

        return X1


def probabilityDist(X, Y, pastX, pastY):
#computeBinnedProbabilityDist Computes probability distribution among all
#combinations of quadruples X(i), pastX(i), Y(i), pastY(i)
#  

	mX=np.max(X)+1
	mY=np.max(Y)+1
	mpastX=np.max(pastX)+1
	mpastY=np.max(pastY)+1

	X=np.squeeze(X)
        Y=np.squeeze(Y)
        pastX=np.squeeze(pastX)
        pastY=np.squeeze(pastY)

	count = len(X)

        v = np.zeros(count)

        for i in range(count):
	        v[i]=np.ravel_multi_index((X[i],Y[i],pastX[i],pastY[i]), (mX,mY,mpastX,mpastY))


	v=v.astype(int)

	result = np.bincount(v,minlength=mX*mY*mpastX*mpastY)/float(count)
  
        result = np.reshape(result,(mX, mY, mpastX, mpastY))

	return result 

def TE(Pxypxpy):
#TI Computes transfer entroy and instantaneous correlation of a probability distribution  
#
# The probability distribution has to have following dimensions in the
# particular order:
# 1: X
# 2: Y
# 3: past of X
# 4: past of Y

        eps=1.E-10

	Ppx = np.squeeze(np.sum(np.sum(np.sum(Pxypxpy,axis=0),axis=0),axis=1))
	Ppy = np.squeeze(np.sum(np.sum(np.sum(Pxypxpy,axis=0),axis=0),axis=0))
        Pxpx = np.squeeze(np.sum(np.sum(Pxypxpy,axis=1),axis=2))
        Pypy = np.squeeze(np.sum(np.sum(Pxypxpy,axis=0),axis=1))
        Ppxpy = np.squeeze(np.sum(np.sum(Pxypxpy,axis=0),axis=0))
        Pxpxpy = np.squeeze(np.sum(Pxypxpy,axis=1))
        Pypxpy = np.squeeze(np.sum(Pxypxpy,axis=0))


	hpx = -np.dot(Ppx[np.nonzero(Ppx)], np.log2(Ppx[np.nonzero(Ppx)] + eps))
        hpy = -np.dot(Ppy[np.nonzero(Ppy)], np.log2(Ppy[np.nonzero(Ppy)] + eps))
        hxpx = -np.dot(Pxpx[np.nonzero(Pxpx)], np.log2(Pxpx[np.nonzero(Pxpx)] + eps))
        hypy = -np.dot(Pypy[np.nonzero(Pypy)], np.log2(Pypy[np.nonzero(Pypy)] + eps))
	hpxpy = -np.dot(Ppxpy[np.nonzero(Ppxpy)], np.log2(Ppxpy[np.nonzero(Ppxpy)] + eps))
        hxpxpy = -np.dot(Pxpxpy[np.nonzero(Pxpxpy)], np.log2(Pxpxpy[np.nonzero(Pxpxpy)] + eps))
        hypxpy = -np.dot(Pypxpy[np.nonzero(Pypxpy)], np.log2(Pypxpy[np.nonzero(Pypxpy)] + eps))
	hxypxpy = -np.dot(Pxypxpy[np.nonzero(Pxypxpy)], np.log2(Pxypxpy[np.nonzero(Pxypxpy)] + eps))


	dixy = hypy - hpy - hypxpy + hpxpy;
	diyx = hxpx - hpx - hxpxpy + hpxpy;
	icxy = hxpxpy - hpxpy + hypxpy  -hxypxpy 

	return dixy,diyx,icxy



def GC(X, mask,lag,method='gcmi'):

	# Data parameters. Size = sources x time points
	nSo=np.shape(X)[0]
	nTi=np.shape(X)[1]	

	ind_t = np.ones((nTi-lag,lag+1))

	gap = np.zeros(nTi) # frames after a gap

	nOrig=np.shape(mask)[0] # number of frames before motion scrubbing


	ncut=0     # number of uncut frames util tt
	isfirst=0     # tt is first point of gap

	for i in range(nTi-lag):
		for j in range(lag+1):
			ind_t[i,j]=ind_t[i,j]+i+(lag-j)

        keep = np.ones(nTi-lag)    # kept rows

	#print ind_t
	
	for t in range(nTi-lag):
		if(mask[t+lag]==0):
			for l in range(lag+1):
				keep[t+l]=0
	
			
	keep = np.where(keep==1)[0]

	ind_t = ind_t[keep,:]	
	ind_t=ind_t.astype(int)


	# Pairs between sources
	nPairs = nSo*(nSo-1)/2
 
	# Init
	GC    = np.zeros((nSo,nSo))
	IC    = np.zeros((nSo,nSo))
 
	# Normalisation coefficient for gaussian entropy
	C = np.log(2*np.pi*np.exp(1));

	# Loop over number of pairs
	for i in range(nSo):
		for j in range(i+1,nSo):
	
			# Extract data for a given pair of sources
			x = X[i,ind_t]
			y = X[j,ind_t]

			correl,pval=pearsonr(x[:,0],y[:,0])
			eps=1.E-10
	
			if(np.max(np.abs(x))>0 and np.max(np.abs(y))>0 and np.abs(correl)<1-eps and np.abs(correl)>eps): 


				if(method=='gcmi'):

					GC[i,j]=gccmi_ccc(y[:,0].T,x[:,1:].T,y[:,1:].T)
					GC[j,i]=gccmi_ccc(x[:,0].T,y[:,1:].T,x[:,1:].T)
					IC[i,j]=gccmi_ccc(x[:,0].T,y[:,0].T, np.column_stack((x[:,1:],y[:,1:])).T)
					IC[j,i]=IC[i,j]

				elif(method=='cov'):
	
					# Hycy: H(Y_i+1|Y_i) = H(Y_i+1,Y_i) - H(Y_i)
					det_yi1  = rdet(np.cov(y.T));
					det_yi   = rdet(np.cov(y[:,1:].T));
					Hycy     = np.log(det_yi1) - np.log(det_yi);
                       			# Hxcx: H(X_i+1|X_i) = H(X_i+1,X_i) - H(X_i)
                        		det_xi1  = rdet(np.cov(x.T));
                        		det_xi   = rdet(np.cov(x[:,1:].T));
					Hxcx     = np.log(det_xi1) - np.log(det_xi);
					# Hycx: H(Y_i+1|X_i,Y_i) = H(Y_i+1,X_i,Y_i) - H(X_i,Y_i)
					det_yi1xi = rdet(np.cov(np.column_stack((y,x[:,1:])).T));
					det_yixi  = rdet(np.cov(np.column_stack((y[:,1:],x[:,1:])).T));
					Hycxy     = np.log(det_yi1xi) - np.log(det_yixi);
					# Hxcy: H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)
					det_xi1yi = rdet(np.cov(np.column_stack((x,y[:,1:])).T));
					Hxcy     = np.log(det_xi1yi1) - np.log(det_yixi);      
    					# Hxi1yi1cxy: H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)
    					det_xi1yi1 = rdet(np.cov(np.column_stack((x,y)).T));
					Hxxcyy   = np.log(det_xi1yi1) - np.log(det_yixi);

    
					GC[i,j] = Hycy - Hycx;
					GC[j,i] = Hxcx - Hxcy;
					IC[i,j] = Hycx + Hxcy - Hxxcyy;
					IC[j,i] = IC[i,j]


				elif(method=='binning'):
				
					nbins=3

					Pr = probabilityDist(bindata1(x[:,0],nbins),bindata1(y[:,0],nbins),bindata1(x[:,1:],nbins),bindata1(y[:,1:],nbins))
					txy,tyx,icxy = TE(Pr)

 		
					GC[i,j] = txy
					GC[j,i] = tyx
					IC[i,j] = icxy
					IC[j,i] = icxy

					# Counter

	return GC,IC


