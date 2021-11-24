import numpy as np
from scipy.interpolate import BSpline
from matplotlib import pyplot as plt
from dlra.utils import gen_BSplines
import scipy.io as spio
import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals
from dlra.algorithms import dlra_parafac
from mscode.methods.algorithms import omp
import pandas as pd
import copy

# Seeding
np.random.seed(seed=0)

# DataFrame to store results
store_pd = pd.DataFrame(columns=["algorithm", "value", "error type"])

# loading the data
data = spio.loadmat('../../data/XP_chemo/aminoacids.mat', squeeze_me=True)  # Data tensor extracted from amino in matlab
Y = data['data']

Y = tl.transpose(Y,[1,2,0])

n = 201
m = 61 #smoothness on the second mode too?
l = 5
r = 3
k = 6

D1 = gen_BSplines(n, 30, 4, shift=0)
D2 = gen_BSplines(n, 30, 4, shift=6)
D3 = gen_BSplines(n, 30, 4, shift=12)
D4 = gen_BSplines(n, 30, 4, shift=18)
D5 = gen_BSplines(n, 30, 4, shift=24)
D6 = gen_BSplines(n, 10, 8, shift=0)
D7 = gen_BSplines(n, 10, 8, shift=2)
D8 = gen_BSplines(n, 10, 8, shift=4)
D9 = gen_BSplines(n, 10, 8, shift=6)
D10 = gen_BSplines(n, 10, 8, shift=8)
D = np.concatenate((D1,D2,D3,D4,D5,D6,D7,D8,D9,D10), axis=1)

D1 = gen_BSplines(m, 15, 4, shift=0)
D2 = gen_BSplines(m, 15, 4, shift=5)
D3 = gen_BSplines(m, 15, 4, shift=10)
D6 = gen_BSplines(m, 7, 8, shift=0)
D7 = gen_BSplines(m, 7, 8, shift=2)
D8 = gen_BSplines(m, 7, 8, shift=4)
D9 = gen_BSplines(m, 7, 8, shift=6)
Dmode2 = np.concatenate((D1,D2,D3,D6,D7,D8,D9), axis=1)

#plt.plot(D)
#plt.show()

# Ruinning the data with noise
sig = 3*1e2
#todo: SNR this
# notes:
# - do loop on noise? Some values that show nice gain: 3*1e2 (k=6), 1e2 (k=6), 1e3
#
Ynoise = Y + sig*np.random.randn(n,m,l)
SNR = 20*np.log(tl.norm(Y)/tl.norm(Ynoise - Y))

#plt.show()

# CPD on unnoised data for ground truth factors
out_true, err_true = non_negative_parafac_hals(Y,r,n_iter_max=1000,init='svd',verbose=True, return_errors=True)
out_true.normalize()


# Running nnCPD without smoothness
out_cp, err_cp = non_negative_parafac_hals(Ynoise,r,n_iter_max=1000,init='svd',verbose=True, return_errors=True)
out_cp.normalize()
Y_est_cp = out_cp.to_tensor()

#copying for init to avoid tl bug
out_cp_init = copy.deepcopy(out_cp)#(np.copy(out_cp[0]), [np.copy(out_cp[1][0]), np.copy(out_cp[1][1]), np.copy(out_cp[1][2])])
out_cp_init2 = copy.deepcopy(out_cp)#(np.copy(out_cp[0]), [np.copy(out_cp[1][0]), np.copy(out_cp[1][1]), np.copy(out_cp[1][2])])
out_cp_init3 = copy.deepcopy(out_cp)#(np.copy(out_cp[0]), [np.copy(out_cp[1][0]), np.copy(out_cp[1][1]), np.copy(out_cp[1][2])])

# Sparse coding cp first factor
DXcp = out_cp[1][0]
Xcp_omp = []
for i in range(r):
    # for each column perform omp
    x_omp = omp(DXcp[:,i], D, k)[0]
    Xcp_omp.append(x_omp)
Xcp_omp = np.array(Xcp_omp).T

out_cp_omp = copy.deepcopy(out_cp)#tl.cp_tensor.CPTensor((np.copy(out_cp[0]), [np.copy(out_cp[1][0]), np.copy(out_cp[1][1]), np.copy(out_cp[1][2])]))
out_cp_omp[1][0] = D@Xcp_omp
Y_est_cpomp = out_cp_omp.to_tensor()

# Sparse coding cp first factor
DXcp = out_cp[1][1]
X2cp_omp = []
for i in range(r):
    # for each column perform omp
    x_omp = omp(DXcp[:,i], Dmode2, k)[0]
    X2cp_omp.append(x_omp)
X2cp_omp = np.array(X2cp_omp).T

out_cp_omp2 = copy.deepcopy(out_cp)#tl.cp_tensor.CPTensor((np.copy(out_cp[0]), [np.copy(out_cp[1][0]), np.copy(out_cp[1][1]), np.copy(out_cp[1][2])]))
out_cp_omp2[1][0] = D@Xcp_omp
out_cp_omp2[1][1] = Dmode2@X2cp_omp
Y_est_cpomp2 = out_cp_omp2.to_tensor()

# Running Dictionary-based (nn todo)CPD for smoothness on the first mode
nonnegative = False
out_dcp, Xest, Sxest, err_dcp = dlra_parafac(Ynoise, r, [D], [k], verbose=True, n_iter_max = 20, init=out_cp_init, return_errors=True, lamb_rel = [1e-3], nonnegative=nonnegative, tau=5, X0=[np.copy(Xcp_omp)])
out_dcp2, Xest2, Sxest2, err_dcp2 = dlra_parafac(Ynoise, r, [D,Dmode2], [k,k], verbose=True, n_iter_max = 20, init=out_cp_init2, return_errors=True, lamb_rel = [1e-3,1e-3], nonnegative=nonnegative, tau=5, X0=[np.copy(Xcp_omp), np.copy(X2cp_omp)])
out_dcp3, Xest3, Sxest3, err_dcp3 = dlra_parafac(Ynoise, r, [D,Dmode2], [k,k], verbose=True, n_iter_max = 20, init=out_cp_init3, return_errors=True, lamb_rel = [1e-3,1e-3], nonnegative=True, tau=5, X0=[np.copy(Xcp_omp), np.copy(X2cp_omp)])
out_dcp.normalize()
Y_est_dcp = out_dcp.to_tensor()
out_dcp2.normalize()
Y_est_dcp2 = out_dcp2.to_tensor()
out_dcp3.normalize()
Y_est_dcp3 = out_dcp3.to_tensor()

# Reconstruction and validation errors
rec_cp = err_cp[-1]
rec_cp_omp = tl.norm(Ynoise - Y_est_cpomp,2)/tl.norm(Ynoise,2)
rec_cp_omp2 = tl.norm(Ynoise - Y_est_cpomp2,2)/tl.norm(Ynoise,2)
rec_dcp = err_dcp[-1]
rec_dcp2 = err_dcp2[-1]
rec_dcp3 = err_dcp3[-1]

val_cp = tl.norm(Y - Y_est_cp,2)/tl.norm(Y,2)
val_cp_omp = tl.norm(Y - Y_est_cpomp,2)/tl.norm(Y,2)
val_cp_omp2 = tl.norm(Y - Y_est_cpomp2,2)/tl.norm(Y,2)
val_dcp = tl.norm(Y - Y_est_dcp, 2)/tl.norm(Y,2)
val_dcp2 = tl.norm(Y - Y_est_dcp2, 2)/tl.norm(Y,2)
val_dcp3 = tl.norm(Y - Y_est_dcp3, 2)/tl.norm(Y,2)

# Storing results
dic = {
    'algorithm':['HALS noiseless']+2*['HALS', 'HALS+Sparse Coding mode 1', 'HALS+Sparse Coding modes 1&2', 'AO-DCPD mode 1', 'AO-DCPD modes 1&2', 'AO-nnDPCD modes 1&2'],
    'value': [err_true[-1], rec_cp, rec_cp_omp, rec_cp_omp2, rec_dcp, rec_dcp2, rec_dcp3, val_cp, val_cp_omp, val_cp_omp2, val_dcp, val_dcp2, val_dcp3],
    'error type': ['noiseless reconstruction error']+6*['train reconstruction error']+6*['test reconstruction error']
}
dic = pd.DataFrame(dic)
store_pd = store_pd.append(dic, ignore_index=True)

print('CP', rec_cp, val_cp)
print('CP_omp', rec_cp_omp, val_cp_omp)
print('CP_omp2', rec_cp_omp2, val_cp_omp2)
print('DCP', rec_dcp, val_dcp)
print('DCP bimode', rec_dcp2, val_dcp2)
print('DCP NN bimode', rec_dcp3, val_dcp3)
#store_val = [val_cp,val_cp_omp,val_cp_omp2,val_dcp,val_dcp2,val_dcp3]
#store_rec = [rec_cp,rec_cp_omp,rec_cp_omp2,rec_dcp,rec_dcp2,rec_dcp3]

# todo: redo nicer plots with plotly

# showing a slice with and without noise, and reconstruction
# here third slice
plt.subplot(181)
plt.imshow(Y[:,:,3])
plt.title('Original Data')
plt.xlabel('rec.err='+str(np.round(err_true[-1],3)))
plt.subplot(182)
plt.imshow(Ynoise[:,:,3])
plt.title('Noisy data')
plt.xlabel('SNR='+str(np.round(SNR,1)))
plt.subplot(183)
plt.imshow(Y_est_cp[:,:,3])
plt.title('HALS')
plt.xlabel(np.round(val_cp,3))
plt.subplot(184)
plt.imshow(Y_est_cpomp[:,:,3])
plt.title('HALS+SC mode 1')
plt.xlabel(np.round(val_cp_omp,3))
plt.subplot(185)
plt.imshow(Y_est_cpomp2[:,:,3])
plt.title('HALS+SC mode 1&2')
plt.xlabel(np.round(val_cp_omp2,3))
plt.subplot(186)
plt.imshow(Y_est_dcp[:,:,3])
plt.title('AO-DCPD mode 1')
plt.xlabel(np.round(val_dcp,3))
plt.subplot(187)
plt.imshow(Y_est_dcp2[:,:,3])
plt.title('AO-DCPD modes 1&2')
plt.xlabel(np.round(val_dcp2,3))
plt.subplot(188)
plt.imshow(Y_est_dcp3[:,:,3])
plt.title('AO-nnDPCD modes 1&2')
plt.xlabel(np.round(val_dcp3,3))
plt.show()

plt.figure()
plt.subplot(371)
plt.plot(out_true[1][0])
plt.title('HALS noiseless')
plt.ylabel('First mode factor')
plt.subplot(372)
plt.plot(out_cp[1][0])
plt.title('HALS')
plt.subplot(373)
plt.plot(out_cp_omp[1][0])
plt.title('HALS+SC mode 1')
plt.subplot(374)
plt.plot(out_cp_omp2[1][0])
plt.title('HALS+SC mode 1&2')
plt.subplot(375)
plt.plot(out_dcp[1][0])
plt.title('AO-DCPD mode 1')
plt.subplot(376)
plt.plot(out_dcp2[1][0])
plt.title('AO-DCPD modes 1&2')
plt.subplot(377)
plt.plot(out_dcp3[1][0])
plt.title('AO-nnDPCD modes 1&2')
plt.subplot(378)
plt.plot(out_true[1][1])
plt.ylabel('Second mode factor')
plt.subplot(379)
plt.plot(out_cp[1][1])
plt.subplot(3,7,10)
plt.plot(out_cp_omp[1][1])
plt.subplot(3,7,11)
plt.plot(out_cp_omp2[1][1])
plt.subplot(3,7,12)
plt.plot(out_dcp[1][1])
plt.subplot(3,7,13)
plt.plot(out_dcp2[1][1])
plt.subplot(3,7,14)
plt.plot(out_dcp3[1][1])
plt.subplot(3,7,15)
plt.plot(out_true[1][2])
plt.ylabel('Third mode factor')
plt.subplot(3,7,16)
plt.plot(out_cp[1][2])
plt.subplot(3,7,17)
plt.plot(out_cp_omp[1][2])
plt.subplot(3,7,18)
plt.plot(out_cp_omp2[1][2])
plt.subplot(3,7,19)
plt.plot(out_dcp[1][2])
plt.subplot(3,7,20)
plt.plot(out_dcp2[1][2])
plt.subplot(3,7,21)
plt.plot(out_dcp3[1][2])

#plt.show()

year = 2021
month = 10
day = 20
path = '../..'
stor_name = '{}-{}-{}'.format(year,month,day)
store_pd.to_pickle('{}/data/XP_chemo/{}_results'.format(path,stor_name))
#np.save('{}/data/XP_chemo/{}_{}_val_{}'.format(path,stor_name,n,sig), store_val)
#np.save('{}/data/XP_chemo/{}_{}_rec_{}'.format(path,stor_name,n,sig), store_rec)
