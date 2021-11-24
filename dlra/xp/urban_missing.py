from dlra.algorithms import dlra_parafac, dlra_mf, dlra_mf_bcd, dlra_mf_iht
from dlra.utils import sam
from mscode.utils.utils import count_support, redundance_count
from mscode.utils.generator import gen_mix, initialize
from mscode.methods.algorithms import ista, omp
from mscode.methods.proxs import HardT
#import tensorly as tl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.io
from dlra.xp.genDCT import genDCT
import copy

# Seeding
np.random.seed(seed=0)

# Loading the data
# root at this file
dictio = scipy.io.loadmat('../../data/XP_completion/Urban.mat')

# dict is a python dictionnary. It contains the matrix we want to NMF
Yall = dictio['A']

# Extracting a 20x20 patch
n = 20
m = 162
HSI = np.transpose(np.reshape(Yall, [307, 307, m]),[1,0,2])
Sliced_HSI = HSI[70:70+n,100:100+n,:]
#plt.imshow(Sliced_HSI[:,:,10])
#plt.show()
Y = np.reshape(Sliced_HSI,[n*n, m])
#Y = Y/np.linalg.norm(Y)
verbose = 0


# Building the 2DCT dictionary
D = genDCT([n,n], 1)
# model parameters
k = 50
r = 4
lamb = 5e-3 # 5e-3

# DataFrame to store results
store_pd = pd.DataFrame(columns=["value", "error type", "sparsity", "algorithm"])

### First, applying DLRA to Y for sanity check
#
#Xinit = np.random.randn(n*n,r)
#Binit = np.random.randn(m,r)
##Scaling B
##DX = D@Xinit
##DXtDX = DX.T@DX
##DXY = DX.T@Y
##Bt = np.linalg.solve(DXtDX, DXY)
##Binit = Bt.T
##Xinit,_,_,_ = ista(Y, D, Binit, lamb, k=k, itermax=1000, verbose=False, X0=Xinit, tol=1e-8)
#
#out0, X0s, _, err0 = dlra_mf_iht(Y, r, D, k, init=copy.deepcopy([Xinit,Binit]),  return_errors=True, verbose=verbose, step_rel=1, n_iter_max=100)
#out, X, _, err = dlra_mf(Y, r, D, k, lamb_rel=lamb, init=copy.deepcopy([X0s, out0[1]]),  return_errors=True, inner_iter_max=10000, n_iter_max=10, verbose=verbose, method='ista', tau=20, itermax_calib=100)
#out, X, _, err2 = dlra_mf(Y, r, D, k, lamb_rel=lamb, init=copy.deepcopy([Xinit,Binit]),  return_errors=True, inner_iter_max=10000, n_iter_max=50, verbose=verbose, method='ista', tau=20, itermax_calib=100)
## Estimated images
#Ye0s = D@X0s@out0[1].T
#Ye = D@X@out[1].T
##HSIe0 = np.reshape(Ye0, [n, n, m])
#HSIe0s = np.reshape(Ye0s, [n, n, m])
#HSIe = np.reshape(Ye, [n, n, m])
#plt.subplot(311)
#plt.imshow(Sliced_HSI[:,:,10])
#plt.subplot(312)
#plt.imshow(HSIe[:,:,10])
#plt.subplot(313)
#plt.imshow(HSIe0s[:,:,10])
#plt.show()
#
# Now we try to infer the missing pixels
#miss = [4,7,40, 200, 266, 479, 800]
miss = np.random.permutation(n**2)[:50]
Ymiss = np.delete(Y, miss, axis=0)
Dmiss = np.delete(D, miss, axis=0)
rec=[]
val=[]
val_sam=[]
klist = [10, 30, 50, 70, 100, 120, 150, 200, 250]


N = 20
for toto in range(N):
    for k in klist:

        # initialization
        Xinit = np.random.randn(n*n,r)
        Binit = np.random.randn(m,r)

        #out0, X0s,_, err0 = dlra_mf_iht(Ymiss, r, Dmiss, k, init=[Xinit,Binit],  return_errors=True, verbose=verbose, step_rel=0.5, n_iter_max=10)
        #out, X, _, err = dlra_mf(Ymiss, r, Dmiss, k, lamb_rel=lamb, init=[X0s, out0[1]],  return_errors=True, inner_iter_max=1000, n_iter_max=10, verbose=verbose, method='ista', tau=10)
        out, X, _, err = dlra_mf(Ymiss, r, Dmiss, k, lamb_rel=lamb, init=copy.deepcopy([Xinit,Binit]),  return_errors=True, inner_iter_max=1000, n_iter_max=40, verbose=verbose, method='ista', tau=20, itermax_calib=100)

        B = out[1]
        # Reconstructing missing pixels
        Yrec = D@X@B.T
        val = np.linalg.norm(Y[miss,:] - Yrec[miss,:])/np.linalg.norm(Y[miss,:])

        #plt.semilogy(err)

        # Compute SAM
        val_samt = []
        for j in range(miss.shape[0]):
            val_samt.append(sam(Yrec[miss[j],:], Y[miss[j],:]))
        val_sam = np.mean(val_samt)

        print(np.min(err), val, val_sam)

        # Storing results in DataFrame
        dic = {
        "value": [np.min(err), val, val_sam],
        'error type': ['relative train error', 'relative test error', 'SAM' ],
        'sparsity': [k,k,k],
        'algorithm': 3*['AO-DLRA']
        }
        data = pd.DataFrame(dic)
        store_pd = store_pd.append(data, ignore_index=True)

    #miss_image = np.zeros(n**2)
    #miss_image[miss] = 1
    #miss_image = np.reshape(miss_image,[n,n])
    #plt.subplot(6,4,11)
    #plt.imshow(miss_image)

    #plt.subplot(6,4,12)
    #plt.imshow(Sliced_HSI[:,:,70])
    #plt.subplot(6,4,24)
    #plt.plot(Y[miss[:5],:].T)

# Comparison with image per image sparse coding using omp

print(' Running OMP Columnwise ')
print('-------------------------')

for k in klist[:6]:

    X_omp = []

    for i in range(Ymiss.shape[1]):
        # for each column perform omp
        X_omp_temp = omp(Ymiss[:,i], Dmiss, k)[0]
        X_omp.append(X_omp_temp)

    X_omp = np.array(X_omp).T
    #X_omp = HardT(DtY_miss, k)

    Yrec_omp = D@X_omp
    val = np.linalg.norm(Y[miss,:] - Yrec_omp[miss,:])/np.linalg.norm(Y[miss,:])
    rec = np.linalg.norm(Ymiss - Dmiss@X_omp)/np.linalg.norm(Ymiss)

    # Compute SAM
    val_samt_omp = []
    for j in range(miss.shape[0]):
        val_samt_omp.append(sam(Yrec_omp[miss[j],:], Y[miss[j],:]))
    val_sam = np.mean(val_samt_omp)

    print(rec, val, val_sam)


    # Storing results in DataFrame
    dic = {
    "value": [rec, val, val_sam],
    'error type': ['relative train error', 'relative test error', 'SAM' ],
    'sparsity': [k,k,k],
    'algorithm': 3*['Columnwise OMP']
    }
    data = pd.DataFrame(dic)
    store_pd = store_pd.append(data, ignore_index=True)


fig = px.box(store_pd[store_pd.T.iloc[1]=='relative test error'], x="sparsity", y="value", points='all', color="algorithm",
labels={
    'value': 'Relative testing reconstruction error',
    "sparsity": "Sparsity level k"
}, title="Reconstuction error on missing pixels")
fig.update_xaxes(type="category")
fig.update_layout(
    font_family="HelveticaBold",
    font_size=20,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    showlegend=False,
    width=800,
    height=650,
)
fig.show()
fig2 = px.box(store_pd[store_pd.T.iloc[1]=='SAM'], x="sparsity", y="value", points='all', color="algorithm",
labels={
    'value': 'Spectral Angular Mapper',
    "sparsity": "Sparsity level k"
}, title="Average SAM for missing spectra" )
fig2.update_xaxes(type="category")
fig2.update_layout(
    font_family="HelveticaBold",
    font_size=20,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    width=800,
    height=650,
    #showlegend=False
)
fig2.show()
fig3 = px.box(store_pd[store_pd.T.iloc[1]=='relative train error'], x="sparsity", y="value", points='all', color="algorithm",
labels={
    'value': 'Relative training reconstruction error',
    "sparsity": "Sparsity level k"
}, title="Reconstruction error on known pixels" )
fig3.update_xaxes(type="category")
fig3.update_layout(
    font_family="HelveticaBold",
    font_size=20,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    width=800,
    height=650,
    showlegend=False
)
fig3.show()

year = 2021
month = 10
day = 20
path = '../..'
stor_name = '{}-{}-{}'.format(year,month,day)
#store_pd.to_pickle('{}/data/XP_completion/{}_results'.format(path,stor_name))
#fig.write_image('{}/data/XP_completion/{}_plot1.pdf'.format(path,stor_name))
#fig2.write_image('{}/data/XP_completion/{}_plot2.pdf'.format(path,stor_name))
#fig3.write_image('{}/data/XP_completion/{}_plot3.pdf'.format(path,stor_name))
# For Frontiers export
#fig.write_image('{}/data/XP_completion/{}_plot1.jpg'.format(path,stor_name))
#fig2.write_image('{}/data/XP_completion/{}_plot2.jpg'.format(path,stor_name))
#fig3.write_image('{}/data/XP_completion/{}_plot3.jpg'.format(path,stor_name))
#
# to load data
#store_pd = pd.read_pickle('{}/data/XP_completion/{}_results'.format(path,stor_name))
