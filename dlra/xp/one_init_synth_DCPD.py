from dlra.algorithms import dlra_parafac, palm_Dparafac
from mscode.utils.utils import count_support, redundance_count
from mscode.utils.generator import gen_mix, initialize, gen_mix_tensor
import tensorly as tl
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from mscode.methods.algorithms import omp
from itertools import permutations
import copy
import pandas as pd

# Seeding
np.random.seed(seed=0)

# Generation
k = 8 #8
r = 6 #5
m = 20 #20
n = 21 #30
l = 22 #10
d = 30 #41
SNR = 30
cond = 2*1e2
distr = 'Gaussian'
tol = 1e-6

#store = []
#store_sup = []

store_pd = pd.DataFrame(columns=["value", "error type", "algorithm"])

N = 100

for iter in range(N):
    # Data generation
    Y, Ytrue, D, B, C, X, S, sig, condB = gen_mix_tensor([m, n, l], [d, r], k, snr=SNR, cond=cond, cond_rand=False, distr=distr)

    # to optimize
    lamb=1*1e-4

    #cp_tensor_oracle, Xoracle, Soracle, err_oracle = dlra_mf(Y, r, D, k, lamb_rel=lamb, init=init_cp,  return_errors=True, inner_iter_max=1000, verbose=True)
    #cp_tensor_est, _, _, err = dlra_mf(Y, r, D, k, lamb_rel=lamb, init='random',  return_errors=True, inner_iter_max=1000, n_iter_max=100, verbose=True)
    verbose = False

    # Initialize with Parafac, which first mode is sparse coded in D
    init, err_init = tl.decomposition.parafac(Y, r, n_iter_max=100, init='svd', verbose=verbose, return_errors=True)
    DXinit = init[1][0]
    Xinit = np.zeros([d,r])
    Sinit = np.zeros([k,r])
    for i in range(r):
        Xinit[:,i],Sinit[:,i] = omp(DXinit[:,i], D, k)
    init[1][0] = D@Xinit
    Yinit = tl.cp_tensor.cp_to_tensor(init)
    err_init.append(tl.norm(Y-Yinit,2)/tl.norm(Y,2))

    # brute force my way out of lazyness for permutation, use Munkres instead
    val_omp = 0
    toto = [l for l in range(r)]
    perms = set(permutations(toto))
    for perm in perms:
        val_new = count_support(S, Sinit[:,perm])
        if val_new>val_omp:
            val_omp=val_new

    # CPD Init + PALM
    cp_tensor_palm, X_palm, S_palm, err_palm = palm_Dparafac(Y, r, D, k, init=copy.deepcopy(init), X0=np.copy(Xinit),  return_errors=True, n_iter_max=1000, verbose=verbose, tol=1e-16, step_rel=1)
    # does not work, stepsize too hard to tune

    # CPD Init + AO-DLRA
    # weird syntax to allow several constrained modes
    cp_tensor_est3, X3, S3, err3 = dlra_parafac(Y, r, [D], [k], lamb_rel=[lamb], init=copy.deepcopy(init), X0=[np.copy(Xinit)],  return_errors=True, inner_iter_max=1000, n_iter_max=100, verbose=verbose, tol=1e-16)


    # brute force my way out of lazyness for permutation
    S3 = S3[0]
    val = 0
    val1 = 0
    for perm in perms:
        val_new = count_support(S, S3[:,perm])
        val1_new = count_support(S, S_palm[:,perm])
        if val_new>val:
            val=val_new
        if val1_new>val1:
            val1=val1_new

    #print('error solution', tl.norm(Y - Ytrue,2)/tl.norm(Ytrue,2))
    print('initial error', err_init[-1], val_omp)
    #print('final error random init BCD', err2[-1])
    print('final error CP init AO-DLRA', np.min(err3), val)
    print('final error CP init iPALM', err_palm[-1], val1)

    # Random Init
    #----------------
    # Generating same random init
    init_random = tl.random.random_cp((m,n,l), r)
    init_random[0] = tl.ones(r) #unit weights

    # PALM, random Init
    cp_tensor_est_palm_r, X_palm_r, S_palm_r, err_palm_r = palm_Dparafac(Y, r, D, k, init=copy.deepcopy(init_random), return_errors=True, n_iter_max=1000, verbose=verbose, tol=1e-16, step_rel=1)

    # AO-DLRA, random init
    cp_tensor_est4, X4, S4, err4 = dlra_parafac(Y, r, [D], [k], lamb_rel=[lamb], init=copy.deepcopy(init_random), return_errors=True, inner_iter_max=1000, n_iter_max=100, verbose=verbose, tol=1e-16)

    # AO-DLRA, PALM init from random
    cp_tensor_est5, X5, S5, err5 = dlra_parafac(Y, r, [D], [k], lamb_rel=[lamb], init=copy.deepcopy(cp_tensor_est_palm_r), X0 = [np.copy(X_palm_r)], return_errors=True, inner_iter_max=1000, n_iter_max=100, verbose=verbose, tol=1e-16)

    # Counting support recovery
    S4 = S4[0]
    S5 = S5[0]
    val2 = 0
    val5 = 0
    val_palm = 0
    for perm in perms:
        val2_new = count_support(S, S4[:,perm])
        val5_new = count_support(S, S5[:,perm])
        val_new_palm = count_support(S, S_palm_r[:,perm])
        if val2_new>val2:
            val2=val2_new
        if val5_new>val5:
            val5=val5_new
        if val_new_palm>val_palm:
            val_palm=val_new_palm


    print('final error random init AO', np.min(err4), val2)
    print('final error PALM init AO', np.min(err5), val5)
    print('final error random init PALM', err_palm_r[-1], val_palm)

    # Storing in Pandas Dataframe
    dic = {
    "value": [err_init[-1], np.min(err3), np.min(err4),np.min(err5), err_palm[-1], err_palm_r[-1], val_omp, val, val2, val5, val1, val_palm],
    'error type': 6*['relative error']+6*['support recovery'],
    'algorithm':2*['ALS + Sparse Coding', 'AO-DLRA, ALS init', 'AO-DLRA, random init', 'AO-DLRA, iPALM init', 'iPALM, ALS init', 'iPALM, random init']
    }
    data = pd.DataFrame(dic)
    store_pd = store_pd.append(data, ignore_index=True)

    #store.append([err_init[-1], np.min(err3), np.min(err4),np.min(err5),err_palm_r[-1]])
    #store_sup.append([val_omp, val, val2, val5, val_palm])


#store = np.array(store)
#store_sup = np.array(store_sup)

# Post-processing

#fig = px.box(data_frame=store, points='all', y="Relative reconstruction error")
fig = px.box(store_pd[store_pd.T.iloc[1]=='relative error'], x="algorithm", y="value", points='all', color="algorithm",
labels={
    'value': 'Relative Reconstruction Error',
    'algorithm':''
}, title="DCPD, relative reconstruction error" )
fig.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    showlegend=False
)
fig.show()
fig2 = px.box(store_pd[store_pd.T.iloc[1]=='support recovery'], x="algorithm", y="value", points='all', color="algorithm",
labels={
    'value': 'Support Recovery (%)',
    'algorithm':''
}, title="DPCD, support recovery" )
fig2.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    showlegend=False
)
fig2.show()


#plt.subplot(121)
#plt.semilogy(err2)
#plt.semilogy(err4)
#plt.legend(['plain', 'unbiais'])
#plt.subplot(122)
#plt.semilogy(err_init)
#plt.semilogy(err3)
#plt.semilogy(err4)
#plt.semilogy(err5)
#plt.semilogy(err6)
#plt.legend(['CP','CPtoAO', 'randomAO'])
#plt.show()

year = 2021
month = 10
day = 20
path = '../..'
stor_name = '{}-{}-{}'.format(year,month,day)
#store_pd.to_pickle('{}/data/XP_synth/{}_DCPD'.format(path,stor_name))
#fig.write_image('{}/data/XP_synth/{}_DCPD_plot1.pdf'.format(path,stor_name))
#fig2.write_image('{}/data/XP_synth/{}_DCPD_plot2.pdf'.format(path,stor_name))
# Frontiers export
fig.write_image('{}/data/XP_synth/{}_DCPD_plot1.jpg'.format(path,stor_name))
fig2.write_image('{}/data/XP_synth/{}_DCPD_plot2.jpg'.format(path,stor_name))

# to load data
#store_pd = pd.read_pickle('{}/data/XP_synth/{}_DCPD'.format(path,stor_name))
