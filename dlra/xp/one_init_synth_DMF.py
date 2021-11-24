from dlra.algorithms import dlra_parafac, dlra_mf, dlra_mf_bcd, dlra_mf_iht
from mscode.utils.utils import count_support, redundance_count
from mscode.utils.generator import gen_mix, initialize
import tensorly as tl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import permutations
import plotly.express as px
import copy

# Seeding
np.random.seed(seed=0)

# Generation
k = 8 #2
r = 6 #2
n = 50 #10
m = 50 #20
d = 60 #50
#noise = 0.03 # 0.03
SNR = 100  # dB # almost no noise
cond = 2*1e2
distr = 'Gaussian'
tol = 1e-6

verbose = 0
store = []
store_sup = []

N = 100

store_pd = pd.DataFrame(columns=["value", "error type", "algorithm"])

for i in range(N):

    # Data gen
    Y, Ytrue, D, B, X, S, sig, condB = gen_mix([m, n, d, r], k, snr=SNR, cond=cond, cond_rand=False, distr=distr)

    # Initialize with true factors for sanity check
    #init = [X,B]
    lamb=1*1e-2

    # Fix random init
    X0 = np.random.randn(d,r)
    B0 = np.random.randn(m,r)
    init = [X0,B0]

    # PALM, random init # how many iters
    cp_tensor_est2, X2, S2, err2 = dlra_mf_iht(Y, r, D, k, init=copy.deepcopy(init),  return_errors=True, verbose=verbose, step_rel=0.50, n_iter_max=3000)
    # AO-DLRA, random init
    cp_tensor_est6, X6, S6, err6 = dlra_mf(Y, r, D, k, lamb_rel=lamb, init=copy.deepcopy(init),  return_errors=True, inner_iter_max=1000, n_iter_max=100, verbose=verbose)
    # AO-DLRA, PALM init from random
    cp_tensor_est3, X3, S3, err3 = dlra_mf(Y, r, D, k, lamb_rel=lamb, init=copy.deepcopy([X2,cp_tensor_est2[1]]),  return_errors=True, inner_iter_max=1000, n_iter_max=100, verbose=verbose)

    # Solving for permutations and estimating support recovery
    val_2 = 0
    val_6 = 0
    val_3 = 0
    toto = [l for l in range(r)]
    perms = set(permutations(toto))
    for perm in perms:
        val_2_new = count_support(S, S2[:,perm])
        val_6_new = count_support(S, S6[:,perm])
        val_3_new = count_support(S, S3[:,perm])
        if val_2_new>val_2:
            val_2=val_2_new
        if val_6_new>val_6:
            val_6=val_6_new
        if val_3_new>val_3:
            val_3=val_3_new

    print('error solution', np.linalg.norm(Y - D@X@B.T))
    print('final error PALM, random init', err2[-1], val_2)
    print('final error AO-DLRA, PALM init', err3[-1], val_3)
    print('final error AO-DLRA, random init', err6[-1], val_6)

    #store.append([err2[-1],err3[-1], err6[-1]])
    #store_sup.append([val_2,val_3,val_6])

    # Storing in Pandas Dataframe
    dic = {
    "value": [err2[-1], np.min(err3), np.min(err6), val_2, val_3, val_6],
    'error type': 3*['relative error']+3*['support recovery'],
    'algorithm':2*['iPALM, random init', 'AO-DLRA, random init', 'AO-DLRA, iPALM init']
    }
    data = pd.DataFrame(dic)
    store_pd = store_pd.append(data, ignore_index=True)


fig = px.box(store_pd[store_pd.T.iloc[1]=='relative error'], x="algorithm", y="value", points='all', color="algorithm",
color_discrete_sequence= ['#00cc96', '#ab63fa','#19d3f3'],
labels={
    'value': 'Relative Reconstruction Error',
    'algorithm':''
}, title="DMF, relative reconstruction error", category_orders={"algorithm":["AO-DLRA, random init", "AO-DLRA, iPALM init", "iPALM, random init"]})
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
color_discrete_sequence=['#00cc96', '#ab63fa','#19d3f3'], labels={
    'value': 'Support Recovery (%)',
    'algorithm':''
}, title="DMF, support recovery", category_orders={"algorithm":["AO-DLRA, random init", "AO-DLRA, iPALM init", "iPALM, random init"]})
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
#store = np.array(store)
#store_sup = np.array(store_sup)
#fig = px.box(store, points="all")
#fig.show()

#fig2 = px.box(store_sup, points="all")
#fig2.show()

#plt.subplot(121)
#plt.semilogy(err2)
#plt.semilogy(err4)
#plt.legend(['plain', 'unbiais'])
#plt.subplot(122)
#plt.semilogy(err3)
#plt.semilogy(err5)
#plt.semilogy(err6)
#plt.legend(['BCDtoAO',  'BCDutoAO', 'randomAO'])
#plt.show()


year = 2021
month = 10
day = 20
path = '../..'
stor_name = '{}-{}-{}'.format(year,month,day)
#store_pd.to_pickle('{}/data/XP_synth/{}_DMF'.format(path,stor_name))
#fig.write_image('{}/data/XP_synth/{}_DMF_plot1.pdf'.format(path,stor_name))
#fig2.write_image('{}/data/XP_synth/{}_DMF_plot2.pdf'.format(path,stor_name))
# Frontiers export
#fig.write_image('{}/data/XP_synth/{}_DMF_plot1.jpg'.format(path,stor_name))
#fig2.write_image('{}/data/XP_synth/{}_DMF_plot2.jpg'.format(path,stor_name))
#
# to load data
#store_pd = pd.read_pickle('{}/data/XP_synth/{}_DMF'.format(path,stor_name))
