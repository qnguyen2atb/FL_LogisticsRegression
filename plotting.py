from lib import *

def plot_comparision(f1_l_final_global, f1_l):
    #comparision between global model and local model
    figname = 'comparision_global_local'
    n_bins = 10
    fig, ((ax0)) = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
    colors = ['red', 'tan', 'lime']
    ax0.hist(np.array(f1_l_final_global) - np.array(f1_l), n_bins, density=False, range=(0,10), histtype='bar')#, color=colors, label=colors)
    ax0.legend(prop={'size': 10})
    ax0.set_title('Intercept', fontsize=18)
    fig.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if figname == 'default':
        plt.savefig('plots/comparision_global_local.png')
    else:
        plt.savefig('plots/compa_'+figname)    


def plot_f1(f1_local, f1_preunlearn_final,  f1_global=None, plot_l= [], figname='default'):
    
    print('QUANG test', plot_l)
    local_T = list(f1_local.T)
    preunlearn_T = list(f1_preunlearn_final.T)
    fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime'] 
    if 'global' in plot_l:
        if f1_global is not None:
            global_T = list(f1_global.T)
            ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
    if 'local' in plot_l: 
        ax1.scatter(range(np.shape(local_T)[0]), local_T, alpha=0.9, color='red', label='Local Models')
    if 'preunlearn' in plot_l:
        ax1.scatter(range(np.shape(local_T)[0]), preunlearn_T, alpha=0.9, color='green', label='Pre-Unlearned Global Model')

    ax1.set_xlabel('Clients #',fontsize=16)
    ax1.set_ylabel('F1 Score (%)',fontsize=16)
    ax1.legend(fontsize=16)
    ax1.set_ylim([60, 90])
    plt.title(figname)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if figname == 'default':
        plt.savefig('plots/f1_score.png')
    else:
        plt.savefig('plots/'+figname)

    
def plot_improve(f1_local, f1_preunlearn_final,  f1_global=None, plot_l= [], figname='default'):
    
    print('COMPARISION', f1_global - f1_local)
    local_T = list(f1_local.T)
    preunlearn_T = list(f1_preunlearn_final.T)
    fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime'] 
    if 'global' in plot_l:
        if f1_global is not None:
            global_T = list(f1_global.T)
            ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
    if 'local' in plot_l: 
        ax1.scatter(range(np.shape(local_T)[0]), local_T, alpha=0.9, color='red', label='Local Models')
    if 'preunlearn' in plot_l:
        ax1.scatter(range(np.shape(local_T)[0]), preunlearn_T, alpha=0.9, color='green', label='Pre-Unlearned Global Model')

    ax1.set_xlabel('Clients #',fontsize=16)
    ax1.set_ylabel('F1 Score (%)',fontsize=16)
    ax1.legend(fontsize=16)
    ax1.set_ylim([60, 90])
    plt.title(figname)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


def plot_hist(prob, figname='default'):

    fig, ax = plt.subplots(figsize=[15,10])
    x = prob
    plt.hist(x, density=True, bins=20)
    plt.xlabel('Predicted Probability')
    ax.set_ylabel('Number of clients')
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()
    if figname=='default':
        plt.savefig('plots/prob_hist.png')
    else:
        plt.savefig('plots/'+figname+'.png')


def plot_coef_dist(f1_l, error, coef_l, intercept_l, figname='default'):

    n_bins = 10
    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5),(ax6, ax7)) = plt.subplots(nrows=4, ncols=2, figsize=(20,15))
    colors = ['red', 'tan', 'lime']
    ax0.hist(intercept_l, n_bins, density=False, histtype='bar')#, color=colors, label=colors)
    ax0.legend(prop={'size': 10})
    ax0.set_title('Intercept', fontsize=18)

    ax1.hist(coef_l[:,:,0], n_bins, density=False, histtype='bar')
    ax1.set_title('Coefficient 0', fontsize=18)

    ax2.hist(coef_l[:,:,1], n_bins, density=False, histtype='bar')
    ax2.set_title('Coefficient 1', fontsize=18)

    ax3.hist(coef_l[:,:,2], n_bins, density=False, histtype='bar')
    ax3.set_title('Coefficient 2', fontsize=18)

    ax4.hist(coef_l[:,:,3], n_bins, density=False, histtype='bar')
    ax4.set_title('Coefficient 3', fontsize=18)
    #ax4.set_xlim([-1,1])

    ax5.hist(coef_l[:,:,4], n_bins, density=False, histtype='bar')
    ax5.set_title('Coefficient 4', fontsize=18)
    #ax5.set_xlim([-1,1])

    ax6.hist(f1_l, n_bins, density=False, histtype='bar')
    ax6.set_title('F1 score of local models', fontsize=18)
    #ax5.set_xlim([-1,1])

    ax7.hist(error, n_bins, density=False, histtype='bar')
    ax7.set_title('Error of the coeffiecient', fontsize=18)
    #ax5.set_xlim([-1,1])

    fig.tight_layout()
    if figname=='default':
        plt.savefig('plots/coeff_dist.png')
    else:
        plt.savefig('plots/'+figname+'.png')


def plot_f1(f1_local, f1_preunlearn_final,  f1_global=None, plot_l= [], figname='default'):
    
    print('QUANG test', plot_l)
    local_T = list(f1_local.T)
    preunlearn_T = list(f1_preunlearn_final.T)
    fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime'] 
    #ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
    if 'global' in plot_l:
        if f1_global is not None:
            global_T = list(f1_global.T)
            ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
    if 'local' in plot_l: 
        ax1.scatter(range(np.shape(local_T)[0]), local_T, alpha=0.9, color='red', label='Local Models')
    if 'preunlearn' in plot_l:
        ax1.scatter(range(np.shape(local_T)[0]), preunlearn_T, alpha=0.9, color='green', label='Pre-Unlearned Global Model')

    ax1.set_xlabel('Clients #',fontsize=16)
    ax1.set_ylabel('F1 Score (%)',fontsize=16)
    ax1.legend(fontsize=16)
    ax1.set_ylim([60, 90])
    plt.title(figname)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if figname == 'default':
        plt.savefig('plots/f1_score.png')
    else:
        plt.savefig('plots/'+figname)

    
def plot_improve(f1_local, f1_preunlearn_final,  f1_global=None, plot_l= [], figname='default'):
    
    print('COMPARISION', f1_global - f1_local)
    local_T = list(f1_local.T)
    preunlearn_T = list(f1_preunlearn_final.T)
    fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime'] 
    #ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
    if 'global' in plot_l:
        if f1_global is not None:
            global_T = list(f1_global.T)
            ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
    if 'local' in plot_l: 
        ax1.scatter(range(np.shape(local_T)[0]), local_T, alpha=0.9, color='red', label='Local Models')
    if 'preunlearn' in plot_l:
        ax1.scatter(range(np.shape(local_T)[0]), preunlearn_T, alpha=0.9, color='green', label='Pre-Unlearned Global Model')

    ax1.set_xlabel('Clients #',fontsize=16)
    ax1.set_ylabel('F1 Score (%)',fontsize=16)
    ax1.legend(fontsize=16)
    ax1.set_ylim([60, 90])
    plt.title(figname)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #fig.suptitle(figname)# {binary_or_multiclass}')
    '''
    if figname == 'default':
        plt.savefig('plots/f1_score.png')
    else:
        plt.savefig('plots/compa_'+figname)
    '''



