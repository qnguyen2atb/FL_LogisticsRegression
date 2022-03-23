from lib import *

def plot_f1(f1_local, f1_preunlearn_final, f1_global=None, figname='default'):
    local_T = list(f1_local.T)
    preunlearn_T = list(f1_preunlearn_final.T)

    fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime']
    if f1_global is not None:
        global_T = list(f1_global.T)
        ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
    
    #ax1.scatter(range(np.shape(local_T)[0]), local_T, alpha=0.9, color='red', label='Local Models')
    ax1.scatter(range(np.shape(local_T)[0]), preunlearn_T, alpha=0.9, color='green', label='Pre-Unlearned Global Model')

    
    ax1.set_xlabel('Clients #',fontsize=16)
    ax1.set_ylabel('F1 Score (%)',fontsize=16)
    ax1.legend(fontsize=16)
    ax1.set_ylim([60, 90])
    
    plt.sca(ax1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if figname == 'default':
        plt.savefig('plots/f1_score.png')
    else:
        plt.savefig('plots/'+figname)

