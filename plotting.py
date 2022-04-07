from lib import *


def plot_f1(f1_local, f1_preunlearn_final,  f1_global=None, plot_l= [], figname='default'):
    

    local_T = list(f1_local.T)
    preunlearn_T = list(f1_preunlearn_final.T)
    fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime'] 
    #ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')

    for i in range(np.shape(local_T)[0]):
        if f1_global[i] > local_T[i]:
            global_T = list(f1_global.T)
            ax1.scatter(i,  global_T[i], alpha=0.9,  color='blue')
        else: 
            ax1.scatter(i, local_T[i], alpha=0.9, color='red')    
     
    '''
    if 'global' in plot_l:
        if f1_global is not None:
            global_T = list(f1_global.T)
            ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
    if 'local' in plot_l: 
        ax1.scatter(range(np.shape(local_T)[0]), local_T, alpha=0.9, color='red', label='Local Models')
    if 'preunlearn' in plot_l:
        ax1.scatter(range(np.shape(local_T)[0]), preunlearn_T, alpha=0.9, color='green', label='Pre-Unlearned Global Model')
    '''
    ax1.set_xlabel('Clients #',fontsize=16)
    ax1.set_ylabel('F1 Score (%)',fontsize=16)
    ax1.legend(['Unlearned Global Model', 'Local Models'], fontsize=16)
    ax1.set_ylim([60, 90])
    plt.title(figname)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #fig.suptitle(figname)# {binary_or_multiclass}')
   
    if figname == 'default':
        plt.savefig('plots/f1_score.png')
    else:
        plt.savefig('plots/'+figname)


    fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime'] 
    #ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')

    ax1.scatter(range(np.shape(local_T)[0]), local_T, alpha=0.9, color='red')    
    ax1.scatter(range(np.shape(local_T)[0]),  preunlearn_T, alpha=0.9,  color='green')

     
    ax1.set_xlabel('Clients #',fontsize=16)
    ax1.set_ylabel('F1 Score (%)',fontsize=16)
    ax1.legend(['Pre-Unlearned Global Model', 'Local Models'], fontsize=16)
    ax1.set_ylim([60, 95])
    plt.title(figname)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #fig.suptitle(figname)# {binary_or_multiclass}')
   
    if figname == 'default':
        plt.savefig('plots/f1_local_vs_preunlearn_score.png')
    else:
        plt.savefig('plots/local_vs_preunlearn_'+figname)



def plot_improve(f1_local, f1_preunlearn_final,  f1_global=None, plot_l= [], figname='default'):
    
    print('COMPARISION', f1_global - f1_local)
    local_T = list(f1_local.T)
    preunlearn_T = list(f1_preunlearn_final.T)
    fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))
    X = np.arange(6)
    color_l = ['b','g', 'y','r','c', 'm', 'k', 'black','purple', 'pink', 'olive', 'gray', 'orange', 'lime'] 
    #ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
    for i in range(np.shape(local_T)[0]):
        if global_T[i] > local_T[i]:
            if f1_global is not None:
                global_T = list(f1_global.T)
                ax1.scatter(i,  global_T[i], alpha=0.9,  color='blue', label='Unlearned Global Model')
        else: 
            ax1.scatter(i, local_T[i], alpha=0.9, color='red', label='Local Models')
       
        '''
        if 'global' in plot_l:
            if f1_global is not None:
                global_T = list(f1_global.T)
                ax1.scatter(range(np.shape(local_T)[0]),  global_T, alpha=0.9,  color='blue', label='Unlearned Global Model')
        if 'local' in plot_l: 
            ax1.scatter(range(np.shape(local_T)[0]), local_T, alpha=0.9, color='red', label='Local Models')
        if 'preunlearn' in plot_l:
            ax1.scatter(range(np.shape(local_T)[0]), preunlearn_T, alpha=0.9, color='green', label='Pre-Unlearned Global Model')
        '''


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


def plot_hist(prob, figname='default'):
    #worse_clients = [26, 25, 22, 31, 25, 26, 31, 20, 26, 27, 21, 26, 19, 24, 25, 17, 27, 26, 30, 25, 31, 22, 29, 32, 26, 24, 26, 31, 27, 25, 25, 25, 28, 21, 17, 26, 32, 22, 25, 28, 24, 23, 27, 23, 23, 22, 23, 23, 21, 21, 22, 19, 23, 22, 18, 28]
    #better_clients = [30, 31, 34, 25, 31, 30, 25, 36, 30, 29, 35, 30, 37, 32, 31, 39, 29, 30, 26, 31, 25, 34, 27, 24, 30, 32, 30, 25, 29, 31, 31, 31, 28, 35, 39, 30, 24, 34, 31, 28, 32, 33, 29, 33, 33, 34, 33, 33, 35, 35, 34, 37, 33, 34, 38, 28]


    #labels = 3*np.arange(np.size(better_clients) )
    #print(labels)
    #x = np.arange(len(labels))  # the label locations
    #width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=[15,10])
    x = prob
    plt.hist(x, density=True, bins=20)  # density=False would make counts
    #plt.ylabel('Probability')
    plt.xlabel('Predicted Probability')
    #rects1 = ax.bar(x - width/2, worse_clients, width, label='Worse')
    #rects2 = ax.bar(x + width/2, better_clients, width, label='Better')

    #trend_line = plt.plot(x - width/2, worse_clients,marker='o', color='#5b74a8', label='Worse')
    #trend_line = plt.plot(x + width/2, better_clients,marker='o', color='black', label='Better')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of clients')
    #ax.set_title('Comparision between unlearn models and local models ')
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    #autolabel(rects1)
    #autolabel(rects2)

    fig.tight_layout()
    if figname=='default':
        plt.savefig('plots/prob_hist.png')
    else:
        plt.savefig('plots/'+figname+'.png')


        

#intercept_l = np.array([array([1.41166305]), array([0.49314751]), array([1.82301233]), array([1.35900513]), array([2.15408251]), array([1.76271734]), array([2.0104497]), array([2.86649155]), array([1.39490666]), array([0.99682581]), array([0.31411773]), array([1.34150505]), array([1.42038632]), array([0.67752146]), array([1.50428133]), array([1.71799887]), array([1.23939513]), array([1.31783761]), array([2.17847566]), array([1.08156985]), array([1.84326593]), array([0.97085333]), array([0.78218743]), array([1.61694662]), array([0.77613503]), array([1.21673312]), array([0.68226542]), array([-1.03884462]), array([1.06213971]), array([1.05349376]), array([1.13416497]), array([0.94303878]), array([0.54512575]), array([2.54732446]), array([0.7515886]), array([-0.25820929]), array([2.0001421]), array([1.122153]), array([-0.27319614]), array([0.46755343]), array([0.13327208]), array([0.30696797]), array([0.9578501]), array([0.04109376]), array([-0.4468961]), array([1.24227637]), array([1.40537611]), array([0.85537737]), array([1.36182067]), array([1.44970943]), array([0.21840497]), array([0.99745584]), array([0.30433192]), array([0.63806133]), array([0.68756817]), array([1.92016358])])
#coef_l =  np.array([array([[-4.45933712,  3.01521226,  7.98189986, -1.12080376,  3.81393013]]), array([[-3.71020381,  3.03936295,  6.86486774, -0.14919135,  1.18988373]]), array([[-5.96470497,  3.56691752, 10.23224747, -1.59530216, 13.64478263]]), array([[-5.4748932 ,  3.3884347 ,  9.63195803, -1.10755514, 13.31670706]]), array([[-6.41459841,  3.95567247, 10.95236246, -1.83627126,  9.63976509]]), array([[-5.94142   ,  3.65063417, 10.67288483, -1.60256382, 11.34096319]]), array([[-6.28501396,  3.98617753,  9.36368986, -1.54556124, 13.92986482]]), array([[-7.3414269 ,  3.72813883,  9.38892424, -2.31477088, 20.52614808]]), array([[-4.33479858,  0.60752603,  8.88436631, -0.90596955,  1.03027706]]), array([[-5.53543749,  4.3040412 , 10.18433607, -0.79609989, 11.82233578]]), array([[-4.91013303,  4.6302328 , 10.25711957, -0.08099732,  6.55059259]]), array([[-6.41047353,  5.03830495, 10.80286126, -1.2134978 , 17.92357723]]), array([[-5.75029568,  3.51983675, 11.27714412, -1.19581787,  5.00664559]]), array([[-5.00008366,  3.46260741,  9.34428363, -0.232258  ,  3.86641437]]), array([[-5.39120998,  3.73923463, 10.24760112, -1.21560541,  6.14995475]]), array([[-6.3723127 ,  4.22116048, 12.7686076 , -1.61080838, 15.20009868]]), array([[-5.45641161,  4.14720686,  8.58562319, -0.93755463, 14.4027635 ]]), array([[-5.05457723,  3.46519878,  8.8978085 , -1.11108364, 13.73792426]]), array([[-6.54002429,  4.52687271, 11.98752955, -1.99328246,  3.39906648]]), array([[-3.39346969,  2.28382487,  6.07740611, -0.91808655,  0.31616212]]), array([[-6.47355609,  4.70459955, 11.13958905, -1.59002802, 14.29534335]]), array([[-5.50299632,  3.75823355, 10.78442809, -0.7947285 , 13.9566322 ]]), array([[-4.46851025,  3.72355674, 11.41247913, -0.94532692,  0.73721318]]), array([[-5.29051794,  2.60734641,  8.21714918, -1.21121907, 12.16778844]]), array([[-5.13109092,  4.21621346,  9.06661151, -0.67544183, 13.66330261]]), array([[-5.30478823,  3.9398641 ,  9.40690072, -1.03191984,  5.2423274 ]]), array([[-5.50775678,  5.20295433, 11.05573653, -0.53431772, 10.34807246]]), array([[-2.20308204,  3.01594169,  7.12113764,  0.93680673,  1.68228543]]), array([[-5.81822275,  4.24403854, 11.6663896 , -1.02806555, 12.71922321]]), array([[-5.34569028,  4.69081645,  8.87851189, -0.89559633,  9.84350516]]), array([[-5.73486198,  4.48162199, 10.62685434, -0.98140255,  8.80023264]]), array([[-5.69904142,  5.22019269, 10.56031302, -0.80984699, 11.27743279]]), array([[-4.90033034,  5.46449432,  7.33688368, -0.5616613 ,  5.19947367]]), array([[-5.00210086,  0.73254425,  8.56030263, -2.34272295,  1.49544709]]), array([[-4.66041874,  3.99593161,  9.33633106, -0.45752377,  1.52901264]]), array([[-3.68372272,  3.89237928, 10.55250916,  0.16086353,  7.53207278]]), array([[-5.69472724,  2.78329079,  8.82356039, -1.40812126,  1.70678083]]), array([[-5.56367599,  4.75465155, 10.52660675, -0.99493853, 12.78197414]]), array([[-4.07427227,  4.46853755,  8.53185953,  0.64533472,  3.85188005]]), array([[-5.11689294,  4.86057063, 10.21851578, -0.32740777,  6.83188932]]), array([[-4.99519966,  6.15644509,  9.00384322, -0.08970987, 10.14554664]]), array([[-4.47622603,  4.74386752,  9.05177486, -0.25327394,  7.25255947]]), array([[-5.45181858,  4.63027057, 10.61189803, -0.77419296, 11.93870485]]), array([[-3.54008244,  3.49641489,  8.62307174, -0.04387444,  0.72135804]]), array([[-4.63882292,  6.94581198,  9.4196414 ,  0.1044283 ,  2.8145948 ]]), array([[-6.10684046,  7.03899749, 10.97746656, -0.98735584,  8.04912041]]), array([[-4.06830295,  3.13589406,  6.48515242, -1.08822322,  0.69028644]]), array([[-5.2109303 ,  5.22190167,  8.27727054, -0.30214662,  1.14780293]]), array([[-4.89849325,  5.16796087,  8.09248147, -0.99038189,  0.85012264]]), array([[-6.15200856e+00,  4.79100451e+00,  6.51127626e+00,
#         2.07194131e-03,  6.68913360e-01]]), array([[-4.88269704e+00,  5.02173895e+00,  7.53400962e+00,
#         2.56841170e-03,  1.90129326e+01]]), array([[-5.06679264,  5.25806307, 10.59066339, -1.02625137, 13.04860403]]), array([[-4.99383515,  5.28998953,  9.0043816 , -0.12940191, 12.8154998 ]]), array([[-4.59706366,  5.0171997 ,  9.7457764 , -0.54233451,  8.46171808]]), array([[-3.3247043 ,  2.148743  ,  7.4728036 , -0.64629149,  5.7597993 ]]), array([[-6.44576105,  3.63836415,  9.26900842, -1.46107097, 17.31845781]])])

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

