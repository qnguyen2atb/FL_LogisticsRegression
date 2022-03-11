# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:48:34 2022

@author: quang
"""

import os
from unicodedata import name
os.chdir('./')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.utils import resample

plots_path = './plots/'
path = 'data'
file_name = 'churnsimulateddata.csv'
file = os.path.join(path, file_name)


class Data_Exploration():
    def __init__(self, path, balance=False):
        self.balance = balance
        if self.balance:
            print('Balancing data')
            df = pd.read_csv(path)
            X = df.dropna()
            print(f'shape of the data before balancing {np.shape(X)}')
            # downsample majority
            # concatenate our training data back together
            
            low = X[X.Churn_risk=='Low']   #1
            high = X[X.Churn_risk=='High']
            medium = X[X.Churn_risk=='Medium']
            
            X.Churn_risk.value_counts()

            low_downsampled = resample(low,
                                        replace = False, # sample without replacement
                                        n_samples = len(medium), # match minority n
                                        random_state = 27) # reproducible results

            # combine minority and downsampled majority
            downsampled = pd.concat([low_downsampled, high, medium])

            # checking counts
            downsampled.Churn_risk.value_counts()

            self.df = downsampled
            print(f'shape of the data after balancing {np.shape(downsampled)}')

        else:
            df = pd.read_csv(path)
            #nan = df.isnull().values.any()
            #if nan == True:
            #    df.dropna(inplace = True) # drop columns with NaN values
            df = df.dropna()
            self.df = df


    
    def boxplot_graph(self, name):
        i = 1
        for col_name in name:
            plt.subplot(2,4,i)
            self.df[col_name].plot.box(title = col_name, figsize = (20,13), grid = True)
            plt.xticks(rotation = 0, fontsize = 25)
            plt.yticks(fontsize = 25)
            plt.tight_layout()
            i = i + 1
            plt.savefig(plots_path+'boxplot')
            #plt.show()
            
    def dist_graph(self, name):
        plt.figure()
        plt.figure(figsize=(16,9))
        #plt.title('Boxplot of features')
        #dataframe.boxplot()
        # plot of each score
        i = 1
        for col_name in name:
            plt.hist(self.df[col_name].values, bins = 20, density = True)
            plt.xlabel(col_name, fontsize = 40)
            plt.xlim(self.df[col_name].values.min(), self.df[col_name].values.max())
            #sns.displot(dataframe[col_name])
            plt.tight_layout()
            plt.xticks(fontsize = 35)
            plt.yticks(fontsize = 35)
            plt.savefig(plots_path+'Distribution of '+col_name)
            plt.show()
            
    def coefficient(self, name):
        # correlation matrix
        corr = self.df[name].corr(method = 'spearman')
        plt.figure()
        sns.set(rc={'figure.figsize':(40,40)})
        matrix = np.tril(corr, k = -1)
        im = sns.heatmap(corr, annot = True, square = True, cmap = 'coolwarm', annot_kws={"size":45}, mask = matrix)
        plt.yticks(fontsize = 50, rotation = 0)
        plt.xticks(fontsize = 50, rotation = 90)
        cbar = im.collections[0].colorbar
        tick_font_size = 40
        cbar.ax.tick_params(labelsize = tick_font_size)
        
        plt.savefig(plots_path+'Heatmap')
        plt.show()


    def hist_graph_all(self, name): # original dataframe
        sns.set(rc={'figure.figsize':(16,9)})
    
        for n in name:
            fig, axs = plt.subplots()
            x_min = self.df[n].values.min()
            if n == 'Trnx_count':
                x_max = 1200
            elif n == 'num_products':
                x_max = 12
            else:
                x_max = self.df[n].values.max()
            sns.histplot(data = self.df, 
                         hue = 'Churn_risk', 
                         x = n, 
                         multiple = 'stack',
                         #binwidth = 0.25,  
                         stat = 'count')
            if self.balance:
                axs.set_title('Feature Distribution of ' + n + ' (balanced)', fontsize = 40)
            else:
                axs.set_title('Feature Distribution of ' + n, fontsize = 40)
            axs.set_xlabel(n, fontsize = 40)
            axs.set_ylabel('Count', fontsize = 40)
            plt.xlim((x_min, x_max))
            # set up legend
            legend = axs.get_legend()
            handles = legend.legendHandles
            legend.remove() 
            axs.legend(handles, ['Low', 'Medium', 'High'], title = 'Churn_risk', loc = 0, title_fontsize = 30, fontsize = 30)
            plt.xticks(fontsize = 35)
            plt.yticks(fontsize = 35)
            if self.balance:
                plt.savefig(plots_path+n+'_balanced_')
            else:
                plt.savefig(plots_path+n)


    def hist_graph(self, name): # original dataframe
        sns.set(rc={'figure.figsize':(16,9)})

        for i in range(0, 60, 10):
            _df_split = self.df[(self.df.PSYTE_Segment >= i) & (self.df.PSYTE_Segment < i+10)]
            for n in name:
                fig, axs = plt.subplots()
                x_min = _df_split[n].values.min()
                if n == 'Trnx_count':
                    x_max = 1200
                elif n == 'num_products':
                    x_max = 12
                else:
                    x_max = _df_split[n].values.max()
                    
                sns.histplot(data = _df_split, 
                            hue = 'Churn_risk', 
                            x = n, 
                            multiple = 'stack',
                            #binwidth = 0.25,  
                            stat = 'count')
                if self.balance:
                    axs.set_title('Feature Distribution of ' + n + ' (balanced)', fontsize = 40)
                else:
                    axs.set_title('Feature Distribution of ' + n, fontsize = 40)
                axs.set_xlabel(n, fontsize = 40)
                axs.set_ylabel('Count', fontsize = 40)
                plt.xlim((x_min, x_max))
                # set up legend
                legend = axs.get_legend()
                handles = legend.legendHandles
                legend.remove() 
                axs.legend(handles, ['Low', 'Medium', 'High'], title = 'Churn_risk', loc = 0, title_fontsize = 30, fontsize = 30)
                plt.xticks(fontsize = 35)
                plt.yticks(fontsize = 35)
                
                if self.balance:
                    plt.savefig(plots_path+n+'_batch_balanced_'+str(int(i/10)))
                else:
                    plt.savefig(plots_path+n+'_batch_'+str(int(i/10)))


'''def bivariate_rep(dataframe, cate):
    for n in cate:
        fig, axs = plt.subplots(nrows = 1, ncols = 3)
        fig.suptitle(n)
        sns.scatterplot(ax = axs[0], x = 'math score', y = 'writing score', hue = n, data = dataframe)
        sns.scatterplot(ax = axs[1], x = 'math score', y = 'reading score', hue = n, data = dataframe)
        sns.scatterplot(ax = axs[2], x = 'reading score', y = 'writing score', hue = n, data = dataframe)'''


pl = Data_Explo(file, balance=True)
name_l = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products', 'Churn_risk']
pl.hist_graph(name_l)
pl.hist_graph_all(name_l)

pl = Data_Explo(file)
name_l = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products', 'Churn_risk']
pl.hist_graph(name_l)
pl.hist_graph_all(name_l)


