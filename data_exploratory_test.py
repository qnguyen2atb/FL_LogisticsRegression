# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:48:34 2022

@author: quang
"""

import os
#os.chdir('F:\Machine_Unlearning\Code')
os.chdir('./')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plots_path = 'F:/Machine_Unlearning/Code/MUL_Model/plots/'

#%%

class Data_Explo():
    def __init__(self, path):
        df = pd.read_csv(path)
        nan = df.isnull().values.any()
        if nan == True:
            df.dropna(inplace = True) # drop columns with NaN values
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

    def hist_graph(self, name): # original dataframe
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
            axs.set_title('Feature Distribution of ' + n, fontsize = 50)
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
            plt.savefig(plots_path+n)
            plt.show()
    
#%%
#t['Age'].plot.bar(x = [10,20,30,40,50,60,70,80,90,100], figsize = (16,9))
#plt.xticks(rotation = 90)


'''def bivariate_rep(dataframe, cate):
    for n in cate:
        fig, axs = plt.subplots(nrows = 1, ncols = 3)
        fig.suptitle(n)
        sns.scatterplot(ax = axs[0], x = 'math score', y = 'writing score', hue = n, data = dataframe)
        sns.scatterplot(ax = axs[1], x = 'math score', y = 'reading score', hue = n, data = dataframe)
        sns.scatterplot(ax = axs[2], x = 'reading score', y = 'writing score', hue = n, data = dataframe)'''
