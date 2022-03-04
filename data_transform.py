#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:55:14 2022

@author: quang
"""
from lib import *

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline


class oModel():
    
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3) 

        
    def train_model(self):
    
        
        clf=RandomForestClassifier(n_estimators=100, max_leaf_nodes=None, n_jobs=-1) #100
        

        starttime = timeit.default_timer()
        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(self.X_train,self.y_train)
        model_params = clf.get_params() 
        print(model_params)

        training_time = timeit.default_timer() - starttime
        print("The training time is :", training_time)
        starttime = timeit.default_timer()
        self.y_pred=clf.predict(self.X_test)
        precison = metrics.precision_score(self.y_test, self.y_pred, average='weighted')
        print('Precison: ', precison)
        recall = metrics.recall_score(self.y_test, self.y_pred, average='weighted')
        print('Recall: ', recall)
        f1 = metrics.f1_score(self.y_test, self.y_pred, average='weighted')
        print('F1: ', f1)
        accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        print('Accuracy: ', accuracy)
        testing_time = timeit.default_timer() - starttime
        print("The testing time is :", testing_time)
        
        #plot feature important
        #Check important feature
        # check Important features
        feature_importances_df = pd.DataFrame(
            {"feature": list(self.X_test.columns), "importance": clf.feature_importances_}
            ).sort_values("importance", ascending=False)
        
        # Display
        #print(feature_importances_df)
        # visualize important featuers
        # Creating a bar plot
        fig = plt.figure(figsize=[15,15])
        ax = fig.add_axes([0.1,0.3,0.8,0.65])
        ax.bar(x=feature_importances_df.feature, height=feature_importances_df.importance)
        # Add labels 
        plt.ylabel("Feature Importance Score",fontsize=20)
        plt.xlabel("Features",fontsize=20)
        #plt.title("Visualizing Important Features",fontsize=20)
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        #    , horizontalalignment="right", fontweight="light", fontsize="x-large")
        plotname='feature_important'
        plt.savefig('../plots/'+plotname+'.png')
        #plt.show()
        
        
        # save the model to disk
        filename = 'original_model.sav'
        pickle.dump(clf, open('../models/'+filename, 'wb'))
        
        # load the model from disk
        loaded_model = pickle.load(open('../models/'+filename, 'rb'))
        #result = loaded_model.score(X_test, y_test)
        return str(str(format(training_time, ".3f")) + ',' +
               str(format(testing_time, ".3f")) + ',' +
               str(format(self.y_train.shape[0], ".0f")) +  ',' +
               str(format(self.y_test.shape[0], ".0f")) + ',' +
               str(format(accuracy, ".3f")) +  ',' +
               str(format(precison, ".3f")) + ',' +
               str(format(f1, ".3f"))+ ',' +
               str(format(recall, ".3f")))
