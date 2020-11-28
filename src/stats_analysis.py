import pandas as pd
import numpy as np
from numpy import sqrt,mean,std
import scipy.stats as scs
import matplotlib.pyplot as plt
import seaborn as sns

class two_samples(object):
    def __init__(self, df):
        self.pval = None
        self.df = df
        self.df1 = None
        self.df2 = None
        self.threshold = 0.05

    def studentTtest(self,condition1,condition2,target):
        self.df1 = self.df[condition1][target]
        self.df2 = self.df[condition2][target]

        _, self.pval = scs.ttest_ind(self.df1,self.df2,equal_var=False)

    def mannWhitneyUtest(self,condition1,condition2,target):
        self.df1 = self.df[condition1][target]
        self.df2 = self.df[condition2][target]

        _, self.pval = scs.mannwhitneyu(self.df1,self.df2,alternative='two-sided')
        
    def plot_hist(self):
        dfs = [self.df1,self.df2]
        fig,axs = plt.subplots(2,1)

        for df,ax in zip(dfs,axs.flatten()):
            ax = sns.histplot(df)
        plt.tight_layout()
        ax.vlines(self.threshold,)
        return fig,ax

    # def plot_sa(self):
    #     self.pval = self.studentTtest(self.condition1,self.condition2,self.target)
    #     fig,ax=plt.subplots()
    #     plt.tight_layout()

    #     return fig,ax
if __name__ == '__main__':
    df = pd.read_csv('../data/performances.csv')
    l_h = (df.GS >= 7) & (df.throw == 'L')
    r_h = (df.GS >= 7) & (df.throw == 'R')
    ts = two_samples(df)
    ts.studentTtest(l_h,r_h, 'ER')