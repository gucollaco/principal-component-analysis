# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:08:25 2018

@author: gustavo.collaco
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import colors as mcolors

def dataset():
    # read csv file
    data = pd.read_csv("dataset_glass.csv", header=None)

    # inputs
    values = data.iloc[:, :-1]
    values = values.values
    
    answers = data[np.size(values,1)]

    # standardizing the features
    new_values = StandardScaler().fit_transform(values)

    return new_values, answers, values.shape[1]

def execute_pca(dataset, answers, attributes, desired_components):
    pca = PCA(n_components=desired_components)
    
    columns_name = ['Component ' + str(j+1) for j in range(desired_components)]

    principal_components = pca.fit_transform(dataset)
    principal_data = pd.DataFrame(data = principal_components, columns = columns_name)
    final_data = pd.concat([principal_data, answers], axis = 1)
    
    variance_ratio = pca.explained_variance_ratio_

    variance_ratio_total = sum(variance_ratio)

    print('Variance ratio (per component): ', variance_ratio)
    print('Variance ratio (total): ', variance_ratio_total)
    
    return final_data

def plot_2d_pca(final_data, attributes):
    colors_all = [value for key, value in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).items()]
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    
    ax.set_xlabel('Principal Component 1', fontsize = 12)
    ax.set_ylabel('Principal Component 2', fontsize = 12)
    ax.set_title('2 component - PCA', fontsize = 20)
    
    targets = [np.unique(final_data[attributes])[i] for i in range(len(np.unique(final_data[attributes])))]
    colors = [colors_all[i] for i in range(len(np.unique(final_data[attributes])))]
    
    #print('Targets', targets)
    #print('Final', final_data)
    
    for target, color in zip(targets, colors):
        indexes = final_data[attributes] == target
        ax.scatter(final_data.loc[indexes, 'Component 1'], final_data.loc[indexes, 'Component 2'], c = color, s = 50)
    
    ax.legend(targets)
    ax.grid()
    
# main function
if __name__ == "__main__":
    new_dim = 9
    dataset, answers, attributes = dataset()
    final_data = execute_pca(dataset, answers, attributes, new_dim)
    
    if new_dim == 2: plot_2d_pca(final_data, attributes)
