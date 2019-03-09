#!/usr/bin/env python

# 1.load a dataset from a file
# 2."organize that file so we can access columns *or* rows of it easily
# 3.compute some "summary statistics" about the dataset
# 4.print those summary statistics

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from argparse import ArgumentParser
from itertools import combinations

hist_dir_path='./hist_fig/'
scatter_dir_path='./scatter_fig/'
corr_dir_path='./corr_fig/'
dim_dir_path='./3d_fig/'

features_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'BK', 'LSTAT', 'MEDV']
#load data

parser = ArgumentParser(description='A CVS reader + stats maker')
parser.add_argument('csvfile',
                    help='Path to the input csv file.')

parsed_args = parser.parse_args()

my_csv_file = parsed_args.csvfile

assert os.path.isfile(my_csv_file),"please give us a real file"
print('woot, the file exists')

data= pd.read_csv(my_csv_file, sep='\s+|,', header = None)
print(data.head())

print(data.shape)

#organize data
# 2A. ACCESS any row

print ("*Display the third and fourth rows*")
print (data.iloc[3:5,:]) # inclusive the first value 3 but exclusive the last value 5

# 2b access any column

print ("*Display the last two columns*")
print (data.iloc[:3,-2:]) #-2 the last two

# 2c access any value

print("*Display a specific value*")
print(data.iloc[3,4])

#compute summary statistics

print ("*Calculate mean*")
print (np.mean(data))
print ("*Calculate STD*")
print (np.std(data))


#D-Visualize the data, 1-feature (column) at a time

def plot_histogram():

 if not os.path.exists(hist_dir_path):
     os.makedirs(hist_dir_path)

 sns.set() #for a nicer histogram
 
 #loop on columns and generate histograms
 for col in range(len(features_list)):
 
     df=(data.iloc[:,col])
 
     plt.figure()
     plt.hist(df, bins=20)
     plt.xlabel(features_list[col])
     plt.ylabel('occurrence')
     plt.title('Housing Data:{}'.format(features_list[col]))
 
     print ("Generating histogram file for {} in {} dir".format(features_list[col], hist_dir_path))
 
     plt.savefig('{}{}_hist.png'.format(hist_dir_path,features_list[col]))
     #plt.show() #we can uncomment to display all one by one
     plt.close() #avoid warnings


#E) Visualize the data, 2-features (columns) at a time

def plot_scatter_pairs():

 if not os.path.exists(scatter_dir_path):
     os.makedirs(scatter_dir_path)

 #for col in range(len(features_list)):
 for col1 in range(len(features_list)):
     df1=(data.iloc[:,col1])
     col2=1
     for col2 in range((col2+col1), len(features_list)):
         df2=(data.iloc[:,col2])
         #print ("{}/{}".format(features_list[col1],features_list[col2]))
         plt.figure()
         plt.scatter(df1,df2)
         plt.xlabel(features_list[col1])
         plt.ylabel(features_list[col2])
         plt.title('Housing Data:{}/{}'.format(features_list[col1], features_list[col2]))
 
         print ("Generating scatter file for {}/{} pair in {} dir".format(features_list[col1], features_list[col2], scatter_dir_path))
 
         plt.savefig('{}{}_{}_scatter.png'.format(scatter_dir_path,features_list[col1],features_list[col2]))
         #plt.show() #we can uncomment to display all one by one
         plt.close() #to avoid warnings


#F) Assign a header to a dataset

data_h= pd.read_csv(my_csv_file, sep='\s+|,', names = features_list)

print ("*Test dataset with header*")
print(data_h.head())


#G) Pseudocode for an additional type of plot: correlation matrix

def plot_correlation():

 if not os.path.exists(corr_dir_path):
     os.makedirs(corr_dir_path)

 correlations = data.corr()
 fig, ax = plt.subplots(figsize=(len(features_list), len(features_list)))
 cax = ax.matshow(correlations, vmin=-1, vmax=1)
 fig.colorbar(cax)
 
 ax.set_xticks(range(0, len(features_list)))
 ax.set_yticks(range(0, len(features_list)))

 ax.set_xticklabels(features_list)
 ax.set_yticklabels(features_list)

 print ("Generating corrolation file for all features in {} dir".format(corr_dir_path))
 plt.savefig('{}corrolation.png'.format(corr_dir_path))
 #plt.show()
 plt.close()


#H)plot 3d
def plot_3d():

 if not os.path.exists(dim_dir_path):
     os.makedirs(dim_dir_path)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

#a pick of the most important features (for less combinations)
 features_sublist=['RM', 'LSTAT', 'TAX', 'CRIM', 'MEDV', 'DIS', 'PTRATIO', 'INDUS', 'AGE']

 for comb in combinations(features_sublist,6):

     #plt.figure()
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')

     grp1_ft1,grp1_ft2,grp1_ft3,grp2_ft1,grp2_ft2,grp2_ft3=comb

     xs =(data.iloc[:,features_list.index(grp1_ft1)])
     ys =(data.iloc[:,features_list.index(grp1_ft2)])
     zs =(data.iloc[:,features_list.index(grp1_ft3)])

     xt =(data.iloc[:,features_list.index(grp2_ft1)])
     yt =(data.iloc[:,features_list.index(grp2_ft2)])
     zt =(data.iloc[:,features_list.index(grp2_ft3)])

     ax.scatter(xs, ys, zs, c='r', marker='o', label='Group1: {},{},{}'.format(grp1_ft1,grp1_ft2,grp1_ft3))
     ax.scatter(xt, yt, zt, c='b', marker='^', label='Group2: {},{},{}'.format(grp2_ft1,grp2_ft2,grp2_ft3))

     ax.set_xlabel('{}/{}'.format(grp1_ft1, grp2_ft1))
     ax.set_ylabel('{}/{}'.format(grp1_ft2, grp2_ft2))
     ax.set_zlabel('{}/{}'.format(grp1_ft3, grp2_ft3))
     ax.legend(loc='upper left')

     print ("Creating 3d file for {} {} {} {} {} {} in {} dir".format(grp1_ft1,grp1_ft2,grp1_ft3, grp2_ft1,grp2_ft2,grp2_ft3, dim_dir_path))
     #plt.show()
     plt.savefig('{}3D_{}_{}_{}_{}_{}_{}.png'.format(dim_dir_path,grp1_ft1,grp1_ft2,grp1_ft3, grp2_ft1,grp2_ft2,grp2_ft3))
     plt.close()
 
plot_histogram()
plot_scatter_pairs()
plot_correlation()
plot_3d()

