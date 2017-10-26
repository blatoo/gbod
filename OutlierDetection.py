"""
@author: Ying Gu
@copyright: Copyright 2017 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.
@license: This is free software: you can redistribute it and/or modify it under
          the terms of the GNU General Public License as published by the Free
          Software Foundation, either version 3 of the License, or (at your
          option) any later version. You should have received a copy of the
          GNU General Public License along with this software (COPYING).
          If not, see <http://www.gnu.org/licenses/>.
"""
# __author__ = "Ying Gu"
# __email__ = "connygy@gmail.com"

import matplotlib
import pandas as pd
import numpy as np
from sklearn import neighbors
import bisect
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy import spatial

def normalize(df, scale=10, rename=True):
    """Normalize a dataframe for all column
    Input:
    df (pd.DataFrame): to normalized dataframe
    scale (int): to scaled value
    rename (boolean): if rename is ture, then each column name will be appended with a surfix "_norm"
    
    Return:
    df_norm (pd.DataFrame): a normalized dataframe, values are in [0, scale] interval
    """
    df_norm=df.copy()
    df_max = df.max()
    df_min = df.min()
    df_norm=(df-df_min)*scale/(df_max-df_min)
    
    if rename==True:
        df_norm.columns = df_norm.columns+"_norm"
    
    return df_norm, scale

def sample_df(df, n=None, frac=None, random_state = None, sort=True):
    """sample dataset with given number or percent of records.
    
    Input:
    df (pd.DataFrame): input dataset
    n (int): given number of records
    frac (float): given percent of records, it should between 0-1
    random_state (int): the random seed
    sort (boolean): whether sort the dataset after the index after the sampling.
    
    Return:
    resulte dataset (pd.DataFrame)
    
    Notice:
    This is sampling without replacement
    """
    if sort==True:
        return df.sample(n = n, frac = frac, random_state = random_state).sort_index()
    else:
        return df.sample(n=n, frac=frac, random_state=ranodm_state)
    
def get_KDKNN_score(df, k=5, leaf_size=10, scale=10):
    df_norm,_=normalize(df, scale=10)
    kdtree=neighbors.KDTree(df_norm, leaf_size)
    def cal_score(point, k):
        return kdtree.query(point.reshape(1,-1), k+1)[0].sum()/k
    return pd.DataFrame({"score":df_norm.apply(cal_score, args=(k,), axis=1)}, index=df.index)

class GBOD(object):
    """ A Claas which contains a set of GBOD functions.
    Input:
    df (pd.DataFrame): input dataset all attributes has numerical values and without NA.
    n_partition (int): number of partition for each attributes.
    outlier_percent (float, a value between 0 and 100)  
    
    Attribues:
    df, (pd.DataFrame): input dataset
    n_partition (int): number of partition for each attributes.
    outlier_percent (float, a value between 0 and 100)
    n_dim (int): number of attributes of the input dataframe df.
    res (pd.DataFrame): contains df, normalized df, grid_id for each point.
    np_in_grid (pd.DataFrame): number of points in each grid.
    resGroup (grouped pd.DataFrame): the grouped res by the partitions
    pointGridCenter (pd.DataFrame): the points mean for each grid.
    """
    
    def __init__(self, df, n_partition=10, outlier_percent=0.25):
        self.df = df
        self.n_partition=n_partition
        self.outlier_percent=outlier_percent
        self.n_dim = len(df.columns)
        self.res, self.np_in_grid = self.findPartition()
        self.resGroup = self.group_partition()
        self.pointGridCenter = self.get_point_grid_mean()
        
    def get_point_grid_mean(self):
        pointGridCenter = self.resGroup.mean()
        pointGridCenter.columns = pointGridCenter.columns+"_center"
        for i, name in enumerate(pointGridCenter.index.names):
            pointGridCenter[name+"_center"] = pointGridCenter.index.labels[i]+0.5
        
        x2 = 0
        for i in range(self.n_dim):
            x2 += (pointGridCenter.iloc[:, self.n_dim+i]-pointGridCenter.iloc[:, 2*self.n_dim+i])**2
        
        
        pointGridCenter["center_diff"]= np.sqrt(x2)
        return pointGridCenter    
        
    def group_partition(self, res=None):
        """ group the result dataframe by the partitions.
        Input:
        res (pd.DataFrame): a result dataframe. If it is not given, then it is the `self.res`
        
        Return:
        pd.DataFrame, with group by partition.
        """
        if res==None:
            res = self.res
            
        keyColumns = list(res.columns[np.arange(2*self.n_dim, 3*self.n_dim)])
        return res.groupby(by=keyColumns)
    
    def findPartition(self):
        """normalize the dataset and find the partition for each record
        
        Return:
        1. res (pd.DataFrame): Contains the original dataset, the normalized values and the partition id for each record
        2. np_in_grid (pd.DataFrame): Count the number of points in each grid
        """        
        
        def countPoints(res):
            k = len(self.df.columns)
            keyColumns = list(res.columns[np.arange(2*k, 3*k)])
            return res.groupby(by=keyColumns).size()
        
        # normalize the dataset
        myMaxs = self.df.max()
        myMins = self.df.min()
        df_norm=((self.df-myMins)*self.n_partition/(myMaxs-myMins))

        # rename the norm columns
        df_norm.columns = self.df.columns+'_norm'

        # find the grid partition
        df_temp=df_norm.copy()
        for myCol in df_temp:
            df_temp.loc[df_temp[myCol] == self.n_partition] = self.n_partition-1

        # rename the partition columns
        df_temp.columns=self.df.columns+'_partition'
        df_temp = df_temp.astype(int)

        res = pd.concat([self.df,df_norm,df_temp],axis=1)
        
        np_in_grid = countPoints(res)
        
        return res, np_in_grid
    
    def predLabel(self, outlier_percent=None, label="yes"):
        """ detect which grid is outlier grid
        Input:
        outlier_percent (float, between 0 and 100): default is original setted number

        Return:
        1. grid_info (pd.DataFrame)
        2. grid_info_origin (pd.DataFrame)

        In principle, the two outputs are the same, but the 'grid_info_origin' use the partitions as the multi index. So you can
        with help of get_grid_info

        """
        # calculate the number of outlier points
        total_points = self.np_in_grid.sum()
        
        if outlier_percent==None:
            outlier_percent = self.outlier_percent
        else:
            self.outlier_percent = outlier_percent
        
        n_outliers = total_points*outlier_percent/100

        # sort the grids after the number of points in grid
        np_in_grid_sorted = self.np_in_grid.sort_values()

        # calculate which grids are outlier grids.
        np_in_grid_cumsum = np_in_grid_sorted.cumsum()
        grid_info = pd.concat([np_in_grid_sorted, np_in_grid_cumsum], axis=1)
        grid_info.columns = ["points","cum_sum"]
        # Conny: 
        # print(grid_info)
        # grid_info.reset_index(inplace=True)
        
        n_outlier_grids = bisect.bisect_right(grid_info.cum_sum.tolist(), n_outliers)
        # Conny
        # print "n_outlier_grids = ", n_outlier_grids
        
        # Conny: 11.4.2017
        # write the outlier label for the grids
        if label=="yes":
            grid_info["pred_grid_label"] = "no"
            grid_info.ix[0:n_outlier_grids, "pred_grid_label"] = "yes"
        else:
            grid_info["pred_grid_label"] = 0
            grid_info.ix[0:n_outlier_grids, "pred_grid_label"] = 1
            
        grid_info_mIndex = grid_info.copy()
        # grid_info.reset_index(level=tuple(np.arange(len(np_in_grid_sorted.index.names))), inplace=True)
        grid_info.reset_index(inplace=True)
        return grid_info, grid_info_mIndex
    
    def pred_center_bias_label(self, outlier_grid_frac=None):
        """detect which grid is outlier grid after the bias between point center and grid center
        Input:
        outlier_grid_frac (float): between 0 to 1, default value is calculated from self.outlier_percent
            Notice: The total grid number is the grid which contains at least one point.
        
        Return:
        None
        """
        if outlier_grid_frac == None:
            outlier_grid_frac = self.get_n_outlier_grids()[1]
        
        total_n_grids = len(self.pointGridCenter)
        n_outlier_grids = int(outlier_grid_frac*total_n_grids)
        
        myPointGridCenter = self.pointGridCenter.sort_values(by=["center_diff"], ascending=False).copy()
        myPointGridCenter["center_bias_label"] = "no"
        myPointGridCenter.ix[0:n_outlier_grids, "center_bias_label"] = "yes"
        return myPointGridCenter
    
    def combine_result(self, grid_info):
        """ combine the results, so that you will get all the information
        Input:
        grid_info (pd.DataFrame): you get it from predLabel(...)[0]

        Return:
        res (DataFrame): with all information 
        """
        return pd.merge(self.res, grid_info, how='left').set_index(self.res.index)
    
    def run_GBOD(self, outlier_percent=None, label='yes', writeCSV=False):
        """ Get the end result.
        Input:
        outlier_percent (float): a value between 0 and 100, determine how many points are outliers

        Return:
        1. result (pd.DataFrame): the DataFrame which contain all the information of the result
        2. grid_info_mIndex (pd.DataFrame with multi-index): The DataFrame which contain the information, whether the grids are outliers
        the indexes are the partition-id for each index.
        """
        if outlier_percent==None:
            outlier_percent = self.outlier_percent
        else:
            self.outlier_percent = outlier_percent
        
        grid_info, grid_info_mIndex = self.predLabel(outlier_percent, label=label)
        
        result = self.combine_result(grid_info).drop(["points", "cum_sum"], axis=1)
        # result.drop(['index'], axis=1, inplace=True)
        
        if writeCSV == True:
            result.to_csv("gbod_result_outlierPercent_"+str(self.outlier_percent)+".txt")
            grid_info.to_csv("grid_info_outlierPercent_"+str(self.outlier_percent)+".txt")
            
        return result, grid_info_mIndex
    
    def get_n_outlier_grids(self, outlier_percent=None):
        """Calculate the number of outlier grids after a given outlier_percent
        Input:
        outlier_percent (int): between 0 to 100, if outlier_percent==None, then it take the value from the self.outlier_percent
        
        Return:
        n_out_grids (int): number of outlier grids
        outlier_grid_frac (float): the outlier grid frac of total grids which contain at least one point.
        
        Notice: The total grid number is the grid which contains at least one point.
        """
        if outlier_percent == None:
            outlier_percent = self.outlier_percent
        n_outliers = outlier_percent*len(self.df)/100
        
        # sort the grids after the number of points in grid
        np_in_grid_sorted = self.np_in_grid.sort_values()

        # calculate which grids are outlier grids.
        np_in_grid_cumsum = np_in_grid_sorted.cumsum()
        grid_info = pd.concat([np_in_grid_sorted, np_in_grid_cumsum], axis=1)
        grid_info.columns = ["points","cum_sum"]
        grid_info.reset_index(inplace=True)
        n_outlier_grids = bisect.bisect_right(grid_info.cum_sum, n_outliers)
        return n_outlier_grids, float(n_outlier_grids)/len(grid_info), outlier_percent

    def run_GBOD_center_bias(self, outlier_grid_frac=None, writeCSV=False):
        """ get the center biased result
        Input:
        outlier_grid_frac (float): between 0 to 1, default value is calculated from self.outlier_percent
            Notice: The total grid number is the grid which contains at least one point.
        writeCSV (boolean): if True, then it will write two files:
            1. gbod_center_bias_result_frac_xxx.txt
            2. grid_info_center_bias_frac_xxx.txt
            
        Return:
        result (pd.DataFrame): the result dataframe with "center_bias_label"
        centerBiasLabel_origin (pd.DataFrame): detailed information of the grid center and point center.
        """
        if outlier_grid_frac == None:
            print ("outlier_percent = "+str(self.outlier_percent))
            outlier_grid_frac = self.get_n_outlier_grids()[1]
            
        print ("outlier_gird_frac = "+ str(outlier_grid_frac))

        centerBiasLabel_origin = self.pred_center_bias_label(outlier_grid_frac)
        centerBiasLabel = centerBiasLabel_origin.loc[:, "center_bias_label"].reset_index()
        result = self.combine_result(centerBiasLabel)
        if writeCSV==True:
            result.to_csv("gbod_center_bias_result_frac_"+str(outlier_grid_frac)+".txt")
            centerBiasLabel_origin.to_csv("grid_info_center_bias_frac_"+str(outlier_grid_frac)+".txt")
        
        return result, centerBiasLabel_origin
    
    def plotOutliers(self, x=None, y=None, outlier_percent=None, grid=True):
        """ plot the outliers
        Input:
        1. x (array like): the x-axis, it is result.xxx_norm
        2. y (array like): the y-axis, it is result.yyy_norm
        3. outlier_percent: float, between 0 to 100

        Return:
        None    
        """
        result = self.run_GBOD(outlier_percent)[0]
        
        # find the default position of x and y
        if x == None:
            x = result.iloc[:,self.n_dim]
            
        if y == None:
            y = result.iloc[:,self.n_dim+1]
        
        # define the image size
        width = 10
        height = 10
        fig = plt.figure(figsize=(width, height))

        ax = fig.gca()
        ax.set_xticks(np.arange(self.n_partition+1))
        ax.set_yticks(np.arange(self.n_partition+1))
        plt.xlim(0,self.n_partition)
        plt.ylim(0,self.n_partition)
        
        if grid==True:
            plt.grid(color='r', linestyle='--', linewidth=3)

        plt.scatter(x, y, 
                    c = ['red' if label == 'yes' else 'blue' for label in result.pred_grid_label], 
                    s = [100 if label=='yes' else 30 for label in result.pred_grid_label],
                    alpha=0.5)
        plt.show()
        
    def plotOutliers_CenterBias(self, x=None, y=None, outlier_grid_frac=None, grid=True):
        """Plot the outler grids after the center bias method
        Input:
        x (str): the column name for x-axis, default self.res.xxx_norm 
        y (str): the column name for y-axix, default self.res.yyy_norm
        outlier_grid_frac (float): between 0 to 1, default value is calculated from self.outlier_percent
            Notice: The total grid number is the grid which contains at least one point.
        
        Return:
        None        
        """
        
        if outlier_grid_frac == None:
            print ("outlier_percent = "+str(self.outlier_percent))
            outlier_grid_frac = self.get_n_outlier_grids()[1]
        
        result = self.run_GBOD_center_bias(outlier_grid_frac)[0]
        
        # find the default position of x and y
        if x == None:
            x = result.iloc[:,self.n_dim]
            
        if y == None:
            y = result.iloc[:,self.n_dim+1]
        
        # define the image size
        width = 10
        height = 10
        fig = plt.figure(figsize=(width, height))

        ax = fig.gca()
        ax.set_xticks(np.arange(self.n_partition+1))
        ax.set_yticks(np.arange(self.n_partition+1))
        plt.xlim(0,self.n_partition)
        plt.ylim(0,self.n_partition)
        
        if grid==True:
            plt.grid(color='r', linestyle='--', linewidth=3)

        plt.scatter(x, y, 
                    c = ['red' if label == 'yes' else 'blue' for label in result.center_bias_label], 
                    s = [100 if label=='yes' else 30 for label in result.center_bias_label],
                    alpha=0.5)
        plt.show()
        
    def plotOutliers_origin(self, x=None, y=None, outlier_percent=None, grid=True):
        """ plot the outliers
        Input:
        1. x (array like): the x-axis, it is result.xxx
        2. y (array like): the y-axis, it is result.yyy
        3. outlier_percent: float, between 0 to 100

        Return:
        None    
        """
        result = self.run_GBOD(outlier_percent)[0]
        
        # find the default position of x and y
        if x == None:
            x = result.iloc[:,0]
            
        if y == None:
            y = result.iloc[:,1]
        
        # define the image size
        width = 10
        height = 10
        fig = plt.figure(figsize=(width, height))
        
        # set the ticks for the partitions
        XTicks = np.linspace(x.min(), x.max(), self.n_partition+1, endpoint=True)
        YTicks = np.linspace(y.min(), y.max(), self.n_partition+1, endpoint=True)
        

        ax = fig.gca()
        ax.set_xticks(XTicks)
        ax.set_yticks(YTicks)
        
        plt.xlim(x.min(),x.max())
        plt.ylim(y.min(),y.max())
        
        if grid==True:
            plt.grid(color='r', linestyle='--', linewidth=3)

        plt.scatter(x, y, 
                    c = ['red' if label == 'yes' else 'blue' for label in result.pred_grid_label], 
                    s = [100 if label=='yes' else 30 for label in result.pred_grid_label],
                    alpha=0.5)
        plt.show()
        
    def plotGridHist(self, bins = 50, **kwargs):
        """ Plot the histogram of number of points in each grid.
        Input:
        bins (int): number of bins
        
        Return:
        None        
        """
        width = 10
        height = 5
        fig = plt.figure(figsize=(width, height))
        plt.hist(self.np_in_grid, bins=50, **kwargs)
        plt.show()
        
    def plotGridHeatMap(self, fontsize=16):
        """ Plot the heatmap of the number of points in each grid.
        Input: None
        
        Return: 
        None        
        """
        width = 12
        height = 13
        fig = plt.figure(figsize=(width, height))
        ax = fig.gca()
        ax.set_xticks(np.arange(self.n_partition+1))
        ax.set_yticks(np.arange(self.n_partition+1))
        # ticks fontsize
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(24)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(24)
        plt.xlim(0,self.n_partition)
        plt.ylim(0,self.n_partition)
        
        font = matplotlib.font_manager.FontProperties(family='times new roman', style='italic', size=24)

        cellcolor = np.arange(self.n_partition*self.n_partition).reshape(self.n_partition,self.n_partition)
        cellcolor.fill(0.0)
        
        for a,b in self.np_in_grid.iteritems():
            cellcolor[a[1], a[0]]=b

        # define the colormap
        mycmap=plt.cm.Blues
        #mycmap = plt.cm.terrain

        # extract all colors from the .terrain map
        cmaplist = [mycmap(i) for i in range(mycmap.N)]

        # force the first color entry to be grey
        cmaplist[0] = (0.50,0.50,1.0,1.0)
        # create the new map
        #mycmap = mycmap.from_list('Custom cmap', cmaplist, mycmap.N)

        mycmap = colors.ListedColormap([cmaplist[i] for i in range(1, mycmap.N, 20)])

        # define the bins and normalize
        bounds = [0.0]
        self.np_in_grid.sort_values()
        bounds.extend(np.percentile(self.np_in_grid,range(0,100,10))) 
        bounds.extend([self.np_in_grid.max()])

        norm = colors.BoundaryNorm(bounds, len(bounds))
        plot3 = plt.pcolor(cellcolor,cmap=mycmap,norm=norm)
        plt.grid(color='r', linestyle='--', linewidth=1)

        cb4 = plt.colorbar(plot3,orientation='horizontal', cmap=mycmap, norm=norm, ticks=bounds, label="Cell density distribution, divided into percentiles of 10.", pad=0.08)
        cb4.ax.xaxis.label.set_font_properties(font)
        for t in cb4.ax.get_xticklabels():
            t.set_fontsize(16)

        for i in range(0,self.n_partition):
            for j in range(0,self.n_partition):
                z = cellcolor[j,i]
                ax.text( i+0.5, j+0.4, z, fontsize=fontsize, color='red', horizontalalignment='center' )
                
        plt.show()        
        
    def plot_df_norm(self, x=None, y=None, pointCenter=True, gridCenter=True, grid=True, width=10, height=10):
        """ plot the outliers
        Input:
        1. x (str): column name of the result dataframe. x-axis, it is result.xxx_norm
        2. y (str): column name of the result dataframe. y-axis, it is result.yyy_norm
        3. PointCenter (boolean): whether plot the point center for each grid.

        Return:
        None    
        """
        result = self.res
        
        # find the default position of x and y
        if x == None:
            x = result.iloc[:,self.n_dim]
            xMean = self.pointGridCenter.iloc[:, self.n_dim]
            xGridMean = self.pointGridCenter.iloc[:, 2*self.n_dim]
        else:
            x = result[x]
            xMean = self.pointGridCenter[x]
            xGridMean = self.pointGridCenter[x+"_center"]
            
            
        if y == None:
            y = result.iloc[:,self.n_dim+1]
            yMean = self.pointGridCenter.iloc[:, self.n_dim+1]
            yGridMean = self.pointGridCenter.iloc[:, 2*self.n_dim+1]
        else: 
            y = result[y]
            yMean = self.pointGridCenterMean[y]
            yGridMean = self.pointGridCenter[y+"_center"]
        
        # define the image size
        fig = plt.figure(figsize=(width, height))

        ax = fig.gca()
        ax.set_xticks(np.arange(self.n_partition+1))
        ax.set_yticks(np.arange(self.n_partition+1))
        plt.xlim(0,self.n_partition)
        plt.ylim(0,self.n_partition)
        
        if grid==True:
            plt.grid(color='r', linestyle='--', linewidth=3)

        plt.scatter(x, y, c = 'blue', s = 30, alpha=0.5)
        
        if pointCenter == True:
            plt.scatter(xMean, yMean, c="yellow", marker="*", s = 130)
        if gridCenter == True:
            plt.scatter(xGridMean, yGridMean, c="g", marker="x", s=80, linewidths=2)
        plt.show()
        
    def compare_KDKNN(self, k=5, leaf_size=10, outlier_percent=None):
        
        myKDKNN = KDKNN(self.df, k, leaf_size)
        
        if outlier_percent==None:
            outlier_percent=self.outlier_percent
        
        label = myKDKNN.pred_label(outlier_percent)
        
            
        res,_=self.run_GBOD(outlier_percent)
        lable = myKDKNN.pred_label(outlier_percent)
        myRes = pd.concat([res, label], axis=1)
        
        truePositive = myRes[(myRes.pred_grid_label=="yes") & (myRes.label=="yes")]
        tp = truePositive.shape[0]
        falsePositive = myRes[(myRes.pred_grid_label=="yes") & (myRes.label=="no")]
        fp = falsePositive.shape[0]
        falseNegative = myRes[(myRes.pred_grid_label=="no") & (myRes.label=="yes")]
        fn = falseNegative.shape[0]

        precision = tp*1.0/(tp+fp)
        recall = tp*1.0/(tp+fn)
        F1 = 2.0*precision*recall/(precision+recall)
        
        return {"precision": precision, "recall": recall, "F1": F1}
    
def GBOD_P(df, partition_range=[5,20], outlier_percent=1):
    '''Get the detailed information about outlier score
    input:
        df: dataframe
        partition_range: [begin, end]
        outlier_percent: 1-100 (numberic)
    output:
        df_score: a dataframe with outlier result for each p
    '''
    
    def getScore(df_score_row):
        res = df_score_row.tolist().count('yes')
        return res

    df_score=pd.DataFrame()
    
    for x in range(partition_range[0], partition_range[1]):
        #Conny
        #print 'round: '+str(x)
        myGBOD = GBOD(df, n_partition=x,outlier_percent=outlier_percent)
        result, _ = myGBOD.run_GBOD()
        df_score['p='+str(x)]=result.pred_grid_label
        
    temp = df_score.apply(lambda row: getScore(row), axis=1)
    df_score['score']=temp
    
    return df_score

def GBOD_P_2(df, partition_range=[5,20], outlier_percent=1):
    '''Get the outlier score
    input:
        df: dataframe
        partition_range: [begin, end]
        outlier_percent: 1-100 (numberic)
    output:
        df_score: a numpy array with outlier scores (integer)
    '''

    df_score=np.zeros(df.shape[0])
    
    for x in range(partition_range[0], partition_range[1]):
#         print 'round: '+str(x)
        myGBOD = GBOD(df, n_partition=x,outlier_percent=outlier_percent)
        # result, _ = myGBOD.run_GBOD(label=1)
        # df_score += result.pred_grid_label
        df_score += myGBOD.run_GBOD(label=1)[0].pred_grid_label
        
    return df_score
        
        
class KDKNN(object):
    def __init__(self, df, k=5, leaf_size=10):
        self.df = df
        self.df_norm, self.scale = normalize(self.df)
        self.k = k
        self.normalize=normalize
        self.leaf_size=leaf_size
        self.kdtree = neighbors.KDTree(self.df_norm, leaf_size)
        self.score = self.get_score()
        self.res = pd.concat([self.df, self.df_norm, self.score], axis=1)
        
    def get_score(self, k=None, leaf_size=None):
        if k == None:
            k = self.k
        else:
            self.k=k
        if leaf_size==None:
            leaf_size=self.leaf_size
        else:
            self.leaf_size=leaf_size

        k2 = k+1
        score = [self.kdtree.query([point], k=k2)[0].sum()/k for point in self.df_norm.values]
        self.score = pd.DataFrame(score, columns=["score"], index=self.df_norm.index)
        
        return self.score
    
    
    def plot_outliers(self, width=10, height=10, grid=True, score_scale=200, origin = False, scale=None, outlier_percent=None, size=None):
        fig = plt.figure(figsize=(width, height))
        ax = fig.gca()
        
        if size==None:
            size=self.score*score_scale
        
        if scale==None:
            scale = self.scale
  
        if outlier_percent == None:
            color = ["blue"]*self.df.shape[0]
        else:
            label = self.pred_label(outlier_percent)
            label["color"] = map(lambda x: "red" if x == "yes" else "blue", label.label)
            label = pd.concat([self.df, label], axis=1)
            color=label.color
            
        
        if origin == False:
            df_norm,_=normalize(self.df, scale=scale)
            x = df_norm.iloc[:,0]
            y = df_norm.iloc[:,1]
            ax.set_xticks(np.arange(scale+1))
            ax.set_yticks(np.arange(scale+1))
            plt.xlim(0,scale)
            plt.ylim(0,scale)
        else:
            x = self.df.iloc[:,0]
            y = self.df.iloc[:,1]
            x_max, y_max = self.df.max()
            x_min, y_min = self.df.min()
            
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            ax.set_xticks(np.linspace(x_min, x_max, num=scale))
            ax.set_yticks(np.linspace(y_min, y_max, num=scale))
            
        if grid==True:
            plt.grid(color='r', linestyle='--', linewidth=3)
        
        plt.scatter(x, y, c = color, s = size, alpha=0.5)

        plt.show()
        
    def pred_label(self, outlier_percent):
        n_outliers=int(outlier_percent*self.df.shape[0]/100)
        label = self.score.sort_values(by=["score"], ascending=False)
        label["label"] = "no"
        label.label.iloc[0:n_outliers]="yes"
        return label.drop("score", axis=1).copy()
    
    def run_KDKNN(self, outlier_percent):
        """run the KDKNN and get end result regarding a given outlier_percent
        input:
        outlier_percent (int): between 0 and 100
        
        resutl:
        a dataframe with all the inoformation of the outliers.
        """
        label = self.pred_label(outlier_percent)
        return pd.concat([self.res, label], axis=1).copy()


    