'''
Used to calculate cross correlations amongst specified regions
from an ambermd trajectory. This functionality is not baked into
AMBERMD, specifically cpptraj, by default. The documentation is
extremely thourough. If you have any questions regarding the
script please contact me using my ECU email

    lindsays15@students.edu.edu

Best of luck.
 
    -Gubbin Eel
'''

#- Import required modules -#

import pytraj as pt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import messages as mes
import argparse as arg

from scipy.cluster import hierarchy as hc

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def n_n_corr(v1, v2):
    '''
    Purpose:   To calculae the correlation between two atomic 
               trajectories.
    Arguments: v1: 2d array where each entry is a position vector.
               v2: 2d array where each entry is a position vector.
    Returns:   The value of the cross-correlation as a float.
    NOTES:     (1) v1 and v2 must be the same length.
               (2) Man, this really isn't a class method. I
                   keep it here more for documentation sake.
                   I might remove it and add a section to the
                   `Corr()` Docstring that lists the functions
                   not in the module. Actually, I should check
                   if calling `help()` on a module returns the
                   docstring. That would make things easier.
    '''
    avg_of_dot = np.average(np.sum(v1 * v2, axis = 1))

    v1_avg = np.average(v1, axis = 0)
    v2_avg = np.average(v2, axis = 0)

    avg_dot_v1 = np.average(np.sum(v1 * v1, axis = 1))
    avg_dot_v2 = np.average(np.sum(v2 * v2, axis = 1))

    v1_avg_dot = np.dot(v1_avg, v1_avg)
    v2_avg_dot = np.dot(v2_avg, v2_avg)

    numerator = avg_of_dot - np.dot(v1_avg, v2_avg)
    denom1 = avg_dot_v1 - v1_avg_dot
    denom2 = avg_dot_v2 - v2_avg_dot
    denominator = np.sqrt(denom1 * denom2)
    
    return numerator / denominator    

class Corr():
    '''
    Description: A class used to house all of the data and methods 
                 associated with generating correlation heatmaps 
                 and/or feature-space figures for a given trajectory.
                 Specifically, it is intended to deal with scenarios 
                 when the user is interested in calculating cross-
                 correlation heatmaps between whole regions of a
                 protein or protein-complex, rather than for every
                 atom or all atoms in a selection.
    Instance
    Attributes:  traj_file:     File housing the trajectory that will
                                be analyzed.
                 top_file:      File housing the topology for the
                                specified trajectory.
                 traj:          The pytraj `Tracectory` instance.
                 correl_matrix: The correlation matrix as a pandas
                                dataframe.
                 com_traj:      The center of mass trajectory for
                                the regions which a correlation is
                                calculated for.
                 cluster_info:  Information about which clusters
                                each region belongs to.
    '''
    def __init__(self, traj_file, top_file, traj = None,
                 correl_matrix = None, com_traj = None,
                 cluster_info = None, start = None, 
                 end = None, stride = None):
        '''
        Purpose: 
        Arguments: self:
                   traj_file:     Name of the trajectory file.
                   top_file:      Name of the topology file.
                   traj:          An instance of the pytraj
                                  `Trajectory` class.
                   correl_matrix: A symmetrix correlation matrix
                                  stored in a pd df. Feature names
                                  are in the columns and indicies.
                   com_traj:      An N x F x 3 np array. Where N is
                                  the number of center of mass 
                                  trajectories, F is the number of
                                  frames in the trajectory, and 3
                                  corresponds to the xyz coordinates
                                  of the center of mass.
                   cluster_info:  A dictionary that translates from 
                                  the correlation matrix column 
                                  names to cluster names.
                   start:         Starting frame, used when loading
                                  traj.
                   end:           End frame, used when loading traj.
                   stride:        Number of frames to skip, used 
                                  when loading traj.
        Returns:   Initiated class instance.
        '''
        self.traj_file     = traj_file
        self.top_file      = top_file  
        self.correl_matrix = correl_matrix 
        self.com_traj      = com_traj
        self.cluster_info  = cluster_info
        if traj != None:
            self.traj = traj
        else:
            self.traj = pt.load(traj_file, top = top_file, 
                                frame_indices = slice(start, end, 
                                                       stride))

    def collect_com_traj(self, selection, selection_name):
        ''' 
        Purpose:   To take a trajectory and calculate center-of-mass 
                   trajectories for each provided selection and return 
                   the result as a pandas dataframe.
        Arguments: self:           `Corr` object containing an 
                                   associated pytraj trajectory. 
                   selection:      List of group selections. Must be 
                                   the same length as 
                                   `selection_name`.
                   selection_name: Name of each selection. Must be 
                                   the same length as `selection`.
        Returns:   NOTHING. Stores a pandas dataframe with the column
                   names as the name of each center of mass and each
                   entry as a numpy array of the xyz coordinates for
                   each frame. Stored in the `Corr()` class instance 
                   attribute `com_traj` of the provided `Corr()`
                   instance.
        NOTES:     `selection` and `selection_name` must be the same 
                   length.
        '''
        com_traj = dict()
        for ii in range(len(selection)):
            com_traj[selection_name[ii]] = list(
                                           pt.vector.center(self.traj, 
                                           selection[ii]))
        com_traj = pd.DataFrame.from_dict(com_traj)
        self.com_traj = com_traj


    def df_corr(self, col_names):
        '''
        Purpose:   Calculates the correlations amongst all columns of 
                   a dataframe. Specifically, one containing atomic 
                   trajectories.
        Arguments: self:      `Corr()` instance containing a pandas 
                              dataframe containing an appropriate 
                              entry for the `com_traj` attribute 
                              where each datapoint is an np array 
                              representing an xyz coordinate.
                   col_names: Names of the columns in the pd df.
        Returns:   NOTHING. Updates the instance attribute,
                   `correl_matrix`, of the provided `Corr()` 
                   instance to a symmetric dataframe containing the 
                   correlations among columns.
        '''
        # Collect all average values of vectors.
        all_cols = []
        for col in self.com_traj.columns:
            all_cols.append(np.array(list(self.com_traj[col])))
        all_cols = np.array(all_cols)
        
        # Create to numpy arrays that house all possible combinations 
        # of elements.
        vec1 = []
        vec2 = []
        ii = 0
        for v1 in all_cols:
            #print(ii)
            for v2 in all_cols:
                vec1.append(v1)
                vec2.append(v2)
            ii += 1
            
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
    
        corr = []
        for ii in range(len(vec1)):
            v1 = vec1[ii]
            v2 = vec2[ii]
            corr.append(n_n_corr(v1, v2))
        
        corr = np.array(corr)
        axis_len = int(np.sqrt(len(vec1)))
        corr = [list(ii) for ii in corr.reshape(axis_len, axis_len)]
        corr = dict(zip(col_names, corr))
        
        corr = pd.DataFrame.from_dict(corr, orient = 'index', 
                                           columns = col_names)
    
        self.correl_matrix = corr

    def reorder_corr(self, sequence):
        '''
        Purpose:   Take a correlation matrix provided as a dataframe and
                   reshape it using a user-provided sequence.
        Arguments: self:     `Corr()` instance containing a symmetric 
                             correlation matrix stored in a dataframe.
                   sequence: A list used to reorder the matrix.
        Returns:   NOTHING. Replaces the `correl_matrix` attribute with
                   the reordered matrix in the dataframe.
        '''
        # Get df as np array and extract column names.
        corr_array = self.correl_matrix.to_numpy()
        indicies = self.correl_matrix.index
        col_names = self.correl_matrix.columns
        
        # Reshape the array according to the specified sequence.
        corr_array = np.array(corr_array)
        corr_array = corr_array[sequence, :]
        corr_array = corr_array[:, sequence]
    
        # Sort the column names so they match.
        col_names = list(np.array(col_names)[sequence])
        
        # Convert the np.array to a list of lists.
        corr_array = list(corr_array)
        corr_array = [list(entry) for entry in corr_array]
    
        # Create a dictionary to be turned into a dataframe.
        corr_array = dict(zip(col_names, corr_array))
        
        # Convert the dictionary to a symmetric dataframe.
        corr_array = pd.DataFrame.from_dict(corr_array, 
                                            orient = 'index', 
                                            columns = col_names)
        self.correl_matix = corr_array

    def get_clusters(self):
        '''
        Purpose:   A crude method for extracting cluster names from pd 
                   df given the naming convention.
        Arguments: self: Instance of `Corr()` containing a properly
                         formatted `correl_matrix` instance attribute.
        Returns:   NOTHING. Updates the `cluster_info` attribute
                   to be a dictionary translating dataframe column 
                   names to their cluster.
        '''
        clusters = dict()
        for col in self.correl_matrix.columns:  
            clusters[col] = col.split('.')[0]
        self.cluster_info = clusters    

    def visualize_feature_correlation(self, save_f = None):
        '''
        Purpose:   To generate a correlation feature plot for the
                   provided data.
        Arguments: self:   Instance of `Corr()` containing correct
                           values for the `correl_matrix` and
                           `cluster_info` instance attributes.
                   save_f: Saves the plot to the provided file name.
                           Just returns the plot if save = `None`.
        '''
        df = self.com_traj.copy()
        corr = self.correl_matrix
        default_feature_order = sorted(list(df.columns))
        # convert to condensed
        corr_condensed = hc.distance.squareform(1 - np.abs(self.correl_matrix))
        z = hc.linkage(corr_condensed, method='average');
        feature_order = hc.dendrogram(z, labels = df.columns, 
                                      no_plot = True)["ivl"];
        
        #compute PCA and store as X,Y coordinates for each feature
        pca = PCA(n_components = 3)
        pca.fit(np.abs(corr))
        names = list(df.columns)
        coords = pca.transform(np.abs(corr)).tolist()
        pca_coords = dict(zip(names, coords))
        pca_coords = pd.DataFrame.from_dict(pca_coords, orient = "index")
        pca_coords = pca_coords.reset_index()
        pca_coords = pca_coords.rename({0: "X", 1: "Y", 2: "Z", 
                                        "index": "feature"}, axis = 1)
    
        num_labels = self.cluster_info.values()
        num_labels = set(num_labels)
        num_labels = len(num_labels)
        silhouette_scores = [
            {
                "cluster_num": num_labels,
                "silhouette_score": 1,
                "feature": col,
                "cluster": self.cluster_info[col]
            }
            for col in df.columns
        ]
        
        
        cluster_label_df = pd.DataFrame(silhouette_scores)
        # Please father forgive me for this garbage.
        cluster_label_df["cluster_size"] = cluster_label_df.\
                groupby(["cluster_num", "cluster"])["feature"].\
                transform("count")
    
        cluster_label_df["key"] = cluster_label_df["cluster_num"].\
                astype(str).str.cat(cluster_label_df["feature"].\
                astype(str), sep=":")
    
        
        cluster_label_df["Subunit"] = cluster_label_df.\
                groupby(["cluster_num","cluster"])["feature"].\
                transform("first")
        
        default_cluster_num = cluster_label_df.\
                groupby("cluster_num")["silhouette_score"].\
                max().idxmax()
    
        # Set to 1 because our modification of the original has no 
        # targets.
        pca_coords = pca_coords.reset_index()
        pca_coords["target_corr"] = 1
            
        # Add cluster name to pca_coords.
        pca_coords["Subunit"] = [ii.split('.')[0] for ii in \
                                 list(pca_coords["feature"])]
        pca_coords["Repeat"] = [ii.split('.')[1] for ii in \
                                list(pca_coords["feature"])]
        
        # get dataset for lines between features (if they have higher 
        # correlation than corr_threshold)
        corr_lines = corr.reset_index(drop=False).\
                rename({"index":"feature"}, axis=1)\
                .melt(id_vars = ["feature"], 
                      var_name = "feature_2", 
                      value_name = "corr")\
                .query("feature > feature_2")
    
        # Create a dictionary from pca_coords to lookup coordinates 
        # based on feature name.
        coord_lookup = dict()
        for ii in range(len(pca_coords)):
            key = pca_coords["feature"][ii]
            value = [pca_coords["X"][ii], \
                     pca_coords["Y"][ii], \
                     pca_coords["Z"][ii]]
            coord_lookup[key] = value
        
        # Keep only those lines with strong corelations 
        # specified by user.
        corr_lines["corr_abs"] = np.abs(corr_lines["corr"])
        corr_lines = corr_lines.loc[corr_lines["corr_abs"] > 0.6]
        
        # Lookup coordinates for each feature and add them to the 
        # dataframe.
        X1 = []
        X2 = []
        Y1 = []
        Y2 = []
        for ii in range(len(corr_lines)):
            x1 = coord_lookup[corr_lines["feature"].iloc[ii]][0]
            x2 = coord_lookup[corr_lines["feature_2"].iloc[ii]][0]
            y1 = coord_lookup[corr_lines["feature"].iloc[ii]][1]
            y2 = coord_lookup[corr_lines["feature_2"].iloc[ii]][1]
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)
    
        corr_lines["X1"] = X1
        corr_lines["X2"] = X2
        corr_lines["Y1"] = Y1
        corr_lines["Y2"] = Y2
        
        # Generate column containings distance between points.
        corr_lines["X.diff"] = corr_lines["X2"] - corr_lines["X1"]
        corr_lines["Y.diff"] = corr_lines["Y2"] - corr_lines["Y1"]
        corr_lines["dist"]   = ((corr_lines["X.diff"]) ** 2 + \
                               (corr_lines["Y.diff"]) ** 2)**0.5
    
        '''----------
        | Plot Data |
        ----------'''
    
        #sns.set()
        # Base plot with cluster shown by color.
        sns.relplot(data = pca_coords, x = "X", y = "Y", 
                    hue = "Subunit", height = 8)
        
        for line in range(0, pca_coords.shape[0]):
            plt.text(pca_coords.X[line] + 0.02, pca_coords.Y[line], 
                     pca_coords.Repeat[line], 
                     horizontalalignment='left', size='medium', 
                     color='black', weight='semibold')
        
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', \
                  '#f58231', '#911eb4', '#46f0f0', '#f032e6', \
                  '#bcf60c', '#fabebe', '#008080', '#e6beff', \
                  '#9a6324', '#fffac8']
        clusters = list(set(pca_coords["Subunit"]))
        color_map = dict(zip(clusters, colors))
        
        # Draw lines between datapoints with correlations above 0.7.
        X = np.array([np.array(corr_lines["X1"]), 
                      np.array(corr_lines["X2"])])
        Y = np.array([np.array(corr_lines["Y1"]), 
                      np.array(corr_lines["Y2"])])
        plt.plot(X, Y, zorder = 0, color = "grey")
        plt.ylabel("PC2")
        plt.xlabel("PC1")

        if save_f == None:
            pass
        else:
            plt.savefig(save_f)

    def visualize_heatmap(self, save_h):
        '''
        Purpose:   To generate a correlation heatmap plot for the
                   provided data.
        Arguments: self:   Instance of `Corr()` containing correct
                           values for the `correl_matrix` and
                           `cluster_info` instance attributes.
                   save_h: Controls whether a plot is saved or just
                           returned. If a filename is provided then
                           a plot will be saved.
        Returns:   NOTHING. Either an image is rendered, if
                   used in a jupyter-notebook or saved to a
                   file.
        '''
        # Reorder so that it goes from top left to right.
        corr_mat = self.correl_matrix.iloc[::-1, :]
        
        mask = np.ones_like(corr_mat, dtype=np.bool)
        
        plt.subplots(figsize=(10,10))
        sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='plasma',
                    vmin=-1, vmax=1, cbar_kws={"shrink": .8}, 
                    square = True)

        plt.savefig("heatmap.png")

        if save_h == None:
            pass
        else:
            plt.savefig(save_h)

if __name__ == "__main__":
    parser = arg.ArgumentParser(description = mes.Summary)

    requiredNamed = parser.add_argument_group('Required Named Arguments')


    requiredNamed.add_argument('--traj',    help = mes.traj_help, 
                               required=True)
    requiredNamed.add_argument('--top',     help = mes.top_help, 
                               required=True)
    requiredNamed.add_argument('--region',  help = mes.region_help, 
                               required=True)
    requiredNamed.add_argument('--reorder', help = mes.reorder_help, 
                               required=True)
    requiredNamed.add_argument('--names',   help = mes.names_help, 
                               required=True)
    requiredNamed.add_argument('--cut',     help = mes.cut_help, 
                               required=True)
    requiredNamed.add_argument('--saveh',   help = mes.saveh_help, 
                               required=True)
    requiredNamed.add_argument('--savef',   help = mes.savef_help, 
                               required=True)
    requiredNamed.add_argument('--start',   help = mes.start_help, 
                               required=True)
    requiredNamed.add_argument('--end',     help = mes.end_help,  
                               required=True)
    requiredNamed.add_argument('--stride',  help = mes.stride_help,  
                               required=True)

    args = parser.parse_args()

    #- Handle Arguments -#
    trajectory_file = args.traj
    topology_file   = args.top
    region_sele     = args.region.split()
    reorder         = args.reorder.split()
    reorder         = [int(num) for num in reorder]
    region_names    = args.names.split()
    corr_cut        = float(args.cut)
    save_h          = args.saveh
    save_f          = args.savef
    start           = int(args.start)
    end             = int(args.end)
    stride          = int(args.stride)

    # Initiate instance of the `Corr()` class.
    corr = Corr(trajectory_file, topology_file, start = start, 
                end = end, stride = stride)

    # Add COM information to the `Corr()` instance.
    corr.collect_com_traj(region_sele, region_names)

    # Calculate cross correlation matix and add to
    # `Corr()` instance.
    corr.df_corr(region_names)

    # Get the cluster names and store them in the,
    # you guessed it `Corr()` instance.
    corr.get_clusters()

    # Generate the feature-space plot for the 
    # `Corr()` instance.
    # SAVE IT TO A FILE.
    corr.visualize_feature_correlation(save_f = save_f)

    # Reorder the correlation matrix in the order
    # specified by the user.
    corr.reorder_corr(reorder)

    # Generate the correlation heatmap.
    # SAVE IT TO A FILE.
    corr.visualize_heatmap(save_h = save_h)
