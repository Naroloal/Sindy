#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 07:44:12 2020

@author: lorenzo
"""


from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import itertools
from scipy import cluster


def get_cluster_indexs(number,fl):
    if isinstance(number,list):
        indexes_list = []
        for i in number:
            indexes_list.append(np.where(fl == i)[0])
        return indexes_list
    else: return np.where(fl == number)[0]

def plot_branch(cluster,fl,s_plot,df,label = False):
    tup = get_cluster_indexs(cluster,fl)
    spikes = df.Mean[tup]
    labels1 = list(df.bNoise[tup])
    try:
        labels2 = list(df.bUnSure[tup])
    except:pass
    i = 0
    for element in spikes:
        if label:
            try:s_plot.plot(element,label = '{},{}'.format(labels1[i],labels2[i]))
            except:s_plot.plot(element,label = '{}'.format(labels1[i]))
            i+=1
            s_plot.legend()
        else:s_plot.plot(element)
    return tup

def Mat_to_dataframe(path,data = 'NoiseClusters'):
    Data = loadmat(path)
    Data = Data[data]
    Data = Data[0]
    Column_names = Data.dtype.names
    Dataframe = {}
    for i in range(len(Column_names)):
        Dataframe[Column_names[i]] = []
    for i in range(len(Data)):
        row = Data[i]
        for j in range(len(row)):
            column = row[j]
            Dataframe[Column_names[j]].append(column)
    Data = pd.DataFrame.from_dict(Dataframe)
    Data = pd.DataFrame.from_dict(Data)

    return Data

def see_teams(lista_de_teams,fl):
    cmap = plt.get_cmap('gnuplot')
    if isinstance(lista_de_teams,list):
        colors = [cmap(i) for i in np.linspace(0, 1, len(lista_de_teams))]
        speiksheipes = get_cluster_indexs(lista_de_teams,fl)
        fig = plt.figure(6)
        i = 0
        for element in speiksheipes:
            for element2 in element:
                plt.plot(df.Mean[int(element2)],c = colors[i])
                i+=1
    else:
        speiksheipes = get_cluster_indexs(lista_de_teams,fl)
        colors = [cmap(i) for i in np.linspace(0, 1, len(speiksheipes))]
        fig = plt.figure(6)
        i = 0
        for element in speiksheipes:
            plt.plot(df.Mean[int(element)],c = colors[i])
            i+=1
            
def dataframe_to_math(path,df):
    a = {name: col.values for name, col in df.items()}
    sp.io.savemat(path,a)
        
    
def Search_candidates(i,spikes,fl,N_teams,Correlation_p,size,Good_candidates = False):
    r = []
    Not_good_candidate = []
    for Candidates in itertools.combinations(spikes,r = size):
            while not Good_candidates:
                Good_candidates = True
                for j in range(1,N_teams+1):
                    if not j==i:
                        other_spikes = get_cluster_indexs(j,fl)
                        for k in range(len(other_spikes)):
                            check = [Correlation_p[i][k] > 0.9 for q in Candidates]
                            if not all(check):
                                Good_candidates = False
                                break
                            if not Good_candidates:break
                    
                if Good_candidates: r.append([i,Candidates])         
                else:
                   print('No good candidates found for Team {}'.format(i))
                   Good_candidates = True
                   Not_good_candidate.append(i)
    return [r,Not_good_candidate]

def print_inf(dataframe,i,show_bNoise_bUnsure = False):
    Info = dataframe.iloc[i]
    Data = []
    Data.append(Info.PatientExperiment[0])
    Data.append(Info.Channel[0][0])
    Data.append(Info.Cluster[0][0])
    if show_bNoise_bUnsure:
        Data.append(Info.bNoise)
        Data.append(Info.bUnSure)
    return Data
def plot_teams(df,fl,NUM_PLOTS_PER_FIG = 6):
    num_figs = max(fl)//NUM_PLOTS_PER_FIG+1
    Figures = [plt.figure(i) for i in range(num_figs)]
    axes = []
    for figure in Figures:
        axes.append(figure.subplots(3,3).flat)
        for j in range(1,max(fl)+1):
            Fig_to_plot = j//NUM_PLOTS_PER_FIG
            fig = axes[Fig_to_plot]
            ax_to_plot = j%NUM_PLOTS_PER_FIG
            subplot = fig[ax_to_plot]   
            indexes = get_cluster_indexs(j,fl)
            for i in indexes:
                subplot.plot(df.Mean.iloc[i],label = '{}\n{}'.format(df.bNoise.iloc[i],df.Data.iloc[i]))
                subplot.title.set_text('Team {}'.format(i))
                subplot.legend()

def find_mix(df,fl):
    Mix_teams = []
    for team in np.arange(1,max(fl)+1):
        spikes_shapes = get_cluster_indexs(team,fl)
        aux = list(df.bNoise[spikes_shapes])
        aux2 = []
        for element in aux:
                aux2.append(element)
                if  (aux2.count(1) !=0) and (aux2.count(0)!=0):
                    Mix_teams.append(team)
    return Mix_teams
        
def plot_Bulk(Bulk,ax,label = ''):
    for spike in Bulk:
        ax.plot(spike,color = 'b',linewidth = 0.1)
        
def load_time_files(mat_filename):
    timefile = loadmat(mat_filename)
    cluster_class = timefile['cluster_class'].T
    cluster = np.array(cluster_class[0],dtype = int)
    time = cluster_class[1]
    s = dict({'cluster':cluster,'time':time})
    dataframe = pd.DataFrame(s)
    return dataframe

def get_isi_from_timefiles(Input,PatientExperiment):
    if isinstance(Input,list):
        ISI_info = pd.DataFrame()
        for timefile_name in Input:
            channel = timefile_name[9:-4]
            df = load_time_files(timefile_name)
            df_grouped = df.groupby('cluster')
            q = df_grouped.apply(np.diff)
            q = q.apply(np.histogram,bins = 100,range = {0,100})
            q = pd.DataFrame(q).T
            q['Channel'] = channel
            q['PatientExperiment'] = PatientExperiment
            ISI_info = pd.concat([ISI_info,q],ignore_index = True)
        ISI_info = ISI_info[['Channel','PatientExperiment'] + [c for c in ISI_info if c not in ['Channel','PatientExperiment']]]
        return ISI_info
    else:
        channel = Input[9:-4]
        df = load_time_files(Input)
        df_grouped = df.groupby('cluster')
        q = df_grouped.apply(np.diff)
        q = q.apply(np.histogram,bins = 100,range = {0,100})
        q = pd.DataFrame(q).T 
        q['Channel'] = channel
        q['PatientExperiment'] = PatientExperiment
        q = q[['Channel','PatientExperiment'] + [c for c in q if c not in ['Channel','PatientExperiment']]]
        return q
            
def plots_hierarchichal_clustering(df,threshold2 = 0.1):
    '''Stadarization of the data'''
    df['Mean'] = (df['Mean'] - df.Mean.apply(lambda row:np.mean(row)))/df.Mean.apply(lambda  row:np.std(row))
    Mean = pd.DataFrame(np.array(df.Mean.to_list())).values
    '''Threshold = sqrt(-0.5ln(alpha)*(m + n )/(m*n)'''
 
    '''Plot matrix correlation clustered'''
    linkage_p = cluster.hierarchy.linkage(Mean,method = 'complete',metric = 'correlation')
    fl = cluster.hierarchy.fcluster(linkage_p,threshold2,criterion = 'distance')
    '''Plot Teams'''
    fig = plt.figure(3,figsize=(10,10))
    axes = fig.subplots(int(max(fl)/4)+1,4)
    axes_flt = axes.flat
    for i in range(1,max(fl)+1):
        plot_branch(i,fl,axes_flt[i-1],df)
        axes_flt[i-1].title.set_text('Team {}'.format(i))
    return fig,axes,fl

            
            
            
        