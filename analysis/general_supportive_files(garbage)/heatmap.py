#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:25:09 2018

@author: Gr3at
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# create the heat map
# deleted code
# heatmap code source : https://github.com/jeanmidevacc/udacity_mlen/blob/master/capstone/analytics/pynb/visualisation.ipynb
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_heatmap(fig, ax,df,title):
   
    m, n = len(df.index),len(df.columns)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel(title, fontsize=15)
    #ax.set_ylabel('Candidates' , fontsize=15)
    #ax.set_title('Who with who', fontsize=15, fontweight='bold')
    ax = plt.imshow(df, interpolation='nearest', cmap='seismic',aspect='auto').axes

    _ = ax.set_xticks(np.linspace(0, n-1, n))
    _ = ax.set_xticklabels(df.columns,rotation=45)
    _ = ax.set_yticks(np.linspace(0, m-1, m))
    _ = ax.set_yticklabels(df.index)


    ax.grid('off')
    ax.xaxis.tick_top()
    path_effects = [patheffects.withSimplePatchShadow(shadow_rgbFace=(1,1,1))]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(cax=cax)
#     for i, j in product(range(m), range(n)):
#         _ = ax.text(j, i, '{0:.2f}'.format(df.iloc[i, j]),
#             size='medium', ha='center', va='center',path_effects=path_effects)
    
    return fig,ax

heatmap_=pd.pivot_table(df_daily_energy_consumption, values="AcornA_sum", index=['month'],columns=['weekday'], aggfunc=np.mean)
heatmap_.columns=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
heatmap_.index=["January","February","March","April","May","June","July","August","September","October","November","December"]

fig, ax = plt.subplots(figsize=(12,12))
fig,ax=plot_heatmap(fig, ax,heatmap_,"Daily Energy Consumption")
ax.set_xlabel("Weekday",fontsize=15)
ax.set_ylabel("Month of the year",fontsize=15)
plt.show()
ax.figure.savefig("heatmap_weekday_month_London.png")
