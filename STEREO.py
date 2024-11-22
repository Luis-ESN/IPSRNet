# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:32:20 2024

@author: Luís Eduardo Sales do Nascimento
"""

import matplotlib.pyplot as plt
import os
import numpy as np
#import pandas as pd
from matplotlib.ticker import MaxNLocator
import cv2
from datetime import datetime, timedelta
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries


# Define Methods to download STEREO data
def Get_STEREO_data(start, end, spacecraft='STA'):
    """
        Download timeseries of solar wind plasma and interplanetary magnetic 
        field parameters from STEREO Spacecrafts

        Parameters
        ----------
        start : datetime
            Start time of observed period.
        end : datetime
            End time of observed period.
        spacecraft : string, optional (default='STA')
            Spacecraft from which in-situ data will be collected
            Either "STA" or "STB" (case-insensitive)
            "STA" means get data from STEREO-A spacecraft
            "STB" means get data from STEREO-B spacecraft
            
        Returns
        -------
        IMF_df : pd.DataFrame
            DataFrame of shape (n, 1). Interplanetary Magnetic Field (IMF) parameter,
            with cadence of 1 minute. With n equal to number of time-steps.
        plasma_df : pd.DataFrame
            DataFrame of shape (n, 3). Solar Wind Plasma parameters, density, speed and temperature,
            with cadence of 1 minute. With n equal to number of time-steps.
    """
    sc = spacecraft.upper()
    if sc!='STA' and sc!='STB':
        raise Exception("Only 'STA' or 'STB' are valid values for 'spacecraft' parameter.")

    trange = a.Time(start, end)
    if sc=='STA':
        dataset = a.cdaweb.Dataset.sta_l2_magplasma_1m
    if sc=='STB':
        dataset = a.cdaweb.Dataset.stb_l2_magplasma_1m
    result = Fido.search(trange, dataset)
    
    downloaded_files = Fido.fetch(result, progress=False)
    df = TimeSeries(downloaded_files)
    df = df.to_dataframe()
    df = df.loc[start:end]
    df = df.asfreq('min')
    
    df[df < 0] = np.nan
    
    IMF_df = df[['BTOTAL']]
    plasma_df = df[['Np', 'Vp', 'Tp']]
    
    return IMF_df, plasma_df

def plot_STEREO(df_b, df_p, s, e, folder_to_save):
    """
        Plot timeseries of solar wind plasma and interplanetary magnetic 
        field parameters from STEREO Spacecraft

        Parameters
        ----------
        df_b : pd.DataFrame
            DataFrame of shape (n, 1). Interplanetary Magnetic Field (IMF) parameter,
            with cadence of 1 minute. With n equal to number of time-steps. 
            Column name should be "BTOTAL".
        df_p : pd.DataFrame
            DataFrame of shape (n, 3). Solar Wind Plasma parameters, density, speed and temperature,
            with cadence of 1 minute. With n equal to number of time-steps.
            Column names should be "Np", "Vp" and "Tp", respectively.
        s : datetime
            Start time of observed period.
        e : datetime
            End time of observed period.
        folder_to_save : string
            filename to which the image is saved. 
            A .png extension will be appended to the filename.
            
        Returns
        -------
        None
            Plot and save figure with the given data but does not return anything.
    """
    fig = plt.figure(figsize=(4,4))
    plt.subplots_adjust(wspace=0, hspace=0.2)
    
    ax1_plt = fig.add_subplot(4, 1, 1)
    ax2_plt = fig.add_subplot(4, 1, 2)
    ax3_plt = fig.add_subplot(4, 1, 3)
    ax4_plt = fig.add_subplot(4, 1, 4)

    for axis in ['top','bottom','left','right']:
        ax1_plt.spines[axis].set_linewidth(0.25)
        ax2_plt.spines[axis].set_linewidth(0.25)
        ax3_plt.spines[axis].set_linewidth(0.25)
        ax4_plt.spines[axis].set_linewidth(0.25)

    ax1_plt.plot(df_b.index, df_b['BTOTAL'], color='black', linewidth=0.3, marker='+', markersize=1, mec='black', mew=0.1)
    ax1_plt.set_xlim(s, e)
    ax1_plt.yaxis.set_major_locator(MaxNLocator(5))
    ax1_plt.tick_params(bottom=False, labelbottom=False)
    ax1_plt.tick_params(axis='y', labelsize=5, pad=0.5, length=1, width=0.25)
    
    ax2_plt.plot(df_p.index, df_p['Np'], color='black', linewidth=0.3, marker='+', markersize=1, mec='black', mew=0.1)
    ax2_plt.set_xlim(s, e)
    ax2_plt.yaxis.set_major_locator(MaxNLocator(5))
    ax2_plt.tick_params(bottom=False, labelbottom=False)
    ax2_plt.tick_params(axis='y', labelsize=5, pad=0.5, length=1, width=0.25)
    
    ax3_plt.plot(df_p.index, df_p['Vp']/(10**2), color='black', linewidth=0.3, marker='+', markersize=1, mec='black', mew=0.1)
    ax3_plt.set_xlim(s, e)
    ax3_plt.yaxis.set_major_locator(MaxNLocator(5))
    ax3_plt.tick_params(bottom=False, labelbottom=False)
    ax3_plt.tick_params(axis='y', labelsize=5, pad=0.5, length=1, width=0.25)
    
    ax4_plt.plot(df_p.index, df_p['Tp']/(10**5), color='black', linewidth=0.3,marker='+', markersize=1, mec='black', mew=0.1)
    ax4_plt.set_xlim(s, e)
    ax4_plt.yaxis.set_major_locator(MaxNLocator(5))
    ax4_plt.tick_params(bottom=False, labelbottom=False)
    ax4_plt.tick_params(axis='y', labelsize=5, pad=0.5, length=1, width=0.25)
    
    fig.savefig(folder_to_save+'.png', format='png', bbox_inches='tight', dpi=512)
    plt.cla()
    plt.close('all')
    
    #opencv
    image = cv2.imread(folder_to_save+'.png')
    image = cv2.resize(image, (2048, 2048), interpolation=cv2.INTER_CUBIC)
    
    if os.path.exists(folder_to_save+'.png'):
        os.remove(folder_to_save+'.png')
    cv2.imwrite(folder_to_save+'.png', image)


def STEREO(shock_date, folder_to_save, time_window=15, spacecraft='STA'):
    """
        Download and plot timeseries of solar wind plasma and interplanetary magnetic 
        field parameters from STEREO Spacecraft

        Parameters
        ----------
        shock_date : string or datetime
            Date to analyze the occurrence of an interplanetary shock wave. 
            If shock_date is a string must be in the format "%Y-%m-%d %H:%M:%S".
        folder_to_save : string
            filename to which the image is saved. 
        time_window : int, optional (default=15)
            Duration of observation time for the interplanetary shock wave, 
            covering both upstream and downstream parameters.
        spacecraft : string, optional (default='STA')
            Spacecraft from which in-situ data will be collected
            Either "STA" or "STB" (case-insensitive)
            "STA" means get data from STEREO-A spacecraft
            "STB" means get data from STEREO-B spacecraft
            
            
        Returns
        -------
        None
            Plot and save figure with the given data but does not return anything.
    """
    if type(shock_date) == str:
        date =  datetime.strptime(shock_date, '%Y-%m-%d %H:%M:%S')
    elif type(shock_date) == datetime:
        date = shock_date
    else:
        raise Exception("Only String or datetime values are valid for 'shock_date' parameter")
    
    sc = spacecraft.upper()
    if sc!='STA' and sc!='STB':
        raise Exception("Only 'STA' or 'STB' are valid values for 'spacecraft' parameter.")
    
    date_start = date - timedelta(minutes = time_window)
    date_end = date + timedelta(minutes = time_window-1)
    
    df_b, df_p = Get_STEREO_data(date_start, date_end, spacecraft=sc)
    
    plot_STEREO(df_b,df_p, date_start, date_end, folder_to_save)
