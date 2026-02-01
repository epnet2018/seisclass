# -*- coding: utf-8 -*-
"""
Seismic phase file parser

@author: lezhao.jia@gmail.com

2025.09.20 update
"""
import pandas as pd
from io import StringIO
import time,datetime
import os

def Dis_seismic_phaseRTS(path):
    df_phase = pd.read_csv(
        path,
        sep='\t',
        encoding='utf-8',
        parse_dates=False
        )
    return df_phase

def Dis_seismic_phase(path):
    '''
    Parse seismic phase file

    Parameters
    ----------
    path : str
        Path to phase file, supports Jopens format or simple tab-separated format

    Returns
    -------
    DataFrame
        Phase arrival data in DataFrame format

    '''
    path=os.path.abspath(path)
    with open(path,'r',encoding='utf-8') as f:
      read_data=f.read()
      f.closed
    
    try:
        Seis_info=read_data.split('#Phase Arrivals:')  
        Phase_info=Seis_info[1].split('#Station Magnitudes:')
        Seis_info=Seis_info[0]
        Mag_info=Phase_info[1]
        Phase_info=Phase_info[0]
        
        df_phase=pd.read_csv(StringIO(Phase_info), delim_whitespace=True,header=0,names=["id", "dist", "azi", "phase","date","time","res","wt"],index_col=0)
        df_mag=pd.read_csv(StringIO(Mag_info), delim_whitespace=True,header=0,names=["id", "dist", "azi", "type","date","time","value","res","amp","per"]) 
        
        Seis_info=Seis_info.split(sep=None)
        O_time=datetime.datetime(int(Seis_info[0]),int(Seis_info[1]),int(Seis_info[2]),int(Seis_info[3]),int(Seis_info[4]),int(float(Seis_info[5])))
        lat=Seis_info[6]
        lon=Seis_info[8]
        dep=Seis_info[10]
        mag=Seis_info[12]
        mag_flag=Seis_info[13]
        
        df_phase['O_time']=O_time
        df_phase['lat']=lat
        df_phase['lon']=lon
        df_phase['dep']=dep
        df_phase['mag']=mag
        df_phase['mag_flag']=mag_flag
    except (IndexError, ValueError):
        df_phase=pd.read_csv(path, sep='\t', encoding='utf-8', header=0)
        if 'Phase_name' in df_phase.columns:
            df_phase=df_phase.rename(columns={'Phase_name':'phase', 'Phase_time':'date', 'Phase_time_frac':'time', 'Distance':'dist', 'Azi':'azi'})
        if 'id' not in df_phase.columns:
            if all(col in df_phase.columns for col in ['Net_code', 'Sta_code', 'Loc_id', 'Chn_code']):
                df_phase['Net_code']=df_phase['Net_code'].astype(str)
                df_phase['Sta_code']=df_phase['Sta_code'].astype(str)
                df_phase['Loc_id']=df_phase['Loc_id'].astype(str)
                df_phase['Chn_code']=df_phase['Chn_code'].astype(str)
                df_phase['id']=df_phase['Net_code']+'.'+df_phase['Sta_code']+'.'+df_phase['Loc_id']+'.'+df_phase['Chn_code']
            else:
                df_phase['id']=[str(i) for i in range(len(df_phase))]
        if 'O_time' not in df_phase.columns:
            df_phase['O_time']=datetime.datetime.now()
        if 'lat' not in df_phase.columns:
            df_phase['lat']=0.0
        if 'lon' not in df_phase.columns:
            df_phase['lon']=0.0
        if 'dep' not in df_phase.columns:
            df_phase['dep']=0.0
        if 'mag' not in df_phase.columns:
            df_phase['mag']=0.0
        if 'mag_flag' not in df_phase.columns:
            df_phase['mag_flag']=''
    
    return df_phase


