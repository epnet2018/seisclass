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
        # Split by '#Phase Arrivals:'
        parts = read_data.split('#Phase Arrivals:')
        
        if len(parts) < 2:
            raise ValueError("No '#Phase Arrivals:' marker found")
        
        Seis_info = parts[0].strip()
        
        # Split phase and magnitude info
        phase_parts = parts[1].split('#Station Magnitudes:')
        Phase_info = phase_parts[0].strip()
        
        if len(phase_parts) > 1:
            Mag_info = phase_parts[1].strip()
        else:
            Mag_info = ""
        
        # Parse phase data
        phase_lines = Phase_info.split('\n')
        # Remove header line if present
        data_lines = []
        for line in phase_lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            data_lines.append(line)
        
        Phase_info_clean = '\n'.join(data_lines)
        
        # Determine column count
        if len(data_lines) > 0:
            first_data_line = data_lines[0]
            num_cols = len(first_data_line.split())
            
            if num_cols == 7:
                # Format: id, dist, azi, phase, time, res, wt
                df_phase=pd.read_csv(StringIO(Phase_info_clean), sep=r'\s+',header=None,names=["id", "dist", "azi", "phase","time","res","wt"])
            elif num_cols == 8:
                # Format: id, dist, azi, phase, date, time, res, wt
                df_phase=pd.read_csv(StringIO(Phase_info_clean), sep=r'\s+',header=None,names=["id", "dist", "azi", "phase","date","time","res","wt"])
            else:
                df_phase=pd.read_csv(StringIO(Phase_info_clean), sep=r'\s+',header=None)
                if df_phase.shape[1] >= 4:
                    df_phase.columns = ['id', 'dist', 'azi', 'phase'] + [f'col{i}' for i in range(4, df_phase.shape[1])]
        else:
            raise ValueError("No phase data found")
        
        # Parse event info from first line
        # Format: year month day hour minute second lat lat_err lon lon_err dep dep_err mag
        Seis_info_parts = Seis_info.split()
        if len(Seis_info_parts) >= 13:
            year = int(Seis_info_parts[0])
            month = int(Seis_info_parts[1])
            day = int(Seis_info_parts[2])
            hour = int(Seis_info_parts[3])
            minute = int(Seis_info_parts[4])
            second = int(float(Seis_info_parts[5]))
            
            O_time=datetime.datetime(year, month, day, hour, minute, second)
            lat=Seis_info_parts[6]
            lon=Seis_info_parts[8]
            dep=Seis_info_parts[10]
            mag=Seis_info_parts[12]
            mag_flag='' if len(Seis_info_parts) < 14 else Seis_info_parts[13]
        else:
            O_time=datetime.datetime.now()
            lat=0.0
            lon=0.0
            dep=0.0
            mag=0.0
            mag_flag=''
        
        df_phase['O_time']=O_time
        df_phase['lat']=lat
        df_phase['lon']=lon
        df_phase['dep']=dep
        df_phase['mag']=mag
        df_phase['mag_flag']=mag_flag
        
    except (IndexError, ValueError, pd.errors.ParserError):
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