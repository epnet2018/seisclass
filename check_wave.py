# -*- coding: utf-8 -*-
"""
Seismic waveform classification module
@author: jialuozhao
@mail: 18429320@qq.com

"""
import os
import sys
import warnings

# Selective warning suppression for TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*TensorFlow.*')
warnings.filterwarnings('ignore', message='.*Keras.*')

import pandas as pd
from datetime import datetime,timedelta 
import numpy as np 
import obspy
from obspy.core import UTCDateTime
import joblib

package_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from .packages import read_phase
except ImportError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from packages import read_phase

def return_result(x):
    x=np.argmax(x)
    if(x==0):
        return 'Earthquake'
    else:
        return 'Non-earthquake'

def check_seed(seed_path, phase_path, model_str='251111nw'):
    '''
    Classify seismic event from SEED file

    Parameters
    ----------
    seed_path : str
        Path to SEED file
    phase_path : str
        Path to phase file
    model_str : str, optional
        Model name, default is '251111nw'

    Returns
    -------
    str
        Comma-separated result string: event_type,earthquake_prob,explode_prob,collapse_prob

    '''

    model_name = model_str
    model_file = os.path.join(package_dir, 'model', model_name, 'event_model.json')
    model_weight = os.path.join(package_dir, 'model', model_name, 'event_model.h5')
    model_pkl = os.path.join(package_dir, 'model', model_name, 'event_model.pkl')

    file_path = seed_path
    phase_path = phase_path

    (Event_id, Event_type) = os.path.splitext(os.path.basename(file_path))

    with open(model_file, 'r') as file:
        model_json = file.read()

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

    tf.config.optimizer.set_jit(True)
    with tf.device('/CPU:0'):
        try:
            new_model = tf.keras.models.load_model(model_weight, compile=False)
        except Exception as e:
            try:
                custom_objects = {
                    'Sequential': Sequential,
                    'Input': Input,
                    'Conv2D': Conv2D,
                    'MaxPooling2D': MaxPooling2D,
                    'Dropout': Dropout,
                    'Flatten': Flatten,
                    'Dense': Dense
                }
                new_model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
                new_model.load_weights(model_weight)
            except Exception as e2:
                try:
                    from keras.models import model_from_json
                    from keras.layers import Input as KInput, Conv2D as KConv2D, MaxPooling2D as KMaxPooling2D, Dropout as KDropout, Flatten as KFlatten, Dense as KDense
                    from keras.models import Sequential as KSequential
                    
                    custom_objects_keras = {
                        'Sequential': KSequential,
                        'Input': KInput,
                        'Conv2D': KConv2D,
                        'MaxPooling2D': KMaxPooling2D,
                        'Dropout': KDropout,
                        'Flatten': KFlatten,
                        'Dense': KDense
                    }
                    new_model = model_from_json(model_json, custom_objects=custom_objects_keras)
                    new_model.load_weights(model_weight)
                except Exception as e3:
                    raise RuntimeError(f"Failed to load model: {e}, {e2}, {e3}")

    wave_array = np.empty(0)
    pic_result = ''
    
    st = obspy.read(file_path)
    
    df_phase = read_phase.Dis_seismic_phase(phase_path)
    df_p_phase = df_phase[df_phase['phase'].str.startswith('P')]
    
    df_temp = df_p_phase['id'].str.split('.', expand=True)
    df_p_phase.loc[:, 'Net_code'] = df_temp[0]
    df_p_phase.loc[:, 'Sta_code'] = df_temp[1]
    
    if df_p_phase['time'].dtype != 'object':
        df_p_phase['time'] = df_p_phase['time'].astype(str)
    if df_p_phase['date'].dtype != 'object':
        df_p_phase['date'] = df_p_phase['date'].astype(str)
    
    def format_time(row):
        date_str = row['date']
        time_str = row['time']
        combined_str = f"{date_str} {time_str}"
        if ' ' in combined_str:
            parts = combined_str.split(' ')
            if len(parts) >= 3:
                date_part = parts[0]
                time_part = parts[1]
                ms_part = parts[2]
                ms_part = ms_part.ljust(4, '0')[:4]
                return f"{date_part} {time_part}.{ms_part}"
        return combined_str
    
    df_p_phase['Phase_mtime'] = df_p_phase.apply(format_time, axis=1)
    df_p_phase['Phase_mtime'] = pd.to_datetime(df_p_phase['Phase_mtime'], errors='coerce')
    
    i = 0
    for row in df_p_phase.itertuples():
        phase_mtime = getattr(row, 'Phase_mtime')
        if pd.isna(phase_mtime):
            continue
        rowst = st.select(network=getattr(row, 'Net_code'), station=getattr(row, 'Sta_code'))
        start = UTCDateTime(phase_mtime - timedelta(seconds=3) - timedelta(hours=8))
        
        j = 0
        channel_count = 0
        for j in range(0, rowst.count()):
            if channel_count == 3:
                break
            if rowst[j].stats.endtime < (start + 60):
                continue
            x = rowst[j].slice(starttime=start, endtime=start + 60).data
            if len(x) != 6001:
                continue
            wave_array = np.append(wave_array, x)
            channel_count += 1
        i += 1

    if wave_array.shape[0] == 0:
        return ('Earthquake,0,0,0', pic_result)

    wave_array = wave_array.reshape(-1, 18003)
    wave_array = wave_array[:, 0:18000]
    
    col = np.around(np.mean(wave_array, axis=1).reshape(-1, 1))
    ins = np.where(wave_array == 0)
    wave_array[ins] = np.take(col, ins[0])

    try:
        import warnings
        
        # Selective warning suppression - only suppress specific scikit-learn warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*unpickle.*')
            warnings.filterwarnings('ignore', message='.*version.*')
            warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')
            warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
            
            scaler = joblib.load(model_pkl)
            wave_array = scaler.transform(wave_array)
    except Exception as e:
        # Fallback normalization if scaler fails
        wave_array = (wave_array - np.mean(wave_array)) / (np.max(wave_array) - np.min(wave_array) + 1e-8)
    
    wave_array = wave_array.reshape(-1, 100, 180, 1)
        
    np.set_printoptions(suppress=True)

    predictions = new_model.predict(wave_array)

    eathquake = round((np.sum(predictions[:, 0]) / predictions.shape[0]) * 100, 2)
    explode = round((np.sum(predictions[:, 1]) / predictions.shape[0]) * 100, 2)
    collapse = round((np.sum(predictions[:, 2]) / predictions.shape[0]) * 100, 2)
    
    return '%s,%s,%s,%s' % (return_result((eathquake, explode, collapse)), eathquake, explode, collapse)

