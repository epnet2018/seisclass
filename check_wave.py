# -*- coding: utf-8 -*-
"""
Seismic waveform classification module (3-class only)
@author: jialuozhao
@mail: 18429320@qq.com

Models:
    - models/2606/model_3class.h5 (Earthquake, Explosion, Collapse)

Input format: (100, 60, 3) - 3-channel spectrogram-like format
Preprocessing: Replace zeros -> Normalize per channel -> Reshape
"""
import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import obspy
from obspy.core import UTCDateTime
from scipy.signal import butter, filtfilt

package_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from .packages import read_phase
except ImportError:
    sys.path.insert(0, package_dir)
    from packages import read_phase

CLASS_NAMES_3 = ['Earthquake', 'Explosion', 'Collapse']

MODEL_3CLASS_PATH = os.path.join(package_dir, 'models', '2606', 'model_3class.h5')

_model_3class = None


def demean(data):
    """Remove DC offset (mean) from waveform data"""
    return data - np.mean(data)


def bandpass_filter(data, freqmin=1.0, freqmax=20.0, df=100.0, corners=4):
    """Apply 1-20Hz bandpass filter to waveform data"""
    from obspy.signal.filter import bandpass as obspy_bandpass
    return obspy_bandpass(data, freqmin=freqmin, freqmax=freqmax, df=df, corners=corners, zerophase=True)


def replace_zeros_per_channel(waveform):
    """
    Replace zero values with channel mean (per-channel processing)
    Consistent with training code in train_final_model_3channel.py
    """
    waveform_fixed = waveform.copy()
    
    for ch_start, ch_end in [(0, 6000), (6000, 12000), (12000, 18000)]:
        channel = waveform[ch_start:ch_end].copy()
        zero_mask = channel == 0
        if np.any(zero_mask):
            non_zero = channel[~zero_mask]
            if len(non_zero) > 0:
                mean = np.mean(non_zero)
                channel[zero_mask] = mean
            else:
                channel[zero_mask] = 0
        waveform_fixed[ch_start:ch_end] = channel
    
    return waveform_fixed


def normalize_per_channel(waveform):
    """
    Normalize waveform per channel (Z-score normalization)
    Consistent with training code in train_final_model_3channel.py
    """
    waveform_normalized = np.zeros_like(waveform, dtype=np.float32)
    
    for ch_start, ch_end in [(0, 6000), (6000, 12000), (12000, 18000)]:
        channel = waveform[ch_start:ch_end]
        mean = np.mean(channel)
        std = np.std(channel)
        if std > 1e-8:
            waveform_normalized[ch_start:ch_end] = (channel - mean) / std
        else:
            waveform_normalized[ch_start:ch_end] = channel - mean
    
    return waveform_normalized


def reshape_to_3channel(waveform):
    """
    Reshape 1D waveform to 3-channel input format
    Consistent with training code in train_final_model_3channel.py
    """
    X_3ch = np.zeros((100, 60, 3), dtype=np.float32)
    X_3ch[:, :, 0] = waveform[0:6000].reshape(100, 60)
    X_3ch[:, :, 1] = waveform[6000:12000].reshape(100, 60)
    X_3ch[:, :, 2] = waveform[12000:18000].reshape(100, 60)
    return X_3ch


def preprocess_waveform(waveform, apply_filter=False):
    """
    Preprocess single waveform for model input
    
    Parameters
    ----------
    waveform : array, shape (18000,)
        1D array containing Z/N/E 3 channels concatenated
    apply_filter : bool, default False
        Whether to apply bandpass filter and demean
    
    Returns
    -------
    array, shape (1, 100, 60, 3)
        Preprocessed waveform ready for model input
    """
    waveform = waveform.astype(np.float32)
    
    if apply_filter:
        for ch_start, ch_end in [(0, 6000), (6000, 12000), (12000, 18000)]:
            channel = waveform[ch_start:ch_end].copy()
            try:
                channel = demean(channel)
                channel = bandpass_filter(channel, freqmin=1.0, freqmax=20.0, df=100.0)
            except Exception:
                pass
            waveform[ch_start:ch_end] = channel
    
    waveform = replace_zeros_per_channel(waveform)
    waveform = normalize_per_channel(waveform)
    waveform = reshape_to_3channel(waveform)
    
    return np.expand_dims(waveform, axis=0)


def load_3class_model():
    """Load 3-class model (Earthquake, Explosion, Collapse)"""
    global _model_3class
    
    if _model_3class is not None:
        return _model_3class
    
    if not os.path.exists(MODEL_3CLASS_PATH):
        return None
    
    import tensorflow as tf
    with tf.device('/CPU:0'):
        try:
            _model_3class = tf.keras.models.load_model(MODEL_3CLASS_PATH, compile=False)
        except Exception:
            try:
                from keras.models import load_model as keras_load_model
                _model_3class = keras_load_model(MODEL_3CLASS_PATH, compile=False)
            except Exception as e:
                print("Warning: Failed to load 3-class model - {}".format(str(e)))
                _model_3class = None
    
    return _model_3class


def load_model():
    """Load 3-class model"""
    return load_3class_model()


def predict_3class(waveform, apply_filter=False):
    """
    Predict using 3-class model
    
    Parameters
    ----------
    waveform : array, shape (18000,)
        1D array containing Z/N/E 3 channels concatenated
    apply_filter : bool
        Whether to apply bandpass filter
    
    Returns
    -------
    dict
        {'result': class_name, 'probs': [prob1, prob2, prob3]}
    """
    model = load_3class_model()
    if model is None:
        return {'result': 'Earthquake', 'probs': [33.33, 33.33, 33.33]}
    
    X = preprocess_waveform(waveform, apply_filter=apply_filter)
    pred = model.predict(X, verbose=0)[0]
    
    return {
        'result': CLASS_NAMES_3[np.argmax(pred)],
        'probs': [round(p * 100, 2) for p in pred]
    }


def read_waveform_from_seed(seed_path, phase_path, max_dist_km=100):
    """
    Read waveform data from SEED file using phase file
    
    Parameters
    ----------
    seed_path : str
        Path to SEED file (supports MiniSEED, Full SEED, and ZIP archives)
    phase_path : str
        Path to phase file
    max_dist_km : float, default 100
        Maximum epicentral distance in kilometers to filter stations
    
    Returns
    -------
    dict
        {'event_id': str, 'wave_array': array, 'n_samples': int, 'stations_used': list}
    """
    import zipfile
    import tempfile
    import shutil
    
    (Event_id, Event_type) = os.path.splitext(os.path.basename(seed_path))
    
    wave_array = np.empty(0)
    
    # Check if file is a ZIP archive
    is_zip = False
    temp_dir = None
    
    try:
        with open(seed_path, 'rb') as f:
            header = f.read(4)
            if header == b'PK\x03\x04':
                is_zip = True
    except Exception:
        pass
    
    if is_zip:
        # Extract ZIP to temporary directory
        temp_dir = tempfile.mkdtemp(prefix='seisclass_')
        try:
            with zipfile.ZipFile(seed_path, 'r') as zf:
                # Extract all mseed files
                mseed_files = [f for f in zf.namelist() if f.endswith('.mseed')]
                if not mseed_files:
                    return {'error': 'No MiniSEED files found in ZIP archive'}
                
                zf.extractall(temp_dir)
                
                # Read all mseed files
                st = obspy.Stream()
                for mseed_file in mseed_files:
                    mseed_path = os.path.join(temp_dir, mseed_file)
                    try:
                        st += obspy.read(mseed_path)
                    except Exception:
                        continue
        except Exception as e:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return {'error': 'Failed to read ZIP archive: {}'.format(str(e))}
    else:
        # Try reading as regular SEED file
        try:
            st = obspy.read(seed_path)
        except Exception:
            try:
                st = obspy.read(seed_path, format='SEED')
            except Exception:
                try:
                    st = obspy.read(seed_path, format='MSEED')
                except Exception as e:
                    return {'error': 'Failed to read SEED file: {}'.format(str(e))}
    
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
        combined_str = "{} {}".format(date_str, time_str)
        if ' ' in combined_str:
            parts = combined_str.split(' ')
            if len(parts) >= 3:
                date_part = parts[0]
                time_part = parts[1]
                ms_part = parts[2]
                ms_part = ms_part.ljust(4, '0')[:4]
                return "{} {}.{}".format(date_part, time_part, ms_part)
        return combined_str
    
    df_p_phase['Phase_mtime'] = df_p_phase.apply(format_time, axis=1)
    df_p_phase['Phase_mtime'] = pd.to_datetime(df_p_phase['Phase_mtime'], errors='coerce')
    
    max_dist_deg = max_dist_km / 111.2
    
    stations_used = []
    
    data_start = min([tr.stats.starttime for tr in st])
    data_end = max([tr.stats.endtime for tr in st])
    
    for row in df_p_phase.itertuples():
        phase_mtime = getattr(row, 'Phase_mtime')
        if pd.isna(phase_mtime):
            continue
        
        dist = getattr(row, 'dist', None)
        if dist is None or pd.isna(dist):
            continue
        if float(dist) > max_dist_deg:
            continue
        
        rowst = st.select(network=getattr(row, 'Net_code'), station=getattr(row, 'Sta_code'))
        
        if rowst.count() == 0:
            continue
        
        rowst_start = min([tr.stats.starttime for tr in rowst])
        rowst_end = max([tr.stats.endtime for tr in rowst])
        
        start_candidates = []
        start_candidates.append(UTCDateTime(phase_mtime - timedelta(seconds=3) - timedelta(hours=8)))
        start_candidates.append(UTCDateTime(phase_mtime - timedelta(seconds=3)))
        start_candidates.append(UTCDateTime(phase_mtime - timedelta(seconds=3) + timedelta(hours=8)))
        
        valid_start = None
        for candidate in start_candidates:
            if rowst_end >= (candidate + 60):
                valid_start = candidate
                break
        
        if valid_start is None:
            if rowst_end >= rowst_start + 60:
                valid_start = rowst_start
            else:
                continue
        
        channel_count = 0
        for j in range(rowst.count()):
            if channel_count == 3:
                break
            slice_end = min(valid_start + 60, rowst_end)
            x = rowst[j].slice(starttime=valid_start, endtime=slice_end).data
            if len(x) < 6000:
                continue
            if len(x) > 6001:
                x = x[:6001]
            wave_array = np.append(wave_array, x)
            channel_count += 1
        
        if channel_count > 0:
            sta_info = {
                'station': '{}.{}'.format(getattr(row, 'Net_code'), getattr(row, 'Sta_code')),
                'dist_km': round(float(dist) * 111.2, 2),
                'channels': channel_count
            }
            stations_used.append(sta_info)
    
    if wave_array.shape[0] == 0:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return {'error': 'No valid waveform data within {}km'.format(max_dist_km)}
    
    wave_array = wave_array.reshape(-1, 18003)
    wave_array = wave_array[:, 0:18000]
    
    # Clean up temporary directory
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    return {
        'event_id': Event_id,
        'wave_array': wave_array,
        'n_samples': wave_array.shape[0],
        'stations_used': stations_used
    }


def check_seed_3class(seed_path, phase_path, apply_filter=True, max_dist_km=200):
    """
    Classify seismic event using 3-class model
    
    Parameters
    ----------
    seed_path : str
        Path to SEED file
    phase_path : str
        Path to phase file
    apply_filter : bool
        Whether to apply 1-20Hz bandpass filter
    max_dist_km : float, default 200
        Maximum epicentral distance in kilometers
    
    Returns
    -------
    dict
        {'event_id': str, 'n_samples': int, 'result': str, 'probs': [p1, p2, p3], 
         'stations_used': list, 'confidence': float, 'sample_results': list}
    """
    wave_data = read_waveform_from_seed(seed_path, phase_path, max_dist_km=max_dist_km)
    
    if 'error' in wave_data:
        return {
            'error': wave_data['error'],
            'result': 'Earthquake',
            'probs': [0, 0, 0],
            'stations_used': [],
            'confidence': 0.0,
            'sample_results': []
        }
    
    wave_array = wave_data['wave_array']
    
    predictions = []
    sample_results = []
    for i in range(wave_array.shape[0]):
        pred = predict_3class(wave_array[i], apply_filter=apply_filter)
        predictions.append(pred['probs'])
        sample_results.append(pred['result'])
    
    avg_probs = np.mean(predictions, axis=0)
    final_result = CLASS_NAMES_3[np.argmax(avg_probs)]
    
    confidence = sum(1 for r in sample_results if r == final_result) / len(sample_results) * 100
    
    return {
        'event_id': wave_data['event_id'],
        'n_samples': wave_data['n_samples'],
        'result': final_result,
        'probs': [round(p, 2) for p in avg_probs],
        'stations_used': wave_data['stations_used'],
        'confidence': round(confidence, 2),
        'sample_results': sample_results
    }


def check_seed(seed_path, phase_path, apply_filter=True, max_dist_km=200):
    """
    Classify seismic event using 3-class model
    
    Parameters
    ----------
    seed_path : str
        Path to SEED file
    phase_path : str
        Path to phase file
    apply_filter : bool
        Whether to apply 1-20Hz bandpass filter
    max_dist_km : float, default 200
        Maximum epicentral distance in kilometers
    
    Returns
    -------
    dict
        {'event_id': str, 'n_samples': int, 'result': str, 'probs': [p1, p2, p3], 
         'stations_used': list, 'confidence': float}
    """
    return check_seed_3class(seed_path, phase_path, apply_filter=apply_filter, max_dist_km=max_dist_km)


def check_seed_old(seed_path, phase_path, model_str='251111nw'):
    """
    Original check_seed function (kept for backward compatibility)
    """
    model_name = model_str
    model_file = os.path.join(package_dir, 'model', model_name, 'event_model.json')
    model_weight = os.path.join(package_dir, 'model', model_name, 'event_model.h5')
    model_pkl = os.path.join(package_dir, 'model', model_name, 'event_model.pkl')

    (Event_id, Event_type) = os.path.splitext(os.path.basename(seed_path))

    with open(model_file, 'r') as file:
        model_json = file.read()

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

    with tf.device('/CPU:0'):
        try:
            new_model = tf.keras.models.load_model(model_weight, compile=False)
        except Exception:
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

    wave_data = read_waveform_from_seed(seed_path, phase_path)
    
    if 'error' in wave_data:
        return ('Earthquake,0,0,0', '')

    wave_array = wave_data['wave_array']
    
    col = np.around(np.mean(wave_array, axis=1).reshape(-1, 1))
    ins = np.where(wave_array == 0)
    wave_array[ins] = np.take(col, ins[0])

    try:
        import joblib
        scaler = joblib.load(model_pkl)
        wave_array = scaler.transform(wave_array)
    except Exception:
        wave_array = (wave_array - np.mean(wave_array)) / (np.max(wave_array) - np.min(wave_array) + 1e-8)
    
    wave_array = wave_array.reshape(-1, 100, 180, 1)

    predictions = new_model.predict(wave_array)

    earthquake = round((np.sum(predictions[:, 0]) / predictions.shape[0]) * 100, 2)
    explode = round((np.sum(predictions[:, 1]) / predictions.shape[0]) * 100, 2)
    collapse = round((np.sum(predictions[:, 2]) / predictions.shape[0]) * 100, 2)
    
    def return_result(x):
        x = np.argmax(x)
        return 'Earthquake' if x == 0 else 'Non-earthquake'
    
    return '{} {},{},{}'.format(return_result((earthquake, explode, collapse)), earthquake, explode, collapse), ''


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Seismic waveform classification')
    parser.add_argument('--seed', required=True, help='Path to SEED file')
    parser.add_argument('--phase', required=True, help='Path to phase file')
    parser.add_argument('--filter', action='store_true', default=True, help='Apply 1-20Hz bandpass filter')
    parser.add_argument('--no-filter', action='store_true', help='Disable bandpass filter')
    parser.add_argument('--model', choices=['3class', '2class', 'both'], default='both', help='Model to use')
    
    args = parser.parse_args()
    apply_filter = not args.no_filter
    
    print("=" * 60)
    print("Seismic Waveform Classification (2606 Models)")
    print("=" * 60)
    print("SEED file: {}".format(args.seed))
    print("Phase file: {}".format(args.phase))
    print("Apply filter: {}".format(apply_filter))
    print("Model: {}".format(args.model))
    
    if args.model == '3class':
        result = check_seed_3class(args.seed, args.phase, apply_filter=apply_filter)
        print("\n" + "-" * 60)
        print("3-Class Classification Result")
        print("-" * 60)
        if 'error' in result:
            print("Error: {}".format(result['error']))
        else:
            print("Event ID: {}".format(result['event_id']))
            print("Samples: {}".format(result['n_samples']))
            print("Result: {}".format(result['result']))
            print("Probabilities: Earthquake={}%, Explosion={}%, Collapse={}%".format(
                result['probs'][0], result['probs'][1], result['probs'][2]))
    
    elif args.model == '2class':
        result = check_seed_2class(args.seed, args.phase, apply_filter=apply_filter)
        print("\n" + "-" * 60)
        print("2-Class Classification Result")
        print("-" * 60)
        if 'error' in result:
            print("Error: {}".format(result['error']))
        else:
            print("Event ID: {}".format(result['event_id']))
            print("Samples: {}".format(result['n_samples']))
            print("Result: {}".format(result['result']))
            print("Probabilities: Earthquake={}%, Non-earthquake={}%".format(
                result['probs'][0], result['probs'][1]))
    
    else:
        result = check_seed(args.seed, args.phase, apply_filter=apply_filter)
        print("\n" + "-" * 60)
        print("Classification Results")
        print("-" * 60)
        if 'error' in result:
            print("Error: {}".format(result.get('error', 'Unknown error')))
        else:
            print("Event ID: {}".format(result['event_id']))
            print("Samples: {}".format(result['n_samples']))
            print("\n3-Class: {} [{}, {}, {}]%".format(
                result['3class_result'],
                result['3class_probs'][0],
                result['3class_probs'][1],
                result['3class_probs'][2]))
            print("2-Class: {} [{}, {}]%".format(
                result['2class_result'],
                result['2class_probs'][0],
                result['2class_probs'][1]))