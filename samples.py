# -*- coding: utf-8 -*-
"""
Seisclass Library Usage Example (3-class model only)

Inference on samples/test2.mseed file

@author: jialuozhao
@mail: 18429320@qq.com
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

import numpy as np
import obspy

package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from check_wave import (
    load_3class_model,
    preprocess_waveform,
    demean,
    bandpass_filter,
    CLASS_NAMES_3
)

SAMPLE_RATE = 100
CLIP_LENGTH = 6000
TOTAL_WAVEFORM_LENGTH = 18000


def classify_mseed(mseed_file, apply_filter=True, verbose=True):
    """
    Classify MiniSEED file using 3-class model
    
    Parameters
    ----------
    mseed_file : str
        MiniSEED file path
    apply_filter : bool
        Apply bandpass filter and demean
    verbose : bool
        Print detailed output
    
    Returns
    -------
    dict
        Classification result with probabilities
    """
    if verbose:
        print("=" * 70)
        print("3-Class Classification - MiniSEED")
        print("=" * 70)
        print("MiniSEED file: {}".format(mseed_file))
        print("Apply filter: {}".format(apply_filter))
        print("-" * 70)
    
    # Read MiniSEED file
    st = obspy.read(mseed_file)
    
    # Group by station
    stations = {}
    for tr in st:
        station = tr.stats.station
        channel = tr.stats.channel[-1]  # Z, N, E
        if station not in stations:
            stations[station] = {}
        stations[station][channel] = tr
    
    if verbose:
        print("Found {} stations".format(len(stations)))
    
    # Load model
    model = load_3class_model()
    if model is None:
        print("Error: Failed to load model")
        return {'error': 'Failed to load model'}
    
    predictions = []
    sample_results = []
    stations_used = []
    
    # Process each station
    for station, traces in stations.items():
        # Check for complete 3-channel data
        if not ('Z' in traces and 'N' in traces and 'E' in traces):
            if verbose:
                print("  {}: Missing channels (Z={}, N={}, E={})".format(
                    station, 'Z' in traces, 'N' in traces, 'E' in traces))
            continue
        
        # Extract waveform (18000 points: Z+N+E, each 6000)
        waveform = np.zeros(TOTAL_WAVEFORM_LENGTH, dtype=np.float32)
        for i, ch in enumerate(['Z', 'N', 'E']):
            data = traces[ch].data.astype(np.float32)
            
            # Apply filter
            if apply_filter:
                try:
                    data = demean(data)
                    data = bandpass_filter(data, freqmin=1.0, freqmax=20.0, df=SAMPLE_RATE)
                except Exception:
                    pass
            
            # Clip to 6000 points
            n = min(len(data), CLIP_LENGTH)
            waveform[i * CLIP_LENGTH : i * CLIP_LENGTH + n] = data[:n]
        
        # Preprocess and predict
        X = preprocess_waveform(waveform, apply_filter=False)
        pred = model.predict(X, verbose=0)[0]
        pred_class = CLASS_NAMES_3[np.argmax(pred)]
        pred_probs = [round(p * 100, 2) for p in pred]
        
        predictions.append(pred_probs)
        sample_results.append(pred_class)
        stations_used.append(station)
        
        if verbose:
            print("  {}: {} (EQ={}%, EX={}%, CO={}%)".format(
                station, pred_class, pred_probs[0], pred_probs[1], pred_probs[2]))
    
    if len(predictions) == 0:
        print("Error: No valid waveform data")
        return {'error': 'No valid waveform data'}
    
    # Calculate average probabilities
    avg_probs = np.mean(predictions, axis=0)
    final_result = CLASS_NAMES_3[np.argmax(avg_probs)]
    confidence = sum(1 for r in sample_results if r == final_result) / len(sample_results) * 100
    
    if verbose:
        print("\n" + "-" * 70)
        print("Total samples: {}".format(len(predictions)))
        print("Stations used: {}".format(stations_used))
        print("\nClassification Result: {}".format(final_result))
        print("Confidence: {}%".format(round(confidence, 2)))
        print("\nAverage Probabilities:")
        print("  Earthquake:  {}%".format(round(avg_probs[0], 2)))
        print("  Explosion:   {}%".format(round(avg_probs[1], 2)))
        print("  Collapse:    {}%".format(round(avg_probs[2], 2)))
        
        print("\nSample Voting Details:")
        for i, result in enumerate(sample_results):
            vote_mark = "(+)" if result == final_result else ""
            print("  Sample {} ({}): {} {}".format(i+1, stations_used[i], result, vote_mark))
        
        print("=" * 70)
    
    return {
        'n_samples': len(predictions),
        'result': final_result,
        'probs': [round(p, 2) for p in avg_probs],
        'confidence': round(confidence, 2),
        'sample_results': sample_results,
        'stations_used': stations_used
    }


def get_earthquake_probability(mseed_file, apply_filter=True):
    """
    Output earthquake probability only
    
    Parameters
    ----------
    mseed_file : str
        MiniSEED file path
    apply_filter : bool
        Apply bandpass filter and demean
    
    Returns
    -------
    dict
        {'earthquake_prob': float, 'confidence': float, 'is_earthquake': bool}
    """
    result = classify_mseed(mseed_file, apply_filter=apply_filter, verbose=False)
    
    if 'error' in result:
        print("Error: {}".format(result['error']))
        return {'earthquake_prob': 0.0, 'confidence': 0.0, 'is_earthquake': False}
    
    earthquake_prob = result['probs'][0]
    is_earthquake = result['result'] == 'Earthquake'
    
    # Confidence = certainty of the judgment (max of prob and 100-prob)
    # High prob -> high confidence (definitely earthquake)
    # Low prob -> high confidence (definitely NOT earthquake)
    # ~50% prob -> low confidence (uncertain)
    confidence = max(earthquake_prob, 100 - earthquake_prob)
    
    print("Earthquake Probability: {}%".format(round(earthquake_prob, 2)))
    print("Confidence: {}%".format(round(confidence, 2)))
    print("Is Earthquake: {}".format(is_earthquake))
    
    return {
        'earthquake_prob': round(earthquake_prob, 2),
        'confidence': round(confidence, 2),
        'is_earthquake': is_earthquake
    }


def get_non_earthquake_probability(mseed_file, apply_filter=True):
    """
    Output non-earthquake probability only (Explosion + Collapse)
    
    Parameters
    ----------
    mseed_file : str
        MiniSEED file path
    apply_filter : bool
        Apply bandpass filter and demean
    
    Returns
    -------
    dict
        {'non_earthquake_prob': float, 'confidence': float, 'is_non_earthquake': bool}
    """
    result = classify_mseed(mseed_file, apply_filter=apply_filter, verbose=False)
    
    if 'error' in result:
        print("Error: {}".format(result['error']))
        return {'non_earthquake_prob': 0.0, 'confidence': 0.0, 'is_non_earthquake': False}
    
    explosion_prob = result['probs'][1]
    collapse_prob = result['probs'][2]
    non_earthquake_prob = explosion_prob + collapse_prob
    
    is_non_earthquake = result['result'] in ['Explosion', 'Collapse']
    
    # Confidence = certainty of the judgment (max of prob and 100-prob)
    # High prob -> high confidence (definitely non-earthquake)
    # Low prob -> high confidence (definitely earthquake)
    # ~50% prob -> low confidence (uncertain)
    confidence = max(non_earthquake_prob, 100 - non_earthquake_prob)
    
    print("Non-Earthquake Probability: {}%".format(round(non_earthquake_prob, 2)))
    print("  Explosion: {}%".format(round(explosion_prob, 2)))
    print("  Collapse:  {}%".format(round(collapse_prob, 2)))
    print("Confidence: {}%".format(round(confidence, 2)))
    print("Is Non-Earthquake: {}".format(is_non_earthquake))
    
    return {
        'non_earthquake_prob': round(non_earthquake_prob, 2),
        'explosion_prob': round(explosion_prob, 2),
        'collapse_prob': round(collapse_prob, 2),
        'confidence': round(confidence, 2),
        'is_non_earthquake': is_non_earthquake
    }


def get_non_earthquake_simple(mseed_file, apply_filter=True):
    """
    Simple output: non-earthquake probability and confidence only
    
    Parameters
    ----------
    mseed_file : str
        MiniSEED file path
    apply_filter : bool
        Apply bandpass filter and demean
    
    Returns
    -------
    dict
        {'non_earthquake_prob': float, 'confidence': float}
    """
    result = classify_mseed(mseed_file, apply_filter=apply_filter, verbose=False)
    
    if 'error' in result:
        print("Error: {}".format(result['error']))
        return {'non_earthquake_prob': 0.0, 'confidence': 0.0}
    
    non_earthquake_prob = result['probs'][1] + result['probs'][2]
    confidence = max(non_earthquake_prob, 100 - non_earthquake_prob)
    
    print("Non-Earthquake Probability: {}%  Confidence: {}%".format(
        round(non_earthquake_prob, 2), round(confidence, 2)))
    
    return {
        'non_earthquake_prob': round(non_earthquake_prob, 2),
        'confidence': round(confidence, 2)
    }


def main():
    """Main function"""
    samples_dir = os.path.join(package_dir, 'samples')
    mseed_file = os.path.join(samples_dir, 'test2.mseed')
    
    if not os.path.exists(mseed_file):
        print("Error: MiniSEED file not found - {}".format(mseed_file))
        return
    
    # Full classification
    classify_mseed(mseed_file, apply_filter=True, verbose=True)
    
    # Earthquake probability only
    print("\n--- Earthquake Probability ---")
    get_earthquake_probability(mseed_file, apply_filter=True)
    
    # Non-earthquake probability only
    print("\n--- Non-Earthquake Probability ---")
    get_non_earthquake_probability(mseed_file, apply_filter=True)
    
    # Simple non-earthquake output
    print("\n--- Simple Non-Earthquake Output ---")
    get_non_earthquake_simple(mseed_file, apply_filter=True)
    
    print("\nInference completed.")


if __name__ == '__main__':
    main()