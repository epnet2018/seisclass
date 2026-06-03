# seisclass

Seismic event classification package for identifying natural and non-natural earthquakes.

## Overview

`seisclass` is a Python package designed to classify seismic events using machine learning models. It provides two classification models:

- **3-Class Model**: Classifies events as Earthquake, Explosion, or Collapse
- **2-Class Model**: Classifies events as Earthquake or Non-earthquake

### Key Features

- Uses models from `models/2606/` directory
- Automatic 1-20Hz bandpass filtering (consistent with training data)
- DC offset correction (demean)
- Per-channel normalization (Z-score)
- 3-channel input format (Z/N/E components)

For detailed information about the program and to cite it in your research publications, please refer to the following papers:

[1] Jia, L., Chen, H., & Xing, K. (2022). Rapid classification of local seismic events using machine learning. Journal of Seismology, 26(5), 897-912.

[2] Jia, L., Chen, S., Li, Y., & Zheng, P. (2025). A Semisupervised Seismic Events Classifier Based on Generative Adversarial Network. Seismological Research Letters, 96(3), 2039-2051.

Git-Repository: [https://github.com/epnet2018/]

## Waveform Requirements

**IMPORTANT: The following preprocessing requirements must be met for accurate classification, and we further recommend adopting the three-class classification model.**

### 1. Bandpass Filtering
- **Frequency range**: 1-20Hz
- **Filter type**: Butterworth bandpass filter
- **Corners**: 4
- **Zero-phase**: True
- Applied per channel (Z, N, E separately)

### 2. DC Offset Correction (Demean)
- Remove the mean value from each channel
- Applied before bandpass filtering

### 3. Waveform Window
- **Start time**: 1 second before P-wave arrival
- **Duration**: 60 seconds total (1 second before P + 59 seconds after P)
- **Channels**: 3 components (Z, N, E) concatenated
- **Sampling rate**: 100 Hz (6000 samples per channel, 18000 total)

### 4. Input Format
- Shape: `(100, 60, 3)` - 3-channel spectrogram-like format
- Each channel: 6000 samples reshaped to (100, 60)
- Z-score normalization applied per channel

### 5. Station Distance Selection
- **Recommended maximum epicentral distance**: 200 km
- Stations within 200 km provide more reliable classification results
- Near-field data (<100 km) may bias toward earthquake classification
- Far-field data (>200 km) may introduce noise and reduce accuracy
- Use `max_dist_km` parameter to filter stations by distance:

```python
# Use stations within 200 km
result = check_seed_3class('seed_file', 'phase_file', max_dist_km=200)
```

**Distance threshold comparison (test2.seed example)**:

| Distance | Stations | 3-Class Result | Explosion Prob |
|----------|----------|----------------|----------------|
| 100 km   | 4        | Earthquake     | 36.4%          |
| 150 km   | 5        | Earthquake     | 42.1%          |
| 200 km   | 7        | Explosion ✓    | 56.3%          |

The 200 km threshold provides optimal balance for explosion detection accuracy.

## Usage

### Basic Usage

```python
from seisclass import check_seed

# Classify seismic event from SEED file (both models)
result = check_seed('path/to/seed/file', 'path/to/phase/file', apply_filter=True)

print("3-Class Result: {}".format(result['3class_result']))
print("3-Class Probs: Earthquake={}%, Explosion={}%, Collapse={}%".format(
    result['3class_probs'][0], result['3class_probs'][1], result['3class_probs'][2]))

print("2-Class Result: {}".format(result['2class_result']))
print("2-Class Probs: Earthquake={}%, Non-earthquake={}%".format(
    result['2class_probs'][0], result['2class_probs'][1]))
```

### Independent 3-Class and 2-Class Functions

```python
from seisclass import check_seed_3class, check_seed_2class

# Use only 3-class model
result_3class = check_seed_3class('path/to/seed/file', 'path/to/phase/file', apply_filter=True)
print("3-Class: {} [{}, {}, {}]%".format(
    result_3class['result'],
    result_3class['probs'][0],
    result_3class['probs'][1],
    result_3class['probs'][2]))

# Use only 2-class model
result_2class = check_seed_2class('path/to/seed/file', 'path/to/phase/file', apply_filter=True)
print("2-Class: {} [{}, {}]%".format(
    result_2class['result'],
    result_2class['probs'][0],
    result_2class['probs'][1]))
```

### Output Format

```python
{
    'event_id': '20260423151835',
    'n_samples': 15,
    '3class_result': 'Earthquake',        # Earthquake / Explosion / Collapse
    '3class_probs': [42.52, 34.76, 22.72], # [Earthquake, Explosion, Collapse] in %
    '2class_result': 'Earthquake',        # Earthquake / Non-earthquake
    '2class_probs': [53.58, 46.42]         # [Earthquake, Non-earthquake] in %
}
```

### Disable Filtering

If your waveform data is already preprocessed (filtered and demeaned), you can disable automatic filtering:

```python
result = check_seed('path/to/seed/file', 'path/to/phase/file', apply_filter=False)
```

### Custom Inference

```python
from seisclass.check_wave import load_models, preprocess_waveform_channel

import numpy as np

# Load models
model_3class, model_2class = load_models()

# Prepare waveform (18000 samples: Z/N/E concatenated)
waveform = np.random.randn(18000)  # Replace with your data

# Preprocess (apply_filter=True for raw data, False for preprocessed data)
X = preprocess_waveform_channel(waveform, apply_filter=True)

# Predict
pred_3class = model_3class.predict(X, verbose=0)[0]
pred_2class = model_2class.predict(X, verbose=0)[0]

print("3-Class: Earthquake={}%, Explosion={}%, Collapse={}%".format(
    pred_3class[0]*100, pred_3class[1]*100, pred_3class[2]*100))
```

### Batch Inference

```python
from seisclass.check_wave import load_models, preprocess_waveform_channel
import numpy as np

model_3class, model_2class = load_models()

# Multiple waveforms
waveforms = np.random.randn(10, 18000)  # 10 samples

predictions = []
for i in range(waveforms.shape[0]):
    X = preprocess_waveform_channel(waveforms[i], apply_filter=True)
    pred = model_3class.predict(X, verbose=0)[0]
    predictions.append(pred)

# Average probabilities
avg_probs = np.mean(predictions, axis=0)
print("Average: Earthquake={}%, Explosion={}%, Collapse={}%".format(
    avg_probs[0]*100, avg_probs[1]*100, avg_probs[2]*100))
```

## Phase File Format

The phase file should contain P-wave arrival times in the following format:

```
Net_code  Sta_code  Loc_id  Chn_code  Phase_name  Phase_time           Phase_time_frac  Resi       Mag_val    Distance   Azi
XX        XXXXX     00      HHZ       P           2025-09-20 03:20:39  7600             -1.903390  2.391120   53.238400  339.952000
YY        YYYYY     00      HHZ       P           2025-09-20 03:20:46  4100             -2.548810  2.651560   104.083000  90.712900
```

Required columns:
- `Net_code`: Network code
- `Sta_code`: Station code
- `Phase_time`: P-wave arrival time (datetime format)
- `Phase_time_frac`: Milliseconds fraction

## Installation

### From PyPI

```bash
pip install seisclass
```

### From Source

```bash
pip install .
```
##This version requires Python 3.9 or above; use the previous release if you have an older Python version.
## Dependencies

- numpy
- obspy
- tensorflow (>= 2.0)
- pandas
- scipy

## Models

Models are located in `models/2606/` directory:

| Model | File | Classes |
|-------|------|---------|
| 3-Class | `model_3class.h5` | Earthquake, Explosion, Collapse |
| 2-Class | `model_2class.h5` | Earthquake, Non-earthquake |

## Testing

Run the sample examples:

```bash
python -m seisclass.samples
```

Or run tests:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Jia Luozhao - [18429320@qq.com](mailto:18429320@qq.com)

## Changelog

### 2026-06-01
- Added 3-class and 2-class models from models/2606
- Added 1-20Hz bandpass filtering (consistent with waveform generation)
- Added DC offset correction (demean)
- Changed to 3-channel input format (100, 60, 3)
- Added per-channel normalization
- Updated all comments and prompts to English
- Added comprehensive usage examples in samples.py