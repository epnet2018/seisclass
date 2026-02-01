# seisclass

Seismic event classification package for identifying natural and non-natural earthquakes.

## Overview

`seisclass` is a Python package designed to classify seismic events using machine learning models. It provides one main function:
- `check_seed`: Analyzes a seismic data file (SEED format) with corresponding phase file, returning a comma-separated result string: event_type,earthquake_prob,explode_prob,collapse_prob

For detailed information about the program and to cite it in your research publications, please refer to the following papers:

[1] Jia, L., Chen, H., & Xing, K. (2022). Rapid classification of local seismic events using machine learning. Journal of Seismology, 26(5), 897-912.

[2] Jia, L., Chen, S., Li, Y., & Zheng, P. (2025). A Semisupervised Seismic Events Classifier Based on Generative Adversarial Network. Seismological Research Letters, 96(3), 2039-2051.

Git-Repository: [https://github.com/epnet2018/]

## Usage
### Basic Usage
```python
from seisclass import check_seed
# Analyze a SEED file with corresponding phase file
result = check_seed('path/to/seed/file', 'path/to/phase/file')
print(result)
```
### phase file format
The phase file should be in the following format:
```
station,channel,time,phase
```
where `station` is the station code, `channel` is the channel code, `time` is the arrival time of the phase, and `phase` is the phase type (e.g., P, S).

Net_code	Sta_code	Loc_id	Chn_code	Phase_name	Phase_time	Phase_time_frac	Resi	Mag_val	Distance	Azi
XX	XXXXX	00	HHZ	P	2025-09-20 03:20:39	7600	-1.903390	2.391120	53.238400	339.952000
YY	YYYYY	00	HHZ	P	2025-09-20 03:20:46	4100	-2.548810	2.651560	104.083000	90.712900

## Installation

You can install the package from PyPI:

```bash
pip install seisclass
```

Or install from source:

```bash
pip install .
```
## Dependencies
- numpy
- obspy
- tensorflow
- keras
- joblib
- scikit-learn
- pandas



### Advanced Usage

```python
# Specify a different model
result = check_seed('path/to/seed/file', 'path/to/phase/file', model_str='251111nw')
```

## Testing

To run the tests:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Jia Luozhao - [18429320@qq.com](mailto:18429320@qq.com)

## Project Structure

```
seisclass/
├── seisclass/
│   ├── __init__.py          # Package initialization and API exposure
│   ├── check_wave.py        # Core functionality
│   ├── model/
│   │   └── 251111nw/         # Model files
│   ├── packages/
│   │   └── read_phase.py     # Phase file parser
│   └── resource/
│       ├── test.phase        # Test phase data
│       └── test.seed         # Test seed data
├── tests/
│   ├── __init__.py
│   └── test_checkwave.py     # Tests
├── setup.py                 # Setup configuration
├── pyproject.toml           # Modern PyPI configuration
├── README.md                # This file
├── LICENSE                  # License file
└── requirements.txt         # Dependencies
```