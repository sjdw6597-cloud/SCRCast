# SCRCast

Standalone implementation for multivariate time-series forecasting with CSCA,
ACR, and CRH.

## Installation

Use Python 3.11 and install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch build.

## Data

Download the benchmark CSVs using the
[Time-Series-Library data instructions](https://github.com/thuml/Time-Series-Library#usage)
or [ETDataset](https://github.com/zhouhaoyi/ETDataset), and arrange them as follows:

```text
dataset/
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
  weather/weather.csv
  exchange_rate.csv
  national_illness.csv
```

Each CSV must contain a `date` column followed by numeric channels. Custom
datasets place target `OT` last. All channels are forecast; normalization is
fitted on the training split.

## Run

Run a dataset script to use its configured parameters:

```bash
bash scripts/ETTh1.sh
```

Scripts are available for `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `Exchange`,
and `ILI`. Each runs four horizons serially with seed 2021. ETT, Weather, and
Exchange use 96/192/336/720; ILI uses 24/36/48/60. All scripts use CRH `active`
normalization.

Select horizons, data location, device, or inspect parameters:

```bash
HORIZONS="96" DATA_ROOT=/path/to/dataset bash scripts/ETTh1.sh
DEVICE=cpu HORIZONS="24" bash scripts/ILI.sh
DRY_RUN=1 bash scripts/ETTh1.sh
```

`PYTHON` selects the Python executable. Use `python run.py --help` for CLI options;
CLI defaults do not automatically load a dataset script's parameters.

Each run saves `config.json` and `metrics.json` under `results/`. Metrics are
MSE/MAE in training-standardized units. The best checkpoint is selected by
validation MSE and removed after testing. Set `SAVE_CHECKPOINT=1` to retain it.

## License

SCRCast is licensed under the [MIT License](LICENSE),
Copyright (c) 2026 Zhang Jin.
Code derived from [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
retains its upstream [MIT License](licenses/MIT-TSLIB.txt),
Copyright (c) 2021 THUML @ Tsinghua University.
