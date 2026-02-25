# Truck cancellation problem

# Description:

Swiss Post schedules several transports between sorting centers per day. The last transport of the day is frequently not needed when volumes are lower than expected. Cancellations today are made manually and often too late, partly because operators cannot reliably anticipate whether the final trip will be required. The goal is to predict whether the last planned route will be needed, based on sorting-center data and historical patterns.

## Quick data plots

You can generate a PDF with one graph per sampled day from generated data:

```bash
python3 plot_generated_data.py vol
```

This writes `rl/outputs/vol_3days.pdf` by default.

Useful options:

```bash
python3 plot_generated_data.py hist_avg_vol_tot --center A --day 0
python3 plot_generated_data.py vol --input toy_sim/example_data/df_raw.pkl --n-days 3 --output-dir rl/outputs
python3 plot_generated_data.py --hist-center-total --center A
python3 plot_generated_data.py --hist-all-centers
python3 plot_generated_data.py --list-columns
```
