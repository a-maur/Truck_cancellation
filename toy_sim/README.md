# First very simple simulation

## Description:
Swiss Post schedules several transports between sorting centers per day.
The last transport of the day is frequently not needed when volumes are lower than
expected. Cancellations today are made manually and often too late, partly because
operators cannot reliably anticipate whether the final trip will be required.
The goal is to predict whether the last planned route will be needed, based on sorting-center
data and historical patterns.

## Principles of the simulation:
- generate random (gaussian) volumes for each route (origin->destination centers)
- simulate cumulative volumes processed by each sorting center along the day
- book a predetermined number of trucks based on historical needs
- determine whether the last truck of the day per route is ultimately needed

## Model:
- Only the total (sum over all destinations) processed volume is measured at sorting centers and should be used to make the prediction - plus historical (average) distributions.
- Sorting centers are independent (no info from center A is usefull to know what will happen in center B).
- Route predictions are evaluated independently (i.e. A->B is evaluated independently on A->C) based on the following inputs:
    - global day features: {day of the week}, {high/low season} (might be redundant given the line below)
    - global sorting center features: average (mean/std) volume of that given sorting center in that given day/season
    - specific sorting center features: processed volume of the day (summed over all destinations)
    - destination specific features: average (mean/std) volume of that specific route in that given day/season, number of booked trucks and average cancellation rate (as before, the last two might be redundant since average destination volumes carry the same/more information)

## (Current) Simplifications:
- Single destination routes. If we want to include multiple destinations routes with shared destination (e.g. A -> B -> C and A -> D -> C) this would require significant changes (if that's the case, we will probably need to build somthing completely different)
- Two types of letters are simulated with different volumes and processing times, but they are actually treated together (no priority difference)
- There is no specific truck schedule. The evaluation of how many trucks are needed/filled is done at the end of the day (which is equivalent to say that we are never short of letters when loading a truck, which seems pretty reasonable, but we can think to make it more general)

---

# Simulation description

- (n=10) sorting centers named ['A', 'B', ... 'J ']
- (n-1(=9)) destinations, every center is connected to all the others
- (k=2) letter types ['type_A', 'type_B']
- (h=5) time steps along the day ['h0, ... 'h4']
- 100 letters per truck

## Features definitions

- {center} : name of the sorting center ['A'...'J']
- {day} : day of the week (integer, 0 to 4 corresponding to Monday to Friday)
- {season} : 0/1 integer (low/high season)
- {hist_avg_vol_tot} : mean of historical volumes sorted in {center} on {day} {season}
- {hist_std_vol_tot} : std of historical volumes sorted in {center} on {day} {season}
- {dest}_{type} :  total daily incoming volume per destination parcel type
  - These are the core underlying features, what I actually generate, but cannot be used for prediction, since we don’t have this info ‘live’ in the sorting center
- vol_h(0,1,2,3,4) : total volume of letter processed at the sorting center along the day (in ’n_h’ steps)
  - This is what we measure at sorting centers and should be used to make our prediction.
- {dest}_n_exp_trucks : number of trucks booked for the day to a certain destination (based on “historical” averages, where historical here means of the generated sample itself)
- {dest}_last_truck_needed: label (0/1), this is what we need to predict, if the per-destination truck is ultimately needed
- {dest}_n_exp_trucks: number of booked trucks for the given route ({center}->{dest}) on {day} {season}
- {dest}_frac_last_truck_needed: average cancellation rate of the last truck for the given route ({center}->{dest}) on {day} {season}
- {dest}_hist_avg_vol: mean of historical route ({center}->{dest}) volumes on {day} {season}
- {dest}_hist_std_vol: std of historical route ({center}->{dest}) volumes on {day} {season}
- {dest}_overflow: overflow from previous day, i.e. number of letters not send the previous day due to volume exceeding the booked trucks capacity
- {dest}_last_truck_needed: label (0/1), this is what we need to predict, whether the per-destination truck is ultimately needed

## Code output

- `main.py` runs the simulation based on the configurables in `config.py` and produces two files that can be saved as pandas dataframe: `df_raw.pkl` and `df_per_dest_train(test).pkl` (small example files con be found in `example/data/`).

- `df_raw.pkl` corresponds to the generated day at the sorting centers, and contains all destinations and relevant features
  
- `df_per_dest_train.pkl` the previous dataset is preprocessed to extract the following hourly statistical features:
  - min(V(t))
  - max(V(t))
  - mean(V(t))
  - std(V(t))
  - delta(V(t)) = V(t)-V(t-1)
  where V(t) is the cumulative processed volume up to time t.
`df_per_dest_train(test).pkl` is structured such as that each entry corresponds to a single route (this is the input for the supervised classification model approach).

