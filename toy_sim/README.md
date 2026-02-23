## First very simple simulation

# Description:
Swiss Post schedules several transports between sorting centers per day.
The last transport of the day is frequently not needed when volumes are lower than
expected. Cancellations today are made manually and often too late, partly because
operators cannot reliably anticipate whether the final trip will be required.
The goal is to predict whether the last planned route will be needed, based on sorting-center
data and historical patterns.

# Note:
Only the total (sum over all destinations) processed volume is measured at sorting centers and should be used to make the prediction - plus historical (average) distributions.

# Principles of the simulation:
- generate random (gaussian) volumes for each route (origin->destination centers)
- simulate cumulative volumes processed by each sorting center along the day
- book a predetermined number of trucks based on historical needs
- determine whether the last truck of the day per route is ultimately needed

# Simplifications (at the moment):
- Two types of letters are simulated with different volumes and processing times, but they are actually treated together (no priority difference)
- There is no specific truck schedule. The evaluation of how many trucks are needed/filled is done at the end of the day (which is equivalent to say that we are never short of letters when loading a truck, which seems pretty reasonable, but we can think to make it more general)

