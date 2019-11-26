![](https://img.shields.io/badge/status-Active-green)

## NEAT Trader on Gym Env

#### This is based on the knowledge gained from previous experiments.
#### And Running on the newly developed Gym Env.

## Trader Gym Env

##### Observation : Price (ohlc) + technical indicators + patterns + account_history
##### Observations scaled at runtime with a `look_back_window_size`
##### Reward: `Weighted_Unrealised_Profit` with some hacks to make the agent perform better.     
##### Output : Buy Hold Sell. 
