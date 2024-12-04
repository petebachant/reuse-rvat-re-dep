"""Run a mock simulation."""

import os

import pandas as pd

df_sim = pd.DataFrame(
    {
        "tsr": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "cp": [0.03, 0.09, 0.21, 0.28, 0.18, 0.06],
    }
)
os.makedirs("results", exist_ok=True)

df_sim.to_csv("results/simulation.csv", index=False)
