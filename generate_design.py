from typing import TypeVar

import pandas as pd
import numpy as np
from scipy.stats import qmc

# maybe continuous, maybe not, categorical even

# 2-tuple (int or float)
#   randomizes between bounds (rounds if 2 int, treats int as float otherwise)

# any other tuple or any list
#   picks uniformly from the list

def generate_lhs_to_csv(param_dict: dict[str, tuple | list], num_samples=10, output_file="lhs_design.csv"):
    sampler = qmc.LatinHypercube(d=len(param_dict), rng=np.random.default_rng(42))
    sample_points = sampler.random(n=num_samples)
    df = pd.DataFrame(columns=list(param_dict.keys()))

    for i, (key, values) in enumerate(param_dict.items()):
        raw_samples = sample_points[:, i]
        
        is_bound = isinstance(values, tuple) and len(values) == 2 and all(isinstance(v, (int, float)) for v in values)
        if is_bound:
            lower, upper = values
            scaled_samples = raw_samples * (upper - lower) + lower

            if isinstance(lower, int) and isinstance(upper, int):
                df[key] = np.round(scaled_samples).astype(int)
            else:
                df[key] = scaled_samples
        else:
            indices = np.clip(
                np.floor(raw_samples * len(values)).astype(int), # multiplicar float estraga [0, 1), viva o float
                0,
                len(values) - 1
            )
            
            df[key] = [values[idx] for idx in indices]
            
    df.to_csv(output_file, index=False)
    print(f"Generated {num_samples} samples to '{output_file}'")
    print(df)
    return df

if __name__ == "__main__":
    pars = {
        "NUM_GENERATIONS":[10, 20],
        "POP_SIZE":[16, 32],
        "NUM_PARENTS":[3, 5],
        "VERSIONS":["-p", "-s"], # parallel, sequential
        "THREADS":[4, 16, -1],
        "INPUT_SIZES":["small", "full"],
    }
    
    generate_lhs_to_csv(
        param_dict=pars, 
        num_samples=25, # ~ 1/4 
        output_file="doe.csv"
    )
    
    generate_lhs_to_csv(
        param_dict=pars, 
        num_samples=6, # ~ 1/17
        output_file="doe_intel.csv"
    )
    