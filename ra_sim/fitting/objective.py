import numpy as np

def compute_cost(integrated_data, reference_profiles):
    # Example simplified code to compute a mean squared error
    squared_errors = []
    for region_data in integrated_data:
        region = region_data['Region']
        sim_theta = region_data['Radial (2θ)']
        sim_intensity = region_data['Intensity']
        
        ref_theta = reference_profiles[region]['Radial (2θ)']
        ref_intensity = reference_profiles[region]['Intensity']

        if len(sim_theta) == len(ref_theta):
            diff = sim_intensity - ref_intensity
            squared_errors.extend(diff**2)
    
    if len(squared_errors) == 0:
        return 1e9
    mse = np.mean(squared_errors)
    return mse
