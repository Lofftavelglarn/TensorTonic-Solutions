import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    if y_true == y_pred:
        return 1.0
        
    y_mean = np.mean(y_true)
    ss_tot = np.power(np.subtract(y_true, y_mean), 2).sum()
    if ss_tot == 0:
        return 0.0
    
    ss_res = np.power(np.subtract(y_true, y_pred), 2).sum()
    
    r2_score = 1 - ss_res / ss_tot
    return r2_score