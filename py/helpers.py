import numpy as np

def dither(x, type='symmetric', value=None):
    if len(x) == 0:
        return x
    if not np.issubdtype(np.array(x).dtype, np.number):
        raise ValueError("'x' must be numeric")
    if value is None:
        unique_x = np.unique(x)
        if len(unique_x) > 1:
            value = np.min(np.diff(np.sort(unique_x)))
        else:
            value = 0.0
    if type == 'symmetric':
        v = x + np.random.uniform(-value/2, value/2, len(x))
    elif type == 'right':
        v = x + np.random.uniform(0, value, len(x))
    else:
        raise ValueError("invalid type")
    return v