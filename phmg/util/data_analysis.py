import numpy as np


def geometric_rate(resids, start=5):
    try:
        tmp = (resids[1:] / resids[:-1])[start:]
        rho = np.prod(tmp) ** (1.0 / tmp.size)
    except:
        rho = -1
    return rho


def write_to_file(filename, content):
    with open(filename, "a") as file:
        file.write(content)
        print(content)
