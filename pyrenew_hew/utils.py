import numpy as np


def convert_to_logmean_log_sd(mean, sd):
    logmean = np.log(
        np.power(mean, 2) / np.sqrt(np.power(sd, 2) + np.power(mean, 2))
    )
    logsd = np.sqrt(np.log(1 + (np.power(sd, 2) / np.power(mean, 2))))
    return logmean, logsd
