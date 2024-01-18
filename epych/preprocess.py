#!/usr/bin/python3

import scipy.stats as stats

def zscore_trials(data, trial_dim=-1):
    return stats.zscore(data, axis=trial_dim,  nan_policy='omit')
