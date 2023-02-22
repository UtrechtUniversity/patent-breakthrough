import pandas as pd
from collections import defaultdict
from hyperopt import space_eval


def dataframe_from_trials(trials, model_class):
    trial_results = defaultdict(list)
    for trial in trials:
        if "loss" not in trial["result"]:
            continue
        new_trial = {key: val[0] for key, val in trial["misc"]["vals"].items()}
        conv_trial = space_eval(model_class.hyper_space(), new_trial)
        for key, val in conv_trial.items():
            trial_results[key].append(val)
        trial_results["loss"].append(trial["result"]["loss"])
    return pd.DataFrame(trial_results)
