"""Some utils for plotting metrics"""
# pylint: disable = C0111


import glob
import numpy as np
import utils
import matplotlib.pyplot as plt
# import seaborn as sns


def int_or_float(val):
    try:
        return int(val)
    except ValueError:
        return float(val)


def get_figsize(is_save):
    if is_save:
        figsize = [6, 4]
    else:
        figsize = None
    return figsize

def get_data(expt_dir):
    data = {}
    measurement_losses = utils.load_if_pickled(expt_dir + '/measurement_losses.pkl')
    l2_losses = utils.load_if_pickled(expt_dir + '/l2_losses.pkl')
    data = {'measurement': measurement_losses.values(),
            'l2': l2_losses.values()}
    return data


def get_metrics(expt_dir):
    data = get_data(expt_dir)

    metrics = {}

    m_loss_mean = np.mean(data['measurement'])
    m_loss_std = np.std(data['measurement']) / np.sqrt(len(data['measurement']))
    metrics['measurement'] = {'mean': m_loss_mean, 'std': m_loss_std}

    l2_loss_mean = np.mean(data['l2'])
    l2_loss_std = np.std(data['l2']) / np.sqrt(len(data['l2']))
    metrics['l2'] = {'mean':l2_loss_mean, 'std':l2_loss_std}

    return metrics


def get_expt_metrics(expt_dirs):
    expt_metrics = {}
    for expt_dir in expt_dirs:
        metrics = get_metrics(expt_dir)
        expt_metrics[expt_dir] = metrics
    return expt_metrics


def get_nested_value(dic, field):
    answer = dic
    for key in field:
        answer = answer[key]
    return answer


def find_best(pattern, criterion, retrieve_list):
    dirs = glob.glob(pattern)
    metrics = get_expt_metrics(dirs)
    best_merit = 1e10
    answer = [None]*len(retrieve_list)
    for _, val in metrics.iteritems():
        merit = get_nested_value(val, criterion)
        if merit < best_merit:
            best_merit = merit
            for i, field in enumerate(retrieve_list):
                answer[i] = get_nested_value(val, field)
    return answer


def plot(base, regex, criterion, retrieve_list):
    keys = map(int_or_float, [a.split('/')[-1] for a in glob.glob(base + '*')])
    means, std_devs = {}, {}
    for i, key in enumerate(keys):
        pattern = base + str(key) + regex
        answer = find_best(pattern, criterion, retrieve_list)
        if answer[0] is not None:
            means[key], std_devs[key] = answer
    plot_keys = sorted(means.keys())
    means = np.asarray([means[key] for key in plot_keys])
    std_devs = np.asarray([std_devs[key] for key in plot_keys])
    (_, caps, _) = plt.errorbar(plot_keys, means, yerr=1.96*std_devs,
                                marker='o', markersize=5, capsize=5)
    for cap in caps:
        cap.set_markeredgewidth(1)
