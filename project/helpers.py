import datetime
import os


def create_results_dir_and_results_predict_dir(root_dir):
    results_dir = root_dir + 'results/' + str(datetime.datetime.now()).replace(' ', '_')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_dir_predict = results_dir + '/' + 'predict'
    if not os.path.exists(results_dir_predict):
        os.makedirs(results_dir_predict)

    return results_dir + '/'
