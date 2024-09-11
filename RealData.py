import os
import sys
import pandas as pd
import numpy as np
import ast
import pickle

def explore_data_path(data_path, which_data):

    df = pd.DataFrame(columns=['shape', 'cm_x', 'cm_y', 'values'])
    for root, dirs, files in os.walk(data_path):
        if files == []: continue
        if dirs != []: continue

        try:
            shape = root[-4]
            cm_estimation = pd.read_csv(root + '/center_of_mass_estimate.csv')['CAMERA'].tolist()
            records = pd.read_csv(root + f'/{which_data}.csv').iloc[:, 1:].values.tolist()
        except:
            print('Something wrong in ' + root)
            continue
      

        sub_data = {
        'shape': [shape] * len(records),
        'cm_x': [cm_estimation[0]] * len(records),
        'cm_y': [cm_estimation[1]] * len(records),
        'values': list(records)
        }

        sub_df = pd.DataFrame(sub_data)
        df = pd.concat([df, sub_df], axis=0)

    def _convert_to_list_or_array(x):
        try:
            return np.array(ast.literal_eval(x)) if isinstance(x, str) else x
        except ValueError:
            return x  # In case there's an issue with the conversion

    df.iloc[:, -1] = df.iloc[:, -1].apply(_convert_to_list_or_array)
    data = df.values

    with open('RealData.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    data_path = 'Dataset'
    which_data = 'calibrated_sensor_data_all_recordings'
    explore_data_path(data_path, which_data)
    