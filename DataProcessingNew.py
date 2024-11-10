import os
import sys
import pandas as pd
import numpy as np
import ast
import pickle

def explore_data_path(data_path, chosen_directory, which_data, name):
    output_path = f'{data_path}/RealData_{name}.pkl'
    df = pd.DataFrame(columns=['shape', 'cm_x', 'cm_y', 'values'])
    data_path = os.path.join(data_path, chosen_directory)
    
    # Populate the dataframe with the data from the csv files
    for root, dirs, files in os.walk(data_path):

        if files == []: continue
        if dirs != []: continue

        #print(f'{root}')
        #for dir in dirs:
        #    print(f'    {dir}')
        #for file in files:
        #    print(f'        {file}')

        try:
            shape = root[-4]
            cm_estimation = pd.read_csv(root + '/center_of_mass_estimate.csv')['CAMERA'].tolist()
            records = pd.read_csv(root + f'/{which_data}.csv').iloc[:, 1:].values.tolist()
        
            print(f'{shape} {cm_estimation} {len(records)}')

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

        #print(f'{sub_df}')
        #print(f'{df.shape}')
    
    
    def _convert_to_list_or_array(x):
        try:
            return np.array(ast.literal_eval(x)) if isinstance(x, str) else x
        except ValueError:
            return x  # In case there's an issue with the conversion

    df.iloc[:, -1] = df.iloc[:, -1].apply(_convert_to_list_or_array)
    data = df.values

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    data_path = 'Dataset'
    chosen_directory = 'Threshold0g'
    which_datas = ['calibrated_sensor_data_all_recordings','uncalibrated_sensor_data']
    names = ['Calibrated', 'Uncalibrated']
    for which_data, name in zip(which_datas, names):
        explore_data_path(data_path, chosen_directory, which_data, name)
    
    #load the data
    data = pickle.load(open(f'{data_path}/RealData_Calibrated.pkl', 'rb'))
    print(data)
    print(data.shape)
