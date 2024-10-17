import pandas as pd
import numpy as np


def get_data(path):
    if path.endwith('.csv'):
        df = pd.read_csv(path)
    elif path.endwith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = None
        print(f'error: {path}, filetype is not supported')
        exit(0)

    # get attribute dictionary
    attr_dict = {}
    for features in df.iloc[:-1]:
        unique_values = df[features].unique()
        attr_dict[features] = unique_values.tolist()

    # get label array & map class id to class name(e.g. 0 -> 'no', 1 -> 'yes')
    label, class_codes = pd.factorize(df['Class'])
    id2name = {v: k for v, k in enumerate(class_codes)}

    # get data array
    data = np.array(df.iloc[:-1])

    return data, label, attr_dict, id2name
