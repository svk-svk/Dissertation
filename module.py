import numpy as np
import pandas as pd
import ipaddress
import itertools

def ipv4_address_to_int(ip_str):
    try:
        ip= ipaddress.IPv4Address(ip_str)
        ip_int = int(ipaddress.IPv4Address(ip))
        return ip_int
    except ValueError:
        return None


def bool_to_int(value):
    try:
        if (type(value) is bool):
            if value == True:
                return 1  # true
            else:
                return 2  # false
        else:
            if value == 1:
                return 1
            elif value == 2:
                return 2
            return 0  # none

    except ValueError:

        return None

def count_appearances(text_list):
    appearances = {}
    for text in text_list:
        if text in appearances:
            appearances[text] += 1
        else:
            appearances[text] = 1
    sorted_appearances = dict(sorted(appearances.items(), key=lambda item: item[1], reverse=True))

    return sorted_appearances


def list_to_nan_first_position_list(list):
    new_list=[]
    if  np.nan in list:
        new_list=[np.nan]
        for x in list:
            if x is np.nan:
                continue
            else:
                new_list.append(x)

        return new_list
    else:
        return list


def to_int_token_from_list(value,list):
    try:
        return list.index(value)

    except ValueError:

        return None


def split_row_client(row,sep="/"):
    if (np.nan is row):
        return pd.Series({'first_element': 0, 'second_element': 0})

    else:
        parts = row.lower().split(sep)
        if len(parts) == 2:
            return pd.Series({'first_element': parts[0], 'second_element': parts[1]})
        elif len(parts)==1:
            return pd.Series({'first_element': parts[0], 'second_element':0 })

def print_column_info(df, column_name):
    distinct_values = df[column_name].unique()
    print("Distinct Values:")
    print(distinct_values)


    unique_counts = df[column_name].value_counts()
    print("\nUnique Value Counts:")
    print(unique_counts)

    value_types = df[column_name].apply(type).value_counts()
    print("\nValue Types and Counts:")
    print(value_types)

    filtered_values = df[df[column_name].apply(lambda x: isinstance(x, float))]
    print("\nFiltered Rows (Where Values are of Type float):")
    print(filtered_values)


def process_column(df, column_name):

    appearance_dict = count_appearances(df[column_name])

    aux_list = list(appearance_dict.keys())
    aux_list = list_to_nan_first_position_list(aux_list)

    df[column_name] = df[column_name].apply(lambda x: to_int_token_from_list(x, aux_list))

    print_column_info(df, column_name)


def generate_combinations_features(items):
    all_combinations = []

    for r in range(7, len(items) + 1):
        combinations = itertools.combinations(items, r)
        all_combinations.extend(combinations)

    all_combinations = [list(comb) for comb in all_combinations]

    return all_combinations
