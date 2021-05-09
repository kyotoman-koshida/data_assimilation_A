# -*- coding: utf-8 -*-
"""
Created on Sun May  9 02:57:26 2021

@author: Morinaga
"""

import numpy as np
import pandas as pd


def main():
    
    filepath_to_input\
    = "task3_output/365-730days_per6h_dt36m_true.tsv"
    filepath_to_output_0\
    = "task4_output/365-730days_per6h_dt36m_meas.tsv"
    filepath_to_output_1\
    = "task4_output/365-730days_per6h_dt36m_nois.tsv"
    
    df_x_true\
    = pd.read_csv(filepath_to_input, sep = "\t", index_col = 0)
    
    df_x_nois\
    = pd.DataFrame(0.0, index = df_x_true.index, columns = df_x_true.columns)
    for i_row in range(len(df_x_nois)):
        df_x_nois.iloc[i_row] = np.random.normal(0, 1, df_x_nois.shape[1])
    
    df_x_meas = df_x_true +df_x_nois
    
    df_x_meas.to_csv(filepath_to_output_0, sep = "\t")
    df_x_nois.to_csv(filepath_to_output_1, sep = "\t")


if __name__ == "__main__":
    main()
