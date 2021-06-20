# -*- coding: utf-8 -*-
"""
Created on Tue May 18 04:52:36 2021

@author: Morinaga
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    
    KF_or_OI = (
                "KF"
                #"OI"
                )
    
    infix_dx = (
                "10"
                #"05"
                #"01"
                )
    
    infix_alpha = (
                   "10"
                   #"11"
                   #"12"
                   #False
                   )
    
    filepath_0 = "task4_output/365-730days_per6h_dt36m_meas.tsv"
    filepath_1 = "task5_output/365-730days_per6h_dt36m_" +KF_or_OI +"FA_"\
                 +"dx" +infix_dx +"milli"\
                 +(("_alpha" +infix_alpha +"deci") if infix_alpha else "")\
                 +".tsv"
    
    i_dim_chosen = (
                    0
                    #1
                    )
    
    df_y_o = pd.read_csv(filepath_0, sep = "\t", index_col = 0)
    df_x_fa = pd.read_csv(filepath_1, sep = "\t", index_col = 0)
    
    df_to_plot = pd.DataFrame(None, index = df_y_o.index)
    df_to_plot["observation"] = df_y_o["x_{:0>2}".format(i_dim_chosen)]
    df_to_plot["forecast"] = df_x_fa["xf{:0>2}".format(i_dim_chosen)]
    df_to_plot["analysis"] = df_x_fa["xa{:0>2}".format(i_dim_chosen)]
    
    df_to_plot.index = np.arange(365.0, 730.01, 0.25)
    df_to_plot.index.name = "day"
    
    df_to_plot.plot(
            xlim = [365, 390],
            xticks = [365, 370, 375, 380, 385, 390],
            ylim = [-7.5, 10.0],
            yticks = [-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0],
            color = ["gold", "darkturquoise", "orchid"],
            #lw = 0.9,
            )
    plt.ylabel("Value of x_" +str(i_dim_chosen))
    
    plt.title(KF_or_OI +" (dx = 0.0" +infix_dx
              +((", alpha = " +infix_alpha[0] +"." +infix_alpha[1:])
                if infix_alpha else "")
              +")")
    
    plt.show()


if __name__ == "__main__":
    main()
