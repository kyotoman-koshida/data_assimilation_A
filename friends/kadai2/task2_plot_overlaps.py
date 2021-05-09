# -*- coding: utf-8 -*-
"""
Created on Sun May  9 00:15:28 2021

@author: Morinaga
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    
    ax_current = None
    
    for i_page in range(4):
        
        filepath_ofs400\
        = "task2_output_Runge4/Runge4_F8_dt05milli_edr_ofs400_"\
          +str(i_page) +".tsv"
        filepath_ofs500\
        = "task2_output_Runge4/Runge4_F8_dt05milli_edr_ofs500_"\
          +str(i_page) +".tsv"
        filepath_ofs600\
        = "task2_output_Runge4/Runge4_F8_dt05milli_edr_ofs600_"\
          +str(i_page) +".tsv"
        
        ary2_edr_avr_ofs400 = np.loadtxt(filepath_ofs400, skiprows = 1)
        ary2_edr_avr_ofs500 = np.loadtxt(filepath_ofs500, skiprows = 1)
        ary2_edr_avr_ofs600 = np.loadtxt(filepath_ofs600, skiprows = 1)
        
        df_edr_avr = pd.DataFrame(None, index = ary2_edr_avr_ofs400[:,0])
        df_edr_avr.index.name = "t"
        df_edr_avr["offset = 2.0, extent = 1.0"] = ary2_edr_avr_ofs400[:,1]
        df_edr_avr["offset = 2.5, extent = 1.0"] = ary2_edr_avr_ofs500[:,1]
        df_edr_avr["offset = 3.0, extent = 1.0"] = ary2_edr_avr_ofs600[:,1]
        
        ax_current\
        = df_edr_avr.plot(grid=True,
                          ax = ax_current,
                          legend = not(i_page),
                          color = ["c", "m", "y"],
                          alpha = 0.5,
                          )
    
    plt.ylabel("Average of Error Developing Rate")
    plt.show()


if __name__ == "__main__":
    main()
