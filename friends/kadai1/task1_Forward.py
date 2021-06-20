# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 06:29:02 2021

@author: Jaku
"""

import numpy as np

def get_Lorenz_v(ary_x, F):
    return\
    (np.roll(ary_x, -1) -np.roll(ary_x, -2)) *np.roll(ary_x, -3) -ary_x +F


def main():
    
    filepath_to_output = "task1_output_Forward/Forward_F8_err.tsv"
    
    dt = 0.01
    n_steps = 200
    F = 8.0
    
    #ary_x = np.array([1.0, 1.0, 1.0, 1.0])
    ary_x = np.array([1.001, 1.0, 1.0, 1.0])
    
    ary_v = get_Lorenz_v(ary_x, F)
    
    with open(filepath_to_output, mode="w") as filedata:
        filedata.write("t\tx_0\tx_1\tx_2\tx_3\tv_0\tv_1\tv_2\tv_3\n")
        filedata.write(
                str(0.00) +"\t"
                +str(ary_x[0]) +"\t" +str(ary_x[1]) +"\t"
                +str(ary_x[2]) +"\t" +str(ary_x[3]) +"\t"
                +str(ary_v[0]) +"\t" +str(ary_v[1]) +"\t"
                +str(ary_v[2]) +"\t" +str(ary_v[3]) +"\n"
                )
        
        for i_step in range(1, n_steps +1):
            
            ary_x += ary_v *dt
            ary_v = get_Lorenz_v(ary_x, F)
            
            filedata.write(
                    "{:.2f}".format(i_step*dt) +"\t"
                    +str(ary_x[0]) +"\t" +str(ary_x[1]) +"\t"
                    +str(ary_x[2]) +"\t" +str(ary_x[3]) +"\t"
                    +str(ary_v[0]) +"\t" +str(ary_v[1]) +"\t"
                    +str(ary_v[2]) +"\t" +str(ary_v[3]) +"\n"
                    )


if __name__ == "__main__":
    main()
