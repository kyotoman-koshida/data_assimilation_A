# -*- coding: utf-8 -*-
"""
Created on Sun May  9 01:42:08 2021

@author: Morinaga
"""

import numpy as np


def get_Lorenz_v(ary_x, F):
    return\
    (np.roll(ary_x, -1) -np.roll(ary_x, +2)) *np.roll(ary_x, +1) -ary_x +F


def generate_Runge_Kutta_4(ary_x_init, dt, func_for_v, skip_init = True):
    
    ary_x = ary_x_init.copy()
    if not skip_init:
        yield ary_x
    
    while 1:
        ary_q_1 = func_for_v(ary_x) *dt
        ary_q_2 = func_for_v(ary_x +ary_q_1/2) *dt
        ary_q_3 = func_for_v(ary_x +ary_q_2/2) *dt
        ary_q_4 = func_for_v(ary_x +ary_q_3) *dt
        ary_x += (ary_q_1 +2*ary_q_2 +2*ary_q_3 +ary_q_4) /6
        yield ary_x


def main():
    
    filepath_to_output = "task3_output/365-730days_per6h_dt36m_true.tsv"
    
    F = 8.0
    dt = 0.005
    n_steps_per_day = 40
    n_steps_per_6h = 10
    n_days_spinup = 365
    n_days_all = 730
    
    ary_x = np.full(40, 8.0)
    ary_x[0] = 8.008
    
    generating_Runge_Kutta_4\
    = generate_Runge_Kutta_4(
            ary_x_init = ary_x, 
            dt = dt,
            func_for_v = (lambda ary_lambda: get_Lorenz_v(ary_lambda, F)),
            skip_init = True,
            )
    
    for i_step in range(1, n_steps_per_day * n_days_spinup):
            
        ary_x = next(generating_Runge_Kutta_4)
    
    with open(filepath_to_output, mode="w") as filedata:
        
        s_to_write = "t\t"
        s_to_write += "\t".join(map(lambda i: 'x_{:0>2}'.format(i), range(40)))
        s_to_write += "\n"
        print(s_to_write)
        filedata.write(s_to_write)
        
        for i_day in range(n_days_spinup, n_days_all):
            for i_step in range(n_steps_per_day):
            
                ary_x = next(generating_Runge_Kutta_4)
                
                if i_step % n_steps_per_6h == 0:
                    s_to_write\
                    = "{0:03}d{1:02}h".format(i_day, i_step//n_steps_per_6h*6)\
                      +"\t"
                    s_to_write += "\t".join(ary_x.astype(str))
                    s_to_write += "\n"
                    print(s_to_write)
                    filedata.write(s_to_write)
        
        ary_x = next(generating_Runge_Kutta_4)
        s_to_write = "730d00h\t"
        s_to_write += "\t".join(ary_x.astype(str))
        s_to_write += "\n"
        print(s_to_write)
        filedata.write(s_to_write)


if __name__ == "__main__":
    main()
