# -*- coding: utf-8 -*-
"""
Created on Sat May  8 19:00:54 2021

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
    
    filepath_to_input\
    = "task2_output_Runge4/Runge4_F8_dt05milli.tsv"
    filepath_to_output\
    = "task2_output_Runge4/Runge4_F8_dt05milli_edr_ofs"\
      +(
        #"400"
        #"500"
        "600"
        )\
      +".tsv"
    
    ary2_x_att = np.loadtxt(filepath_to_input, skiprows = 1)
    
    dt = ary2_x_att[1,0] -ary2_x_att[0,0]
    ary2_x_att = ary2_x_att[:,1:]
    
    F = 8.0
    
    n_offset = (
                #400
                #500
                600
                )
    n_extent = 100
    n_trials = 1000
    n_steps_edr = 200
    
    ary_edr_sum = np.zeros(n_steps_edr +1)
    
    for i_trial in range(n_trials):
        print("now: i_trial = " +str(i_trial))
        
        i_step_chosen = np.random.randint(n_offset, n_offset +n_extent)
        ary_dx = np.random.rand(ary2_x_att.shape[1])\
                 *(-1 +2*np.random.randint(0, 2, ary2_x_att.shape[1]))
                 
        norm_dx = np.linalg.norm(ary_dx, ord = 2)
        
        generating_Runge_Kutta_4\
        = generate_Runge_Kutta_4(
                ary_x_init = ary2_x_att[i_step_chosen] +ary_dx, 
                dt = dt,
                func_for_v = (lambda ary_lambda: get_Lorenz_v(ary_lambda, F)),
                skip_init = True,
                )
        
        for i_step_edr in range(1, n_steps_edr +1):
            
            ary_delta_x\
            = next(generating_Runge_Kutta_4)\
              -ary2_x_att[i_step_chosen +i_step_edr]
            
            ary_edr_sum[i_step_edr]\
            += np.linalg.norm(ary_delta_x, ord = 2) / norm_dx
    
    ary_edr_avr = ary_edr_sum / n_trials
    ary_edr_avr[0] = 1.0
    print(ary_edr_avr)
    
    with open(filepath_to_output, mode="w") as filedata:
        s_to_write = "t\tedr\n"
        s_to_write\
        += "\n".join(map(lambda i_lambda:
                         "{:.3f}\t".format(i_lambda *dt)\
                         +ary_edr_avr.astype(str)[i_lambda],
                         range(n_steps_edr +1)
                         )
                     )
        s_to_write += "\n"
        filedata.write(s_to_write)


if __name__ == "__main__":
    main()
