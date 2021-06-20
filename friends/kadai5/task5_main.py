# -*- coding: utf-8 -*-
"""
Created on Mon May 17 23:54:38 2021

@author: Morinaga
"""

import numpy as np
import pandas as pd


def get_Lorenz_v(ary_x, F):
    return\
    (np.roll(ary_x, -1) -np.roll(ary_x, +2)) *np.roll(ary_x, +1) -ary_x +F


def model(ary_x_init, F, dt, n_sampling):
    
    ary_x = ary_x_init.copy()
        
    for i_step in range(n_sampling):
        ary_q_1 = get_Lorenz_v(ary_x, F) *dt
        ary_q_2 = get_Lorenz_v(ary_x +ary_q_1/2, F) *dt
        ary_q_3 = get_Lorenz_v(ary_x +ary_q_2/2, F) *dt
        ary_q_4 = get_Lorenz_v(ary_x +ary_q_3, F) *dt
        ary_x += (ary_q_1 +2*ary_q_2 +2*ary_q_3 +ary_q_4) /6
    
    return ary_x


def generate_KF_cycle(ary2_y_o, F, dt, n_sampling):
    
    n_dim = ary2_y_o.shape[1]
    
    ary_x_f = np.zeros(n_dim)
    ary_x_f[:] = np.nan
    
    ary_x_a = ary2_y_o[0].copy()
    ary2_P_a = np.identity(n_dim)
    
    yield ary_x_f, ary_x_a
    
    for i_step in range(1, len(ary2_y_o)):
        ary_x_f\
        = model(ary_x_a, F, dt, n_sampling)
        
        func_m\
        = lambda i:\
          [
           -dt*n_sampling*ary_x_a[i-1],
           dt*n_sampling*(ary_x_a[(i+1)%n_dim] -ary_x_a[i-2]),
           1-dt*n_sampling,
           dt*n_sampling*ary_x_a[i-1],
           ]\
          +[0.0 for i in range(n_dim-4)]
        
        ary2_M = np.array([np.roll(func_m(i), i) for i in range(n_dim)])
        
        ary2_P_f = np.dot(np.dot(ary2_M, ary2_P_a), ary2_M.T)
        
        ary2_H = np.identity(n_dim)
        ary2_R = np.identity(n_dim)
        
        ary2_K\
        = np.dot(
                 np.dot(ary2_P_f, ary2_H.T),
                 np.linalg.inv(
                               np.dot(np.dot(ary2_H, ary2_P_f), ary2_H.T)
                               +ary2_R
                               ),
                 )
        
        ary2_P_a\
        = np.dot(
                 np.identity(n_dim) -np.dot(ary2_K, ary2_H),
                 ary2_P_f,
                 )
        
        ary_x_a\
        = ary_x_f +np.dot(ary2_K, ary2_y_o[i_step] -np.dot(ary2_H, ary_x_f))
        
        yield ary_x_f, ary_x_a


def main():
    
    filepath_to_input = "task4_output/365-730days_per6h_dt36m_meas.tsv"
    filepath_to_output = "task5_output/365-730days_per6h_dt36m_KFFA.tsv"
    
    F = 8.0
    dt = 0.005
    n_sampling = 10
    
    df_y_o\
    = pd.read_csv(filepath_to_input, sep = "\t", index_col = 0)
    
    with open(filepath_to_output, mode="w") as filedata:
        
        s_to_write = "t\t"
        s_to_write += "\t".join(map(lambda i: 'xf{:0>2}'.format(i), range(40)))
        s_to_write += "\t"
        s_to_write += "\t".join(map(lambda i: 'xa{:0>2}'.format(i), range(40)))
        s_to_write += "\n"
        filedata.write(s_to_write)
        
        generating_KF_cycle\
        = generate_KF_cycle(np.array(df_y_o), F, dt, n_sampling)
        
        ary_x_f, ary_x_a = next(generating_KF_cycle)
        print("now: " +df_y_o.index[0])
        
        s_to_write = df_y_o.index[0] +"\t"
        s_to_write += "\t".join(ary_x_f.astype(str))
        s_to_write += "\t"
        s_to_write += "\t".join(ary_x_a.astype(str))
        s_to_write += "\n"
        filedata.write(s_to_write)
        
        for i_step in range(1, len(df_y_o)):
            ary_x_f, ary_x_a = next(generating_KF_cycle)
            print("now: " +df_y_o.index[i_step])
            
            s_to_write = df_y_o.index[i_step] +"\t"
            s_to_write += "\t".join(ary_x_f.astype(str))
            s_to_write += "\t"
            s_to_write += "\t".join(ary_x_a.astype(str))
            s_to_write += "\n"
            filedata.write(s_to_write)


if __name__ == "__main__":
    main()
