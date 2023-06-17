import numpy as np
import random
import time
#from scipy.intergrate import odeint
from math import e
import math

#----FUNCTIONS----

#--------k effective---------
def keff(kd1, kd2, i, j):
# calculate the k_effective value given the i,j indices and the kd's (binding affinities)
    keff= kOn*(kd1**i)*(kd2**j)*e**(-(i+j-1)*9/0.6)
    return keff

#--------propensity calculations--------
def propensity(conc):
    props = [
        kOn * conc[0] * (conc[0] - 1)/2, keff(kd1, kd2, 0, 1) * conc[1],
        2*kOn * conc[0] * (conc[0] - 1)/2, keff(kd1, kd2, 1, 0) * conc[2],
        2*kOn * conc[0] * conc[1], keff(kd1, kd2, 1, 0) *conc[3],
        kOn * conc[0] * conc[2], keff(kd1, kd2, 0, 1) * conc[3],
        2*kOn *conc[0] * conc[1], keff(kd1, kd2, 1, 0) * conc[5], 
        kOn * conc[0] * conc[2], keff(kd1, kd2, 0, 1) * conc[5],
        kOn * conc[0] * conc[2], 3 * keff(kd1, kd2, 2, 0) * conc[8],
        kOn * conc[0] * conc[3], 2 * keff(kd1, kd2, 1, 0) * conc[4],
        kOn * conc[0] * conc[5], 2 * keff(kd1, kd2, 1, 0) * conc[6],
        kOn * conc[0] * conc[3], 2 * keff(kd1, kd2, 1, 1) * conc[7],
        kOn * conc[0] * conc[5], 2 * keff(kd1, kd2, 1, 1) * conc[7],
        kOn * conc[0] * conc[3], 2 * keff(kd1, kd2, 2, 0) * conc[9],
        kOn * conc[0] * conc[5], keff(kd1, kd2, 2, 0) * conc[9],
        3 * kOn * conc[0] * conc[8], keff(kd1, kd2, 0, 1) * conc[9],
        2 * kOn * conc[0] * conc[4], keff(kd1, kd2, 2, 1) * conc[10],
        2 * kOn * conc[0] * conc[6], keff(kd1, kd2, 2, 1) * conc[10],
        2 * kOn * conc[0] * conc[7], keff(kd1, kd2, 2, 0) * conc[10],
        2 * kOn * conc[0] * conc[9], 2 * keff(kd1, kd2, 1, 1) * conc[10],
        kOn * conc[0] * conc[10], 6 * keff(kd1, kd2, 2, 1) * conc[11],
        kOn * conc[2] * (conc[2] - 1)/2, keff(kd1, kd2, 0, 1) * conc[4],
        kOn * conc[2] * (conc[2] - 1)/2, keff(kd1, kd2, 0, 1) * conc[6],
        kOn * conc[1] * (conc[1] - 1)/2, keff(kd1, kd2, 2, 0) * conc[7],
        kOn * conc[2] * (conc[2] -1)/2, keff(kd1, kd2, 0, 2) * conc[7],
        2 * kOn * conc[1] * conc[2], keff(kd1, kd2, 2, 0) * conc[9],
        2* kOn * conc[2] * conc[3], keff(kd1, kd2, 3, 0) * conc[10],
        2 * kOn * conc[1] * conc[5], keff(kd1, kd2, 3, 0) * conc[10],
        kOn * conc[2] * conc[3], keff(kd1, kd2, 2, 1) * conc[10],
        kOn * conc[2] * conc[5], keff(kd1, kd2, 2, 1) * conc[2],
        3 * kOn * conc[2] * conc[8], keff(kd1, kd2, 0, 2) * conc[10],
        2 * kOn * conc[1] * conc[7], 3 * keff(kd1, kd2, 4, 0) * conc[11],
        kOn * conc[2] * conc[9], 6 * keff(kd1, kd2, 2, 2) * conc[11],
        kOn * conc[3] * (conc[3] - 1)/2, 3 * keff(kd1, kd2, 4, 1) * conc[11],
        kOn * conc[5] * (conc[5] - 1)/2, 3 * keff(kd1, kd2, 4, 1) * conc[11],
        3 * kOn * conc[8] * (conc[8] - 1)/2, keff(kd1, kd2, 0, 3) * conc[11]]

    return props

#---------Gillespie Algorithm---------

def GA_SR(IC, kd1, kd2, time_end):
    
# initialize conditions
    x_0 = [IC]

    x_1 = [0]; x_2 = [0]; x_3 = [0]; x_4 = [0]; x_5 = [0]; x_6 = [0]; x_7 = [0];  x_8 = [0]; x_9 = [0];  x_10 = [0]; x_11 = [0];
    
    time = [0] #all time
    time_tr = [] #time for tracker 

    # concnetrations on specific time points, list of list
    tracker = [[x_0[-1]], [x_1[-1]], [x_2[-1]], [x_3[-1]], [x_4[-1]], [x_5[-1]], [x_6[-1]], [x_7[-1]], [x_8[-1]], [x_9[-1]], [x_10[-1]], [x_11[-1]]]

    p = 0
    time_vec = np.logspace(0, (math.log(tEnd)/math.log(1.5)), num = 100, endpoint = True, base = 1.5)

#begin while loop
    while time[-1] <= time_end and p<100:
 #       print("-----------------------------------------------------------------")
        # all concentrations
        conc = [x_0[-1], x_1[-1], x_2[-1], x_3[-1], x_4[-1], x_5[-1], x_6[-1], x_7[-1], x_8[-1], x_9[-1], x_10[-1], x_11[-1] ]

#calculate propensity from concentrations
        props = propensity(conc)
       
        props_cs = np.cumsum(props)
        #cumulative sum of propensities

        props_sum = sum(props)
        #straigh up sum of propensties

        r = np.random.random(1)
        tau = (1/props_sum) * np.log(1/r)
        time.extend( time[-1] + tau ) #time is now time plus tau
        if time_vec[p]< time[-1]:
            p = p + 1
            time_tr.append(time[-2]) 
            print("current time_tr: ", time_tr[-1])

#tracking data for log time, if tau pushes past time point then current conctration is recorded before new concentration calculated
#time recorded is time before tau pushes time past threshold
#        if time[-1] >= 1.5*time_tr[-1]:
#            print("lets look here :)")
#            for p in range(12):
#                tracker[p].append(conc[p])
#            time_tr.extend(time[-1] - tau)
#            print("time : ", time_tr[-1])
#            print("tracker : ", tracker)
#            print("tr_time : ", time_tr)
 

#time using vector 

#selecting rxn 
        rand = random.uniform(0,1)

        num = rand * props_sum

        # rxn 1     0 + 0 = 1 

        if num > 0 and num <= props_cs[0] and conc[0] > 1:
            x_0.append(x_0[-1] - 2)
            x_1.append(x_1[-1] + 1)
            x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]),  x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]),  x_11.append(x_11[-1])

        elif num > props_cs[0] and num <= props_cs[1] and conc[1] > 0: 
            x_0.append(x_0[-1] + 2)
            x_1.append(x_1[-1] - 1)
            x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 2    0 + 0 = 2

        elif num > props_cs[1] and num <= props_cs[2] and conc[0] > 0:
            x_0.append(x_0[-1] - 2)
            x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[2] and num <= props_cs[3] and conc[2] > 0:
            x_0.append(x_0[-1] + 2)
            x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 3     0 + 1 = 3

        elif num > props_cs[3] and num <= props_cs[4] and conc[0] > 0 and conc[1] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1] - 1)
            x_2.append(x_2[-1])
            x_3.append(x_3[-1] + 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[4] and num <= props_cs[5] and conc[3] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1] + 1)
            x_2.append(x_2[-1])
            x_3.append(x_3[-1] - 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 4    0 + 2 = 3

        elif num > props_cs[5] and num <= props_cs[6] and conc[0] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1] + 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]),  x_11.append(x_11[-1])

        elif num > props_cs[6] and num <= props_cs[7] and conc[2] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1] - 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 5    0 + 1 = 5

        elif num > props_cs[7] and num <= props_cs[8] and conc[0] > 0 and conc[1] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1] - 1)
            x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] + 1)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[8] and num <= props_cs[9] and conc[5] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1] + 1)
            x_2.append(x_2[-1]),  x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1] - 1), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1])
            x_11.append(x_11[-1])

        # rxn 6    0 + 2 = 5

        elif num > props_cs[9] and num <= props_cs[10] and conc[0] > 0 and conc[2] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] + 1)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[10] and num <= props_cs[11] and conc[5] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] - 1)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 7    0 + 2 = 8

        elif num > props_cs[11] and num <= props_cs[12] and conc[0] > 0 and conc[2] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] + 1)
            x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[12] and num <= props_cs[13] and conc[8] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] - 1)
            x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 8    0 + 3 =  4

        elif num > props_cs[13] and num <= props_cs[14] and conc[0] > 0 and conc[3] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1])
            x_3.append(x_3[-1] - 1)
            x_4.append(x_4[-1] + 1)
            x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[14] and num <= props_cs[15] and conc[4] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1])
            x_3.append(x_3[-1] + 1)
            x_4.append(x_4[-1] - 1)
            x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 9   0 + 5 = 6

        elif num > props_cs[15] and num <= props_cs[16] and conc[0] > 0 and conc[5] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] - 1)
            x_6.append(x_6[-1] + 1)
            x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[16] and num <= props_cs[17] and conc[5] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] + 1)
            x_6.append(x_6[-1] - 1), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 10  0 + 3 = 7

        elif num > props_cs[17] and num <= props_cs[18] and conc[0] > 0 and conc[3] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1])
            x_3.append(x_3[-1] - 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1] + 1), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[18] and num <= props_cs[19] and conc[7] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1])
            x_3.append(x_3[-1] + 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1])
            x_7.append(x_7[-1] - 1)
            x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 11  0 + 5 = 7
        elif num > props_cs[19] and num <= props_cs[20] and conc[0] > 0 and conc[5] > 0:
            x_0.append(x_0[-1] - 1), x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]),x_4.append(x_4[-1])
            x_5.append(x_5[-1] - 1)
            x_6.append(x_6[-1])
            x_7.append(x_7[-1] + 1)
            x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[20] and num <= props_cs[21] and conc[7] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] + 1)
            x_6.append(x_6[-1])
            x_7.append(x_7[-1] - 1)
            x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 12 0 + 3 = 9
        elif num > props_cs[21] and num <= props_cs[22] and conc[0] > 0 and conc[3] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1])
            x_3.append(x_3[-1] - 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] + 1)
            x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[22] and num <= props_cs[23] and conc[9] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1])
            x_3.append(x_3[-1] + 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] - 1)
            x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 13    0 + 5 = 9
        elif num > props_cs[23] and num <= props_cs[24] and conc[0] > 0 and conc[5] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]),x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] - 1)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] + 1)
            x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[24] and num <= props_cs[25] and conc[9] > 0:
            x_0.append(x_0[-1] + 1), x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] + 1)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] - 1)
            x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 14    0 + 8 = 9
        elif num > props_cs[25] and num <= props_cs[26] and conc[0] > 0 and conc[8] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1])
            x_2.append(x_2[-1])
            x_3.append(x_3[-1])
            x_4.append(x_4[-1])
            x_5.append(x_5[-1])
            x_6.append(x_6[-1])
            x_7.append(x_7[-1])
            x_8.append(x_8[-1] - 1)
            x_9.append(x_9[-1] + 1)
            x_10.append(x_10[-1])
            x_11.append(x_11[-1])

        elif num > props_cs[26] and num <= props_cs[27] and conc[9] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] + 1)
            x_9.append(x_9[-1] - 1)
            x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 15    0 + 4 = 10
        elif num > props_cs[27] and num <= props_cs[28] and conc[0] > 0 and conc[8] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] - 1)
            x_9.append(x_9[-1] + 1)
            x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[28] and num <= props_cs[29] and conc[9] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] + 1)
            x_9.append(x_9[-1] - 1)
            x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 16    0 + 6 = 10
        elif num > props_cs[29] and num <= props_cs[30] and conc[0] > 0 and conc[6] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1])
            x_6.append(x_6[-1] -1)
            x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] + 1)
            x_11.append(x_11[-1])

        elif num > props_cs[30] and num <= props_cs[31] and conc[10] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1])
            x_6.append(x_6[-1] + 1)
            x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] - 1)
            x_11.append(x_11[-1])

        # rxn 17    0 + 7 = 10
        elif num > props_cs[31] and num <= props_cs[32] and conc[0] > 0 and conc[7] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1])
            x_7.append(x_7[-1] - 1)
            x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] + 1)
            x_11.append(x_11[-1])

        elif num > props_cs[32] and num <= props_cs[33] and conc[10] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1])
            x_7.append(x_7[-1] + 1)
            x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] - 1)
            x_11.append(x_11[-1])

        # rxn 18    0 + 9 = 10
        elif num > props_cs[33] and num <= props_cs[34] and conc[0] > 0 and conc[9] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] - 1)
            x_10.append(x_10[-1] + 1)
            x_11.append(x_11[-1])

        elif num > props_cs[34] and num <= props_cs[35] and conc[10] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] + 1)
            x_10.append(x_10[-1] - 1)
            x_11.append(x_11[-1])

        # rxn 19    0 +  10 = 11
        elif num > props_cs[35] and num <= props_cs[36] and conc[0] > 0 and conc[10] > 0:
            x_0.append(x_0[-1] - 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] - 1)
            x_11.append(x_11[-1] + 1)

        elif num > props_cs[36] and num <= props_cs[37] and conc[11] > 0:
            x_0.append(x_0[-1] + 1)
            x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] + 1)
            x_11.append(x_11[-1] - 1)

        # rxn 20    2 +  2 = 4
        elif num > props_cs[37] and num <= props_cs[38] and conc[2] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 2)
            x_3.append(x_3[-1])
            x_4.append(x_4[-1] + 1)
            x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[38] and num <= props_cs[39] and conc[4] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 2)
            x_3.append(x_3[-1])
            x_4.append(x_4[-1] - 1)
            x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 21    2 +  2 = 6
        elif num > props_cs[39] and num <= props_cs[40] and conc[2] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 2)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1])
            x_6.append(x_6[-1] + 1)
            x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[40] and num <= props_cs[41] and conc[6] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 2)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1])
            x_6.append(x_6[-1] - 1)
            x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 22    1 + 1 = 7
        elif num > props_cs[41] and num <= props_cs[42] and conc[1] > 0:
            x_0.append(x_0[-1])
            x_1.append(x_1[-1] - 2)
            x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1])
            x_7.append(x_7[-1] + 1)
            x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[42] and num <= props_cs[43] and conc[7] > 0:
            x_0.append(x_0[-1])
            x_1.append(x_1[-1] + 2)
            x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1])
            x_7.append(x_7[-1] - 1)
            x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 23    2  +  2 = 7 
        elif num > props_cs[43] and num <= props_cs[44] and conc[2] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 2)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1])
            x_7.append(x_7[-1] + 1)
            x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[44] and num <= props_cs[45] and conc[7] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 2)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1])
            x_7.append(x_7[-1] - 1)
            x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 24    1  +  2 = 9 
        elif num > props_cs[45] and num <= props_cs[46] and conc[1] > 0 and conc[2]> 0:
            x_0.append(x_0[-1])
            x_1.append(x_1[-1] - 1)
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] + 1)
            x_10.append(x_10[-1]), x_11.append(x_11[-1])

        elif num > props_cs[46] and num <= props_cs[47] and conc[9] > 0:
            x_0.append(x_0[-1])
            x_1.append(x_1[-1] + 1)
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] -1)
            x_10.append(x_10[-1]), x_11.append(x_11[-1])

        # rxn 25    2  +  3 = 10 
        elif num > props_cs[47] and num <= props_cs[48] and conc[2] > 0 and conc[3]> 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1] - 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] + 1)
            x_11.append(x_11[-1])

        elif num > props_cs[48] and num <= props_cs[49] and conc[10] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1] + 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] - 1)
            x_11.append(x_11[-1])

    # rxn 26    1  +  5 = 10 
        elif num > props_cs[49] and num <= props_cs[50] and conc[1] > 0 and conc[5]> 0:
            x_0.append(x_0[-1])
            x_1.append(x_1[-1] - 1)
            x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] - 1), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] + 1)
            x_11.append(x_11[-1])

        elif num > props_cs[50] and num <= props_cs[51] and conc[10] > 0:
            x_0.append(x_0[-1])
            x_1.append(x_1[-1] + 1)
            x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] + 1)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] - 1)
            x_11.append(x_11[-1])

        # rxn 27    2  +  3 = 10 
        elif num > props_cs[51] and num <= props_cs[52] and conc[2] > 0 and conc[3]> 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1] - 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] + 1)
            x_11.append(x_11[-1])

        elif num > props_cs[52] and num <= props_cs[53] and conc[10] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1] + 1)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] - 1)
            x_11.append(x_11[-1])

        # rxn 28    2  +  5 = 10 
        elif num > props_cs[53] and num <= props_cs[54] and conc[2] > 0 and conc[5]> 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] - 1)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] + 1)
            x_11.append(x_11[-1])

        elif num > props_cs[54] and num <= props_cs[55] and conc[10] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] + 1)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1])
            x_10.append(x_10[-1] - 1)
            x_11.append(x_11[-1])

        # rxn 29    2  +  8 = 10 
        elif num > props_cs[55] and num <= props_cs[56] and conc[2] > 0 and conc[8]> 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] - 1)
            x_9.append(x_9[-1])
            x_10.append(x_10[-1] + 1)
            x_11.append(x_11[-1])

        elif num > props_cs[56] and num <= props_cs[57] and conc[10] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] + 1)
            x_9.append(x_9[-1])
            x_10.append(x_10[-1] - 1)
            x_11.append(x_11[-1])

        # rxn 30    1  +  7 = 11 
        elif num > props_cs[57] and num <= props_cs[58] and conc[1] > 0 and conc[7]> 0:
            x_0.append(x_0[-1])
            x_1.append(x_1[-1] - 1)
            x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1])
            x_7.append(x_7[-1] - 1)
            x_8.append(x_8[-1]),  x_9.append(x_9[-1]), x_10.append(x_10[-1])
            x_11.append(x_11[-1] + 1)

        elif num > props_cs[58] and num <= props_cs[59] and conc[11] > 0:
            x_0.append(x_0[-1])
            x_1.append(x_1[-1] + 1)
            x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] + 1)
            x_9.append(x_9[-1]), x_10.append(x_10[-1])
            x_11.append(x_11[-1] - 1)

        # rxn 31    2  +  9 = 11 
        elif num > props_cs[59] and num <= props_cs[60] and conc[2] > 0 and conc[9]> 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] - 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] - 1)
            x_10.append(x_10[-1])
            x_11.append(x_11[-1] + 1)

        elif num > props_cs[60] and num <= props_cs[61] and conc[11] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1])
            x_2.append(x_2[-1] + 1)
            x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1])
            x_9.append(x_9[-1] + 1)
            x_10.append(x_10[-1])
            x_11.append(x_11[-1] - 1)

        # rxn 32    3 +  3 = 11 
        elif num > props_cs[61] and num <= props_cs[62] and conc[3] > 2:
            x_0.append(x_0[-1]), x_1.append(x_1[-1]), x_2.append(x_2[-1])
            x_3.append(x_3[-1] - 2)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1])
            x_11.append(x_11[-1] + 1)

        elif num > props_cs[62] and num <= props_cs[63] and conc[11] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1]), x_2.append(x_2[-1])
            x_3.append(x_3[-1] + 2)
            x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1])
            x_11.append(x_11[-1] - 1)

        # rxn 33    5 +  5 = 11 
        elif num > props_cs[63] and num <= props_cs[64] and conc[5] > 2:
            x_0.append(x_0[-1]), x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] - 2)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1])
            x_11.append(x_11[-1] + 1)

        elif num > props_cs[64] and num <= props_cs[65] and conc[11] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1])
            x_5.append(x_5[-1] + 2)
            x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1])
            x_11.append(x_11[-1] - 1) 

        # rxn 34    8 +  8 = 11 
        elif num > props_cs[65] and num <= props_cs[66] and conc[8] > 2:
            x_0.append(x_0[-1]), x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] - 2)
            x_9.append(x_9[-1]), x_10.append(x_10[-1])
            x_11.append(x_11[-1] + 1)

        elif num > props_cs[66] and num <= props_cs[67] and conc[11] > 0:
            x_0.append(x_0[-1]), x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1])
            x_8.append(x_8[-1] + 2)
            x_9.append(x_9[-1]), x_10.append(x_10[-1])
            x_11.append(x_11[-1] - 1) 
            

        else:
            x_0.append(x_0[-1]), x_1.append(x_1[-1]), x_2.append(x_2[-1]), x_3.append(x_3[-1]), x_4.append(x_4[-1]), x_5.append(x_5[-1]), x_6.append(x_6[-1]), x_7.append(x_7[-1]), x_8.append(x_8[-1]), x_9.append(x_9[-1]), x_10.append(x_10[-1]), x_11.append(x_11[-1])
    
    #erasing first element in the subspecies
    x_0.pop(0)
    x_1.pop(0)
    x_2.pop(0)
    x_3.pop(0)
    x_4.pop(0)
    x_5.pop(0)
    x_6.pop(0)
    x_7.pop(0)
    x_8.pop(0)
    x_9.pop(0)
    x_10.pop(0)
    x_11.pop(0)
    
    tracker = np.vstack(tracker)
#    concentrations = tracker
    concentrations = np.array(tracker)
    #concentrations = np.vstack(concentrations)
    #cocentrations = np.reshape(t_len, 11)
    #concentrations = concentrations.transpose()
    
    return concentrations, time_tr

print("its running dw")

#--------Assembly Yeild---------
def AY_final(array1):
    x11_AY_fnl = (6 * array1[-1 , -1]/ IC_1)
    return x11_AY_fnl

def AY_time(array1):
    AYTC = (6 * array1[:, 11]/ IC_1)
    return AYTC

def AY_Avg(numrun):
    avg_cumsum = 0
    AY_Avg = avg_cumsum/numrun
    for i in range(numrun):
        array1, time1 = GA_SR(IC_1, kd1, kd2, tEnd)
        avg_cumsum+= AY_final(array1)
    print("cumulative average: ", avg_cumsum)
    return float(AY_Avg)

#--------Heat Map-------
def AY_Matrix(KD1_start, KD1_end, KD1_increm, KD2_start, KD2_end, KD2_increm):
    kd1_list = np.linspace(KD1_start, KD1_end, num = 20, endpoint = True)
    kd2_list = np.linspace(KD2_start, KD2_end, num = 20, endpoint = True)
   # print(kd1_list)
    AY_list = []
    for kd2 in kd2_list:
        for kd1 in kd1_list:
            array1, time1 = GA_SR(IC_1, kd1, kd2, tEnd)   
            AY_list.append(AY_final(array1))
    AY_matrix = np.reshape(AY_list, (len(kd1_list), len(kd2_list)))
    return AY_matrix


# --- PARAMETERS ---
# List of fixed parameters
kOn = 10**6
delta = 0 # degradation rate parameter
Q = 0 # synthesis rate parameter
tEnd = 10000

# List of parameters that will be explored 
Na = 6.022*10**23  #avogadro's number
Vol = 10**-7 #volume in litre
kappa =6*10**-3 #per molecule per second
IC_1 = 6000#(Na * Vol)/kappa
IC_2 = 3.54*10**5
kd1 = 10**-5 # binding affinity WITHIN rings (intra)
kd2 = 10**-8 # binding affinity BETWEEN rings (inter)
KD1_start = 10**-9; KD1_end = 10**-4; KD1_increm = 10**-5; #for heatmap
KD2_start = 10**-9; KD2_end = 10**-4; KD2_increm = 10**-5;

Param_dict={kOn:"kOn", delta:"delta", Q:"Q", tEnd:"tEND",IC_1:"IC_1",IC_2:"IC_2",kd1:"kd1",kd2:"kd2",KD1_start:"KD1_start", KD1_end:"KD1_end", KD1_increm:"KD1_increm", KD2_start:"KD2_start", KD2_end:"KD2_end", KD2_increm:"KD2_increm"}


#keeping track of time
t =time.time()

# Running functions NO SYNTH OR DEG
print(Param_dict)
array1, time1 = GA_SR(IC_1, kd1, kd2, tEnd)
print("GA complete")

elapsed = time.time() - t
print("time leapsed : ", elapsed)

AYF = AY_final(array1)
print("Final AY complete : ", AYF)

AYvT = AY_time(array1)
print("AY v Time  complete")

#AY_AVG = AY_Avg(10)
#print("AY average complete", AY_AVG)

#AY_Matrix = AY_Matrix(KD1_start, KD1_end, KD1_increm, KD2_start, KD2_end, KD2_increm)
#print("matrix complete")

#print("this is raw data : ", array1)
#print("this is time ", time1)
#print("this is the final assembly yeild: ", AYF)
#print("AY for TC: " , AYvT)
#print("average AY: ", AY_AVG)
#print("AY_matrix for heatmap : ", AY_Matrix)


#-------- Running FUnctions WITH synth and deg

#-------- 
#delta = 0 # degradation rate parameter
#Q = 0 # synthesis rate parameter

#np.savetxt("/home/paigem/tr_update/tr_conc0.csv", array1, delimiter=', ', header = "'X0','X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8','X9', 'X10', 'X11'")
np.savetxt("/home/paigem/tr_update/tr_conc.csv", array1, delimiter=',', header = "'X0','X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8','X9', 'X10', 'X11'")
np.savetxt("/home/paigem/tr_update/tr_ctime.csv",time_tr, delimiter=',', header = "time")
#np.savetxt("/home/paigem/tr_update/tr_ay0.csv", AYvT, delimiter=',',header = "AY X11")
#np.savetxt("/home/paigem/tr_update/tr_hm.csv", AY_Matrix, delimiter=',')
