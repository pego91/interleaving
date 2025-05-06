import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import subprocess
from scipy.spatial.distance import *
from Utils_dendrograms_OPT import prune_dendro, from_cloud_to_dendro_sublvl
from top_TED_lineare_multiplicity import top_TED_lineare as TED
from copy import deepcopy
import skfda

def data_loading_pool(LIST):
    
#    print('Entrato!')
    
    [data,start,stop,epsilon,K]=LIST
    
    trees = []
        
    for patient in np.arange(start,stop+1):
        
        f=data[patient]      
    
        f=np.asarray(f).reshape((len(f),))*K           
        D=np.arange(len(f))

        print('Inizio: ', start)
        
        T = from_cloud_to_dendro_sublvl(D,f,1.01, None,\
                                        prec = 0.0000001, prune_param = None, ITris=[])
        
        T.make_mult(f=False)
        
        trees.append(T)
    return trees
        