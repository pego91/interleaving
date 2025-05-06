import numpy as np
from copy import deepcopy
import time

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core.util import sum_product,quicksum

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
# logging.getLogger('pyomo.core').setLevel(logging.WARNING)

from Trees_OPT import Tree
from Utils_dendrograms_OPT import prune_dendro,prune_dendro_N
from itertools import product
import multiprocessing as mp

from sympy import *


T_build=0
T_solve=0

def calculate_Wxy_pool(LIST):
    
    [T_0,T_1,W,x,y,SUB_0,SUB_1,W_0,W_1,V_0,V_1,K_0,K_1,binary, approx, K_const_,check,check_mat_]=LIST
    
    T_x=SUB_0[x]
    T_y=SUB_1[y]
    
    m_x=[SUB_0[i].f_uniq[0] for i in V_0[x]]    
    m_y=[SUB_1[j].f_uniq[0] for j in V_1[y]]
    
    W_aux=np.zeros((T_x.dim-1,T_y.dim-1)).astype(float)
    
    avaiable_x=set(T_x.vertices).copy()-set(np.where(T_x.name_vertices==x)[0])
    avaiable_y=set(T_y.vertices).copy()-set(np.where(T_y.name_vertices==y)[0])
    
    global approx_
    global K_const
    global check_
    global check_mat
    
    check_ = check
    approx_ = approx
    K_const = K_const_
    check_mat = check_mat_    
            
#    print('Dario ', x,y, K_0[x],K_1[y],T_x.f_uniq[0] - T_y.f_uniq[0], np.abs(T_y.f_uniq[0]-T_x.f_uniq[0]))

    
    """
    Quando len(avaiable_x)*len(avaiable_y)==0 potrei avere un problema, perché io sto sempre supponendo che le radici
    vengano accoppiate tra di loro. Ma poi quando calcolo le distanze tra i subtrees in questo caso particolare, non succede!
    
    Ora ho cambiato.
    """
    
    available_x = T_x.vertices[:-1]
    available_x_aux = T_x.name_vertices[:-1]

    available_y = T_y.vertices[:-1]
    available_y_aux = T_y.name_vertices[:-1]

    for i,v0 in enumerate(available_x_aux):
        for j,v1 in enumerate(available_y_aux):
            W_aux[i,j]=W[v0,v1]
             
    if len(avaiable_x)*len(avaiable_y)>0:
        new_W = make_model(x,y,T_x,T_y,W_aux,avaiable_x,avaiable_y,T_x.dim-1,T_y.dim-1,
                                       m_x,m_y,
                                       W_0,W_1,V_0,V_1,binary,True,False)
        return new_W    
    elif len(avaiable_x)>0:
        M={}
#        return np.max([K_0[x], np.abs(T_y.f_uniq[0]-T_x.f_uniq[-1])])
        return np.max([K_0[x], np.abs(T_y.f_uniq[0]-T_x.f_uniq[0])])
    elif len(avaiable_y)>0:
        M={}
#        return np.max([K_1[y], np.abs(T_x.f_uniq[0] - T_y.f_uniq[-1])]) 
        return np.max([K_1[y], np.abs(T_y.f_uniq[0]-T_x.f_uniq[0])]) 
    else:
        M={}
        return np.abs(T_x.f_uniq[-1] - T_y.f_uniq[-1])

    
def calculate_Wxy(T_0,T_1,W,x,y,
                  SUB_0,SUB_1,W_0,W_1,V_0,V_1,K_0,K_1,binary):
    
    T_x=SUB_0[x]
    T_y=SUB_1[y]
    
    m_x=[SUB_0[i].f_uniq[0] for i in V_0[x]]
    m_y=[SUB_1[j].f_uniq[0] for j in V_1[y]]

    W_aux=np.zeros((T_x.dim-1,T_y.dim-1)).astype(float)
    
    available_x = T_x.vertices[:-1]
    available_x_aux = T_x.name_vertices[:-1]

    available_y = T_y.vertices[:-1]
    available_y_aux = T_y.name_vertices[:-1]

    for i,v0 in enumerate(available_x_aux):
        for j,v1 in enumerate(available_y_aux):
            W_aux[i,j]=W[v0,v1]
 
    if len(avaiable_x)*len(avaiable_y)>0:
        new_W = make_model(x,y,T_x,T_y,W_aux,avaiable_x,avaiable_y,T_x.dim-1,T_y.dim-1,
                                       m_x,m_y,
                                       W_0,W_1,V_0,V_1,binary,True,False)
        return new_W
    elif len(avaiable_x)>0:
        M={}
        return np.max([K_0[x], np.abs(T_y.f_uniq[0]-T_x.f_uniq[0])])
    elif len(avaiable_y)>0:
        M={}
        return np.max([K_1[y], np.abs(T_y.f_uniq[0]-T_x.f_uniq[0])]) 
    else:
        M={}
        return np.abs(T_x.f_uniq[0] - T_y.f_uniq[0])

def calculate_H(T_0,T_1,SUB_0,SUB_1):
    
    H=np.zeros((T_0.dim,T_1.dim)).astype(float)

    """
    Le foglie che devo cancellare se prendo x (resp. y) come root: quelle che non stanno sotto x (resp.y)
    """
    LEAVES_0 = [[l for l in T_0.leaves if not l in SUB_0[x].name_vertices] for x in T_0.vertices[:-1]]
    LEAVES_0.append([-1]) # Aggiungo per far girare il codice, ma non dovrebbe cambiare nulla
    LEAVES_1 = [[l for l in T_1.leaves if not l in SUB_1[y].name_vertices] for y in T_1.vertices[:-1]]
    LEAVES_1.append([-1]) # Aggiungo per far girare il codice, ma non dovrebbe cambiare nulla
    
    """
    Il costo massimo di cancellare in due fasi le foglie che non stanno sotto x (resp. y)
    """
    H_0 = [np.max([0]+[calculate_merging_heigh(T_0,x,v)- T_0.f_uniq[v]\
                   for v in LEAVES_0[x]]) for x in T_0.vertices[:-1]]
    H_0.append(0) # Aggiungo per far girare il codice, ma non dovrebbe cambiare nulla
    
    H_1 = [np.max([0]+[calculate_merging_heigh(T_1,y,w)- T_1.f_uniq[w]\
                   for w in LEAVES_1[y]]) for y in T_1.vertices[:-1]]
    H_1.append(0) # Aggiungo per far girare il codice, ma non dovrebbe cambiare nulla
    
    for x in T_0.vertices:
        for y in T_1.vertices:
            
            if len(LEAVES_0[x])>0:
                """
                Il costo di mandare la foglia non sotto x (risp y) più bassa, 
                in quella più bassa sotto y (risp x)
                """
                tmp_0 = T_1.f_uniq[0]-T_0.f_uniq[LEAVES_0[x][0]]
            else:
                tmp_0 = 0
            
            if len(LEAVES_1[y])>0:
                tmp_1 = T_0.f_uniq[0]-T_1.f_uniq[LEAVES_1[y][0]]
            else:
                tmp_1= 0
            
            tmp = np.max([H_0[x]/2,H_1[y]/2,tmp_0,tmp_1])
            
            H[x,y]=tmp
    
    return H
    
    
    
def calculate_merging_heigh(T,x,v):
    """
    T: albero
    x: il punto che tengo fisso
    v: da v risalgo fino a che trovo LCA(x,v)
    """
    for p in T.paths[v,:]:
        if p in T.paths[x,:]:
            q = p
            break
    if q==-1:
        return []
    else:
        return T.f_uniq[q]
    

def cost_eval(i_idx,j_idx,x,y,n,m,W,L,u_x,u_y,m_x,m_y,
                      W_x,f_x,paths_x,len_paths_x,
                      W_y,f_y,paths_y,len_paths_y,
                      W_0,W_1,V_0,V_1,root=True):
        
    T_x = {}
    T_y = {}
    B_x=[[-1]]
    B_y=[[-1]]
    
    T_s_i = []
    T_A_i = []
    T_u_i = []
    T_B_i = []
        
    
#     if i_idx is not None:
#         i=i_idx
    for i in range(n-int(root)):
        aux=np.where(paths_x==i) # tutti i paths verso la roots che passano per i
        """
        Variabili Ausiliarie -> guardo se sotto o sopra di lui ho dei punti accoppiati. 
                                Se non li ho vuol dire che devo cancellare.
        """
        vertices = []
        for w in aux[0]:
            path = paths_x[w,:len_paths_x[w]-1]
            vertices = vertices + list(path)

        vertices = np.unique(vertices) 
        d_i = 0
        for w in vertices:
            for j in range(m):
                d_i += L[w,j] # se è 0 cancello. 
        """
        Costo dei matching -> costo tra i subtrees già calcolato
        """
        s_i = [L[i,j]*(W[i,j]) \
                   for j in range(m)]
        
        """
        Costo della deletion Lambda(i)=0, caso semplice: guardo se ci sono foglie accoppiate 
        lungo i paths dalle foglie di sub(v) alla root. Se non ci sono (i.e. d_i=0) cancello metà di 
        sub(v) e (v,father(v))
        """ 
        f_i = paths_x[i][1]
        # W_x[i,paths_x[i][1] = f(father(i))-f(i) 
        # W_0[V_0[x][i]] = min f-f(i) in sub(i) 
        A_i = 0.5*(f_x[f_i]-m_x[i])*(1-d_i) # ho costo positivo sse d_i=0 
                       
        """
        Costo della deletion Lambda(i)=0, caso complicato: prendo father(v) e tutte le 
        foglie in sub(father(v)). Per ogni foglia l guardo se un punto tra l e father(v) 
        è assegnato, e nel caso, prendo il punto più basso cui è assegnato un punto nel 
        suo sottoalbero.
        """
        B_i = [] 

        if approx_=='up' or approx_ =='middle':

            if f_i == x:
                vertices_f_i = np.arange(n)
            else:
                vertices_f_i = np.where(paths_x==f_i)[0]

            vertices_f_i = np.array([p for p in vertices_f_i if p<f_i])

            for v in vertices_f_i:
                """
                OSS: qua non funziona: il <tmp> max non è per forza quello che sto cercando!!!!!!!!
                """
                tmp = sum([L[v,j]*(m_y[j]-m_x[i]) for j in range(m)])-K_const*d_i
                B_i.append(tmp)

            if len(B_i) >0:
                B_x.append(B_i)

        """
        Costo della deletion Lambda(i)>1: max{f(r_T),g(r_G)}-f(x)
        """    
        if approx_ == 'middle':
            u_i = np.abs(f_x[paths_x[i,1]]-f_x[i])*u_x[i]
        else:
            u_i = np.abs(f_y[-1]-f_x[i])*u_x[i]
            
        T_s_i.append(sum(s_i))
        T_A_i.append(A_i)
        T_u_i.append(u_i)
        T_B_i.append(max([max(b) for b in B_x]))
        
    """
    Creo il dizionario per x
    """    
    T_x['s_x'] = max(T_s_i)
    T_x['A_x'] = max(T_A_i)
    T_x['u_x'] = max(T_u_i)
        
#     if j_idx is not None:
#         j=j_idx
    T_s_j = []
    T_A_j = []
    T_u_j = []
    T_B_j = []
    
    for j in range(m-int(root)):
        aux=np.where(paths_y==j) # tutti i paths verso la roots che passano per j

        vertices = []
        for w in aux[0]:
            path = paths_y[w,:len_paths_y[w]-1]
            vertices = vertices + list(path)

        vertices = np.unique(vertices) 
        d_j = 0
        for w in vertices:
            for i in range(n):
                d_j += L[i,w] # se è 0 cancello. 

        s_j = [L[i,j]*(W[i,j]) \
                   for i in range(n)]
        f_j = paths_y[j][1]
        A_j = 0.5*(f_y[f_j]-m_y[j])*(1-d_j) # ho costo positivo sse d_j=0    
        
        B_j = []
        
        if approx_=='up' or approx_ =='middle':
            if f_j == y:
                vertices_f_j = np.arange(m)
            else:
                vertices_f_j = np.where(paths_y==f_j)[0]

    #             vertices_f_j = np.array(list(set(vertices_f_j)-set(aux[0])))
            vertices_f_j = np.array([q for q in vertices_f_j if q<f_j])

            for w in vertices_f_j:
                tmp = sum([L[i,w]*(m_x[i]-m_y[j]) for i in range(n-1)])-K_const*d_j 
                B_j.append(tmp)

            if len(B_j)>0:
                B_y.append(B_j)
                
        if approx_ == 'middle':
            u_j = np.abs(f_y[paths_y[j,1]]-f_y[j])*u_y[j]
        else:
            u_j = np.abs(f_x[-1]-f_y[j])*u_y[j]

#        if j == -6:
#            print('Mariella ', u_j, u_y[j],f_y[f_j],m_y[f_j])

        T_s_j.append(sum(s_j))
        T_A_j.append(A_j)
        T_u_j.append(u_j)
        T_B_j.append(max([max(b) for b in B_y]))
        
    """
    Creo il dizionario per y
    """    
    T_y['s_y'] = max(T_s_j)
    T_y['A_y'] = max(T_A_j)
    T_y['u_y'] = max(T_u_j)
 
    T_x['B_x'] = max(T_B_i)
    T_y['B_y'] = max(T_B_j)
#     return T_x,B_x, T_y, B_y     
    return T_x, T_y
  
    
def sym_objective_fun(x,y,n,m,W,L,u_x,u_y,m_x,m_y,
                      W_x,f_x,paths_x,len_paths_x,
                      W_y,f_y,paths_y,len_paths_y,
                      W_0,W_1,V_0,V_1,root=True):
    
        
    T_x = []
    T_y = []
    B = []
    
    """
    Per ogni punto di T_x devo prendere
    """
    for i in range(n-int(root)):

        aux=np.where(paths_x==i) # tutti i paths verso la roots che passano per i
        
        """
        Variabili Ausiliarie -> guardo se sotto o sopra di lui ho dei punti accoppiati. 
                                Se non li ho vuol dire che devo cancellare.
        """
        vertices = []
        for w in aux[0]:
            path = paths_x[w,:len_paths_x[w]-1]
            vertices = vertices + list(path)
               
        vertices = np.unique(vertices)
        
        d_i = 0
        
        for w in vertices:
            for j in range(m):
                d_i += L[w,j] # se è 0 cancello. 
        """
        Costo dei matching -> costo tra i subtrees già calcolato
        """
        s_i = [L[i,j]*(W[i,j]) \
                   for j in range(m)]
        """
        Costo della deletion Lambda(i)=0, caso semplice: guardo se ci sono foglie accoppiate 
        lungo i paths dalle foglie di sub(v) alla root. Se non ci sono (i.e. d_i=0) cancello metà di 
        sub(v) e (v,father(v))
        """ 
        # W_x[i,paths_x[i][1] = f(father(i))-f(i) 
        # W_0[V_0[x][i]] = min f-f(i) in sub(i) 
        f_i = paths_x[i][1]
        A_i = 0.5*(f_x[f_i]-m_x[i])*(1-d_i) # ho costo positivo sse d_i=0          
        
        """
        Costo della deletion Lambda(i)=0, caso complicato: prendo father(v) e tutte le 
        foglie in sub(father(v)). Per ogni foglia l guardo se un punto tra l e father(v) 
        è assegnato, e nel caso, prendo il punto più basso cui è assegnato un punto nel 
        suo sottoalbero.
        """
        B_i = [] 
        
        if approx_=='up' or approx_ =='middle':
            if f_i == x:
                vertices_f_i = np.arange(n)
            else:
                vertices_f_i = np.where(paths_x==f_i)[0]

            vertices_f_i = np.array([p for p in vertices_f_i if p<f_i])

            for v in vertices_f_i:
                tmp = sum([L[v,j]*(m_y[j]-m_x[i]) for j in range(m)])-K_const*d_i
                B_i.append(tmp)
                        
            if len(B_i) >0:
                B.append(B_i)
            
        """
        Costo della deletion Lambda(i)>1: max{f(r_T),g(r_G)}-f(x)
        """ 
        
        if approx_ == 'middle':
            u_i = np.abs(f_x[paths_x[i,1]]-f_x[i])*u_x[i]
        else:
            u_i = np.abs(f_y[-1]-f_x[i])*u_x[i]

 
        """
        Creo il vettorone per x
        """    
        T_x = T_x + [sum(s_i),u_i,A_i] 

    for j in range(m-int(root)):

        aux=np.where(paths_y==j) # tutti i paths verso la roots che passano per j
        
        vertices = []
        for w in aux[0]:
            path = paths_y[w,:len_paths_y[w]-1]
            vertices = vertices + list(path)
               
        vertices = np.unique(vertices) 
        d_j = 0
        for w in vertices:
            for i in range(n):
                d_j += L[i,w] # se è 0 cancello. 
                
        s_j = [L[i,j]*(W[i,j]) \
                   for i in range(n)]

        f_j = paths_y[j][1]
        A_j = 0.5*(f_y[f_j]-m_y[j])*(1-d_j) # ho costo positivo sse d_j=0    

        B_j = []

        if approx_=='up' or approx_ =='middle':
            if f_j == y:
                vertices_f_j = np.arange(m)
            else:
                vertices_f_j = np.where(paths_y==f_j)[0]

#             vertices_f_j = np.array(list(set(vertices_f_j)-set(aux[0])))
            vertices_f_j = np.array([q for q in vertices_f_j if q<f_j])

            for w in vertices_f_j:
                tmp = sum([L[i,w]*(m_x[i]-m_y[j]) for i in range(n)])-K_const*d_j 
                B_j.append(tmp)
            
            if len(B_j)>0:
                B.append(B_j)
                
        if approx_ == 'middle':
            u_j = np.abs(f_y[paths_y[j,1]]-f_y[j])*u_y[j]
        else:
            u_j = np.abs(f_x[-1]-f_y[j])*u_y[j]
                    
        T_y = T_y + [sum(s_j),u_j,A_j]
        
    return T_x + T_y, B 

def make_poly(x,y,n,m,W,L,u_x,u_y,m_x,m_y,
              W_x,f_x,paths_x,len_paths_x,
              W_y,f_y,paths_y,len_paths_y,
              W_0,W_1,V_0,V_1,root=False):
    
    out,B=sym_objective_fun(x,y,n,m,W,L,u_x,u_y,m_x,m_y,
                          W_x,f_x,paths_x,len_paths_x,
                          W_y,f_y,paths_y,len_paths_y,
                          W_0,W_1,V_0,V_1,root=False)    
    return out,B



def make_model(x,y,T_x,T_y,W,avaiable_x,avaiable_y,n,m,
               m_x,m_y,
               W_0,W_1,V_0,V_1,binary,no_r,root=False):
    
    t0 = time.time()
    
    cost = pyo.ConcreteModel()
    cost.costr=pyo.ConstraintList()
  
    cost.n=n+1
    cost.m=m+1
    
    if binary:
        cost.L = pyo.Var(np.arange(cost.n),np.arange(cost.m), 
                 domain=pyo.Binary,initialize=0)
        cost.u_x = pyo.Var(np.arange(cost.n), 
                 domain=pyo.Binary,initialize=0)
        cost.u_y = pyo.Var(np.arange(cost.m), 
                 domain=pyo.Binary,initialize=0)
    else:
        cost.L = pyo.Var(np.arange(cost.n),np.arange(cost.m), 
                 domain=pyo.NonNegativeReals,initialize=0)
        cost.u_x = pyo.Var(np.arange(cost.n), 
                 domain=pyo.NonNegativeReals,initialize=0)
        cost.u_y = pyo.Var(np.arange(cost.m), 
                 domain=pyo.NonNegativeReals,initialize=0)
    cost.L[n,m]=1

    for i in np.arange(cost.n):
        cost.L[i,m].fixed=True
        
    for i in T_x.leaves:
        cost.u_x[i].fixed=True
    
    for j in np.arange(cost.m):
        cost.L[n,j].fixed=True

    for j in T_y.leaves:
        cost.u_y[j].fixed=True
        
    cost.aux = pyo.Var(domain=pyo.NonNegativeReals,initialize=0)

    def objective_poly(cost):        
        out,B=make_poly(x,y,n,m,W,cost.L,cost.u_x,cost.u_y,
                       m_x,m_y,
                       T_x.weights,T_x.f_uniq,T_x.paths,T_x.len_paths,
                       T_y.weights,T_y.f_uniq,T_y.paths,T_y.len_paths,
                       W_0,W_1,V_0,V_1,root)       
        return out,B
    
    out,B = objective_poly(cost)
                        
    if approx_ == 'up':
        for b in B:
            out = out+b        
        
    for tmp in out:
        cost.costr.add(cost.aux >= tmp)   

    def objective_wrap(cost):
        return cost.aux
        
    cost.obj=pyo.Objective(rule=objective_wrap, sense=pyo.minimize)
    make_costraints(cost,T_x,T_y,no_r)

    solver_ = 'gurobi'
    
    solver = pyo.SolverFactory(solver_)
    #solver = pyo.SolverFactory('cplex', 
     #                         executable='/mnt/d/Documents/CPLEX_students/cplex/bin/x86-64_linux/cplex')

    t1 = time.time()

    S = solver.solve(cost)
    
    t2 = time.time()
    
    global T_build
    global T_solve    
    
    T_build = T_build + t1-t0
    T_solve = T_solve + t2-t1

    
    from pyomo.opt import SolverStatus, TerminationCondition

    
    new_W = cost.obj()   
    
    root_edit = np.abs(T_x.f_uniq[-1]-T_y.f_uniq[-1])
    w_xy = np.max([new_W,root_edit])
    
#    print('Dario ', S.solver.status,w_xy)
    
#    if x==8 and y==8:
#        cost.pprint()

    
    if check_ :
        
        L_eval=np.zeros((cost.n,cost.m))
        for key,val in cost.L.extract_values().items():
            L_eval[key]=val
            
        u_x_eval=np.zeros((cost.n))
        for key,val in cost.u_x.extract_values().items():
            u_x_eval[key]=val    

        u_y_eval=np.zeros((cost.m))
        for key,val in cost.u_y.extract_values().items():
            u_y_eval[key]=val 
        
        C_x,C_y = cost_eval(1,None,x,y,n,m,W,L_eval,u_x_eval,u_y_eval,
                       m_x,m_y,
                       T_x.weights,T_x.f_uniq,T_x.paths,T_x.len_paths,
                       T_y.weights,T_y.f_uniq,T_y.paths,T_y.len_paths,
                       W_0,W_1,V_0,V_1,root)

        couples = np.where(L_eval==1)
        pairs = [[couples[0][i],couples[1][i]] for i in range(len(couples[0]))]
        
#        print('\nMario ', x,y, L_eval.shape, pairs, w_xy, root_edit, C_x,C_y)
                
        if len(check_mat)>0 and w_xy == max([C_x['B_x'],C_y['B_y']]) and w_xy>max([C_x['s_x'],C_y['s_y'],
                                                              C_x['A_x'],C_y['A_y'],
                                                              C_x['u_x'],C_y['u_y']]):
            check_mat[x,y]=0
        else:
            check_mat[x,y]=1
            
            
    return w_xy
    
    
        
#         L_eval=np.zeros((cost.n,cost.m))

#         min_x = []
#         min_y = []

#         for key,val in cost.L.extract_values().items():
#             L_eval[key]=val
#             min_x.append(T_y.f_uniq[key[1]]) 
#             min_y.append(T_x.f_uniq[key[0]]) 
    
#     if x==40 and y==40:
#         L_eval=np.zeros((cost.n,cost.m))
#         for key,val in cost.L.extract_values().items():
#             L_eval[key]=val    

#         print('prova:',new_W,'\n', L_eval,'\n', T_x.f_uniq,'\n', T_y.f_uniq)
#     T_x.plot_newick()
#     T_y.plot_newick()
    
#     return new_W


def calculate_opt(T_0,T_1,W,W_0,W_1,V_0,V_1):

    avaiable_0=set(T_0.name_vertices)
    avaiable_1=set(T_1.name_vertices)
    
    new_W=make_model(max(avaiable_0),max(avaiable_1,),
                     T_0,T_1,W,avaiable_0,avaiable_1,T_0.dim,T_1.dim,
                     W_0,W_1,V_0,V_1,False,True)

    return new_W

def make_costraints(model,T_x,T_y,no_r,rows=True):
    """
    Devo mettere queste condizioni:
    1) per ogni punto, uno solo dei suoi l_x[i] può essere assegnato
    2) lungo un percorso da foglia a radice, posso avere al massimo un punto assegnato.
       Così metto anche che posso assegnarlo a massimo un punto!
    3) bisogna mettere qualcosa che regoli i ghost! se un punto ha 2 figli non ghostati
       non può essere ghost!
       Per ogni punto devo summare lungo il percorso da lui a root, tutti i
       cammini che passano per di lui!
       
       Voglio che se ho un ghost, allora sotto di lui gli altri sono morti.
       Voglio che in un punto 1-la somma degli l_x[i,j,k] con k>0 sia maggiore della somma
       dei l_x[i_,j_,k_] dei punti sotto. 
       Questo dovrebbe sistemare tutti i problemi dei ghost: infatti se un punto sotto
       ha un l_x[i,j,k'] con k'>k, allora in quel punto sta violando la regola perchè sotto di lui ha una cosa non zero!
    4) appaio il vertice più basso
       
    """

    """
    Lungo il percorso da ogni foglia alla root esclusa voglio massimo una assegnazione!
    """
           
    root=1-int(no_r)
            
    for v in T_x.leaves:
        AUX = 0
        for v_aux in T_x.paths[v,:T_x.len_paths[v]-1]:
            for j in range(model.m-1):
                AUX += model.L[v_aux,j]
        model.costr.add(AUX <= 1)
       
    for v in T_y.leaves:
        AUX = 0
        for v_aux in T_y.paths[v,:T_y.len_paths[v]-1]:
            for i in range(model.n-1):
                AUX += model.L[i,v_aux]
        model.costr.add(AUX <= 1)

    """
    Ora mancano i constraints per u_x e u_y
    """
    
    K = model.n-1
    m = 1/K
    q = -m-0.00001
    
    for i in T_x.vertices[:-1]:
        aux=np.where(T_x.paths==i)[0] # tutti i paths verso la roots che passano per i
        lambda_i = sum([model.L[i_aux,j] for i_aux in aux\
                        for j in range(model.m-1)])
        model.costr.add(model.u_x[i] <= 0.5*lambda_i) 
        if len(T_x.leaves)>1:
            model.costr.add(model.u_x[i] >= m*lambda_i+q) 
       
    K = model.m-1
    m = 1/K
    q = -m-0.00001

    for j in T_y.vertices[:-1]:
        aux=np.where(T_y.paths==j)[0] # tutti i paths verso la roots che passano per i
        lambda_j = sum([model.L[i,j_aux] for j_aux in aux\
                        for i in range(model.n-1)])
        model.costr.add(model.u_y[j] <= 0.5*lambda_j) 
        if len(T_y.leaves)>1:
            model.costr.add(model.u_y[j] >= lambda_j*m+q) 
    
    """
    Costraints per appaiare sempre il vertice piu' basso!
    """
    if approx_ == 'up':
        path = T_x.paths[0,:T_x.len_paths[0]-1]
        U_T = sum([model.L[i,j] for j in range(model.m-1) for i in path])
        model.costr.add(U_T >= 1)
        path = T_y.paths[0,:T_y.len_paths[0]-1]           
        U_G = sum([model.L[i,j] for i in range(model.n-1) for j in path])
        model.costr.add(U_G >= 1)
    
                
def make_W(T_0,T_1,SUB_0,SUB_1,W_0,W_1,V_0,V_1,K_0,K_1,binary,MP):

    W=np.zeros((T_0.dim,T_1.dim)).astype(float)-1

    global check_mat 
    
    if check_:
        check_mat = np.ones((T_0.dim,T_1.dim)).astype(float)
    else:
        check_mat = 0


    N_0=np.max(T_0.len_paths)
    N_1=np.max(T_1.len_paths)
    
    vertices_0=set(T_0.vertices.copy())
    vertices_1=set(T_1.vertices.copy())
    
    cnt_0=0
    cnt_1=0

    lvl_0=[0]
    lvl_1=[0]

    while cnt_0<N_0 or cnt_1<N_1:

        for v in vertices_0:
            if T_0.len_paths[v]>=N_0-cnt_0:
                lvl_0.append(v)
                vertices_0=vertices_0-set([v])
                
        for v in vertices_1:
            if T_1.len_paths[v]>=N_1-cnt_1:
                lvl_1.append(v)
                vertices_1=vertices_1-set([v])

        if cnt_0==0 and cnt_1==0:        
            lvl_0=lvl_0[1:]
            lvl_1=lvl_1[1:]
        
        cnt_0+=1
        cnt_1+=1

        if MP: 
            
            couple_aux = [(v0,v1) for v0 in sorted(lvl_0) for v1 in sorted(lvl_1)
                                 if W[v0,v1]==-1]

            pool = mp.Pool(processes=7)

            RESULTS=pool.map(calculate_Wxy_pool,([T_0,T_1,W,v[0],v[1],
                                               SUB_0,SUB_1,W_0,W_1,V_0,V_1,K_0,K_1,binary,approx_,K_const,check_,check_mat] 
                                     for v in couple_aux))
            pool.close()

            cnt_aux=0
            for v0 in sorted(lvl_0):
                for v1 in sorted(lvl_1):
                    if W[v0,v1]==-1:
                        W[v0,v1] = RESULTS[cnt_aux]
                        cnt_aux+=1
        else:
            for v0 in sorted(lvl_0):
                for v1 in sorted(lvl_1):
                    if W[v0,v1]==-1:
                        results=calculate_Wxy(T_0,T_1,W,v0,v1,
                                               SUB_0,SUB_1,W_0,W_1,V_0,V_1,K_0,K_1,binary) 
                                
                        
                        W[v0,v1] = results
                        
    return W

def make_sub_trees(T):
    SUB=[]  
    V=[]
    W=[]
    K = []
    
    for v in T.vertices:
        father = T.find_father(v)[0]
                
        T_aux = T.sub_tree(v)
        V.append(T.sub_tree(v).name_vertices)
            
        tmp = np.array([calculate_merging_heigh(T_aux,0,x)-T_aux.f_uniq[x] for x in T_aux.leaves])
#        tmp = [T.f_uniq[v]-T_aux.f_uniq[0]] 
        K.append(np.max(tmp)/2)
        lenghts = []
        for i in T_aux.leaves:
            l = 0
            for i_aux in T_aux.paths[i,:T_aux.len_paths[i]-1]:
                j = T_aux.find_father(i_aux)[0]
                l = l + T_aux.weights[i_aux,j]
            lenghts.append(l)
            
        SUB.append(T_aux)   
        
        aux = np.max(lenghts)       
        W.append(aux)
                           
    return SUB,V,W,K


def interleaving(T_0,T_1, binary = False, approx = 'up', MP = False,
                 check=False,verbose = False):

    """
    approx = up -> upper bound
    approx = down -> lower bound
    approx = middle -> deletion con #Lambda=1 contraggo solo (x,father(x))
    
    check = serve per vedere se l'upper bound sulle deletion con #Lambda=0 é esatto o meno
    """
    
    global T_build
    global T_solve
    global approx_
    global K_const
    global verbose_
    global check_
    
    verbose_ = verbose
    approx_ = approx
    check_ = check
        
    K_const = np.max([np.abs(T_0.f_uniq[0]-T_1.f_uniq[-1]),
                np.abs(T_0.f_uniq[-1]-T_1.f_uniq[0]),
                T_0.f_uniq[-1]-T_0.f_uniq[0],
                T_1.f_uniq[-1]-T_1.f_uniq[0]])
    
    T_build = 0
    T_solve = 0
    
    SUB_0,V_0,W_0,K_0=make_sub_trees(T_0)
    SUB_1,V_1,W_1,K_1=make_sub_trees(T_1)

    W = make_W(T_0,T_1,SUB_0,SUB_1,W_0,W_1,V_0,V_1,K_0,K_1, binary=binary, MP=MP)
    
 #   print('Ontario ', W)
    
    H = calculate_H(T_0,T_1,SUB_0,SUB_1)
    
    COST = np.maximum(W,H)    
    cost = np.min(COST)
    
    if MP == True and verbose == True:
        out = {}
        out['d']= cost
        out['W']= W
        out['H']= H
        
    elif MP == False and verbose == True:
        print('\nTempo speso per costruire il modello: ', T_build)
        print('Tempo speso per risolvere il modello: ', T_solve)
        
        out = {}
        out['d']= cost
        out['T_build'] = T_build
        out['T_solve'] = T_solve

        if check_:
            flag = np.max([check_mat[p[0],p[1]] for p in np.argwhere(COST==cost)])            
            out['flag'] = flag
    else:
        if check_:
            flag = np.max([check_mat[p[0],p[1]] for p in np.argwhere(COST==cost)])
            out = {}
            out['d']= cost
            out['flag'] = flag
        else:
            out = cost
            
    return out


def interl_pruning(T,G,max_leaves=20,keep_root=False,return_eps = False, 
                   binary = True, approx = 'up', MP = True,check = False, verbose = False):
    
    T_aux,eps_T = prune_dendro_N(T,max_leaves,keep_root=keep_root,
                                 return_eps = return_eps, approx=False)  
    G_aux,eps_G = prune_dendro_N(G,max_leaves,keep_root=keep_root,
                                 return_eps = return_eps, approx=False)  
    
    d = interleaving(T_aux,G_aux,binary = binary, approx = approx, MP = MP,check=check, verbose = verbose)
    
    if return_eps:        
        out = {}
        if len(d)>1:
            out['d']=d['d']
            out['flag'] = d['flag']
        else:
            out['d']=d
            
        out['eps'] = max([eps_T,eps_G])/2
        out['T_leaves'] = len(T_aux.leaves)
        out['G_leaves'] = len(G_aux.leaves)
        
    else:
        out = d   
    return out
    
def interl_approx(T, G, N = 100, max_leaves = 20, 
                  return_bound = False,
                  binary = True, approx = 'up', MP = True, check = False,
                  verbose = False):
    
    grid = np.linspace(0,1,N)
    eps = (1-grid)**6
    
    if (len(T.leaves)<max_leaves or len(G.leaves)<max_leaves):
        eps = [0]
    else:
        M = np.max([T.f_uniq[-1]-T.f_uniq[0],G.f_uniq[-1]-G.f_uniq[0]])
        eps = eps*M
        
    thresh = eps[0]
    d = 0
    cnt = -1
    
    T_aux = prune_dendro(T,thresh,keep_root=False)
    G_aux = prune_dendro(G,thresh,keep_root=False)
    
    tmp_T = 0
    tmp_G = 0
    
    while cnt+1<len(eps) and (len(T_aux.leaves)<max_leaves or len(G_aux.leaves)<max_leaves):
        cnt+=1
        thresh = eps[cnt] 
        
        T_aux = prune_dendro(T,thresh,keep_root=False)
        G_aux = prune_dendro(G,thresh,keep_root=False)      

    idx = cnt-1
    thresh = eps[idx]
    T_aux = prune_dendro(T,thresh,keep_root=False)
    G_aux = prune_dendro(G,thresh,keep_root=False)
    
    if verbose:
        print('\nComputing the distance between trees with ',len(T_aux.leaves),' and ',len(G_aux.leaves),
              ' leaves; the pruning threshold is: ',thresh)
    
    d = interleaving(T_aux,G_aux,binary = binary, approx = approx, MP = MP, check=check, verbose = False)
        
    if return_bound:
        idx = cnt-1
        thresh = eps[idx]
        T_aux = prune_dendro(T,thresh,keep_root=False)
        G_aux = prune_dendro(G,thresh,keep_root=False)
        
        out = {}
        if check:
            out['d']=d['d']
            out['flag'] = d['flag']
        else:
            out['d']=d
            
        out['eps'] = thresh/2
        out['T_leaves'] = len(T_aux.leaves)
        out['G_leaves'] = len(G_aux.leaves)
        
    else:
        if check:
            out = {}
            out['d']=d['d']
            out['flag'] = d['flag']
            d = d['d']
        out = np.max([thresh/2,d])
    
    return out
    
    