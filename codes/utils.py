import numpy as np
import sympy as sp
from sympy import lambdify,symbols,integrate,solve
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize,curve_fit
import torch
import math
from tqdm import tqdm
x, C,A,B = symbols('x C A B')

def read_data_unsoda(file_names):
    theta = []
    psi = []
    ts = []
    tr = []
    for file_name in tqdm(file_names):
        file_path = f'unsoda_data_auto/{file_name}'
        # print(file_name)
        data = pd.ExcelFile(file_path)
        # Load data from the first sheet (assuming it contains the necessary data)
        df = data.parse(data.sheet_names[0])
        theta.append(df['t-h(t)'].dropna().values)
        psi.append(df['t-h(h)'].dropna().values)
        ts.append(df['ts'].dropna().values)
        tr.append(df['tr'].dropna().values)

    return theta, psi, ts, tr
def read_data():
    file_names = os.listdir('data for saturated flow')
    # file_path = 'data for saturated flow/Beit_Netofa_Clay.xlsx'
    theta=[]
    psi=[]
    ts=[]
    tr=[]

    for file_name in file_names:
        file_path = f'data for saturated flow/{file_name}'
        #print(file_name)
        data = pd.ExcelFile(file_path)
        # Load data from the first sheet (assuming it contains the necessary data)
        df = data.parse(data.sheet_names[0])
        # h = df['h-K(h)']
        # K_h = df['h-K(K)']
        theta.append(df['t-h(t)'].values)
        psi.append(df['t-h(h)'].values)
        ts.append(df['ts'].dropna().values)
        tr.append(df['tr'].dropna().values)
    return theta,psi,ts,tr

def read_h_K(file_name):
    file_path = f'data for saturated flow/{file_name}'
    # print(file_name)
    data = pd.ExcelFile(file_path)
    # Load data from the first sheet (assuming it contains the necessary data)
    df = data.parse(data.sheet_names[0])
    # h = df['h-K(h)']
    # K_h = df['h-K(K)']
    h=df['h-K(h)'].dropna().values
    K=df['h-K(K)'].dropna().values
    return h,K
def read_h_K_unsoda(file_name):
    file_path = f'unsoda_data_auto/{file_name}'
    # print(file_name)
    data = pd.ExcelFile(file_path)
    # Load data from the first sheet (assuming it contains the necessary data)
    df = data.parse(data.sheet_names[0])
    # h = df['h-K(h)']
    # K_h = df['h-K(K)']
    h=df['h-K(h)'].dropna().values
    K=df['h-K(K)'].dropna().values
    K=np.array(K)/np.max(K)
    new_h = []
    new_K = []
    for i in range(h.shape[0]):
        if (h[i] != 0):
            new_h.append(h[i])
            new_K.append(K[i])
    return np.array(new_h),np.array(new_K)
def calculate_Kr(expr,x,postive_index_begin,positive_index_end):
    #keep larger than 0

    numerator_integral = integrate(expr, (x, postive_index_begin, x))
    denominator_integral = integrate(expr, (x, postive_index_begin, positive_index_end))
    Kr=x**(0.5)*(numerator_integral/denominator_integral)**2
    return Kr

def get_non_positive_idds(t,arr):
    index_begin=0
    index_end=arr.shape[0]-1
    for i in range(arr.shape[0]):
        if arr[i]>0:
            index_begin=i
            break

    for i in range(arr.shape[0]):
        if arr[arr.shape[0]-i-1]>0:
            index_end=arr.shape[0]-i-1
            break


    return t[index_begin],t[index_end]


def filter_small_theta(theta,psi):
    new_theta=[]
    new_psi=[]
    for i in range(theta.shape[0]):
        if (theta[i]>=0.01) and (theta[i]<=0.99) and (psi[i]!=0):
            new_theta.append(theta[i])
            new_psi.append(psi[i])
    return np.array(new_theta),np.array(new_psi)

def plot_data(theta_ob,theta_pred,observation_psi,pred_psi):
    # Plot t vs h(t)
    plt.figure()
    plt.scatter(theta_ob, observation_psi, marker='o', label="h(t)-t", color="orange")
    plt.plot(theta_pred, pred_psi, label="h(t)-t", color="black")
    #plt.xscale("log")
    plt.xlabel("theta")
    plt.ylabel("h")
    plt.title("t vs h(t)")
    plt.xlim([0, 1])
    #plt.xlim([1e0, 1e5])
    plt.legend()
    plt.grid()
def plot_free_data(theta_ob,theta_pred,observation_psi,pred_psi):
    # Plot t vs h(t)
    plt.figure()
    plt.scatter(theta_ob, observation_psi, marker='o', label="h(t)-t", color="orange")
    plt.plot(theta_pred, pred_psi, label="h(t)-t", color="black")
    #plt.xscale("log")
    plt.xlabel("theta")
    plt.ylabel("h")
    plt.title("t vs h(t)")
    #plt.xlim([1e0, 1e5])
    plt.legend()
    plt.grid()
def plot_h_theta(psi,psi_for_pred,observation_theta, pred_theta):
    plt.figure()
    plt.scatter(psi, observation_theta, marker='o', label="h(t)-t", color="orange")
    plt.plot(psi_for_pred, pred_theta ,label="h(t)-t", color="black")
    plt.xscale("log")
    plt.xlabel("h")
    plt.ylabel("theta")
    plt.title("h vs theta")
    plt.ylim([0, 0.6])
    plt.xlim([1e0, 1e5])
    plt.legend()
    plt.grid()

def plot_h_Kr(psi,psi_for_pred,observation_Kr, pred_Kr):
    plt.figure()
    plt.scatter(psi, observation_Kr, marker='o', label="h(t)-t", color="orange")
    plt.plot(psi_for_pred, pred_Kr ,label="h(t)-t", color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("h")
    plt.ylabel("Kr")
    plt.title("h vs Kr")
    plt.xlim([1e0, 1e4])
    plt.legend()
    plt.grid()
def f_psi(theta,m,n,alpha):
    psi=alpha*(theta**(-1/m)-1)**(1/n)
    return psi

def clamp_value(c):
    """
    If c > 1, return 1.
    If c < 0, return 0.
    Otherwise, return c.
    """
    if c > 1:
        return 1
    elif c < 0:
        return 0
    else:
        return c
def evaluate_expr_theta(A_value, C1_value, x_array):
    """
    Evaluate the expression:
       theta(x) = 1/C1 * atan( -(x/A + 1) ) - 1
    but adjust by integer multiples of pi so that the final result is in [0, 1].

    :param A_value: Optimized numeric value for A
    :param C1_value: Optimized numeric value for C1
    :param x_array: 1D array (list or np.array) of x values
    :return: A 1D numpy array of the same length as x_array, with each theta(x) in [0,1] if possible.
             If no shift can place theta in [0,1], that element is set to np.nan.
    """
    x_array = np.array(x_array, dtype=float)
    out_array = np.full_like(x_array, np.nan, dtype=float)

    #get domain
    arg_val = -(x_array[int(x_array.shape[0]/2)] / A_value + 1)  # argument of atan
    y0 = math.atan(arg_val)  # principal value in [-pi/2, pi/2]
    delta=(y0 - 2 * C1_value) / np.pi
    if delta>0:
        delta=np.floor((y0 - 2 * C1_value) / np.pi)
    if delta<0:
        delta=np.ceil((y0 - 2 * C1_value) / np.pi)

    for i, x_val in enumerate(x_array):
        # 1) Compute the principal value from arctan(...)
        arg_val = -(x_val / A_value + 1)  # argument of atan
        y0 = math.atan(arg_val)  # principal value in [-pi/2, pi/2]
        # delta=np.floor((y0-2*C1_value)/np.pi)
        # print(delta)
        out_array[i]=clamp_value((1.0 / C1_value) * (y0 -delta * math.pi) - 1.0)
        # 2) Search for an integer k so that final_value is in [0,1]
        #    final_value = (1/C1_value)*(y0 + k*pi) - 1
        #    We only try k in a small range, e.g., [-2, -1, 0, 1, 2]
        # found = False
        # for k in range(-2, 3):
        #     candidate = (1.0 / C1_value) * (y0 + k * math.pi) - 1.0
        #     if 0 <= candidate <= 1:
        #         out_array[i] = candidate
        #         found = True
        #         break

        # If 'found' remains False, it means no integer shift k
        # brought the expression into [0,1]. In that case, out_array[i] is np.nan.

    return out_array

def solve_f_equals_theta(expr,h, initial_guess=0.5):
    """
    Numerically solve f(x) = m_value for x, using Sympy's nsolve.
    :param m_value: Numeric value of m.
    :param initial_guess: Initial guess for the root.
    :return: A single numeric solution (if found).
    """

    lambdified_expr=lambdify(x, expr, 'numpy')

    def objective(x):
        # x_array is typically a 1D array with a single value
        return (lambdified_expr(x) - h) ** 2

    guesses = np.arange(0.01, 1.01, 0.01)

    best_solution = None
    best_objective_val = np.inf

    # Try each guess and keep track of the best solution
    for guess in guesses:
        result = minimize(
            objective,
            x0=np.array([guess]),  # The initial guess
            method='L-BFGS-B',
            bounds=[(0, 1)],
            options={'disp': False}
        )

        if result.fun < best_objective_val:
            best_objective_val = result.fun
            best_solution = result.x # result.x is an array of length 1
    if best_solution==None:
        result = minimize(
            objective,
            x0=np.array([0.5]),  # The initial guess
            method='L-BFGS-B',
            bounds=[(0, 1)],
            options={'disp': False}
        )
        best_solution=result.x
    # best_solution holds the x value that gave the lowest objective among all guesses
    return best_solution


def multiple_solve(expr,h_list):
    solutions=[]
    for h in h_list:
        solution=solve_f_equals_theta(expr, h)
        solutions.append(solution[0])
    return solutions
