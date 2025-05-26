import pickle

from graphic_symbolic_regression import *
from utils import *
from matplotlib.ticker import MaxNLocator,LogLocator
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
C1, C2, C3 = symbols('C1 C2 C3')
h_K_name=['Beit_Netofa_Clay.xlsx','Hygiene sandstone.xlsx','Silt Loam G.E. 3.xlsx','Touchet Silt Loam G.E. 3.xlsx']

def get_results_ours(dataset='unsoda'):
    '''
    The best expr: A * (-tan(C1 * (x + 1)) - 1)
    true_graph = {'nodes': ['add', 'tan', 'add', '1', 'x', '1'], 'edges': [[0, 1], [1, 2], [2, 3], [2, 4], [0, 5]],
                  'edge_attr': [-1, -100000000.0, 1, 1, -1]}
    '''
    if dataset=='mulam':
        file_names = os.listdir('data for saturated flow')
        theta, psi, ts, tr = read_data()
    elif dataset=='unsoda':
        file_names =os.listdir('unsoda_data')
        theta, psi, ts, tr = read_data_unsoda(file_names)

    t_for_prediction = np.arange(0, 1, 0.01)

    theta = [(theta[i] - tr[i]) / (ts[i] - tr[i]) for i in range(len(theta))]
    new_theta = []
    new_psi = []
    for t, p in zip(theta, psi):
        t, p = filter_small_theta(t, p)
        new_theta.append(t)
        new_psi.append(p)

    theta_obs = new_theta

    psi_obs = new_psi
    psi_obs=[1 / p * 100 for p in psi_obs]

    theta_expand = [np.linspace(np.min(new_theta[i]), np.max(new_theta[i]), 100) for i in range(len(new_theta))]

    psi_for_prediction = 10 ** (np.flip(np.arange(1, 5, 0.1), axis=0))
    psi_expand = [psi_for_prediction.copy() for _ in range(len(new_psi))]

    theta = theta_expand
    psi = psi_expand
    psi=[1 / p * 100 for p in psi]

    model = Random_graph_for_expr()
    model_sympy = Graph_to_sympy()
    GA = Genetic_algorithm(x_data=theta_obs, y_data=psi_obs)
    expr = A * (-tan(C1 * (x + 1)) - 1)
    expr_theta = 1 / C1 * atan(-(x / A + 1)) - 1


    h_list=[]
    h_obs_list = []
    true_Kr_list=[]
    predict_Kr_list=[]
    predict_Kr_scatter_list = []
    true_psi_list=[]
    pred_psi_list=[]
    true_theta_list=[]
    pred_theta_list=[]
    pred_theta_scatter_list=[]
    max_theta_list=[]
    min_theta_list=[]

    fitness, optimal_params = GA.get_fitness_from_expr(expr, theta_obs, psi_obs)
    print(optimal_params)
    sim_expr = GA.get_regressed_function_from_expr(expr, optimal_params)
    for n in tqdm(range(len(theta))):
        opt_A = optimal_params[n][0]
        opt_C1 = optimal_params[n][2]
        expand_pred = GA.get_function_prediction(sim_expr[n], t_for_prediction)
        positive_index_begin, positive_index_end = get_non_positive_idds(t_for_prediction,
                                                                         expand_pred)  # this is the true tr
        expr_Kr = calculate_Kr(sim_expr[n], x, positive_index_begin, positive_index_end)

        if dataset=='mulam':
            use_h_K_name=h_K_name
        elif dataset=='unsoda':
            use_h_K_name=file_names
        if file_names[n] in use_h_K_name:
            if dataset=='mulam':
                h, Kr = read_h_K(file_names[n])
            elif dataset == 'unsoda':
                h, Kr = read_h_K_unsoda(file_names[n])
            h_obs = h
            h = np.linspace(np.min(h), np.max(h), 100)

            theta_for_Kr = evaluate_expr_theta(opt_A, opt_C1, 1 / h * 100)
            f_Kr = lambdify(x, expr_Kr, 'numpy')
            f_results = [f_Kr(t_value) for t_value in theta_for_Kr]

            theta_for_Kr_obs = evaluate_expr_theta(opt_A, opt_C1, 1 / h_obs * 100)
            f_results_obs = [f_Kr(t_value) for t_value in theta_for_Kr_obs]


            h_list.append(h)
            h_obs_list.append(h_obs)
            true_Kr_list.append(Kr)
            predict_Kr_list.append(f_results)
            predict_Kr_scatter_list.append(f_results_obs)




        pred_theta_from_psi = evaluate_expr_theta(opt_A, opt_C1, 1 / psi_for_prediction * 100)
        pred_theta= evaluate_expr_theta(opt_A, opt_C1, psi[n])

        true_theta_list.append( theta_obs[n] * (ts[n] - tr[n]) + tr[n])
        pred_theta_list.append(pred_theta_from_psi * (ts[n] - tr[n]) + tr[n])
        true_psi_list.append(1/psi_obs[n]*100)
        pred_psi_list.append(psi_for_prediction)
        max_theta_list.append(ts[n])
        min_theta_list.append(tr[n])
        pred_theta_scatter_list.append(pred_theta * (ts[n] - tr[n]) + tr[n])


    return [h_list,
            h_obs_list,
            true_Kr_list,
            predict_Kr_list,
            predict_Kr_scatter_list,
            true_theta_list,
            pred_theta_list,
            true_psi_list,
            pred_psi_list,
            pred_theta_scatter_list,
            max_theta_list,
            min_theta_list]


def plot_h_vs_Kr(psi, observation_Kr, psi_for_pred, pred_Kr,save_name=''):
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(2, 2), dpi=300)

    # Scatter plot
    plt.scatter(
        psi,
        observation_Kr,
        marker='o',
        facecolors='none',  # No fill for hollow circles
        edgecolors='black',  # Outline color
        alpha=0.8,
        label="Observation",
        s=20
    )
    # Line plot
    plt.plot(psi_for_pred, pred_Kr, label="Prediction", color="#CD5C5C",linestyle='--', linewidth=1.5)
    plt.xticks(fontproperties = 'Arial', size = 7)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0],fontproperties = 'Arial', size = 7)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(6))
    # plt.xlabel("h")
    # plt.ylabel("$K_r$")
    plt.ylim([-0.05,1.05])

    plt.legend(
        prop={"family": "Arial", "size": 7},
        loc="best")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    # plt.savefig(f'plot_save/unsoda_h_Kr/h_Kr_{save_name}.pdf', bbox_inches='tight', dpi=300)
    # plt.savefig(f'plot_save/unsoda_h_Kr/h_Kr_{save_name}.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_h_vs_theta(psi, observation_theta, psi_for_pred, pred_theta,max_y,min_y,save_name=''):
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(2, 2), dpi=300)

    # Scatter plot
    plt.scatter(
        psi,
        observation_theta,
        marker='o',
        facecolors='none',  # No fill for hollow circles
        edgecolors='black',  # Outline color
        alpha=0.8,
        label="Observation",
        s=20
    )
    # Line plot
    plt.plot(psi_for_pred, pred_theta, label="Prediction", color="#CD5C5C",linestyle='--', linewidth=1.5)
    plt.xticks(fontproperties = 'Arial', size = 7)
    plt.yticks(fontproperties = 'Arial', size = 7)
    ax = plt.gca()
    ax.set_xscale('log')

    plt.ylim([min_y-0.05,max_y+0.05])
    # Set the x-axis to have 5 major ticks in log scale
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    # plt.xlabel("h")
    # plt.ylabel("$theta$")

    plt.legend(
        prop={"family": "Arial", "size": 7},
        loc="best")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    # plt.savefig(f'plot_save/unsoda_h_theta/h_theta_{save_name}.pdf', bbox_inches='tight', dpi=300)
    # plt.savefig(f'plot_save/unsoda_h_theta/h_theta_{save_name}.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plots_on_unsoda():
    our_results = get_results_ours(dataset='unsoda')
    file_names_unsoda = os.listdir('unsoda_data')
    for i in range(len(our_results[0])):
        plot_h_vs_Kr(our_results[1][i], our_results[2][i], our_results[0][i], our_results[3][i],
                     save_name=file_names_unsoda[i])

        plot_h_vs_theta(our_results[7][i], our_results[5][i], our_results[8][i], our_results[6][i],
                        max_y=our_results[10][i], min_y=our_results[11][i], save_name=file_names_unsoda[i])

if __name__ == '__main__':
    plots_on_unsoda()

