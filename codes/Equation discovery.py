from graphic_symbolic_regression import *
from trail_repreat import true_graph
from utils import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

h_K_name=['Beit_Netofa_Clay.xlsx','Hygiene sandstone.xlsx','Silt Loam G.E. 3.xlsx','Touchet Silt Loam G.E. 3.xlsx']

if __name__ == '__main__':

    file_names = os.listdir('data for saturated flow')
    mode='Valid_discovered' #['Train','Valid_discovered']


    t_for_prediction=np.arange(0,1,0.01)
    psi_for_prediction=10**(np.flip(np.arange(1,5,0.1),axis=0))
    theta,psi,ts,tr=read_data()
    theta=[(theta[i]-tr[i])/(ts[i]-tr[i]) for i in range(len(theta))]
    new_theta=[]
    new_psi=[]
    for t,p in zip(theta,psi):
        t,p=filter_small_theta(t,p)
        new_theta.append(t)
        new_psi.append(p)
    theta=new_theta
    psi=new_psi
    psi=[1/p*100 for p in psi]



    model = Random_graph_for_expr()
    model_sympy = Graph_to_sympy()
    GA = Genetic_algorithm(x_data=theta, y_data=psi)


    if mode=='Valid':
        true_graph={'nodes': ['add', 'tan', 'add', '1', 'x', '1'], 'edges': [[0, 1], [1, 2], [2, 3], [2, 4], [0, 5]], 'edge_attr': [-1, -100000000.0, 1, 1, -1]}
        expr = model_sympy.graph_to_sympy(true_graph)
        C1=symbols('C1')
        expr=A*(-tan(C1*(x + 1)) - 1)
        fitness, optimal_params = GA.get_fitness_from_expr(expr,theta,psi)
        print(fitness)
        sim_expr = GA.get_regressed_function_from_expr(expr,optimal_params)
        print(sim_expr)

        for n in range(len(theta)):
            print("true theta: ",theta[n])
            print('true psi:', psi[n])
            #============Test True graph===============
            print('pred expr: ',sim_expr[n],'  fitness: ',fitness)
            expand_pred = GA.get_function_prediction(sim_expr[n], t_for_prediction)
            positive_index_begin, positive_index_end = get_non_positive_idds(t_for_prediction,
                                                                             expand_pred)  # this is the true tr
            print("non negative theta:", positive_index_begin, positive_index_end)
            expr_Kr = calculate_Kr(sim_expr[n], x, positive_index_begin, positive_index_end)
            print('pred expr_Kr:', expr_Kr)


            if file_names[n] in h_K_name:
                h,Kr=read_h_K(file_names[n])
                theta_for_Kr = multiple_solve(sim_expr[n], h_list=1 / h * 100)
                print("theta",theta_for_Kr)
                print("true Kr:",Kr)
                f_Kr = lambdify(x, expr_Kr, 'numpy')
                # 4) Evaluate f(x) at each x in x_values
                f_results = [f_Kr(t_value) for t_value in theta_for_Kr]
                print('pred Kr:',f_results)
                plot_h_Kr(h,h,Kr,f_results)

            pred = GA.get_function_prediction(sim_expr[n], theta[n])
            print("pred psi:",pred)

            pred_theta_from_psi=multiple_solve(sim_expr[n],h_list=1/psi_for_prediction*100)
            print("pred theta from psi:", pred_theta_from_psi)

            expand_pred = GA.get_function_prediction(sim_expr[n], t_for_prediction)
            plot_data(theta[n], theta[n], psi[n], pred)
            plot_h_theta(1/psi[n]*100,psi_for_prediction, theta[n] * (ts[n] - tr[n]) + tr[n],pred_theta_from_psi * (ts[n] - tr[n]) + tr[n])
            print('====================================')
            plt.show()


        plt.show()

    if mode=='Valid_discovered':
        best_graph=pickle.load(open(f'result_save/max_var_2/best_graphs.pkl', 'rb'))
        best_fitnesses=pickle.load(open(f'result_save/max_var_2/best_fitness.pkl', 'rb'))
        best_expr=[]
        for graphs in best_graph:
            exprs=[]
            for i in range(5):
                exprs.append(model_sympy.graph_to_sympy(graphs[i]))
            best_expr.append(exprs)

        for j in range(150):
            print(f'The {j} epoch')
            for i in range(5):
                print(f'The #{i} best expr:, The #{i} best fitness:',best_expr[j][i],best_fitnesses[j][i])
                print(f'The #{i} best graph:', best_graph[j][i])
            print('=================================')


    if mode=='Train':
        # ==============Regression====================
        GA.evolution(save_dir='results')