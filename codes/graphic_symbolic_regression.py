'''
graph regression for any expression with unknown form and coefficient
formal version
by HaoXu
'''

import os
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from torch_geometric.data import Data
import torch
from scipy.optimize import minimize,curve_fit
from tqdm import tqdm
import time
import heapq
from sympy import symbols, sin, cos, tan, log, ln, sqrt, exp, csc, sec, cot, sinh, tanh, cosh, atan, asin, acos, atanh, \
    asinh, acosh, sympify,pi, lambdify,E,I
from copy import deepcopy
import warnings
import sympy as sp
import pickle
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", category=RuntimeWarning)
NODE_FEATURE_MAP = {
    "1":1,
    'add': 2,
    'mul': 3,
    'exp': 4,
    'div': 5,
    "log": 6,
    "ln": 7,
    "sqrt": 8,
    "abs": 9,
    "sub": 10,
    "sin": 11,
    "cos": 12,
    "tan":13,
    "csc":14,
    "sec":15,
    "cot":16,
    "sinh":17,
    "tanh":18,
    "cosh":19,
    "atan":20,
    "asin":21,
    "acos":22,
    "atanh":23,
    "asinh":24,
    "acosh":25,
    'x': 26,  # Variable
    'a':27,
    'b':28,
    'c':29,
    'd':30,
    'n':31,
    'm':32,
    'E':33,
    'pi':34
}
Binary_Operator=['add','mul']
Unary_Operator_ln=["log"]
Unary_Operator_exp=['exp']
Triangle_Operator=["sin", "cos","tan"]
Arctriangle_Operator=["csc","sec","cot","sinh", "tanh","cosh","atan","asin","acos","atanh","asinh","acosh" ]
Variable=['x']
Constant=['1','pi']
polynomial_integral={'value':[-2,-1,-0.5,0,0.5,1,2,-1e8],'prob':[2,3,2,1,2,3,2,1]} #-1e8 indicate a varying coefficient to be determined
x, C,A,B = symbols('x C A B')

def set_random_seeds(rand_seed=1101, np_rand_seed=525):
    random.seed(rand_seed)
    np.random.seed(np_rand_seed)

set_random_seeds()

def convert_graph_to_pyG(graph):
    """
    Converts a given graph dictionary into a PyTorch Geometric (PyG) Data object.

    This function processes the input graph by extracting its nodes, edges, and edge attributes.
    It maps the nodes to their corresponding feature representations, converts the data into
    PyTorch tensors, and constructs a PyG Data object with the processed information.

    :param graph: A dictionary containing the graph data. It must include the following keys:
                  - 'nodes': A list of node identifiers.
                  - 'edges': A list of edge pairs represented as tuples or lists of two integers.
                  - 'edge_attr': A list of edge attributes corresponding to the edges.
    :type graph: dict[str, list]

    :return: A PyTorch Geometric Data object representing the graph. The returned object contains
             the following attributes:
             - x: Node feature matrix as a PyTorch tensor.
             - edge_index: Graph connectivity in COO format as a PyTorch tensor.
             - edge_attr: Edge attribute matrix as a PyTorch tensor.
    :rtype: torch_geometric.data.Data
    """
    graph_nodes = graph['nodes']
    graph_edges = graph['edges']
    graph_edge_attr = graph['edge_attr']
    x=[NODE_FEATURE_MAP[node] for node in graph_nodes]
    x=torch.from_numpy(np.array(x).astype(np.int64))
    edge_index=torch.from_numpy(np.array(graph_edges).astype(np.int64)).T
    edge_attr=torch.from_numpy(np.array(graph_edge_attr).astype(np.float32))
    pyG_graph=Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyG_graph

def plot_graph_with_features(graph,show=False):
    """
    Plot a graph with nodes and edges, visualizing their features using a custom layout and color scheme.

    This function takes a graph object, converts it into a PyTorch Geometric data structure, and then
    visualizes it using NetworkX. Nodes are colored based on their features, and edges are colored
    according to their attributes. The visualization uses a circular layout to arrange the nodes and
    includes a legend for node categories and a color bar for edge features.

    :param graph: A graph object that can be converted into a PyTorch Geometric data structure.
                  It should contain node features and edge information.
    :type graph: Any
    :param show: A flag to determine whether to display the plot immediately after rendering.
                 If False, the plot will not be shown automatically.
    :type show: bool
    :return: None. The function renders a graph visualization but does not return any value.
    :rtype: None
    """
    plt.rc('font', family='Arial')
    graph_data=convert_graph_to_pyG(graph)
    G = nx.DiGraph()  # 使用有向图

    # 添加节点及其特征
    for i, feature in enumerate(graph_data.x):
        G.add_node(i, feature=int(feature.item()))

    # 添加边及其特征
    edge_features = {}
    for (src, dst), feature in zip(graph_data.edge_index.t().tolist(), graph_data.edge_attr.tolist()):
        G.add_edge(src, dst, feature=feature)
        edge_features[(src, dst)] = feature

    # 布局算法
    pos = nx.shell_layout(G)


    possible_node_categories = ['mul', 'add', 'x', '1', 'log', 'exp', 'sin', 'cos', 'tan']
    custom_color_map = {
        NODE_FEATURE_MAP['mul']: '#B3E2CD',
        NODE_FEATURE_MAP['add']: '#FDCDAC',
        NODE_FEATURE_MAP['x']: '#CBD5E8',
        NODE_FEATURE_MAP['1']: '#F4CAE4',
        NODE_FEATURE_MAP['log']: '#E6F5C9',
        NODE_FEATURE_MAP['exp']: '#FFF2AE',
        NODE_FEATURE_MAP['sin']: '#F1E2CC',
        NODE_FEATURE_MAP['cos']: '#D6CDEA',
        NODE_FEATURE_MAP['tan']: '#FDD5E5',
    }

    node_features = nx.get_node_attributes(G, 'feature')
    unique_categories = sorted(set(node_features.values()))
    num_categories = len(unique_categories)
    color_map= {category: custom_color_map[category] for i, category in enumerate(unique_categories)}
    node_colors = [color_map[node_features[i]] for i in G.nodes]


    edge_colors = [edge_features[edge] for edge in G.edges()]
    edge_color_map = plt.cm.coolwarm  # 改变边的颜色映射


    #fig, ax = plt.subplots(figsize=(12, 10))
    fig, ax = plt.subplots(figsize=(5, 3),dpi=300)
    pos = nx.circular_layout(G)


    ax.set_aspect('equal', adjustable='datalim')

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=300,  # 比原先 700 略小
        edgecolors="black",
        linewidths=0.8,  # 节点边框线宽
        alpha=0.9,  # 节点透明度，可让整体更柔和
        ax=ax
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        edge_cmap=edge_color_map,
        edge_vmin=-10,  # 你自定义的取值范围
        edge_vmax=12,  # 你自定义的取值范围
        arrows=True,
        arrowsize=10,  # 调整箭头大小（默认 10~20 之间）
        arrowstyle='->',  # 箭头样式
        width=1,  # 边宽
        alpha=0.8,  # 透明度
        ax=ax
    )

    nx.draw_networkx_labels(
        G, pos,
        labels={i: str(i) for i in G.nodes()},
        font_size=7,  # 更小的字体在 300 dpi 也可读
        font_color='black',
        font_family='Arial',
        ax=ax
    )


    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f"{list(NODE_FEATURE_MAP.keys())[category-1]}",
                   markersize=6, markerfacecolor=color, markeredgecolor="black")
        for category, color in color_map.items()
    ]

    ax.legend(
        handles=legend_handles,
        #title="Node Categories",
        loc='center left',  # 以图的左中位置为锚点
        bbox_to_anchor=(0.85, 0.5),  # 往右 (1.05) 偏移
        fontsize=7,  # 图例文字大小
        #title_fontsize=7  # 图例标题大小
    )

    sm_edges = plt.cm.ScalarMappable(cmap=edge_color_map,
                                     norm=plt.Normalize(vmin=-10, vmax=12))
    #sm_edges = plt.cm.ScalarMappable(cmap=edge_color_map, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
    cbar_edges = fig.colorbar(sm_edges, ax=ax, fraction=0.046, pad=0.04)
    cbar_edges.ax.tick_params(labelsize=7)      # 调整刻度标签字体
    cbar_edges.set_label("Edge Features", fontsize=7)  # 色标标题字体


    plt.axis("off")  # 去掉坐标轴
    plt.tight_layout()
    if show==True:
        plt.show()

    # print("Node features (x):\n", graph_data.x)
    # print("Edge indices (edge_index):\n", graph_data.edge_index)
    # print("Edge features (edge_attr):\n", graph_data.edge_attr)

class Random_graph_for_expr():
    """
    functions for generating random graphs
    """
    def concate_subgraph_to_node(self, graph, subgraph, concate_node_index, concate_node,set_maximum_node_num=-1,with_node_indice=[]):
        '''
        :param graph: a tuple of [nodes, edges, edge_attr] for the existing graph
        :param concate_node: concate the graph to which node
        :param subgraph: a tuple of [nodes, edges, edge_attr] for the generated subgraph
        :return: new graph
        '''
        if set_maximum_node_num==-1:
            maximun_node_num = len(graph['nodes'])
        else:
            maximun_node_num =set_maximum_node_num
        subgraph_nodes = subgraph['nodes']
        subgraph_edges = subgraph['edges']
        subgraph_edge_attr = subgraph['edge_attr']

        graph_nodes = graph['nodes']
        graph_edges = graph['edges']
        graph_edge_attr = graph['edge_attr']

        for sublist in subgraph_edges:
            for i in range(len(sublist)):
                sublist[i] += maximun_node_num
        graph_nodes += subgraph_nodes
        graph_edges.append([concate_node_index, maximun_node_num])
        graph_edges += subgraph_edges
        if concate_node =='exp':
            graph_edge_attr.append(random.choices([-2,-1,-0.5,0.5,1,2,-1e8],[0.1,0.25,0.1,0.1,0.25,0.1,0.1],k=1)[0])
        elif concate_node=='add':
            graph_edge_attr.append(random.choices([1,-1,-1e8],[0.45,0.45,0.1],k=1)[0])
        elif concate_node=='mul':
            graph_edge_attr.append(random.choices([1, -1, -1e8], [0.3, 0.3,0.4], k=1)[0])
        else:
            graph_edge_attr.append(1)
        graph_edge_attr += subgraph_edge_attr
        graph = {'nodes': graph_nodes, 'edges': graph_edges, 'edge_attr': graph_edge_attr}

        if len(with_node_indice)==0:
            return graph
        else:
            with_node_indice=[maximun_node_num+i for i in range(len(subgraph_nodes))]
            return graph,with_node_indice
    def generate_single_poly(self,polynomial_integral=polynomial_integral):
        """
        Generates a subgraph representing a single polynomial based on the provided
        integral configuration. The function randomly selects a polynomial degree or
        type according to the given probabilities and constructs a corresponding
        subgraph.

        :param polynomial_integral: A dictionary containing two keys: 'value' and
                                     'prob'. The 'value' key maps to a list of possible
                                     polynomial degrees or types, while the 'prob' key
                                     maps to a list of probabilities associated with
                                     each value. The sum of probabilities in 'prob'
                                     should be positive.
        :type polynomial_integral: dict[str, list]

        :return: A dictionary representing the generated subgraph. The subgraph
                 contains three keys: 'nodes', 'edges', and 'edge_attr'. The 'nodes'
                 key maps to a list of node labels, 'edges' maps to a list of edge
                 connections (each represented as a list of two nodes), and
                 'edge_attr' maps to a list of attributes for each edge.
        :rtype: dict[str, list]
        """
        poly_integ =random.choices(polynomial_integral['value'],weights=np.array(polynomial_integral['prob'])/sum(polynomial_integral['prob']),k=1)[0]
        if poly_integ == 0:
            subgraph = {'nodes': ['1'], 'edges': [], 'edge_attr': []}
        elif poly_integ == 1:
            subgraph = {'nodes': ['x'], 'edges': [], 'edge_attr': []}
        else:
            subgraph = {'nodes': ['exp', 'x'], 'edges': [[0, 1]], 'edge_attr': [poly_integ]}
        return subgraph
    def generate_polynomial_template(self,polynomial_integral=polynomial_integral,max_poly_term=2):
        """
        Generates a polynomial template as a subgraph with a specified number of terms.

        The method creates a polynomial structure represented as a subgraph. The number
        of terms in the polynomial is randomly determined within the range of 1 to the
        maximum allowed terms. For polynomials with more than one term, the subgraph is
        constructed by concatenating single-term polynomial subgraphs into an additive
        structure.

        :param polynomial_integral: A predefined integral value or structure used for
                                     generating single-term polynomials. It serves as a
                                     base for constructing individual terms.
        :type polynomial_integral: IntegralType
        :param max_poly_term: The maximum number of terms allowed in the generated
                              polynomial. Defaults to 2 if not provided.
        :type max_poly_term: int
        :return: A subgraph representing the generated polynomial. The subgraph contains
                 nodes, edges, and edge attributes that define the polynomial's structure.
        :rtype: dict
        """
        n_terms=random.randint(1,max_poly_term)

        if n_terms==1:
           subgraph=self.generate_single_poly(polynomial_integral)
        else:
            subgraph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
            for i in range(n_terms):
                subgraph=self.concate_subgraph_to_node(subgraph,self.generate_single_poly(),0,'add')
        return subgraph

    def generate_log_template(self):
        """
        Generates a subgraph template for logarithmic and natural logarithmic expressions. The function
        randomly selects one of the predefined templates: log(A/B), log(A+B), ln(A/B), or ln(A+B).
        Depending on the selected template, it constructs a graph by combining subgraphs representing
        polynomial expressions with operations such as multiplication, addition, and exponentiation.

        The function utilizes helper methods to generate polynomial templates and concatenate subgraphs
        to form the final graph structure. The resulting graph represents the mathematical expression
        corresponding to the chosen template.

        :return: A dictionary representing the constructed graph. The graph contains keys 'nodes',
                 'edges', and 'edge_attr' that define the structure and attributes of the graph.
        :rtype: dict
        """
        Template = ['log(A+B)', 'ln(A+B)']
        use_template = random.choice(Template)
        if use_template == 'log(A/B)':
            initial_graph = {'nodes': ['log','mul'], 'edges': [[0,1]], 'edge_attr': [10]}
            A_graph=self.generate_polynomial_template(polynomial_integral={'value':[0,1,2,3],'prob':[1,2,1,1]})
            B_graph=self.generate_polynomial_template(polynomial_integral={'value':[0,1,2,3],'prob':[1,2,1,1]})
            subgraph=self.concate_subgraph_to_node(initial_graph,A_graph,1,'mul')
            inverse_B={'nodes': ['exp'], 'edges': [], 'edge_attr': []}
            B_graph=self.concate_subgraph_to_node(inverse_B,B_graph,0,'exp')
            B_graph['edge_attr'][0]=-1
            graph=self.concate_subgraph_to_node(subgraph,B_graph,1,'mul')
        if use_template == 'log(A+B)':
            initial_graph = {'nodes': ['log'], 'edges': [], 'edge_attr': []}
            A_graph = self.generate_polynomial_template(
                polynomial_integral={'value': [0, 1, 2, 3], 'prob': [1, 2, 1, 1]})
            graph = self.concate_subgraph_to_node(initial_graph, A_graph, 0, 'log')
            graph['edge_attr'][0]=10
        if use_template == 'ln(A/B)':
            initial_graph = {'nodes': ['log', 'mul'], 'edges': [[0, 1]], 'edge_attr': [math.e]}
            A_graph = self.generate_polynomial_template(
                polynomial_integral={'value': [0, 1, 2, 3], 'prob': [1, 2, 1, 1]})
            B_graph = self.generate_polynomial_template(
                polynomial_integral={'value': [0, 1, 2, 3], 'prob': [1, 2, 1, 1]})
            subgraph = self.concate_subgraph_to_node(initial_graph, A_graph, 1, 'mul')
            inverse_B = {'nodes': ['exp'], 'edges': [], 'edge_attr': []}
            B_graph = self.concate_subgraph_to_node(inverse_B, B_graph, 0, 'exp')
            B_graph['edge_attr'][0] = -1
            graph = self.concate_subgraph_to_node(subgraph, B_graph, 1, 'mul')
        if use_template == 'ln(A+B)':
            initial_graph = {'nodes': ['log'], 'edges': [], 'edge_attr': []}
            A_graph = self.generate_polynomial_template(
                polynomial_integral={'value': [0, 1, 2, 3], 'prob': [1, 2, 1, 1]})
            graph = self.concate_subgraph_to_node(initial_graph, A_graph, 0, 'log')
            graph['edge_attr'][0] = math.e

        return graph

    def generate_triangle_template(self):
        """
        Generates a random graph representation based on predefined triangle function templates.
        The function randomly selects a template and an operator to construct the graph, which
        includes nodes, edges, and edge attributes.

        :param self: The instance of the class containing the method.
        :type self: object

        :return: A dictionary representing the graph structure with keys 'nodes', 'edges', and
                 'edge_attr'. 'nodes' contains the list of nodes in the graph, 'edges' contains
                 pairs of node indices defining the connections, and 'edge_attr' contains
                 attributes associated with each edge.
        :rtype: dict

        """
        Template = ['tri(Cx)', 'tri(C*pi*x)', 'tri(pi*x+C)', 'tri(2*pi*x+C)']
        use_template = random.choice(Template)
        Operator=random.choice(Triangle_Operator)
        if use_template == 'tri(Cx)':
            graph = {'nodes': [Operator,'x'], 'edges': [[0,1]], 'edge_attr': [random.choice([-2,-1,1,2,-1e8])]}
        if use_template == 'tri(C*pi*x)':
            graph = {'nodes': [Operator,'mul','x'], 'edges': [[0,1],[1,2]], 'edge_attr': [math.pi,random.choice([-2,-1,1,2,-1e8])]}
        if use_template == 'tri(pi*x+C)':
            graph = {'nodes': [Operator,'add', 'mul', 'x', '1'], 'edges': [[0,1],[1,2],[2,3],[1,4]],
                     'edge_attr': [1,1,math.pi,-1e8]}
        if use_template == 'tri(2*pi*x+C)':
            graph = {'nodes': [Operator, 'add', 'mul', 'x', '1'], 'edges': [[0, 1], [1, 2], [2, 3], [1, 4]],
                     'edge_attr': [1, 1, 2*math.pi, -1e8]}
        return graph
    def generate_exp_template(self):
        """
        Generates a graph template for an exponential expression by randomly selecting
        a sub-template and combining it with the base graph structure.

        The function initializes a base graph with a single node labeled 'exp'. It then
        randomly selects one of three possible templates ('exp-log', 'exp-tri', or
        'exp-x') based on predefined weights. Depending on the selected template, a
        corresponding subgraph is generated using helper methods. The selected
        subgraph is then concatenated to the base graph at the 'exp' node.

        :param: None

        :return: A dictionary representing the final graph structure after concatenating
                 the selected subgraph to the base graph. The graph contains 'nodes',
                 'edges', and 'edge_attr' as keys.
        :rtype: dict
        """
        graph = {'nodes': ['exp'], 'edges': [], 'edge_attr': []}
        Template = ['exp-log', 'exp-tri','exp-x']
        use_template = random.choices(Template,weights=[0.2,0.2,0.6],k=1)[0]
        if use_template=='exp-x':
            template=self.generate_polynomial_template()
        elif use_template=='exp-log':
            template = self.generate_log_template()
        elif use_template=='exp-tri':
            template = self.generate_triangle_template()
        graph = self.concate_subgraph_to_node(graph, template, 0, 'exp')
        return graph

    def generate_graph_template(self):
        """
        Generates a random graph template based on predefined templates and their associated probabilities.

        This method selects a graph template randomly from a list of predefined templates, weighted by
        their respective probabilities. The selected template is then used to generate a graph structure
        by invoking various helper methods that construct subgraphs and combine them into a final graph.

        :return: A dictionary representing the generated graph structure, containing 'nodes', 'edges',
                 and 'edge_attr' keys.
        :rtype: dict

        :raises ValueError: If an invalid template is selected or if the helper methods fail to generate
                            valid subgraphs.
        """
        template=['x','x/x','expx','log','tri','x*expx','x+expx','x+log','x+tri']
        choice_prob=np.array([3,1,3,3,3,1,1,1,1])
        #template=['expx','x']
        #choice_prob = np.array([3, 2])
        temp=random.choices(template,weights=choice_prob/np.sum(choice_prob),k=1)[0]
        #print('use_temp:',temp)
        if temp=='x':
            graph=self.generate_polynomial_template()
        elif temp=='x*x':
            graph= {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph=self.concate_subgraph_to_node(graph,self.generate_polynomial_template(),0,'mul')
            graph=self.concate_subgraph_to_node(graph,self.generate_polynomial_template(),0,'mul')
        elif temp=='x*tri':
            graph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'mul')
            graph = self.concate_subgraph_to_node(graph, self.generate_triangle_template(), 0, 'mul')
        elif temp == 'x*log':
            graph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'mul')
            graph = self.concate_subgraph_to_node(graph, self.generate_log_template(), 0, 'mul')
        elif temp=='expx':
            graph=self.generate_exp_template()
        elif temp=='x/x':
            subgraph = {'nodes': ['exp'], 'edges': [], 'edge_attr': []}
            template = self.generate_polynomial_template()
            subgraph = self.concate_subgraph_to_node(subgraph, template, 0, 'exp')
            subgraph['edge_attr'][0]=-1
            graph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'mul')
            graph = self.concate_subgraph_to_node(graph, subgraph, 0, 'mul')
        elif temp=='x*expx':
            graph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'mul')
            graph = self.concate_subgraph_to_node(graph, self.generate_exp_template(), 0, 'mul')
        elif temp=='x+expx':
            graph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'add')
            graph = self.concate_subgraph_to_node(graph, self.generate_exp_template(), 0, 'add')
        elif temp=='x+log':
            graph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'add')
            graph = self.concate_subgraph_to_node(graph, self.generate_log_template(), 0, 'add')
        elif temp == 'x+tri':
            graph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'add')
            graph = self.concate_subgraph_to_node(graph, self.generate_triangle_template(), 0, 'add')
        elif temp=='log':
            graph=self.generate_log_template()
        elif temp=='tri':
            graph=self.generate_triangle_template()

        return graph

    def generate_random_graph(self):
        """
        Generates a random graph based on a predefined template.

        This method creates a graph structure by utilizing the `generate_graph_template`
        method, which provides the foundational structure. The generated graph is then
        returned as the output of this method.

        :return: A graph object generated from the template.
        :rtype: Graph
        """
        graph=self.generate_graph_template()
        return graph


class Graph_to_sympy():
    """
      functions for converting graphs to sympy expressions.
      """
    def graph_to_sympy(self,graph):
        operator_map = {
            'add': lambda a, b: a + b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b,
            'log': lambda a: log(a, 10),
            'ln': lambda a: ln(a),
            'sqrt': lambda a: sqrt(a),
            'exp': lambda a: a,  # exp(x) -> x^C, where C is an unknown coefficient
            'sin': lambda a: sin(a),
            'cos': lambda a: cos(a),
            'tan': lambda a: tan(a),
            'csc': lambda a: csc(a),
            'sec': lambda a: sec(a),
            'cot': lambda a: cot(a),
            'sinh': lambda a: sinh(a),
            'tanh': lambda a: tanh(a),
            'cosh': lambda a: cosh(a),
            'atan': lambda a: atan(a),
            'asin': lambda a: asin(a),
            'acos': lambda a: acos(a),
            'atanh': lambda a: atanh(a),
            'asinh': lambda a: asinh(a),
            'acosh': lambda a: acosh(a),
        }


        nodes, edges, edge_attr=graph['nodes'],graph['edges'],graph['edge_attr']
        # Create a dictionary to hold the expressions for each node
        expressions = {}
        visited = set()  # To track nodes that are currently being processed
        symbol_index = 0  # To track how many unknown coefficients we've generated

        # Assuming new symbols are generated for edge_attr = -1
        unknown_symbol_dict = {i: symbols(f'C{i}') for i in range(len(edges)+5)}  #

        # Function to evaluate the expression for a given node
        def evaluate_node(node_index):
            # If the node has already been evaluated, return its expression
            if node_index in expressions:
                return expressions[node_index]

            # If the node is currently being processed, we've detected a cycle
            if node_index in visited:
                raise RecursionError(f"Cyclic dependency detected at node {node_index}")

            # Mark the current node as being processed
            visited.add(node_index)

            # Get the current node's operation or variable
            node_value = nodes[node_index]
            # If the node is a constant (e.g., pi or 1)
            if node_value == 'pi':
                expressions[node_index] = pi
            elif node_value == 'x':
                expressions[node_index] = x
            elif node_value == '1':
                expressions[node_index] = 1
            # If the node is a unary operator
            elif node_value in Unary_Operator_ln + Unary_Operator_exp + Triangle_Operator + Arctriangle_Operator:
                # For unary operators, only one edge should point to it
                child_node_index = [edges[i][1] for i in range(len(edges)) if edges[i][0] == node_index][
                    0]  # Find the parent node
                edge_index=[i for i in range(len(edges)) if edges[i][0] == node_index][0]
                if node_value=='log':
                    if evaluate_node(child_node_index) in [0,-1]:
                        expressions[node_index] =1
                    elif (type(evaluate_node(child_node_index)).__name__.lower() in ['int','float','integer','rational']) and (float(evaluate_node(child_node_index))<0):
                        expressions[node_index]=1
                    else:
                        if abs(edge_attr[edge_index]-math.e)<1e-5:
                            expressions[node_index] = operator_map['ln'](evaluate_node(child_node_index))
                        else:
                            expressions[node_index] = operator_map[node_value](evaluate_node(child_node_index))

                elif node_value in Triangle_Operator+Arctriangle_Operator:
                    if edge_attr[edge_index]==-1e8:
                        expressions[node_index] = operator_map[node_value](
                            unknown_symbol_dict[edge_index] * evaluate_node(child_node_index))
                    else:
                        expressions[node_index] = operator_map[node_value](edge_attr[edge_index]*evaluate_node(child_node_index))
                else:
                    expressions[node_index] = operator_map[node_value](evaluate_node(child_node_index))
            # If the node is a binary operator (add, mul)
            elif node_value in Binary_Operator:
                # Collect the child nodes of this node
                child_nodes = [edges[i][1] for i in range(len(edges)) if edges[i][0] == node_index]
                child_egdes=[i for i in range(len(edges)) if edges[i][0] == node_index]

                # Apply the appropriate operation (add, mul, div)
                if node_value == 'add':
                    addition=0
                    for iter, child in enumerate(zip(child_nodes, child_egdes)):
                        if edge_attr[child[1]]==-1e8:
                            addition += evaluate_node(child[0]) * unknown_symbol_dict[child[1]]
                        else:
                            addition+=evaluate_node(child[0])*edge_attr[child[1]]
                    expressions[node_index]=addition
                elif node_value == 'mul':
                    product = 1
                    for iter,child in enumerate(zip(child_nodes,child_egdes)):
                        if edge_attr[child[1]] == -1e8:
                            product *= (evaluate_node(child[0]) *unknown_symbol_dict[child[1]])
                        else:
                            product *= (evaluate_node(child[0])*edge_attr[child[1]])
                    expressions[node_index] = product

            # If the edge_attr is -1, we introduce an unknown coefficient (C)
            if nodes[node_index] == 'exp':  # For non-exponentiation nodes
                child_egdes = [i for i in range(len(edges)) if edges[i][0] == node_index]

                if edge_attr[child_egdes[0]]!=-1e8:
                    if  expressions[node_index]==0 and edge_attr[child_egdes[0]]<=0:
                        expressions[node_index] = 1
                    elif expressions[node_index]==-1 and edge_attr[child_egdes[0]] in [0.5,-0.5]:
                        expressions[node_index] = 1
                    elif type(expressions[node_index]).__name__.lower() in ['int','float','integer','rational'] and float(evaluate_node(child_node_index))<0 and edge_attr[child_egdes[0]] in [0.5,-0.5]:
                        expressions[node_index] = 1
                    else:
                        expressions[node_index] = expressions[node_index] **edge_attr[child_egdes[0]]
                else:
                    if  expressions[node_index]==0:
                        expressions[node_index] = 1
                    else:
                        expressions[node_index] = expressions[node_index]** unknown_symbol_dict[child_egdes[0]]


            # Remove the node from the visited set as we're done processing it
            visited.remove(node_index)

            # Return the evaluated expression for the node
            return expressions[node_index]

        # Start the evaluation from the root nodes (ones that do not have incoming edges)
        for node_index in range(len(nodes)):
            if node_index not in [edge[1] for edge in edges]:
                evaluate_node(node_index)

        if type(expressions[0]).__name__.lower() not in ['int','float','integer','rational']:
            if expressions[0].has(I)==True:
                expressions[0]=1
        # The root node will be the last evaluated expression
        #print(expressions)
        return expressions[0]*A#expressions[0]*A+B

        # Example input (nodes, edges, edge_attr as described)


class Genetic_algorithm(Random_graph_for_expr,Graph_to_sympy):
    '''
    Optimization in the graphic-symbolic regression
    '''
    def __init__(self,x_data,y_data):
        super().__init__()  # 调用父类的构造函数
        self.x_data=x_data
        self.y_data=y_data
        self.size_pop=300
        self.generation_num=150
        self.distinction_epoch=5
        self.max_edge_num=30
        self.max_variable_num=7
        self.use_parallel_computing=True
        self.seek_best_initial=True
        self.epi=0.2
        self.max_unconstant=2


    def renumber_subgraph(self,graph,node_indice):
        """
        Renumber the nodes of a subgraph starting from 0, and adjust the edges accordingly.

        Args:
        nodes (list): A list of nodes in the subgraph.
        edges (list): A list of edges, where each edge is represented by a pair of node indices.

        Returns:
        tuple: A tuple containing:
            - The renumbered nodes (list)
            - The renumbered edges (list)
        """
        nodes=node_indice
        edges=graph['edges']
        # Step 1: Create a mapping from the original nodes to the new renumbered nodes.
        node_mapping = {node: i for i, node in enumerate(sorted(nodes))}
        # Step 2: Renumber the nodes in the edges based on the new node_mapping
        renumbered_edges = [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in edges]

        # Step 3: Return the renumbered nodes (sorted and starting from 0) and edges.
        renumbered_nodes = list(node_mapping.values())  # New nodes are just the renumbered indices

        graph['edges']=renumbered_edges
        return graph

    def extract_subgraph(self,graph, root):
        """
        Extracts a subgraph from the given graph starting at the specified root node.

        The function performs a breadth-first traversal starting at the root node to
        discover all reachable nodes and associated edges. It collects the nodes, edges,
        and edge attributes that belong to the subgraph. The resulting subgraph nodes
        are returned in sorted order.

        :param graph: A dictionary containing the graph data with keys 'nodes', 'edges',
                      and 'edge_attr'. The 'nodes' key maps to a list of nodes, 'edges'
                      maps to a list of edge tuples, and 'edge_attr' maps to a list of
                      attributes corresponding to the edges.
        :type graph: dict[str, list]
        :param root: The node from which the subgraph extraction begins. This node must
                     exist in the graph's node list.
        :type root: int | str
        :return: A tuple containing three elements: a sorted list of subgraph nodes, a
                 list of subgraph edges, and a list of attributes for the subgraph edges.
        :rtype: tuple[list, list[tuple], list]

        """
        nodes, edges, edge_attr = graph['nodes'], graph['edges'], graph['edge_attr']
        subgraph_nodes = []
        subgraph_edges = []
        subgraph_edge_attr = []

        queue = [root]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            subgraph_nodes.append(current)

            for edge, attr in zip(edges, edge_attr):
                if edge[0] == current:
                    subgraph_edges.append(edge)
                    subgraph_edge_attr.append(attr)
                    queue.append(edge[1])
        subgraph_nodes=sorted(subgraph_nodes)
        return subgraph_nodes, subgraph_edges, subgraph_edge_attr

    def delete_subgraph_from_node(self, graph, node_index):
        """
        Deletes a subgraph originating from a specified node within the given graph.

        This method extracts the subgraph associated with the provided node index, removes all
        nodes and edges belonging to the subgraph from the main graph, and updates the remaining
        structure accordingly. It also returns a mapping of the updated node indices after deletion.

        :param graph: A dictionary representing the graph structure containing 'nodes', 'edges',
                      and 'edge_attr' as keys. The 'nodes' is a list of nodes, 'edges' is a list
                      of tuples representing connections between nodes, and 'edge_attr' is a list
                      of attributes corresponding to each edge.
        :type graph: dict[list, list[tuple], list]
        :param node_index: The index of the node in the graph from which the subgraph extraction
                           and subsequent deletion will be performed.
        :type node_index: int
        :return: A tuple containing the updated graph (with the subgraph removed) and a list of
                 updated node indices reflecting the changes made to the graph structure.
        :rtype: tuple[dict[list, list[tuple], list], list[int]]

        """
        subgraph_nodes, subgraph_edges, subgraph_edge_attr = self.extract_subgraph(graph, node_index)
        graph_node_index = [i for i in range(len(graph['nodes']))]
        graph['nodes'] = [element for index, element in enumerate(graph['nodes']) if index not in subgraph_nodes]
        graph_node_index = [element for index, element in enumerate(graph_node_index) if index not in subgraph_nodes]
        new_edges = []
        new_edge_attr = []
        for index, edge_info in enumerate(zip(graph['edges'], graph['edge_attr'])):
            if (edge_info[0] not in subgraph_edges) and (edge_info[0][1] not in subgraph_nodes):
                new_edges.append(edge_info[0])
                new_edge_attr.append(edge_info[1])
        graph['edges'] = new_edges
        graph['edge_attr'] = new_edge_attr
        return graph,graph_node_index

    def cross_over(self,graph1, graph2):
        """
        Perform crossover by exchanging subgraphs between two graphs.
        """
        graph1 = deepcopy(graph1)
        graph2 = deepcopy(graph2)

        # Select a random node in graph1 as the root of the subgraph to replace
        node1 = random.randint(0, len(graph1['nodes']) - 1)
        max_node1_num=len(graph1['nodes'])
        if node1!=0:
            parent_node_index= [graph1['edges'][i][0] for i in range(len(graph1['edges'])) if graph1['edges'][i][1] == node1][0]
            parent_node_attr= [graph1['edge_attr'][i] for i in range(len(graph1['edges'])) if graph1['edges'][i][1] == node1][0]

        # Select a random node in graph2 as the root of the subgraph to use
        node2 = random.randint(0, len(graph2['nodes']) - 1)

        # Extract subgraph from graph2 starting at node2


        # Extract subgraph from graph2
        sub_nodes_2, sub_edges_2, sub_edge_attr_2 = self.extract_subgraph(graph2, node2)
        sub_nodes_1,sub_edges_1,sub_edge_attr_1=self.extract_subgraph(graph1, node1)

        #Delete the subgraph
        graph1_node_index = [i for i in range(len(graph1['nodes']))]
        graph1['nodes'] = [element for index, element in enumerate(graph1['nodes']) if index not in sub_nodes_1]
        graph1_node_index=[element for index, element in enumerate(graph1_node_index) if index not in sub_nodes_1]
        new_edges_1=[]
        new_edge_attr_1=[]
        for index,edge_info in enumerate(zip(graph1['edges'],graph1['edge_attr'])):
            if edge_info[0] not in sub_edges_1:
                if edge_info[0][1] not in sub_nodes_1:
                    new_edges_1.append(edge_info[0])
                    new_edge_attr_1.append(edge_info[1])
        graph1['edges']=new_edges_1
        graph1['edge_attr']=new_edge_attr_1

        if node1==0:
            graph1_nodes=sub_nodes_2
            graph1['nodes']=[graph2['nodes'][i] for i in graph1_nodes]
            graph1['edges']=sub_edges_2
            graph1['edge_attr']=sub_edge_attr_2
            graph1=self.renumber_subgraph(graph1,graph1_nodes)

        else:

            if len(sub_edges_2)!=0:
                sub_edges_2=(np.array(sub_edges_2)+max_node1_num).tolist()
                nodes2_min=np.min(np.array(sub_edges_2))
                graph1['edges'].append([parent_node_index,nodes2_min])
                graph1_node_index.extend((np.array(sub_nodes_2) + max_node1_num).tolist())
            else:
                graph1['edges'].append([parent_node_index, parent_node_index+max_node1_num])
                graph1_node_index.extend([parent_node_index+max_node1_num])

            graph1['edge_attr'].append(parent_node_attr)

            graph1['nodes'].extend([graph2['nodes'][i] for i in sub_nodes_2])
            graph1['edges'].extend(sub_edges_2)
            graph1['edge_attr'].extend(sub_edge_attr_2)

            graph1=self.renumber_subgraph(graph1,graph1_node_index)

        return graph1

    def mutate(self,graph, node_mutation_rate=0.3,graph_mutation_rate=0.5,graph_delete_graph=0.2,mutate_edge_attr_prob=0.5):
        """
        Perform mutation by modifying nodes and edges randomly.
        """
        graph = deepcopy(graph)
        num_nodes = len(graph['nodes'])
        edges=graph['edges']

        # Mutate nodes
        for i in range(num_nodes):
            if random.random() < node_mutation_rate:
                #print('mutate nodes')
                if graph['nodes'][i]=='log':
                    edge_index=[j for j in range(len(edges)) if edges[j][0] == i][0]
                    graph['edge_attr'][edge_index]= random.choice([10,math.e])
                if graph['nodes'][i] in Triangle_Operator:
                    graph['nodes'][i]=random.choice(Triangle_Operator)

        # Mutate subgraphs--search more complex forms:
        if (random.random()<graph_mutation_rate) and (len(graph['nodes']))>1:
            num_nodes= len(graph['nodes'])
            node_indices=[i for i in range(len(graph['nodes']))]
            mutate_node= random.randint(1, len(graph['nodes']) - 1)
            parent_node_index = \
                [graph['edges'][i][0] for i in range(len(graph['edges'])) if graph['edges'][i][1] == mutate_node][0]

            if graph['nodes'][mutate_node] in ['mul','add']:
                child_edge_indice= [i for i in range(len(edges)) if edges[i][0] == mutate_node]
                mutate_edge_index=random.choice(child_edge_indice)
                mutate_node_index=edges[mutate_edge_index][1]
                graph,node_indices=self.delete_subgraph_from_node(graph,mutate_node_index)
                select_template = random.choices(['log','tri','exp','poly','x'],weights=[0.1,0.1,0.25,0.25,0.3],k=1)[0]
                if select_template=='log':
                    graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_log_template(),
                                                                                mutate_node,
                                                                                graph['nodes'][mutate_node],
                                                                                set_maximum_node_num=num_nodes,
                                                                                with_node_indice=node_indices)
                elif select_template=='tri':
                    graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_triangle_template(),
                                                                                mutate_node,
                                                                                graph['nodes'][mutate_node],
                                                                                set_maximum_node_num=num_nodes,
                                                                                with_node_indice=node_indices)
                elif select_template=='poly':
                    graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(),
                                                                                mutate_node,
                                                                                graph['nodes'][mutate_node],
                                                                                set_maximum_node_num=num_nodes,
                                                                                with_node_indice=node_indices)
                elif select_template=='exp':
                    graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                self.generate_exp_template(),
                                                                                mutate_node,
                                                                                graph['nodes'][mutate_node],
                                                                                set_maximum_node_num=num_nodes,
                                                                                with_node_indice=node_indices)
                elif select_template=='x':
                    graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                self.generate_single_poly(),
                                                                                mutate_node,
                                                                                graph['nodes'][mutate_node],
                                                                                set_maximum_node_num=num_nodes,
                                                                                with_node_indice=node_indices)
                node_indices=node_indices+concate_node_indices
                graph=self.renumber_subgraph(graph,node_indices)

            elif graph['nodes'][mutate_node] in ['log']+Triangle_Operator:
                graph, node_indices = self.delete_subgraph_from_node(graph, mutate_node)
                select_template=random.choice(['log','poly','tri','exp','x'])
                if select_template=='log':
                    graph,concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_log_template(), parent_node_index,
                                                          graph['nodes'][parent_node_index],set_maximum_node_num=num_nodes,with_node_indice=node_indices)
                if select_template=='poly':
                    graph,concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), parent_node_index,
                                                           graph['nodes'][parent_node_index],
                                                          set_maximum_node_num=num_nodes,with_node_indice=node_indices)
                if select_template=='tri':
                    graph,concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_triangle_template(), parent_node_index,
                                                           graph['nodes'][parent_node_index],
                                                          set_maximum_node_num=num_nodes,with_node_indice=node_indices)
                if select_template=='exp':
                    graph,concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_exp_template(), parent_node_index,
                                                           graph['nodes'][parent_node_index],
                                                          set_maximum_node_num=num_nodes,with_node_indice=node_indices)
                if select_template=='x':
                    graph,concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_single_poly(), parent_node_index,
                                                           graph['nodes'][parent_node_index],
                                                          set_maximum_node_num=num_nodes,with_node_indice=node_indices)
                node_indices = node_indices + concate_node_indices
                graph = self.renumber_subgraph(graph,node_indices)
            elif graph['nodes'][mutate_node]=='exp':
                graph, node_indices = self.delete_subgraph_from_node(graph, mutate_node)
                graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_single_poly(),
                                                                            parent_node_index,
                                                                            graph['nodes'][parent_node_index],
                                                                            set_maximum_node_num=num_nodes,
                                                                            with_node_indice=node_indices)
                node_indices = node_indices + concate_node_indices
                graph = self.renumber_subgraph(graph, node_indices)



        #delete edges
        if random.random()<graph_delete_graph:
            if len(graph['nodes'])>1:
                mutate_node= random.randint(1, len(graph['nodes']) - 1)
                parent_node_index = \
                [graph['edges'][i][0] for i in range(len(graph['edges'])) if graph['edges'][i][1] == mutate_node][0]
                parent_node_attr = \
                [graph['edge_attr'][i] for i in range(len(graph['edges'])) if graph['edges'][i][1] == mutate_node][0]
                mutate_node_value=graph['nodes'][mutate_node]
                max_node_num=len(graph['nodes'])
                subgraph_nodes, subgraph_edges, subgraph_edge_attr=self.extract_subgraph(graph,mutate_node)
                graph_node_index = [i for i in range(len(graph['nodes']))]
                graph['nodes'] = [element for index, element in enumerate(graph['nodes']) if index not in subgraph_nodes]
                graph_node_index = [element for index, element in enumerate(graph_node_index) if index not in subgraph_nodes]
                new_edges_1 = []
                new_edge_attr_1 = []
                for index, edge_info in enumerate(zip(graph['edges'], graph['edge_attr'])):
                    if edge_info[0] not in subgraph_edges:
                        if edge_info[0][1] not in subgraph_nodes:
                            new_edges_1.append(edge_info[0])
                            new_edge_attr_1.append(edge_info[1])
                graph['edges'] = new_edges_1
                graph['edge_attr'] = new_edge_attr_1
                graph['nodes'].append('1')
                graph_node_index.append(max_node_num+1)
                graph['edges'].append([parent_node_index,max_node_num+1])
                graph['edge_attr'].append(1)
                graph=self.renumber_subgraph(graph,graph_node_index)

        #mutate edge_attr
        for mutate_edge_attr_index in range(len(graph['edge_attr'])):
            if random.random()<mutate_edge_attr_prob:
                mutate_edge=graph['edges'][mutate_edge_attr_index]
                begin_node=graph['nodes'][mutate_edge[0]]
                if begin_node=='add':
                    graph['edge_attr'][mutate_edge_attr_index]=random.choices([1,-1,-1e8],[0.45,0.45,0.1],k=1)[0]
                elif begin_node=='exp':
                    graph['edge_attr'][mutate_edge_attr_index] =random.choices([-2,-1,-0.5,0.5,1,2,-1e8],[0.1,0.15,0.1,0.1,0.15,0.1,0.3],k=1)[0]
                elif begin_node=='log':
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([10,math.e])
                elif begin_node in Triangle_Operator:
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([1,2,math.pi,2*math.pi,-1e8])
                elif begin_node=='mul':
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([1, -1,-1e8])

        return graph

    def generate_initial_guesses(self,len_variables, num_samples=20):
        """
        Generate initial guesses for parameters.
        :param bounds: List of (min, max) tuples for each parameter.
        :param num_samples: Number of initial guesses to generate.
        :return: List of initial guesses.
        """
        guesses = []
        for _ in range(num_samples):
            guess=[1,1]
            for _ in range(len_variables):
                guess.append(np.random.choice([-2,-1,1,2]))
            guesses.append(guess)
        return guesses

    def find_best_initial_guess(self,len_variables, lambdified_expr,x_data, y_data):
        """
        Find the best initial guess for optimization.
        :param bounds: List of (min, max) tuples for each parameter.
        :param lambdified_expr: Lambdified sympy expression.
        :param x_data: Array of x values.
        :param y_data: Array of y values.
        :param num_samples: Number of guesses to evaluate.
        :return: Best initial guess.
        """
        guesses = self.generate_initial_guesses(len_variables)

        best_guess = None
        best_score = float('inf')

        def objective(params, lambdified_expr, x_data, y_data):
            y_pred = lambdified_expr(x_data, *params)
            mse = np.mean((y_pred - y_data) ** 2)
            return mse

        for guess in guesses:
            result = minimize(objective, guess, args=(lambdified_expr, x_data, y_data), method='BFGS',
                              options={'disp': False})
            score= result.fun
            #score = objective(guess, lambdified_expr, x_data, y_data)
            #print('guess:',guess,score)
            if score < best_score:
                best_score = score
                best_guess = guess
        if best_guess==None:
            best_guess=guesses[0]
        # print('best_score:',best_score)
        # print('best_guess:',best_guess)
        return best_guess

    def get_fitness_from_graph(self,graph):
        """
        Calculates the fitness score and optimal parameters for a given graph by converting it to a symbolic expression,
        lambdifying the expression, and minimizing the mean squared error (MSE) between predicted and actual data.

        :param graph: A dictionary representing the graph structure to be converted into a symbolic expression.
                      The graph should conform to the expected format for `graph_to_sympy`.
        :return: A tuple containing the average fitness score (float) and a list of optimal parameter arrays (numpy.ndarray)
                 for each dataset pair in `x_data` and `y_data`.

        :raises TypeError: If the input graph is not in the expected format or if the symbolic conversion fails.
        :raises ValueError: If the optimization process encounters invalid inputs or fails to converge.

        Notes
        -----
        The fitness score is computed as the average mean squared error (MSE) across all dataset pairs in `x_data` and `y_data`.
        The optimization process uses the BFGS method to minimize the objective function, which evaluates the MSE for
        predicted values based on the lambdified expression.

        The `graph_to_sympy` method is used internally to convert the graph into a symbolic expression. Ensure that this method
        is implemented and functional before using this function.

        The `find_best_initial_guess` method is optionally used to determine an initial guess for optimization parameters if
        `seek_best_initial` is set to True. Otherwise, a default array of ones is used.

        Warnings
        --------
        This function relies on external libraries such as `sympy`, `numpy`, and `scipy.optimize`. Ensure these libraries are
        installed and properly configured in your environment.

        See Also
        --------
        graph_to_sympy : Converts a graph structure into a symbolic mathematical expression.
        lambdify : Converts symbolic expressions into callable functions for numerical evaluation.
        minimize : Performs optimization using various methods, including BFGS.
        """
        expr=self.graph_to_sympy(graph)
        #print(expr)
        variables = list(expr.free_symbols)
        variables=[x,A,B]+[sym for sym in variables if sym not in [x,A,B]]
        def lambdify_expression(expr, symbols):
            return lambdify(symbols, expr, 'numpy')

        lambdified_expr=lambdify_expression(expr, variables)

        def objective(params, lambdified_expr, x_data,y_data):
            # For each expression, calculate the corresponding y values and sum them
            y_pred = lambdified_expr(x_data, *params)  # Sum the contributions from each expression
            MSE=np.mean((y_pred-y_data)**2)
            return MSE
        fitness=0
        optimal_params=[]

        for x_d,y in zip(self.x_data,self.y_data):
            if self.seek_best_initial == True:
                best_guess = self.find_best_initial_guess(len(variables) - 3, lambdified_expr, x_d, y)
            else:
                best_guess = np.ones([len(variables) - 1])

            result = minimize(objective, best_guess, args=(lambdified_expr, x_d, y), method='BFGS', options={'disp': False})

            optimal_params.append(result.x)
            fitness+=result.fun
        fitness=fitness/len(self.y_data)#*(1+self.epi*len(variables)+self.epi*len(graph['nodes']))


        return fitness,optimal_params

    def get_fitness_from_expr(self,expr,x_data,y_data):
        """
        Evaluates the fitness of a given symbolic expression against provided data by optimizing parameters and calculating error metrics.

        This method processes a symbolic mathematical expression, identifies its free variables, and computes an optimal set of
        parameters to minimize the mean squared error (MSE) between predicted and actual data. It uses numerical optimization
        to fit the expression to the data and calculates a fitness score based on the average MSE adjusted by the complexity
        of the expression.

        :param expr: The symbolic mathematical expression to evaluate. Must support free_symbols attribute.
        :type expr: sympy.Expr
        :param x_data: Input data points for the independent variable(s). Each element corresponds to a set of inputs.
        :type x_data: list[numpy.ndarray] or numpy.ndarray
        :param y_data: Output data points corresponding to the dependent variable. Each element corresponds to a target value.
        :type y_data: list[numpy.ndarray] or numpy.ndarray
        :return: A tuple containing the computed fitness score and a list of optimal parameter sets for each data point.
                 The fitness score is a float representing the normalized error metric, and the optimal parameters are
                 stored as lists of floats.
        :rtype: tuple[float, list[list[float]]]

        """
        try:
            variables = list(expr.free_symbols)
            variables = [x, A, B] + [sym for sym in variables if sym not in [x, A, B]]
        except AttributeError:
            return 1e8,0


        def lambdify_expression(expr, symbols):
            return lambdify(symbols, expr, 'numpy')

        lambdified_expr = lambdify_expression(expr, variables)

        def objective(params, lambdified_expr, x_data, y_data):
            # For each expression, calculate the corresponding y values and sum them
            y_pred = lambdified_expr(x_data, *params)  # Sum the contributions from each expression

            MSE = np.mean((y_pred - y_data) ** 2)


            return MSE

        fitness = 0
        eps = 1e-12  # small epsilon to avoid division by zero
        total_relative_error = 0.0
        optimal_params = []

        for x_d,y in zip(x_data,y_data):
            if self.seek_best_initial==True:
                best_guess = self.find_best_initial_guess(len(variables) - 3, lambdified_expr, x_d, y)
            else:
                best_guess = np.ones([len(variables) - 1])
            # print('best_guess_use:',best_guess)
            result = minimize(objective, best_guess, args=(lambdified_expr, x_d, y), method='BFGS',
                              options={'disp': False})
            # print('optimized result:',result.x)
            optimal_params.append(result.x)
            fitness += result.fun

        fitness = fitness / len(y_data)*(1+self.epi*len(variables))

        return fitness, optimal_params

    def get_regressed_function(self,graph):
        """
        Converts a computational graph into a regressed symbolic function by simplifying
        the expression and substituting optimal parameters. The method extracts free
        symbols from the graph, computes fitness to determine optimal parameters, and
        constructs simplified symbolic expressions for each set of parameters.

        :param graph: A computational graph representing the symbolic structure of the
                      function to be regressed. It is expected to be compatible with
                      the internal conversion methods of this class.
        :type graph: object

        :return: A list of simplified symbolic expressions derived from the input graph.
                 Each expression corresponds to a set of optimal parameters computed
                 during the regression process.
        :rtype: list[sp.Expr]

        """
        expr = self.graph_to_sympy(graph)
        variables = list(expr.free_symbols)
        variables = [A, B] + [sym for sym in variables if sym not in [x, A, B]]
        fitness,optimal_params=self.get_fitness_from_graph(graph)
        simplified_expr=[]
        for j in range(len(self.y_data)):
            param_dict={}
            for i in range(len(variables)):
                param_dict[variables[i]]=optimal_params[j][i]
            substituted_expr = expr.subs(param_dict)
            simplified_expr.append(sp.simplify(substituted_expr))
        return simplified_expr

    def get_regressed_function_from_expr(self,expr,optimal_params):
        """
        Constructs a list of simplified symbolic expressions by substituting optimal
        parameters into the given symbolic expression.

        This method iterates over the provided data points and substitutes the optimal
        parameters into the symbolic expression for each data point. The substituted
        expressions are then simplified using symbolic computation. The resulting
        simplified expressions are returned as a list.

        :param expr: The symbolic mathematical expression to be regressed. It should
                     contain free symbols representing variables and parameters.
        :type expr: sympy.Expr
        :param optimal_params: A list of lists containing the optimal parameter values
                               corresponding to each data point. Each inner list
                               contains values for the free symbols in the same order
                               as they appear in the expression.
        :type optimal_params: list[list[float]]
        :return: A list of simplified symbolic expressions obtained after substituting
                 the optimal parameters and simplifying the result for each data point.
        :rtype: list[sympy.Expr]

        """
        variables = list(expr.free_symbols)
        variables = [A, B] + [sym for sym in variables if sym not in [x, A, B]]
        simplified_expr=[]
        for j in range(len(self.y_data)):
            param_dict={}
            for i in range(len(variables)):
                param_dict[variables[i]]=optimal_params[j][i]
            substituted_expr = expr.subs(param_dict)
            simplified_expr.append(sp.simplify(substituted_expr))
        return simplified_expr

    def get_function_prediction(self,expr,x_data):
        """
        Predicts function values based on a given mathematical expression and input data.

        This method takes a symbolic mathematical expression and a set of input data points,
        then evaluates the expression for each input value to generate corresponding predictions.
        It uses a helper function to convert the symbolic expression into a numerical function
        that can operate on numpy arrays.

        :param expr: The symbolic mathematical expression to evaluate. It should be compatible
                     with SymPy's symbolic representation.
        :type expr: sympy.Expr
        :param x_data: The input data points for which the expression will be evaluated. This
                       should be a numpy array or any iterable structure containing numerical
                       values.
        :type x_data: numpy.ndarray
        :return: An array of predicted values obtained by evaluating the expression for each
                 input data point in `x_data`.
        :rtype: numpy.ndarray

        """
        def lambdify_expression(expr, symbols):
            return lambdify(symbols, expr, 'numpy')

        # Create the lambda function for the expression
        lambdified_expr = lambdify_expression(expr, [x])

        # Use the lambdified function to calculate y_data for each x in x_data
        pred = lambdified_expr(x_data)
        return pred

    def sorted(self,graph_list,fitness_list):
        combined = list(zip(graph_list, fitness_list))

        combined_sorted = sorted(combined, key=lambda x: x[1])

        graph_sorted, fitness_sorted = zip(*combined_sorted)
        graph_sorted = list(graph_sorted)
        fitness_sorted = list(fitness_sorted)
        return graph_sorted,fitness_sorted

    def distinction(self,graphs):
        for i in range(1,len(graphs)):
            graphs[i]=self.generate_random_graph()
        return graphs

    def elimiate_length(self,graph):
        flag=1
        max_iter_num=10
        iter_num=0
        while flag==1:
            edge_length=len(graph['edges'])
            expr = self.graph_to_sympy(graph)
            try:
                variables = list(expr.free_symbols)
            except AttributeError:
                return graph
            len_variable=len(variables)
            graph=self.mutate(graph)
            if (edge_length<self.max_edge_num) and (len_variable<self.max_variable_num):
                flag=0
            if iter_num==max_iter_num:
                flag=0
            iter_num+=1
        return graph

    def elimiate_var_num(self,graph):
        edge_attr=graph['edge_attr']
        # Find indices where the value is -1e8
        neg_inf_indices = [i for i, val in enumerate(edge_attr) if val == -1e8]

        # If we already meet the requirement, no change needed
        if len(neg_inf_indices) <= self.max_unconstant:
            return graph

        # Randomly choose which indices to keep as -1e8
        keep_indices = set(random.sample(neg_inf_indices, self.max_unconstant))

        # Create a new list with modifications
        new_edge_attr = []
        for i, val in enumerate(edge_attr):
            if val == -1e8 and i not in keep_indices:
                # Replace with either 1 or -1
                new_edge_attr.append(random.choice([1, -1]))
            else:
                new_edge_attr.append(val)
        graph['edge_attr']=new_edge_attr
        return graph

    def parallel_get_fitness(self,exprs,x_data,y_data, get_fitness_from_expr):
        """
        Evaluates the fitness of multiple expressions in parallel using a process
        pool executor. This function maps the provided fitness evaluation function
        across the given expressions and corresponding data points to compute results
        efficiently.

        :param exprs: List of expressions to evaluate. Each expression is passed to
                      the fitness evaluation function.
        :type exprs: list
        :param x_data: Input data corresponding to the expressions. This data is used
                       as part of the fitness evaluation.
        :type x_data: list
        :param y_data: Target or output data corresponding to the expressions. This
                       data is used as part of the fitness evaluation.
        :type y_data: list
        :param get_fitness_from_expr: A callable function that computes the fitness
                                      of a single expression given the corresponding
                                      input and output data.
        :type get_fitness_from_expr: Callable[[Any, Any, Any], float]
        :return: A list of fitness values computed for each expression.
        :rtype: list[float]
        """
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(get_fitness_from_expr, exprs, x_data, y_data))
        return results

    def evolution(self,save_dir='default'):
        """
        Performs the evolution process to generate and optimize graphs based on fitness values.

        This method initializes populations of graphs, evaluates their fitness, and evolves them
        through crossover, mutation, and selection over a defined number of generations. It supports
        both parallel and sequential computation for fitness evaluation. Intermediate results are
        saved periodically during the evolution process.

        :param save_dir: Directory name where intermediate results will be saved. Defaults to 'default'.
        :type save_dir: str

        :return: None. Results are saved to files within the specified directory.
        """
        self.graphs=[]
        self.fitnesses=[]
        self.best_graphs_record=[]
        self.best_fitness_record=[]
        self.exprs=[]
        all_x_data=[]
        all_y_data=[]

        if self.use_parallel_computing==True:
            for i in range(self.size_pop):
                graph=self.generate_random_graph()
                graph=self.elimiate_var_num(graph)
                self.graphs.append(graph)
                expr = self.graph_to_sympy(graph)
                self.exprs.append(expr)
                all_x_data.append(self.x_data)
                all_y_data.append(self.y_data)
            results = self.parallel_get_fitness(self.exprs, all_x_data, all_y_data, self.get_fitness_from_expr)
            self.fitnesses = [1e18 if pd.isna(f[0]) else f[0] for f in results]

        if self.use_parallel_computing==False:
            for i in range(self.size_pop):
                graph = self.generate_random_graph()
                self.graphs.append(graph)
                fitness = self.get_fitness_from_graph(graph)[0]
                if pd.isna(fitness) == True:
                    fitness = 1e18
                self.fitnesses.append(fitness)

        self.graphs,self.fitnesses=self.sorted(self.graphs,self.fitnesses)
        print(self.fitnesses[0:5])
        print([self.graph_to_sympy(graph) for graph in self.graphs[0:5]])
        best_graph = {0: self.graphs[0], 1: self.graphs[1], 2: self.graphs[2], 3: self.graphs[3], 4: self.graphs[4]}
        best_fitness = {0: self.fitnesses[0], 1: self.fitnesses[1], 2: self.fitnesses[2], 3: self.fitnesses[3],
                        4: self.fitnesses[4]}
        self.best_graphs_record.append(best_graph)
        self.best_fitness_record.append(best_fitness)

        distinction_flag=0
        for iter_num in tqdm(range(self.generation_num)):
            new_graphs=list(best_graph.values())
            new_fitness_list=list(best_fitness.values())

            for i in range(self.size_pop):
                parent1=self.graphs[i]
                parent2=self.graphs[random.randint(0,self.size_pop-1)]

                # Perform crossover
                offspring = self.cross_over(parent1, parent2)

                # Perform mutation
                offspring = self.mutate(offspring)
                offspring=self.elimiate_length(offspring)
                offspring=self.elimiate_var_num(offspring)

                if self.use_parallel_computing==False:
                    fitness,coef = self.get_fitness_from_graph(offspring)

                    if pd.isna(fitness)==True:
                        fitness=1e8
                    new_fitness_list.append(fitness)

                new_graphs.append(offspring)

            if self.use_parallel_computing==True:
                new_exprs=[self.graph_to_sympy(graph) for graph in new_graphs[5:]]
                results = self.parallel_get_fitness(new_exprs, all_x_data, all_y_data, self.get_fitness_from_expr)
                new_fitness_list += [1e18 if pd.isna(f[0]) else f[0] for f in results]



            #sort
            re1 = list(map(new_fitness_list.index, heapq.nsmallest(int(self.size_pop / 2), new_fitness_list)))

            sorted_graph=[]
            sorted_fitness= []
            for index in re1:
                if new_fitness_list[index] not in sorted_fitness:
                    sorted_graph.append(new_graphs[index])
                    sorted_fitness.append(new_fitness_list[index])
            for index in range(self.size_pop-len(sorted_fitness)):
                sorted_graph.append(self.generate_random_graph())
            self.graphs=sorted_graph
            self.fitnesses=sorted_fitness
            print(self.fitnesses[0:5])
            print([self.graph_to_sympy(graph) for graph in self.graphs[0:5]])
            print('best graph:',self.graphs[0])
            if self.fitnesses[0]==best_fitness[0]:
                distinction_flag+=1
            else:
                distinction_flag=0

            best_graph={0:self.graphs[0],1:self.graphs[1],2:self.graphs[2],3:self.graphs[3],4:self.graphs[4]}
            best_fitness={0:self.fitnesses[0],1:self.fitnesses[1],2:self.fitnesses[2],3:self.fitnesses[3],4:self.fitnesses[4]}
            self.best_graphs_record.append(best_graph)
            self.best_fitness_record.append(best_fitness)
            if distinction_flag==self.distinction_epoch:
                distinction_flag=0
                self.graphs=self.distinction(self.graphs)

            try:
                os.makedirs(f'result_save/{save_dir}/')
            except OSError:
                pass

            if (iter_num+1)%10==0:
                pickle.dump(self.best_graphs_record,open(f'result_save/{save_dir}/best_graphs.pkl', 'wb'))
                pickle.dump(self.best_fitness_record,open(f'result_save/{save_dir}/best_fitness.pkl', 'wb'))