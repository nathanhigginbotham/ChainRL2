import numpy as np
import pickle as pkl

np.random.seed(0)

# A default node just immediately processes stock at no cost
# A default edge just immediately transport stock at no cost

node_attrs = {'inv_init': np.float32, 'prod_cost': np.float32, 'prod_time': np.int32, 
              'node_hold_cost': np.float32, 'prod_capacity': np.float32, 'inv_capacity': np.float32}
edge_attrs = {'trans_time': np.int32, 'cost': np.float32, 'edge_hold_cost': np.float32, 'prod_yield': np.float32}

inf = np.inf

nodes = {
    0: {'inv_init': 4.0, 'prod_cost': 0.1, 'prod_time': 0, 'node_hold_cost': 0.01, 'prod_capacity': inf, 'inv_capacity': inf},
    1: {'inv_init': 12.0, 'prod_cost': 0.1, 'prod_time': 2, 'node_hold_cost': 0.01, 'prod_capacity': inf, 'inv_capacity': inf},
    2: {'inv_init': 10.0, 'prod_cost': 0.1, 'prod_time': 3, 'node_hold_cost': 0.01, 'prod_capacity': inf, 'inv_capacity': inf},
    3: {'inv_init': 7.0, 'prod_cost': 0.1, 'prod_time': 4, 'node_hold_cost': 0.01, 'prod_capacity': inf, 'inv_capacity': inf},
    4: {'inv_init': 8.0, 'prod_cost': 0.1, 'prod_time': 2, 'node_hold_cost': 0.01, 'prod_capacity': inf, 'inv_capacity': inf},
    5: {'inv_init': 6.0, 'prod_cost': 0.1, 'prod_time': 1, 'node_hold_cost': 0.01, 'prod_capacity': inf, 'inv_capacity': inf},
    6: {'inv_init': 0.0, 'prod_cost': 0.1, 'prod_time': 4, 'node_hold_cost': 0.01, 'prod_capacity': inf, 'inv_capacity': inf},
    7: {'inv_init': 3.0, 'prod_cost': 0.0, 'prod_time': 0, 'node_hold_cost': 0.00, 'prod_capacity': inf, 'inv_capacity': inf},
    8: {'inv_init': 10.0, 'prod_cost': 0.0, 'prod_time': 0, 'node_hold_cost': 0.00, 'prod_capacity': inf, 'inv_capacity': inf}}

edges = {
    (8,6): {'trans_time': 0, 'cost': 0.200, 'edge_hold_cost': 0.000, 'prod_yield': 1.0},
    (8,5): {'trans_time': 2, 'cost': 0.070, 'edge_hold_cost': 0.002, 'prod_yield': 1.0},
    (7,5): {'trans_time': 1, 'cost': 0.050, 'edge_hold_cost': 0.005, 'prod_yield': 1.0},
    (7,4): {'trans_time': 0, 'cost': 0.150, 'edge_hold_cost': 0.000, 'prod_yield': 1.0},
    (6,3): {'trans_time': 12, 'cost': 0.800,'edge_hold_cost': 0.004, 'prod_yield': 1.0},
    (6,2): {'trans_time': 11, 'cost': 0.750, 'edge_hold_cost': 0.007, 'prod_yield': 1.0},
    (5,2): {'trans_time': 9, 'cost': 0.700, 'edge_hold_cost': 0.005, 'prod_yield': 1.0},
    (4,3): {'trans_time': 10, 'cost': 0.800, 'edge_hold_cost': 0.006, 'prod_yield': 1.0},
    (4,2): {'trans_time': 8, 'cost': 1.000, 'edge_hold_cost': 0.008, 'prod_yield': 1.0},
    (3,1): {'trans_time': 3, 'cost': 1.600, 'edge_hold_cost': 0.015, 'prod_yield': 1.0},
    (2,1): {'trans_time': 5, 'cost': 1.500, 'edge_hold_cost': 0.010, 'prod_yield': 1.0},
    (1,0): {'trans_time': 2, 'cost': 2.000, 'edge_hold_cost': 0.100, 'prod_yield': 1.0}}

data = {}

data['num_nodes'] = len(nodes)
data['num_edges'] = len(edges)

data['nodes'] = np.empty(shape=(data['num_nodes']), dtype=np.int32)
data['edges'] = np.empty(shape=(2, data['num_edges']), dtype=np.int32)

for attr, dtype in node_attrs.items():
    data[attr] = np.empty(shape=(data['num_nodes']), dtype=dtype)

for attr, dtype in edge_attrs.items():
    data[attr] = np.empty(shape=(data['num_edges']), dtype=dtype)

for k, node_attrs in enumerate(nodes.items()):
    node, attrs = node_attrs
    data['nodes'][k] = node

    for attr, value in attrs.items():
        data[attr][k] = value

for k, edge_attrs in enumerate(edges.items()):
    edge, attrs = edge_attrs
    purchaser, supplier = edge
    data['edges'][0,k] = purchaser
    data['edges'][1,k] = supplier

    for attr, value in attrs.items():
        data[attr][k] = value

with open('setups/test.pkl', 'wb') as f:
    pkl.dump(data, f)