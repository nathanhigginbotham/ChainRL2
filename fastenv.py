import numpy as np
import pickle as pkl
import numba


np.random.seed(42)


class FastSupplyChain():

    def __init__(self, setup_path, batch_size, time_length):
        self.batch_size = batch_size
        self.time_length = time_length

        with open(setup_path, 'rb') as f:
            data = pkl.load(f)

        print(f'\nLoaded {setup_path}:\n')

        for attr, value in data.items():
            setattr(self, attr, value)
            print(f'{attr} = {value}')

        self.node_look_ahead_time = 8
        self.edge_look_ahead_time = 16

        # Define the nodes that will serve demand and the demand time series for each demand node

        self.demand_nodes = np.array([0])
        self.demand = np.random.poisson(lam=10, size=(len(self.demand_nodes), self.time_length))
        self.sale_price = np.array([2.2])

        # Define the nodes that will recieve raw supplied and the supply time series for each supply node
        
        self.supply_nodes = np.array([7, 8])
        self.supply = np.random.poisson(lam=1, size=(len(self.supply_nodes), self.time_length))

        # Specifies which runs in the batch dimension will be recorded
        # We do not record everything by default as to save memory

        self.record_ids = np.array([0])
        self.record_size = len(self.record_ids)

        self.reset()

    def reset(self):        
        self.node_inv = np.zeros(shape=(self.batch_size, self.num_nodes, self.node_look_ahead_time))
        self.edge_inv = np.zeros(shape=(self.batch_size, self.num_edges, self.edge_look_ahead_time))
        self.node_profits = np.zeros(shape=(self.batch_size, self.num_nodes))

        # Initialise the node inventories to the specified initialisation values

        self.node_inv[:,:,0] = np.repeat(self.inv_init[np.newaxis,...], self.batch_size, axis=0)

        # Setup the arrays that will record the specified runs in the batch

        self.record_node_profits = np.zeros(shape=(self.record_size, self.num_nodes, self.time_length))
        self.record_node_inv = np.zeros(shape=(self.record_size, self.num_nodes, self.node_look_ahead_time, self.time_length))
        self.record_edge_inv = np.zeros(shape=(self.record_size, self.num_edges, self.edge_look_ahead_time, self.time_length))
        self.record_actions = np.zeros(shape=(self.record_size, self.num_edges, self.time_length))

        self.time = 0

        # Each float32 uses 32 bits = 4 bytes of memory

        for attr in ['node_inv', 'edge_inv', 'node_profits', 'record_node_profits', 'record_node_inv', 'record_edge_inv', 'record_actions']:
            print(f'{attr}: {getattr(self, attr).size * 4 /(1024 * 1024)} MB')

    def step(self, action):

        assert action.shape[0] == self.batch_size, 'action should have first dimension equal to the batch size'
        assert action.shape[1] == self.num_edges, 'action should second dimension equal to the number of edges'

        self.node_profits = np.zeros(shape=(self.batch_size, self.num_nodes))

        for edge in range(self.num_edges):
            # Get the nodes that are going to do the order

            supplier = self.edges[0, edge]
            purchaser = self.edges[1, edge]

            # Requested stock cannot be bigger than the available stock

            supplier_stock = self.node_inv[:, supplier, 0]

            requested_stock = np.round(action[:,edge], 0).astype(np.float32)
            available_requested_stock = np.min((requested_stock, supplier_stock), axis=0).copy()

            # Requested stock cannot be bigger than the space available in the requesters inventory

            space_available = self.inv_capacity[purchaser] - self.node_inv[:, purchaser, 0]
            purchased_stock = np.min((available_requested_stock * self.prod_yield[purchaser], space_available), axis=0).copy()

            # Move the stock from the supplier node inv to the purchaser edge pipeline

            self.edge_inv[:, edge, self.trans_time[edge]] += purchased_stock
            self.node_inv[:, supplier, 0] -= purchased_stock

            # Settle the cost of purchasing the stock

            stock_cost = purchased_stock * self.cost[edge]

            self.node_profits[:, supplier] += stock_cost 
            self.node_profits[:, purchaser] -= stock_cost 

            # Move any stock that has arrived from an edge into the purchaser inv and shift the edge along

            arrived_stock = self.edge_inv[:, edge, 0].copy()
            self.node_inv[:, purchaser, self.prod_time[purchaser]] += arrived_stock * self.prod_yield[edge]
            self.edge_inv[:, edge, 0] = 0.0
            self.edge_inv[:, edge, :] = np.roll(self.edge_inv[:, edge, :], -1)

        for k in range(self.num_nodes):
            node = self.nodes[k]

            # Move any stock that has completed manufacturing into the node env and shift manufacturing pipeline along

            preexisting_manufactured_stock = self.node_inv[:, node, 0].copy()

            self.node_inv[:, node, 0] = 0.0
            self.node_inv[:, node, :] = np.roll(self.node_inv[:, node, :], -1)
            new_manufactured_stock = self.node_inv[:, node, 0].copy()
            self.node_inv[:, node, 0] += preexisting_manufactured_stock

            # Apply manufacture costs

            self.node_profits[:, node] -= new_manufactured_stock * self.prod_cost[node]

            # Apply node holding costs

            self.node_profits[:, node] -= self.node_inv[:, node,0] * self.node_hold_cost[node]

        # Realise the market demand

        for k in range(len(self.demand_nodes)):
            node = self.demand_nodes[k]

            requested_demand = self.demand[k, self.time].repeat(self.batch_size, axis=0)
            purchased_demand = np.min((requested_demand, self.node_inv[:, node, 0]), axis=0)
  
            self.node_inv[:, node, 0] -= purchased_demand
            self.node_profits[:, node] += purchased_demand * self.sale_price[k]

        # Realise supply

        for k in range(len(self.supply_nodes)):
            node = self.supply_nodes[k]

            self.node_inv[:, node, self.prod_time[node]] = self.supply[k, self.time].repeat(self.batch_size, axis=0)

        self.record_actions[..., self.time] = action[self.record_ids]
        self.record_node_profits[..., self.time] = self.node_profits[self.record_ids]
        self.record_node_inv[..., self.time] = self.node_inv[self.record_ids]
        self.record_edge_inv[..., self.time] = self.edge_inv[self.record_ids]

        self.time += 1

        if self.time < self.time_length:
            demand_state = np.expand_dims(self.demand[:,self.time].repeat(self.batch_size, axis=0), 1)
            node_state = self.node_inv.reshape(self.batch_size, self.num_nodes * self.node_look_ahead_time)
            edge_state = self.edge_inv.reshape(self.batch_size, self.num_edges * self.edge_look_ahead_time)
            state = np.concatenate([demand_state, node_state, edge_state], axis=1)
            done = False
        
        else:
            state = None
            done = True
        
        return state, done

