import json
import numpy as np
import networkx as nx


class GraphLoader:
    '''
    A class to load the graph from a json file

    Attributes
    ----------
    data_path : str
        path to the json file containing the graph data
    actor_id_data_path : str
        path to the json file containing the actor id data
    links : list
        list of links in the graph
    nodes : list
        list of nodes in the graph
    id_actor : dict
        dictionary containing the actor id data
    graph : nx.Graph
        networkx graph object

    Methods
    -------
    load_data()
        Load the data from the json file
    load_actor_id()
        Load the actor id data from the json file
    construct_networks()
        Construct the networkx graph object
    get_graph()
        Get the networkx graph object
    get_id_actor_dict()
        Get the actor id data
    '''

    def __init__(self, data_path, actor_id_data_path):
        self.data_path = data_path
        self.actor_id_data_path = actor_id_data_path
        self.links = []
        self.nodes = []
        self.id_actor = {}

    def load_data(self):
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        self.nodes = data['nodes']
        self.links = data['links']

    def load_actor_id(self):
        with open(self.actor_id_data_path, 'r') as file:
            data = json.load(file)
        self.id_actor = data

    def construct_networks(self):
        self.graph = nx.Graph()
        for node in self.nodes:
            self.graph.add_node(node['id'])
            attrs = {key: value for key, value in node.items() if key != 'id'}
            attrs.update({'x': [float(i) for i in list(attrs.values())[1:3]]}) # array with gender and popularity
            nx.set_node_attributes(self.graph, {node['id']: attrs})
            # nx.set_node_attributes(self.graph, {node['id']: x})

        for link in self.links:
            self.graph.add_edge(link['source'], link['target'], category=link['value'])
        
        # Compute graph stats (pagerank, betweenness, closeness)
        pr = nx.pagerank(self.graph)
        bt = nx.betweenness_centrality(self.graph)
        cn = nx.closeness_centrality(self.graph)
        for n in self.graph.nodes:
            x_stat = self.graph.nodes[n]['x'].copy()
            x_stat.append(pr[n])
            x_stat.append(bt[n])
            x_stat.append(cn[n])
            nx.set_node_attributes(self.graph, {n: {'x_stat': x_stat}})

    def get_graph(self):
        '''
        Get the networkx graph object

        Returns
        -------
        nx.Graph
            networkx graph object
        '''
        self.load_data()
        if self.actor_id_data_path:
            self.load_actor_id()
        self.construct_networks()
        return self.graph

    def get_id_actor_dict(self):
        '''
        Get the actor id data dictionary

        Returns
        -------
        dict
            dictionary containing the actor id data
        '''
        return self.id_actor
