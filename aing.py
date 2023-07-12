# An Adaptive Incremental CLustering Method Based on the Growing Neural Gas Algorithm
# Author: Boubacar Sow
# Date: July 10, 2023
# Copyright (c) 2023 Boubacar Sow


from collections import defaultdict
import random
import numpy as np
import math


class AING:
    
    def __init__(self, up_bound):

        """
        Initializes the AING class.
        
        :param up_bound: The upper bound on the number of neurons in the graph.
        """
        self.up_bound = up_bound
        self.k = 0
        self.G = defaultdict(dict)
        self.d_bar = 0  # mean distance of all existing neurons to the center of mass of the observed data points
    

    def fit(self, data):

        """
        Fits the AING model to the data.
        
        :param data: The data to be clustered.
        """
        # Initializing the graph with the first two nb_points
        self.G[0]['w'] = data[0]
        self.G[1]['w'] = data[1]

        #self.G[0][1] = {'age': 0}
        #self.G[1][0] = {'age': 0}
        
        self.G[0]['nb_points'] = 0
        self.G[1]['nb_points'] = 0

        self.G[0]['sum_distances'] = 0
        self.G[1]['sum_distances'] = 0

        self.G[0]['points'] = []
        self.G[1]['points'] = []

        data_points = []
        
        for x in data[2:]:
            data_points.append(x)
            # Update d_bar
            self.d_bar = np.mean([np.linalg.norm(self.G[i]['w'] - np.mean(data_points, axis=0)) for i in self.G.keys()])
 
            # let y1, y2 the two nearest neurons from x in G
            #print(len(self.G))
            dists = [np.linalg.norm(x - self.G[i]['w']) for i in range(len(self.G))] # if tuple(self.G[i]['w']) != tuple(x)
            y1, y2 = np.argsort(dists)[:2]
            

            # get Ty1 and Ty2 according to formula 1
            Ty1 = self.compute_threshold(y1)
            Ty2 = self.compute_threshold(y2)

            if np.linalg.norm(x - self.G[y1]['w']) > Ty1:
                # create new neuron
                y_new = len(self.G)
                self.G[y_new]['w'] = x
                self.G[y_new]['points'] = []
                self.G[y_new]['nb_points'] = 0
                self.G[y_new]['sum_distances'] = 0
            else:
                if np.linalg.norm(x - self.G[y2]['w']) > Ty2 :
                    # create new neuron and connect to y1
                    y_new = len(self.G)
                    self.G[y_new]['w'] = x
                    self.G[y_new]['points'] = []
                    self.G[y_new]['nb_points'] = 0
                    self.G[y_new]['sum_distances'] = 0
                    self.connect(y_new, y1)
                    #print("COnnecting ", y_new, " to ", y1)
                else:
                    
                    # increase the age of edges emanating from y1
                    for n in self.neighbors(y1):
                        self.G[y1][n]['age'] += 1

                    # update nb_points and sum_distances for y1 and its neighbors
                    self.G[y1]['nb_points'] += 1
                    self.G[y1]['points'].append(x)
                    self.G[y1]['sum_distances'] += np.linalg.norm(x - self.G[y1]['w'])

                    e_b = 1 / (self.assigned_data_nb_points(y1))
                    e_n = 1 / (100 * (self.assigned_data_nb_points(y1)))

                    # update reference vectors for y1 and its neighbors
                    self.G[y1]['w'] += e_b * (x - self.G[y1]['w']) 

                    for n in self.neighbors(y1):
                        self.G[n]['w'] += e_n * (x - self.G[n]['w'])

                    # connect y1 to y2 by a new edge (reset its age to 0 if it already exists)
                    self.connect(y1, y2)

                    # remove old edges from G if any
                    nmax = max([self.assigned_data_nb_points(i) for i in self.G])
                    for i in range(len(self.G)):
                        for j in range(i+1, len(self.G)):
                            if j in self.G[i] and   'age' in self.G[i][j] and self.G[i][j]['age'] > nmax:
                                del self.G[i][j]
                                del self.G[j][i]

            while len(self.G) > self.up_bound:
                # some of the nb_points in the graph
                self.k += self.d_bar
                self.G = self.merge(self.k, self.G)


    def compute_threshold(self, y):

        """
        Computes the threshold for adding a point to a cluster.
        
        :param y: The index of the cluster.
        :return: The threshold for adding a point to the cluster.
        """
        if self.G[y]['nb_points'] > 0 or len(self.neighbors(y)) > 0:
            sum_dists = self.G[y]['sum_distances']
            sum_weighted_dists = sum([self.G[e]['nb_points'] * np.linalg.norm(self.G[y]['w'] - self.G[e]['w']) for e in self.neighbors(y)])
            total_dists = self.G[y]['nb_points'] + sum([self.G[e]['nb_points'] for e in self.neighbors(y)])
            if total_dists == 0:
                Ty = 0
            else: 
                Ty = (sum_dists + sum_weighted_dists) / total_dists

        else:
            dists = [np.linalg.norm(self.G[y]['w'] - self.G[i]['w']) for i in self.G if i != y]
            min_dist = min(dists)
            Ty = min_dist / 2
        return Ty+.5


    def assigned_data_nb_points(self, y):
        """
        Returns the number of data points assigned to a cluster.
        
        :param y: The index of the cluster.
        :return: The number of data points assigned to the cluster.
        """
        return self.G[y]['nb_points']
    

    def neighbors(self, y):
        """
        Returns the neighbors of a cluster.
        
        :param y: The index of the cluster.
        :return: A list of indices of neighboring clusters.
        """
        return [n for n in self.G[y] if n != 'w' and n != 'nb_points' and n != 'sum_distances' and n != 'points']
    

    def connect(self, i, j):
        """
        Connects two clusters by an edge.
        
        :param i: The index of the first cluster.
        :param j: The index of the second cluster.
        """
        if j not in self.G[i]:
            self.G[i][j] = {'age': 0}
            self.G[j][i] = {'age': 0}
        else:
            self.G[i][j]['age'] = 0
            self.G[j][i]['age'] = 0
        
    def merge(self, k, G):
        
        """
        Merges clusters in the graph until the number of clusters is less than or equal to the upper bound.
        
        :param k: The merging threshold.
        :param G: The graph to be merged.
        :return: The merged graph.
        """
        def connect(i, j, graph):

            """
            Connects two clusters by an edge.
            
            :param i: The index of the first cluster.
            :param j: The index of the second cluster.
            :param graph: The graph to be modified.
            """
            if j not in graph[i]:
                graph[i][j] = {'age': 0}
                graph[j][i] = {'age': 0}
            else:
                graph[i][j]['age'] = 0
                graph[j][i]['age'] = 0

        def neighbors(y, graph):
            """
            Returns the neighbors of a cluster.
            
            :param y: The index of the cluster.
            :param graph: The graph to be searched.
            :return: A list of indices of neighboring clusters.
            """
            return [n for n in graph[y] if n != 'w' and n != 'nb_points' and n!='sum_distances' and n !='points']

        G_new = defaultdict(dict)
        # Initialize G_new with two neurons chosen randomly from G
        neurons = list(G.keys())
        random_neurons = random.sample(neurons, 2)

        for i, neuron in enumerate(random_neurons):
            G_new[i] = {}
            G_new[i]['w'] = G[neuron]['w']
            G_new[i]['nb_points'] = G[neuron]['nb_points']
            G_new[i]['points'] = G[neuron]['points']
            G_new[i]['sum_distances'] = G[neuron]['sum_distances']

            G.pop(neuron)

        for y in G:
            # Find the two nearest neurons from y in G_new
            dists = [np.linalg.norm(G[y]['w'] - G_new[i]['w']) for i in G_new.keys()]
            y1, y2 = np.argsort(dists)[:2]

            d1 = dists[y1]
            d2 = dists[y2]
            
            y1, y2 = list(G_new.keys())[y1], list(G_new.keys())[y2]

            if random.uniform(0, 1) < min(G[y]['nb_points'] * d1 / k, 1):
                # Add a new neuron to G_new
                y_new = len(G_new)
                G_new[y_new] = {'w': G[y]['w']}
                G_new[y_new]['nb_points'] = G[y]['nb_points']
                G_new[y_new]['points'] = G[y]['points']
                G_new[y_new]['sum_distances'] = G[y]['sum_distances']
            else:
                if random.uniform(0, 1) < min(G[y]['nb_points'] * d2 / k, 1):
                    # Add a new neuron to G_new and connect it to y1
                    y_new = len(G_new)
                    G_new[y_new] = {'w': G[y]['w']}
                    G_new[y_new]['nb_points'] = G[y]['nb_points']
                    G_new[y_new]['sum_distances'] = G[y]['sum_distances']
                    G_new[y_new]['points'] = G[y]['points']
                    connect(y_new, y1, G_new)
                else:
                    G_new[y1]['points'] += G[y]['points'] + [G[y]['w']]
                    G_new[y1]['nb_points'] += G[y]['nb_points'] + 1 # +1 for the neuron itself
                    G_new[y1]['sum_distances'] += G[y]['sum_distances'] + np.linalg.norm(G[y]['w'] - G_new[y1]['w'])
                    # Increase the age of edges emanating from y1
                    for n in neighbors(y1, G_new):
                        G_new[y1][n]['age'] += 1

                    # Update the reference vectors of y1 and its neighbors
                    e_b = 1 / (G_new[y1]['nb_points'])
                    e_n = 1 / (100 * (G_new[y1]['nb_points']))
                    G_new[y1]['w'] += e_b * (G[y]['w'] - G_new[y1]['w'])

                    for n in neighbors(y1, G_new):
                        G_new[n]['w'] += e_n * (G[y]['w'] - G_new[n]['w'])

                    # Connect y1 to y2 by a new edge
                    connect(y1, y2, G_new)
                    
                    # Remove old edges from G_new if any
                    for i in G_new:
                        for j in G_new:
                            if j > i and j in G_new[i] and 'age' in G_new[i][j] and G_new[i][j]['age'] > 100:
                                del G_new[i][j]
                                del G_new[j][i]

        return G_new


    
