import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pylab

from orderedset import OrderedSet

%pylab inline

#-----------------------Simulating civilisations----------------------------------------------#

# inspired from http://timotheepoisot.fr/2012/05/18/networkx-metapopulations-python/

class Patch:
    def __init__(self, label, status='w', pos=(0,0)):
        self.status = status
        self.pos = pos
        self.label = label

    def __str__(self):
        return(str(self.label))

    def __repr__(self):
        return(str(self.label))

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if not other:
            return False
        return self.label == other.label


class WeightedPatch(Patch):
    BIG_CITY_WEIGHT = 10
    CITY_WEIGHT = 5

    def __init__(self, label, status='w', pos=(0,0), weight=1):
        self.status = status
        self.pos = pos
        self.label = label
        self.weight = weight

    def __str__(self):
        return(str(self.weight))

    def __repr__(self):
        return(str(self.weight))


class Civilisation(object):
    def __init__(self, flag):
        self.flag = flag
        self.patches = set()

    def add_patch(self, patch):
        self.patches.add(patch)

    def remove_patch(self, patch):
        self.patches.remove(patch)


class Simulation(object):
    nr_patches = 100   # Number of patches
    c_distance = 15  # An arbitrary parameter to determine which patches are connected

    def __init__(self, with_history=True):
        self.civs = self.create_civilisations()
        self.step = 0
        self.patches = []
        self.history = []
        self.with_history = with_history

        self.graph = nx.Graph()
        self.generate_patches_2d()
        self.place_civs_on_map()

        # keep track of changes in history
        if self.with_history:
            self.save_history()

    def create_civilisations(self):
        return [Civilisation(flag='r'), Civilisation(flag='b'),
                Civilisation(flag='y')]

    def place_civs_on_map(self):
        civ_posistions = self.generate_civ_start_positions()
        for i, civ_pos in enumerate(civ_posistions):
            # each civ will have 3 neighboring patches
            civ_patch = self.patches[civ_pos]
            civ_patch.status = self.civs[i].flag
            self.civs[i].patches.add(civ_patch)
            for civ_ngb in self.graph[civ_patch]:
                # choose neighbors that are not already taken
                if civ_ngb.status != 'w':
                    continue
                civ_ngb.status = self.civs[i].flag
                self.civs[i].add_patch(civ_ngb)
                if len(self.civs[i].patches) > 2:
                    break

    def generate_civ_start_positions(self):
        return np.random.random_integers(low=0, high=self.nr_patches,
                                            size=(len(self.civs),))

    def generate_patches_2d(self):
        # maybe use some Barabasi-Albert graph instead of random generation?
        positions = np.random.uniform(high=100, size=(self.nr_patches,2))
        # add patches to the graph
        for i in range(self.nr_patches):
            patch = self.generate_patch(label=i, pos=positions[i])
            self.graph.add_node(patch)
            self.patches.append(patch)
        # add edges
        for p1 in self.graph.nodes():
            for p2 in self.graph.nodes():
                if p1 == p2:
                    continue
                if self.distance_2d(p1, p2) <= self.c_distance:
                    self.graph.add_edge(p1,p2)

    def distance_2d(self, p1, p2):
        return np.sqrt((p1.pos[1]-p2.pos[1])**2+(p1.pos[0]-p2.pos[0])**2)

    def generate_patch(self, label, pos):
        return Patch(label=label, pos=pos)

    def run_simulation(self, steps=1):
        for step in range(steps):
            # do actions for each civ
            for civ in self.civs:
                # for each node a civ will try to expand to the neighbors
                # at each step only one attempt to conquer a patch can be made
                attempts = set()
                conquered = set()
                for patch in civ.patches:
                    for neighbor in self.graph[patch]:
                        # make sure this civ doesn't own the patch already
                        if neighbor.status == civ.flag:
                            continue
                        # an attempt was already made this turn
                        if neighbor in attempts:
                            continue
                        # try to conquer the patch
                        result = self.conquer(neighbor, civ)
                        attempts.add(neighbor)
                        if result:
                            if neighbor.status != 'w':
                                # the patch belongs to another civ
                                other_civ = self.get_civ_by_color(neighbor.status)
                                other_civ.remove_patch(neighbor)
                            conquered.add(neighbor)
                # claim conquered patches
                for patch in conquered:
                    patch.status = civ.flag
                    civ.add_patch(patch)

            if self.with_history:
                self.save_history()
            self.step += 1

    def conquer(self, patch, civ):
        # total number of neighbors plus the node itself
        total = len(self.graph[patch]) + 1
        # number of neighbors belonging to this civ
        civ_ngbs = len([ngb for ngb in self.graph[patch]
                        if ngb.status == civ.flag])
        # random component
        return bool(np.random.binomial(1, float(civ_ngbs)/total))

    def get_civ_by_color(self, color):
        for civ in self.civs:
            if civ.flag == color:
                return civ

    def save_history(self):
        self.history.append(
            {civ.flag: len(civ.patches) for civ in self.civs})

    def draw_graph(self):
        pylab.figure(1, figsize=(8, 8))
        nx.draw(self.graph, {patch: patch.pos for patch in self.graph.nodes()},
                with_labels=True,
                node_color=[patch.status for patch in self.graph.nodes()])
        pylab.show()

    def draw_history(self):
        time = range(1, len(self.history)+1)
        args = []
        for civ in self.civs:
            args.append(time)
            args.append([h[civ.flag] for h in self.history])
            args.append(civ.flag)
        kwargs = {'figure': pylab.figure(2, (8,8))}
        pylab.plot(*args, **kwargs)


class CivilisationRandomStrategy(Civilisation):
    def run_strategy(self, graph):
        ''' Will select a random subset of nodes from the neighbors.'''
        neighbors = set()
        for patch in self.patches:
            for neighbor in graph[patch]:
                if neighbor in self.patches:
                    continue
                neighbors.add(neighbor)
        if not neighbors:
            return []
        return random.choice(list(neighbors), len(self.patches)/3 or 1)


class CivilisationNaiveStrategy(Civilisation):
    def run_strategy(self, graph):
        ''' Will select the neighbors which are most connected to the civ.'''
        neighbors = self.get_neighbors(graph)
        if not neighbors:
            return []
        return self.neighbors_move(neighbors)

    def neighbors_move(self, neighbors):
        move = [n for n in neighbors.items()]
        # sort the orders by the number of connections it has with the civ
        move.sort(key=lambda x: -x[1])
        return [node for (node, connex) in move[:int(len(self.patches)/3) or 1]]

    def get_neighbors(self, graph):
        neighbors = {}
        for patch in self.patches:
            for neighbor in graph[patch]:
                if neighbor in self.patches:
                    continue
                if neighbor in neighbors:
                    neighbors[neighbor] += 1
                else:
                    neighbors[neighbor] = 1
        return neighbors


class CivilisationAggressiveStrategy(CivilisationNaiveStrategy):
    '''Will attack the nodes of other civs first.'''
    def run_strategy(self, graph):
        move = super(CivilisationAggressiveStrategy, self).run_strategy(graph)
        move.sort(key=lambda node: node.status == 'w')
        return move


class CivilisationBigCityStrategy(CivilisationNaiveStrategy):
    def run_strategy(self, graph):
        '''Will try to take and keep big cities around it.'''
        neighbors = self.get_neighbors(graph)

        # first look if there is any big city within the civ and secure the borders
        move = OrderedSet((patch for patch in self.secure_borders(graph, neighbors)))
        if len(move) >= len(self.patches)/3:
            return move[:int(len(self.patches)/3) or 1]

        # now look if there are big cities in the vecinity (2 degree) and try to get them
        move.update((patch for patch in self.get_big_cities(graph, neighbors)))
        if len(move) >= len(self.patches)/3:
            return move[:int(len(self.patches)/3) or 1]

        # go to naive strategy
        move.update((patch for patch in self.neighbors_move(neighbors)))
        if len(move) >= len(self.patches)/3:
            return move[:int(len(self.patches)/3) or 1]

        return move

    def get_big_cities(self, graph, neighbors):
        big_cities = []
        for neighbor in neighbors:
            if neighbor.weight == WeightedPatch.BIG_CITY_WEIGHT:
                big_cities.append(neighbor)
                continue # a big city does not have big city neighbors
            for ngb_2d in graph[neighbor]:
                if ngb_2d in self.patches:
                    continue
                if ngb_2d.weight == WeightedPatch.BIG_CITY_WEIGHT:
                    big_cities.append(ngb_2d)
        return self.big_cities_move(graph, neighbors, big_cities)

    def secure_borders(self, graph, neighbors):
        big_cities = []
        for patch in self.patches:
            if patch.weight == WeightedPatch.BIG_CITY_WEIGHT:
                big_cities.append(patch)
        if len(big_cities) == 0:
            return []
        return self.big_cities_move(graph, neighbors, big_cities)

    def big_cities_move(self, graph, neighbors, big_cities):
        move = set()
        for big_city in big_cities:
            # before attacking the big city (if we don't own it already) try to conquer its borders
            for border in nx.single_source_dijkstra_path_length(graph, big_city, 2):
                if border not in self.patches and border in neighbors:
                    move.add(border)
            if big_city in neighbors:
                move.add(big_city)

        move = [patch for patch in move]
        move.sort(key=lambda patch: -neighbors.get(patch))
        return move


class SimulationRandomStrategy(Simulation):
    def create_civilisations(self):
        return [CivilisationRandomStrategy(flag='r'),
                CivilisationRandomStrategy(flag='b'),
                CivilisationRandomStrategy(flag='y')]

    def run_simulation(self, steps=1):
        for step in range(steps):
            # do actions for each civ
            for civ in self.civs:
                # for each node a civ will try to expand to the neighbors
                # at each step only one attempt to conquer a patch can be made
                attempts = set()
                conquered = set()
                move = civ.run_strategy(self.graph)
                if not self.check_move(civ, move):
                    continue
                for patch in move:
                    # make sure this civ doesn't own the patch already
                    if patch.status == civ.flag:
                        continue
                    # an attempt was already made this turn
                    if patch in attempts:
                        continue
                    # try to conquer the patch
                    result = self.conquer(patch, civ)
                    attempts.add(patch)
                    if result:
                        if patch.status != 'w':
                            # the patch belongs to another civ
                            other_civ = self.get_civ_by_color(patch.status)
                            other_civ.remove_patch(patch)
                        conquered.add(patch)
                # claim conquered patches
                for patch in conquered:
                    patch.status = civ.flag
                    civ.add_patch(patch)

            if self.with_history:
                self.save_history()
            self.step += 1

    def check_move(self, civ, patches):
        '''Will check if the move is valid.'''
        # a civ can't attempt to conquer more than a third of the number of patches it already has
        if len(patches) > 1 and len(patches) > len(civ.patches)/3:
            print('Invalid move: %s attempt to conquer to many nodes.' % (civ.flag,))
            return False
        # a civ can only conquer its neighboring patches
        for patch in patches:
            valid = False
            for neighbor in self.graph[patch]:
                if neighbor.status == civ.flag:
                    valid = True
                    break
            if not valid:
                print('Invalid move: %s attempt to conquer nodes that are not neighbors.' % (civ.flag,))
                return False
        return True


class SimulationNaiveStrategy(SimulationRandomStrategy):
    def create_civilisations(self):
        return [CivilisationNaiveStrategy(flag='r'),
                CivilisationRandomStrategy(flag='b'),
                CivilisationRandomStrategy(flag='y')]


class SimulationComplexStrategy(SimulationNaiveStrategy):
    c_distance = 12
    c_distance_extended = 15
    max_big_cities = 7 # max number of big cities on map
    max_cities = 4 # max number of cities for a big city

    def __init__(self, with_history=True):
        self._path_lengths = {}
        self.big_cities = set()
        super(SimulationComplexStrategy, self).__init__(with_history)

    def create_civilisations(self):
        return [CivilisationNaiveStrategy(flag='r'),
                CivilisationAggressiveStrategy(flag='b'),
                CivilisationBigCityStrategy(flag='y')]

    def generate_patches_2d(self):
        super(SimulationComplexStrategy, self).generate_patches_2d()
        # because we might get too many connected components in the graph, we will extend the edges
        self.extend_edges()
        self.add_weights_to_patches()

    def extend_edges(self):
        components = [comp for comp in nx.connected_components(self.graph)]
        if len(components) > 1:
            for i in range(len(components) - 1):
                for j in range(1, len(components)):
                    comp_1 = components[i]
                    comp_2 = components[j]
                    for patch_1 in comp_1:
                        for patch_2 in comp_2:
                            if self.distance_2d(patch_1, patch_2) < self.c_distance_extended:
                                self.graph.add_edge(patch_1, patch_2)

    def add_weights_to_patches(self):
        # max 5% of the nodes will be big cities
        # max 15% of the nodes will be cities
        nodes = set(self.graph.nodes())
        while len(self.big_cities) < self.max_big_cities and len(nodes) > 0:
            big_city = nodes.pop()
            big_city.weight = WeightedPatch.BIG_CITY_WEIGHT
            self.big_cities.add(big_city)

            cities = self.neighborhood(big_city, 1)
            nr_cities = 0
            for city in cities:
                if city not in nodes:
                    continue
                if nr_cities < self.max_cities:
                    city.weight = WeightedPatch.CITY_WEIGHT
                nr_cities += 1
                nodes.remove(city)

            towns = self.neighborhood(big_city, 2)
            for town in towns:
                if town not in nodes:
                    continue
                nodes.remove(town)

    def generate_civ_start_positions(self):
        return [self.big_cities.pop().label for civ in self.civs]

    def conquer(self, patch, civ):
        # total sum of weights of the neighbors plus the weight of the node itself
        total_sum = sum([neighbor.weight for neighbor in self.graph[patch]])
        total_sum += patch.weight
        # sum of weigths of the neighbors belonging to this civ
        civ_sum = sum([ngb.weight for ngb in self.graph[patch]
                                if ngb.status == civ.flag])
        # random component
        return bool(np.random.binomial(1, float(civ_sum)/total_sum))

    def neighborhood(self, node, n):
        path_lengths = self._path_lengths.get(node)
        if not path_lengths:
            path_lengths = nx.single_source_dijkstra_path_length(self.graph, node)
            self._path_lengths[node] = path_lengths
        return [node for node, length in path_lengths.items()
                        if length == n]

    def generate_patch(self, label, pos):
        return WeightedPatch(label=label, pos=pos)