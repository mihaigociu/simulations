import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pylab

from orderedset import OrderedSet

# %pylab inline

#-----------------------Simulating belief updates in a network----------------------------------------------#


class Patch:

    NEUTRAL = 'w'

    def __init__(self, label, status=None, pos=(0, 0)):
        if not status:
            self.status = Patch.NEUTRAL
        else:
            self.status = status
        self.pos = pos
        self.label = label

    def set_status(self, status):
        self.status = status

    def set_neutral(self):
        self.status = Patch.NEUTRAL

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


class Belief(object):
    BELIEVE = 'b'
    DISBELIEVE = 'r'
    UNDECIDED = 'y'

    BELIEVE_THRESHOLD = 0.8
    DISBELIEVE_THRESHOLD = 0.5

    def __init__(self):
        self.believe_patches = set()
        self.disbelieve_patches = set()
        self.undecided_patches = set()

        self.belief_states = [Belief.BELIEVE, Belief.DISBELIEVE, Belief.UNDECIDED]

        self.belief_states_patches = {
            Belief.BELIEVE: self.believe_patches,
            Belief.DISBELIEVE: self.disbelieve_patches,
            Belief.UNDECIDED: self.undecided_patches
        }

    def add_patch(self, patch, state):
        if self.check_state(state):
            patch.set_status(state)
            self.belief_states_patches[state].add(patch)

    def remove_patch(self, patch, state):
        if self.check_state(state):
            patch.set_neutral()
            self.belief_states_patches[state].remove(patch)

    def check_state(self, state):
        if state not in [Belief.BELIEVE, Belief.DISBELIEVE, Belief.UNDECIDED]:
            return False
        return True

    def set_random_belief(self, patch):
        belief_random = np.random.uniform()
        if belief_random > Belief.BELIEVE_THRESHOLD:
            self.add_patch(patch, Belief.BELIEVE)
        elif belief_random > Belief.DISBELIEVE_THRESHOLD:
            self.add_patch(patch, Belief.DISBELIEVE)
        else:
            self.add_patch(patch, Belief.UNDECIDED)


class Simulation(object):
    nr_patches = 100   # Number of patches
    c_distance = 15  # An arbitrary parameter to determine which patches are connected

    # TODO: implement algorithm for generating small-world networks

    def __init__(self, with_history=True):
        self.step = 0
        self.patches = []
        self.history = []
        self.with_history = with_history

        self.graph = nx.Graph()
        self.generate_patches_2d()

        self.belief = Belief()
        self.set_initial_beliefs()

        # keep track of changes in history
        if self.with_history:
            self.save_history()

    def set_initial_beliefs(self):
        for patch in self.patches:
            self.belief.set_random_belief(patch)

    def generate_patches_2d(self):
        # maybe use some Barabasi-Albert graph instead of random generation?
        positions = np.random.uniform(high=100, size=(self.nr_patches, 2))
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
                    self.graph.add_edge(p1, p2)

    def distance_2d(self, p1, p2):
        return np.sqrt((p1.pos[1]-p2.pos[1])**2+(p1.pos[0]-p2.pos[0])**2)

    def generate_patch(self, label, pos):
        return Patch(label=label, pos=pos)

    def save_history(self):
        self.history.append({
            Belief.BELIEVE: len(self.belief.believe_patches),
            Belief.DISBELIEVE: len(self.belief.disbelieve_patches),
            Belief.UNDECIDED: len(self.belief.undecided_patches)})

    def draw_graph(self):
        pylab.figure(1, figsize=(8, 8))
        nx.draw(self.graph, {patch: patch.pos for patch in self.graph.nodes()},
                with_labels=True,
                node_color=[patch.status for patch in self.graph.nodes()])
        pylab.show()

    def draw_history(self):
        time = range(1, len(self.history)+1)
        args = []
        for state in self.belief.belief_states:
            args.append(time)
            args.append([h[state] for h in self.history])
            args.append(state)
        kwargs = {'figure': pylab.figure(2, (8, 8))}
        pylab.plot(*args, **kwargs)
