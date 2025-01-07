import json
import numpy as np

class GeneticNeuralNetwork:
    """
    Parameters
    ----------
    sizes: a list of number of neurons for each layer; the first element is for the input layer, the last element is for the output layer

    weight_clipping: if True, it ensures that each component of the weights and biases are between -1 and 1

    mutation_chance: The chance [0, 1] that a mutation will occur for each layer when the GNN is mutated

    weight_deviation: Applies a deviation factor to the weights during mutation according to this number

    bias_deviation: Applies a deviation factor to the biases during mutation according to this number

    gnn_dict: a dictionary containing all the parameters for the genetic neural network. Is arranged like in copy_info(). 
        Does not need to be entered if the other parameters are entered.
    """
    def __init__(self, sizes = None, weight_clipping = False, mutation_chance = 0.5, weight_deviation = 0.1, bias_deviation = 0.1, gnn_dict = None):
        if gnn_dict is None and sizes is None:
            raise Exception("Must include sizes")
        if sizes:
            self.mutation_chance = mutation_chance
            self.weight_deviation = weight_deviation
            self.bias_deviation = bias_deviation
            self.sizes = sizes
            self.weight_clipping = weight_clipping
            self.num_layers = len(sizes)
            self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
            self.weights = [np.random.randn(num_neurons, num_inputs) * np.sqrt(2. / num_inputs)
                            for num_inputs, num_neurons in zip(sizes[:-1], sizes[1:])]
        else:
            self.load_info(gnn_dict)

    def copy_info(self):
        GNN_dict = {}
        GNN_dict['sizes'] = self.sizes
        GNN_dict['num_layers'] = self.num_layers
        GNN_dict['weights'] = [np.copy(self.weights[i]) for i in range(0, self.num_layers - 1)]
        GNN_dict['biases'] = [np.copy(self.biases[i]) for i in range(0, self.num_layers - 1)]
        GNN_dict['mutation_chance'] = self.mutation_chance
        GNN_dict['weight_deviation'] = self.weight_deviation
        GNN_dict['bias_deviation'] = self.bias_deviation
        GNN_dict['weight_clipping'] = self.weight_clipping
        return dict(GNN_dict)
    
    def load_info(self, GNN_dict):
        self.sizes = GNN_dict['sizes']
        self.num_layers = GNN_dict['num_layers']
        self.weights = [np.copy(w) for w in GNN_dict['weights']]
        self.biases = [np.copy(b) for b in GNN_dict['biases']]
        self.mutation_chance = GNN_dict['mutation_chance']
        self.weight_deviation = GNN_dict['weight_deviation']
        self.bias_deviation = GNN_dict['bias_deviation']
        self.weight_clipping = GNN_dict['weight_clipping']

    def relu(self, x):
        return np.maximum(x, 0.)

    # just forward pass
    def predict(self, x): # x must be of shape (-1, 1)
        z = x.copy()
        for i in range(0, self.num_layers - 1):
            a = np.dot(self.weights[i], z) + self.biases[i]
            z = self.relu(a)
        softz = np.exp(z)
        return softz / np.sum(softz)

    def mutate(self):
        for i in range(0, self.num_layers - 1):
            if np.random.rand() < self.mutation_chance:
                self.weights[i] += self.weight_deviation * np.random.randn(*self.weights[i].shape) * np.random.randn(*self.weights[i].shape)
                self.biases[i] += np.random.normal(0, self.bias_deviation, size = self.biases[i].shape)
                if self.weight_clipping:
                    self.weights[i] = np.clip(self.weights[i], a_min=-1.0, a_max=1.0)
                    self.biases[i] = np.clip(self.biases[i], a_min=-1.0, a_max=1.0)
    
    """
    Saves the genetic neural network to the location of the filename with all of the parameters.
    """
    def save(self, filename):
        data = {"sizes": self.sizes,
                "num_layers": self.num_layers,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "mutation_chance": self.mutation_chance,
                "weight_deviation": self.weight_deviation,
                "bias_deviation": self.bias_deviation,
                "weight_clipping": self.weight_clipping}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

"""
Loads the genetic neural netowrk from the location of the filename which contains all of the parameters.
"""
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    nn = GeneticNeuralNetwork(sizes=data["sizes"],
                               weight_clipping=data['weight_clipping'], 
                               mutation_chance=data["mutation_chance"], 
                               weight_deviation=data["weight_deviation"], 
                               bias_deviation=data["bias_deviation"])
    nn.weights = [np.array(w) for w in data["weights"]]
    nn.biases = [np.array(b) for b in data["biases"]]
    return nn