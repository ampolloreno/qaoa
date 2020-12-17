from es import OpenES
import networkx as nx
import numpy as np
from qaoa.source.orchestrator.common import (OptimizerFactory,
                                             ParameterInstance,
                                             ParameterSpace,
                                             SyncOptimizer)
import skopt
from typing import List, Callable

# Random 3-regular graph on 16 vertices.
dict_of_dicts = [(3, {15: {}, 7: {}, 10: {}}),
 (15, {3: {}, 1: {}, 7: {}}),
 (6, {9: {}, 14: {}, 10: {}}),
 (9, {6: {}, 2: {}, 13: {}}),
 (4, {8: {}, 12: {}, 14: {}}),
 (8, {4: {}, 10: {}, 0: {}}),
 (1, {15: {}, 11: {}, 14: {}}),
 (12, {4: {}, 13: {}, 0: {}}),
 (7, {3: {}, 15: {}, 5: {}}),
 (2, {5: {}, 9: {}, 13: {}}),
 (5, {2: {}, 11: {}, 7: {}}),
 (11, {1: {}, 0: {}, 5: {}}),
 (14, {6: {}, 1: {}, 4: {}}),
 (13, {12: {}, 2: {}, 9: {}}),
 (10, {3: {}, 6: {}, 8: {}}),
 (0, {11: {}, 12: {}, 8: {}})]
GRAPH = nx.from_dict_of_dicts(dict_of_dicts)

MAX_ITERATION = 500
NPOPULATION = 100
NPARAMS = 2
# We will seed the randomness so that we can compare experimental and numerical results.
np.random.seed(137)

class EvolutionStrategiesSyncOptimizer(SyncOptimizer):

    def __init__(self, parameter_space: ParameterSpace):
        """
        An optimizer using the Evolution Strategies algorithm.

        :param parameter_space: the space to optimize within
        :type parameter_space: ParameterSpace
        """
        # This sets the order that the parameters will be passed to the optimizer.
        self.parameter_space = parameter_space
        dimensions = []
        for param in parameter_space.parameters:
            dimension = skopt.space.space.Real(param.minimum, param.maximum, name=param.name)
            dimensions.append(dimension)
        self.dimensions = dimensions
        self.population_size = NPOPULATION
        oes = OpenES(NPARAMS,  # number of model parameters
                     sigma_init=0.025,  # initial standard deviation
                     sigma_decay=0.999,  # don't anneal standard deviation
                     learning_rate=0.05,  # learning rate for standard deviation WE BUMPED THIS UP
                     learning_rate_decay=0.999,  # annealing the learning rate
                     popsize=self.population_size,  # population size
                     rank_fitness=False,  # use rank rather than fitness numbers
                     forget_best=True)
        initial_gamma_beta = np.array([0., 0.])
        oes.mu = initial_gamma_beta
        self.optimizer = oes
        self.count = 0
        self.ask_count = 0
        self.results = []

    def sample(self, callback: Callable[[List[ParameterInstance]], float]):
        if self.ask_count == 0:
            self.asks = self.optimizer.ask()
            self.ask_count = self.population_size
        # The pairs produced from ask() need to have the same ordered as self.parameter_space.parameters.
        dimensions = self.asks[self.ask_count]
        self.ask_count -= 1
        parameters = []
        for i, v in enumerate(dimensions):
            param_inst = ParameterInstance(value=v, space=self.parameter_space.parameters[i])
            parameters.append(param_inst)
        # This is where a call will be made to execute on the hardware.
        result = callback(parameters)
        self.results.append(result.cost)
        if self.ask_count == 0:
            self.optimizer.tell(dimensions, self.results)
            self.results = []
        self.count += 1

    def should_continue(self) -> bool:
        return self.count <  MAX_ITERATION * self.population_size

class EvolutionStrategiesSyncOptimizerFactory(OptimizerFactory):

    def create_optimizer(self, parameter_space: ParameterSpace) -> SyncOptimizer:
        return EvolutionStrategiesSyncOptimizer(parameter_space=parameter_space)