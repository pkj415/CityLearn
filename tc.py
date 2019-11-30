import numpy as np
import math
from tiles3 import tiles, IHT

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array,
                 wrap_around=[],
                 use_standard_tile_coding=False,
                 initial_weight_value=0.0):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # print("States low {0}".format(state_low))
        # print("States high {0}".format(state_high))
        # print("Tile width {0}".format(tile_width))
        self.use_standard_tile_coding = use_standard_tile_coding

        if use_standard_tile_coding:
            self.iht = IHT(4096)

        self.wrap_around = wrap_around

        self.state_low = state_low
        self.state_high = state_high
        self.tile_width = tile_width
        self.num_tilings = num_tilings
        # self.tiling_offsets = []

        self.num_tiles_dim = [self.num_tilings]
        idx = 0
        for low, high, width in zip(list(state_low), list(state_high), list(tile_width)):
            # print("For low {0} high {1} width {2}. Num {3}".format(low, high, width, math.ceil((high - low)/width) + 1))
            # print("Tiles after {0}".format(self.num_tiles_dim))
            if not self.wrap_around[idx]:
                self.num_tiles_dim.append(math.ceil((high - low)/width) + 1)
            else:
                self.num_tiles_dim.append(math.ceil((high - low)/width))

            idx += 1

        # print("Weights dimensions {0}".format(self.num_tiles_dim))
        self.maxSize = 4096
        if self.use_standard_tile_coding:
            self.weight = [initial_weight_value]*self.maxSize
        else:
            self.weight = np.full(self.num_tiles_dim, initial_weight_value)

    def mytiles(self, s):
        values_for_stnd_tiles = []
        for idx, dimen_val in enumerate(s):
            scale_factor = 10.0/(self.state_high[idx] - self.state_low[idx])
            values_for_stnd_tiles.append(dimen_val*scale_factor)

        return tiles(self.iht, self.num_tilings, values_for_stnd_tiles)

    def __call__(self, s):
        if self.use_standard_tile_coding:
            tiles = self.mytiles(s)
            estimate = 0
            for tile in tiles:
                estimate += self.weight[tile]

            return estimate

        val = 0

        for tiling_num in range(self.num_tilings):
            dimension_slabs = [tiling_num]
            for idx, dimen_val in enumerate(s):
                slab = int((dimen_val + tiling_num/self.num_tilings * self.tile_width[idx] - self.state_low[idx])/self.tile_width[idx])
                if self.wrap_around[idx]:
                    slab = slab % self.num_tiles_dim[idx+1]
                dimension_slabs.append(slab)

            # print("Call state {0}, adding weight of slab {1} = {2}".format(s, dimension_slabs, self.weight[tuple(dimension_slabs)]))
            val += self.weight[tuple(dimension_slabs)]

        return val

    def get_weights_refs(self, s):
        weights_refs = []

        # print("Getting weight refs for {0}".format(s))
        for tiling_num in range(self.num_tilings):
            dimension_slabs = [tiling_num]
            for idx, dimen_val in enumerate(s):
                slab = int((dimen_val + tiling_num/self.num_tilings * self.tile_width[idx] - self.state_low[idx])/self.tile_width[idx])
                if self.wrap_around[idx]:
                    slab = slab % self.num_tiles_dim[idx+1]
                dimension_slabs.append(slab)
            weights_refs.append(tuple(dimension_slabs))

        # print("Got weight refs {0}".format(weights_refs))
        return weights_refs

    def update(self, alpha, G, s_tau, a=1):
        delta = G - self.__call__(s_tau)

        if self.use_standard_tile_coding:
            tiles = self.mytiles(s_tau)
            estimate = 0
            for tile in tiles:
                self.weight[tile] += (alpha/self.num_tilings)*delta

            return None

        weight_refs = self.get_weights_refs(s_tau)
        for weight_ref in weight_refs:
            self.weight[weight_ref] += (alpha/self.num_tilings)*delta

        return None
