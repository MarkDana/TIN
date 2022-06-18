#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from utils.graph import DiGraph
from utils.misc import *

class Simulator(object):
    def __init__(self, upper_tria_str=None, samplesize=5000, noise_to_signal_ratio=3.):
        '''
        :param upper_tria_str: str.
            A string contains only 0 and 1 that fully determines a DAG structure.
            Write the adjacent matrix A of DAG: A[i, j]!=0 iff i->j, and assume
                labels are permuted s.t. i->j only when i < j, i.e., A is upper triangular.
            Then write the upper triangular part of the matrix by rows to obtain the upper_tria_str.
            The length of upper_tria_str equals to n(n-1)/2, where n is nodenum.
            E.g.
                "1001100011" is a DAG with 5 vertices and edges 0->1, 0->4, 1->2, 2->4, 3->4
                "010001000110110" is a DAG with 6 vertices and edges 0->2, 1->2, 2->3, 2->4, 3->4, 3->5
            In our experiments we only consider chain structure and fully connected DAG:
                chain structure: e.g. "1000100101" is a chain structure with 5 vertices
                fully connected DAG: e.g. "1111111111" (all ones) is a fully connected DAG with 5 vertices
        :param samplesize: int.
        :param noise_to_signal_ratio: float. In our experiments it is tested over {0.5, 1, 2, 3, 4}
        '''
        self.prob_of_linear_weight_negative = 0.5
        self.linear_weight_minabs = 0.5
        self.linear_weight_maxabs = 0.9
        self.samplesize = samplesize
        self.noise_to_signal_ratio = noise_to_signal_ratio
        self.rank_eps = 1e-5
        self.simulation_data_base_dir = f'./data/NSR_{noise_to_signal_ratio}/'
        os.makedirs(self.simulation_data_base_dir, exist_ok=True)
        self.nodenum = LEN_TO_NODENUM[len(upper_tria_str)]
        self.core_A_01 = np.zeros((self.nodenum, self.nodenum), dtype=int)  # adjmat.T that only considers 1: edge, 0: no. No weight.
        self.upper_tria_str = upper_tria_str
        indu = np.triu_indices(self.nodenum, 1)
        self.core_A_01[indu] = [int(char) for char in upper_tria_str]  # note that here A is the regular i->j with A[i,j]=1 (upper triangular)
        self.core_A_01 = self.core_A_01.T  # thus we need to transpose to lower triangular in lingam

        self.graph = DiGraph(generatedAdjmat=self.core_A_01.T)

        self.weight_A, self.weight_B = self.get_random_generic_B()
        self.exogns_noise, self.measure_noise_raw = self.generate_noise_components()
        self.observ_data_clean = self.weight_B @ self.exogns_noise
        self.measure_noise = self.measure_noise_raw * np.sqrt(self.noise_to_signal_ratio) * \
                             self.observ_data_clean.std(axis=1)[:, None]
        self.observ_data = self.observ_data_clean + self.measure_noise

        self.cached_product_expectation = {}
        self.observ_data_2D_cumulants_empirical = self.get_2D_cumulants()

        self.this_simulation_id = len([x for x in os.listdir(self.simulation_data_base_dir)
                                    if x.startswith(self.upper_tria_str)])
        self.this_data_save_dir = os.path.join(self.simulation_data_base_dir,
                            f'{self.upper_tria_str}_{self.this_simulation_id}')
        self.save_data()

    def save_data(self):
        graph_parameters_dir = os.path.join(self.this_data_save_dir, 'graph_param')
        noises_dir = os.path.join(self.this_data_save_dir, 'noise_data')
        cumulants_dir = os.path.join(self.this_data_save_dir, 'cumulants')

        os.makedirs(graph_parameters_dir, exist_ok=True)
        np.save(os.path.join(graph_parameters_dir, 'A.npy'), self.weight_A)
        np.save(os.path.join(graph_parameters_dir, 'B.npy'), self.weight_B)

        os.makedirs(noises_dir, exist_ok=True)
        np.save(os.path.join(noises_dir, 'exogns_noise.npy'), self.exogns_noise)
        np.save(os.path.join(noises_dir, 'measure_noise.npy'), self.measure_noise)
        np.save(os.path.join(noises_dir, 'observ_data.npy'), self.observ_data)

        os.makedirs(cumulants_dir, exist_ok=True)
        np.save(os.path.join(cumulants_dir, 'observ_data_2D_cumulants_empirical.npy', ), self.observ_data_2D_cumulants_empirical)

    def get_random_generic_B(self):
        generic_try_count = 0
        while True:
            generic_try_count += 1
            weight_mask = np.random.uniform(self.linear_weight_minabs, self.linear_weight_maxabs, (self.nodenum, self.nodenum))
            tmpinds = np.random.choice(np.arange(self.core_A_01.size), replace=False,
                                       size=int(self.core_A_01.size * self.prob_of_linear_weight_negative))
            weight_mask[np.unravel_index(tmpinds, weight_mask.shape)] *= -1.
            weight_A = self.core_A_01 * weight_mask
            weight_B = np.linalg.inv(np.eye(self.nodenum) - weight_A)
            if self.is_generic(weight_B): return weight_A, weight_B

    def is_generic(self, weight_B):
        select_Z_from = [list(copi) for copi in
                chain.from_iterable(combinations(range(self.nodenum), r) for r in range(1, self.nodenum))]
        for Z in select_Z_from:
            AncZ = self.graph.getAncestorsOfSet(set(Z))
            if len(AncZ) > len(Z): continue # only use the self contained AncZ
            for Y in select_Z_from:
                min_vertexcut_size = self.graph.find_minimum_vertex_cut_size_from_AncZ_to_Y(AncZ, Y)
                rank_BY_AncZ_submatrix = get_rank(weight_B[sorted(Y)][:, list(AncZ)], eps=self.rank_eps)
                if rank_BY_AncZ_submatrix != min_vertexcut_size: return False # assert rank_BY_AncZ_submatrix < min_vertexcut_size
        return True

    def generate_noise_components(self):
        EXO_NOISES_LIST = []
        MEA_NOISES_LIST = []
        for _ in range(self.nodenum):
            EXO_NOISES_LIST.append(
                standard_scaler(
                    np.random.uniform(low=0., high=1., size=self.samplesize) ** np.random.uniform(5, 7),
                    stdval=np.random.uniform(1., 3.))) # to make variances not the same
            MEA_NOISES_LIST.append(
                standard_scaler(
                    gen_truncated_gaussian(mu=0., sigma=1., lower=0., upper=np.Inf, numsamples=self.samplesize) \
                        ** np.random.uniform(2, 4)
                ))
        return np.vstack(EXO_NOISES_LIST), np.vstack(MEA_NOISES_LIST)

    def get_cached_expected_product(self, ids_group_str_key):
        '''
        calculate (and check cache) the expected product value (moment)
        @param ids_group_str_key: e.g., '0.0.0.0.1'
        @return: E[data[0] * data[0] * data[0] * data[0] * data[1]]
        '''
        if ids_group_str_key in self.cached_product_expectation:
            return self.cached_product_expectation[ids_group_str_key]
        ids_group = list(map(int, ids_group_str_key.split('.')))
        expected_value = np.mean(np.prod(self.observ_data[ids_group], axis=0))
        self.cached_product_expectation[ids_group_str_key] = expected_value
        return expected_value

    def get_2D_cumulants(self):
        cumulants_dict = np.zeros((self.nodenum - 1, self.nodenum, self.nodenum))
        for orderk in range(2, self.nodenum + 1):
            cum_mat = np.zeros((self.nodenum, self.nodenum))
            for i in range(self.nodenum):
                for j in range(self.nodenum):
                    slice_ids = [i] * (orderk - 1) + [j]
                    cnter_dict = Counter(['_'.join(sorted(['.'.join(list(map(str, sorted(onegrp)))) for onegrp in onepar]))
                                        for onepar in partition(slice_ids)])
                    cell_value = 0.
                    for strkey, cnt in cnter_dict.items():
                        list_of_ids_group_str_key = strkey.split('_')
                        len_of_this_partition = len(list_of_ids_group_str_key)
                        cell_value += cnt * ((-1) ** (len_of_this_partition - 1)) * \
                                      np.math.factorial(len_of_this_partition - 1) * \
                                      np.prod([self.get_cached_expected_product(ids_group_str_key)
                                               for ids_group_str_key in list_of_ids_group_str_key])
                    cum_mat[i, j] = cell_value
            cumulants_dict[orderk - 2] = np.copy(cum_mat)
        return cumulants_dict


if __name__ == '__main__':
    for nodenum in [3, 4, 5, 6, 7, 8, 9, 10]:
        UPSTRS = [
            ''.join(['1' + '0' * _ for _ in range(nodenum - 2, -1, -1)]),
            '1' * NODENUM_TO_LEN[nodenum],
        ]
        for upper_tria_str in UPSTRS:
            for noise_to_signal_ratio in [0.5, 1, 2, 3, 4]:
                for gid in range(50):
                    print(f'now sampling {upper_tria_str}, NSR={noise_to_signal_ratio}, ID={gid}/50')
                    me = Simulator(upper_tria_str=upper_tria_str, noise_to_signal_ratio=noise_to_signal_ratio)

