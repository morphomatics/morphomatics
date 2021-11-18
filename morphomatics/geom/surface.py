################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2021 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################


import numpy as np
from scipy import sparse

from .misc import memoize, gradient_matrix_ambient



class Surface(object):
    """ Triangular surface geom. """

    def __init__(self, v, f):
        """
        :arg v: #v-by-dim array of vertex coordinate
        :arg f: #f-by-3 array of triangles (each given as indices into \a v)
        """
        # cache for memoization
        self._cache_v = dict()
        self._cache_f = dict()

        self.v = v.astype(np.double, copy=False)
        self.f = f

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, v):
        self.__v = v
        self._cache_v.clear()

    @property
    def f(self):
        return self.__f

    @f.setter
    def f(self, f):
        self.__f = f
        self._cache_v.clear()
        self._cache_f.clear()

    @property
    @memoize('_cache_v')
    def grad(self):
        """ Gradient of a scalar function defined on piecewise linear elements
        (see: geom.misc.gradient_matrix_ambient) """
        return gradient_matrix_ambient(self.v, self.f)

    @property
    @memoize('_cache_v')
    def div(self):
        """Divergence operator"""
        mass = self.face_areas
        d = self.grad.shape[0] / len(mass) # ambient dim.
        return self.grad.T @ sparse.diags(np.repeat(mass, d), 0)

    @property
    @memoize('_cache_v')
    def face_areas(self):
        """Area of triangles."""
        # Compute cross-product of edges
        ppts = self.v[self.f]
        nnfnorms = np.cross(ppts[:, 1] - ppts[:, 0],
                            ppts[:, 2] - ppts[:, 0])
        # Compute vector length
        return np.sqrt((nnfnorms ** 2).sum(-1)) / 2

    @property
    @memoize('_cache_f')
    def inner_edges(self):
        """
        Extract inner edges of an oriented, triangle surface.
        :return: sparse matrix with inner edge ids as entries (indexed by resp. faces).
        """
        m = len(self.f)
        n = self.f.max() + 1

        e1 = sparse.coo_matrix((np.ones(m), (self.f[:, 0], self.f[:, 1])), (n, n))
        e2 = sparse.coo_matrix((np.ones(m), (self.f[:, 1], self.f[:, 2])), (n, n))
        e3 = sparse.coo_matrix((np.ones(m), (self.f[:, 2], self.f[:, 0])), (n, n))
        D = (e1 + e2 + e3).tocsr()

        # boundary edges as sparse matrix
        E = (D - D.T).tocsr()
        # inner edges as sparse matrix
        F = 0.5*(D + D.T - np.absolute(D - D.T))
        F.eliminate_zeros()

        off = 1
        f1 = sparse.coo_matrix((np.arange(off, m+off), (self.f[:, 0], self.f[:, 1])), (n, n)).tocsr()
        f2 = sparse.coo_matrix((np.arange(off, m+off), (self.f[:, 1], self.f[:, 2])), (n, n)).tocsr()
        f3 = sparse.coo_matrix((np.arange(off, m+off), (self.f[:, 2], self.f[:, 0])), (n, n)).tocsr()

        ff = f1 +f2 +f3

        FF = F.multiply(ff)
        FFU = sparse.triu(FF).tocsr()

        edgesI = FFU.tocoo().row
        edgesJ = FFU.tocoo().col

        edgeTriI = np.reshape(np.asarray(FF[edgesI[:], edgesJ[:]]), edgesI.shape[0]) - 1
        edgeTriJ = np.reshape(np.asarray(FF[edgesJ[:], edgesI[:]]), edgesJ.shape[0]) - 1

        # k-th (inner) edge, s.t. FFU(edgesI[k], edgesJ[k]) -> FFU(edgesJ[k], edgesI[k])
        tris2InnerEdgeId = sparse.coo_matrix((np.arange(1, FFU.getnnz()+1), (edgeTriI[:], edgeTriJ[:])), (m, m))
        tris2InnerEdgeId = tris2InnerEdgeId + tris2InnerEdgeId.T
        tris2InnerEdgeId.data = tris2InnerEdgeId.data -1

        return tris2InnerEdgeId.tocsr()

    @property
    @memoize('_cache_v')
    def edge_areas(self):
        inner_edge = sparse.triu(self.inner_edges)
        areas=np.zeros(inner_edge.data.shape[0])
        areas[inner_edge.data[:]] = 1.0/3.0 * ( self.face_areas[inner_edge.row[:]] + self.face_areas[inner_edge.col[:]] )
        return areas

    @property
    @memoize('_cache_f')
    def neighbors(self):
        """
         Provides information on face neighbors.
         :return: idx_1, idx_2, idx_3, n_1, n_2, n_3,
         where idx_1, idx_2, idx_3 are boolean index arrays for faces with at least one (idx_1), at least two (idx_2) and three (idx_3) neighbors;
         where n_1, n_2, n_3 are n x k array storing the respective neighbors per face for all faces with at least one (n_1), at least two (n_2)
         and three (n_3) neighbors.
         """
        # neighbors per face
        neighs = np.asarray(np.split(self.inner_edges.indices, self.inner_edges.indptr[1:-1]), dtype=object)
        nNeighs = np.asarray(list(map(np.shape, neighs))).ravel()

        idx_1 = (nNeighs == 1)
        idx_2 = (nNeighs == 2)
        idx_3 = (nNeighs == 3)

        neigh_pad = np.zeros((neighs.shape[0], 3)).astype(int)

        if np.count_nonzero(idx_1):
            neigh_pad[idx_1, 0:1] = np.asarray(list(neighs[idx_1]))
        if np.count_nonzero(idx_2):
            neigh_pad[idx_2, 0:2] = np.asarray(list(neighs[idx_2]))
        if np.count_nonzero(idx_3):
            neigh_pad[idx_3, 0:3] = np.asarray(list(neighs[idx_3]))

        idx_1 = (nNeighs > 0)
        idx_2 = (nNeighs > 1)
        idx_3 = (nNeighs > 2)

        n_1 = neigh_pad[idx_1]
        n_2 = neigh_pad[idx_2]
        n_3 = neigh_pad[idx_3]

        return idx_1, idx_2, idx_3, n_1, n_2, n_3

    def copy(self):
        """
        :return: copy of this surface
        """
        return Surface(self.v.copy(), self.f.copy())

    def boundary(self):
        """
        Extract boundary curves of an oriented, triangle surface.
        :return: List of boundary curves (list of vertex indices).
        """
        m = len(self.f)
        n = self.f.max() + 1
        # boundary edges as sparse matrix
        e1 = sparse.coo_matrix((np.ones(m), (self.f[:, 0], self.f[:, 1])), (n, n))
        e2 = sparse.coo_matrix((np.ones(m), (self.f[:, 1], self.f[:, 2])), (n, n))
        e3 = sparse.coo_matrix((np.ones(m), (self.f[:, 2], self.f[:, 0])), (n, n))
        E = (e1 - e1.T + e2 - e2.T + e3 - e3.T).tocsr()
        E.sort_indices()

        if E.nnz == 0:
            return None

        # extract boundary curves
        boundaries = []

        nnz = E.getnnz(1)
        for i in range(n):
            if nnz[i] == 0: continue
            # new boundary curve
            bnd = [i]
            while True:
                j = (E.getrow(bnd[-1]) > 0).indices[0]
                nnz[j] = 0  # mark as part of a processed bnd. curve
                if j == i: break
                bnd.append(j)
            boundaries.append(bnd)

        return boundaries
