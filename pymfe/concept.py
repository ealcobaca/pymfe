"""Module dedicated to extraction of Concept Metafeatures."""

import typing as t

import numpy as np
import scipy.spatial
import sklearn


class MFEConcept:
    """Keep methods for metafeatures of ``Concept`` group.

    The convention adopted for metafeature extraction related methods is to
    always start with ``ft_`` prefix to allow automatic method detection. This
    prefix is predefined within ``_internal`` module.

    All method signature follows the conventions and restrictions listed below:

    1. For independent attribute data, ``X`` means ``every type of attribute``,
       ``N`` means ``Numeric attributes only`` and ``C`` stands for
       ``Categorical attributes only``. It is important to note that the
       categorical attribute sets between ``X`` and ``C`` and the numerical
       attribute sets between ``X`` and ``N`` may differ due to data
       transformations, performed while fitting data into MFE model,
       enabled by, respectively, ``transform_num`` and ``transform_cat``
       arguments from ``fit`` (MFE method).

    2. Only arguments in MFE ``_custom_args_ft`` attribute (set up inside
       ``fit`` method) are allowed to be required method arguments. All other
       arguments must be strictly optional (i.e., has a predefined default
       value).

    3. The initial assumption is that the user can change any optional
       argument, without any previous verification of argument value or its
       type, via kwargs argument of ``extract`` method of MFE class.

    4. The return value of all feature extraction methods should be a single
       value or a generic List (preferably a :obj:`np.ndarray`) type with
       numeric values.

    There is another type of method adopted for automatic detection. It is
    adopted the prefix ``precompute_`` for automatic detection of these
    methods. These methods run while fitting some data into an MFE model
    automatically, and their objective is to precompute some common value
    shared between more than one feature extraction method. This strategy is a
    trade-off between more system memory consumption and speeds up of feature
    extraction. Their return value must always be a dictionary whose keys are
    possible extra arguments for both feature extraction methods and other
    precomputation methods. Note that there is a share of precomputed values
    between all valid feature-extraction modules (e.g., ``class_freqs``
    computed in module ``statistical`` can freely be used for any
    precomputation or feature extraction method of module ``landmarking``).
    """

    @classmethod
    def precompute_concept_dist(
        cls, N: np.ndarray, concept_dist_metric: str = "euclidean", **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute some useful things to support complexity measures.

        Parameters
        ----------
        N : :obj:`np.ndarray`, optional
            Numerical fitted data.

        concept_dist_metric : str, optional
            Metric used to compute distance between each pair of examples. See
            cdist from scipy for more options.

        **kwargs
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`dict`
            With following precomputed items:
                - ``concept_distances`` (:obj:`np.ndarray`): Distance matrix of
                  examples from N.
        """
        precomp_vals = {}

        if N is not None and "concept_distances" not in kwargs:
            # 0-1 scaling
            N = sklearn.preprocessing.MinMaxScaler(
                feature_range=(0, 1)
            ).fit_transform(N)

            # distance matrix
            concept_distances = scipy.spatial.distance.cdist(
                N, N, metric=concept_dist_metric
            )

            precomp_vals["concept_distances"] = concept_distances

        return precomp_vals

    @classmethod
    def ft_conceptvar(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        conceptvar_alpha: float = 2.0,
        concept_dist_metric: str = "euclidean",
        concept_minimum: float = 10e-10,
        concept_distances: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the concept variation that estimates the variability of
        class labels among examples.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        conceptvar_alpha : float, optional
            The alpha value to adjust the weight. The higher the alpha less
            is the effect of the weight in the computation.

        concept_dist_metric : str, optional
            Metric used to compute distance between each pair of examples. See
            cdist from scipy for more options. Used only if the argument
            ``concept_distances`` is None.

        concept_minimum: float, optional
            This variable is the minimum value considered in the computation.
            It will be sum when necessary to avoid division by zero.

        concept_distances : :obj:`np.ndarray`, optional
            Distance matrix of examples from N. Argument used to take
            advantage of precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the concept variation for each example.

        References
        ----------
        .. [1] Vilalta, R. (1999). Understanding accuracy performance through
           concept characterization and algorithm analysis. In Proceedings of
           the ICML-99 workshop on recent advances in meta-learning and future
           work (pp. 3-9).
        """
        if concept_distances is None:
            sub_dic = cls.precompute_concept_dist(N, concept_dist_metric)
            concept_distances = sub_dic["concept_distances"]

        n_col = N.shape[1]

        div = np.sqrt(n_col) - concept_distances
        div[div <= 0] = concept_minimum  # to guarantee that minimum will be 0
        weights = np.power(2, -conceptvar_alpha * (concept_distances / div))
        np.fill_diagonal(weights, 0.0)

        rep_class_matrix = np.repeat(np.expand_dims(y, 0), y.shape[0], axis=0)
        # check if class is different
        class_diff = np.not_equal(rep_class_matrix.T, rep_class_matrix).astype(
            int
        )

        conceptvar_by_example = np.sum(weights * class_diff, axis=0) / np.sum(
            weights, axis=0
        )

        # The original meta-feature is the mean of the return.
        # It will be done by the summary functions.
        return conceptvar_by_example

    @classmethod
    def ft_wg_dist(
        cls,
        N: np.ndarray,
        wg_dist_alpha: float = 2.0,
        concept_dist_metric: str = "euclidean",
        concept_minimum: float = 10e-10,
        concept_distances: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the weighted distance, that captures how dense or sparse
        is the example distribution.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        wg_dist_alpha : float, optional
            The alpha value to adjust the weight. The higher the alpha less
            is the effect of the weight in the computation.

        concept_dist_metric : str, optional
            Metric used to compute distance between each pair of examples. See
            cdist from scipy for more options. Used only if the argument
            ``concept_distances`` is None.

        concept_minimum : float, optional
            This variable is the minimum value considered in the computation.
            It will be sum when necessary to avoid division by zero.

        concept_distances : :obj:`np.ndarray`, optional
            Distance matrix of examples from N. Argument used to take
            advantage of precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the weighted distance for each example.

        References
        ----------
        .. [1] Vilalta, R. (1999). Understanding accuracy performance through
           concept characterization and algorithm analysis. In Proceedings of
           the ICML-99 workshop on recent advances in meta-learning and future
           work (pp. 3-9).
        """
        if concept_distances is None:
            sub_dic = cls.precompute_concept_dist(N, concept_dist_metric)
            concept_distances = sub_dic["concept_distances"]

        n_col = N.shape[1]

        div = np.sqrt(n_col) - concept_distances
        div[div <= 0] = concept_minimum  # to guarantee that minimum will be 0
        weights = np.power(2, -wg_dist_alpha * (concept_distances / div))
        np.fill_diagonal(weights, 0.0)

        wg_dist_example = np.sum(weights * concept_distances, axis=0) / np.sum(
            weights, axis=0
        )

        # The original meta-feature is the mean of the return.
        # It will be done by summary functions.
        return wg_dist_example

    @classmethod
    def ft_impconceptvar(
        cls,
        N: np.ndarray,
        y: np.ndarray,
        impconceptvar_alpha: float = 1.0,
        concept_dist_metric: str = "euclidean",
        concept_distances: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the improved concept variation that estimates the
        variability of class labels among examples.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        y : :obj:`np.ndarray`
            Target attribute.

        impconceptvar_alpha : float, optional
            The alpha value to adjust the weight. The higher the alpha less
            is the effect of the weight in the computation.

        concept_dist_metric : str, optional
            Metric used to compute distance between each pair of examples. See
            cdist from scipy for more options. Used only if the argument
            ``concept_distances`` is None.

        concept_distances : :obj:`np.ndarray`, optional
            Distance matrix of examples from ``N``. Argument used to take
            advantage of precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the improved concept variation for each example.

        References
        ----------
        .. [1] Vilalta, R and Drissi, Y (2002). A Characterization of Difficult
           Problems in Classification. Proceedings of the 2002 International
           Conference on Machine Learning and Applications (pp. 133-138).
        """
        if concept_distances is None:
            sub_dic = cls.precompute_concept_dist(N, concept_dist_metric)
            concept_distances = sub_dic["concept_distances"]

        radius = np.ceil(concept_distances).astype(int)
        radius[radius == 0] = 1

        weights = np.power(2, -impconceptvar_alpha * radius)
        np.fill_diagonal(weights, 0.0)

        rep_class_matrix = np.repeat(np.expand_dims(y, 0), y.shape[0], axis=0)
        # check if class is different
        class_diff = np.not_equal(rep_class_matrix.T, rep_class_matrix).astype(
            int
        )

        impconceptvar_by_example = np.sum(weights * class_diff, axis=0)

        # The original meta-feature is the mean of the return.
        # It will be done by summary functions.
        return impconceptvar_by_example

    @classmethod
    def ft_cohesiveness(
        cls,
        N: np.ndarray,
        cohesiveness_alpha: float = 1.0,
        concept_dist_metric: str = "euclidean",
        concept_distances: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the improved version of the weighted distance, that
        captures how dense or sparse is the example distribution.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Numerical fitted data.

        cohesiveness_alpha : float, optional
            The alpha value to adjust the weight. The higher the alpha less
            is the effect of the weight in the computation.

        concept_dist_metric : str, optional
            Metric used to compute distance between each pair of examples. See
            cdist from scipy for more options. Used only if the argument
            ``concept_distances`` is None.

        concept_distances : :obj:`np.ndarray`, optional
            Distance matrix of examples from ``N``. Argument used to take
            advantage of precomputations.

        Returns
        -------
        :obj:`np.ndarray`
            An array with the cohesiveness for each example.

        References
        ----------
        .. [1] Vilalta, R and Drissi, Y (2002). A Characterization of Difficult
           Problems in Classification. Proceedings of the 2002 International
           Conference on Machine Learning and Applications (pp. 133-138).
        """
        if concept_distances is None:
            sub_dic = cls.precompute_concept_dist(N, concept_dist_metric)
            concept_distances = sub_dic["concept_distances"]

        radius = np.ceil(concept_distances).astype(int)
        radius[radius == 0] = 1

        weights = np.power(2, -cohesiveness_alpha * radius)
        np.fill_diagonal(weights, 0.0)

        cohesiveness_by_example = np.sum(weights, axis=0)

        # The original meta-feature is the mean of the return.
        # It will be done by summary functions.
        return cohesiveness_by_example
