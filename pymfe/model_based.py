"""Module dedicated to extraction of Model-Based Metafeatures."""

from collections import Counter
import typing as t
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class MFEModelBased:
    """Keep methods for metafeatures of ``model-based`` group.

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
       arguments must be strictly optional (i.e., has a predefined
       default value).

    3. The initial assumption is that the user can change any optional
       argument, without any previous verification of argument value or its
       type, via kwargs argument of ``extract`` method of MFE class.

     4. The return value of all feature extraction methods should be a single
        value or a generic Sequence (preferably a :obj:`np.ndarray`)
        type with numeric values.

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
    def precompute_model_based_class(cls, N: np.ndarray, y: np.ndarray,
                                     random_state: t.Optional[int],
                                     **kwargs) -> t.Dict[str, t.Any]:
        """Precompute ``dt_model``, ``dt_info_table`` and ``tree_depth``.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        random_state : :obj:`int`, optional
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        dict
            With following precomputed items:
                - ``dt_model`` (:obj:`DecisionTreeClassifier`): decision tree
                  classifier.
                - ``dt_info_table`` (:obj:`np.ndarray`): tree property dt_info_table.
                - ``tree_depth`` (:obj: `np.ndarray`): the depth of each tree
                  node ordered by node (e.g., index one contain the node one
                  depth, the index two the node two depth and so on).
        """
        prepcomp_vals = {}  # type: t.Dict[str, t.Any]

        if (N is not None and y is not None
                and not {"dt_model", "dt_info_table", "tree_depth"
                         }.issubset(kwargs)):
            dt_model = MFEModelBased._built_dt_model(
                N=N, y=y, random_state=random_state)
            dt_info_table = MFEModelBased._extract_table(N, y, dt_model)
            tree_depth = MFEModelBased.tree_depth(dt_model)
            prepcomp_vals["dt_model"] = dt_model
            prepcomp_vals["dt_info_table"] = dt_info_table
            prepcomp_vals["tree_depth"] = tree_depth

        return prepcomp_vals

    @classmethod
    def _built_dt_model(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            random_state: t.Optional[int] = None,
    ) -> DecisionTreeClassifier:
        """Build a Decision Tree Classifier model."""
        dt_model = DecisionTreeClassifier(random_state=random_state)
        dt_model.fit(N, y)
        return dt_model

    @classmethod
    def _extract_table(cls, N: np.ndarray, y: np.ndarray,
                       dt_model: DecisionTreeClassifier) -> np.ndarray:
        """Precompute ``dt_model``, ``dt_info_table`` and ``tree_depth``.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        random_state : :obj:`int`, optional
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`np.ndarray`
            Decision tree properties table.
                - Each line represents a node.
                - Column 0: It is the id of the attribute splitted in that
                  node.
                - Column 1: It is 1 if the node is a leaf node, otherwise 0.
                - Columns 2: It is the number of examples that fall on that
                  node.
                - Columns 3: It is 0 if the node is not a leaf, otherwise is
                  the class number represented by that leaf node.
        """
        dt_info_table = np.zeros((dt_model.tree_.node_count,
                                  4))  # type: np.ndarray
        dt_info_table[:, 0] = dt_model.tree_.feature
        dt_info_table[:, 2] = dt_model.tree_.n_node_samples

        leaves = dt_model.apply(N)  # type: DecisionTreeClassifier

        if not isinstance(y, np.number):
            _, y = np.unique(y, return_inverse=True)

        tmp = np.array([leaves, y + 1])  # type: np.ndarray

        x = 0  # type: int
        for x in set(leaves):
            dt_info_table[x, 3] = list(Counter(
                tmp[1, tmp[0, :] == x]).keys())[0] + 1
            dt_info_table[x, 1] = 1

        return dt_info_table

    @classmethod
    def ft_leaves(cls, dt_info_table: np.ndarray) -> int:
        """Number of leaves of the DT model.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        Returns
        -------
        :obj:`np.ndarray`
            Number of leaves.
        """

        return np.sum(dt_info_table[:, 1], dtype=int)

    @classmethod
    def ft_tree_depth(cls, tree_depth: np.ndarray) -> np.ndarray:
        """Tree depth, which is the level of all tree nodes and leaves of the
        DT model.

        Parameters
        ----------
        tree_depth : :obj:`np.ndarray`
            Tree depth from ``tree_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            Tree depth.
        """

        return tree_depth

    @classmethod
    def tree_depth(cls, dt_model: DecisionTreeClassifier) -> np.ndarray:
        """Compute the depth of each node.

        Parameters
        ----------
        dt_model : :obj:`DecisionTreeClassifier`
            The DT model.

        Returns
        -------
        :obj:`np.ndarray`
            The depth of each node.
        """

        def node_depth(node_ind: int, cur_depth: int) -> None:
            if not 0 <= node_ind < depths.size:
                return

            depths[node_ind] = cur_depth
            node_depth(son_id_l[node_ind], cur_depth + 1)
            node_depth(son_id_r[node_ind], cur_depth + 1)

        son_id_l = dt_model.tree_.children_left
        son_id_r = dt_model.tree_.children_right

        depths = np.zeros(dt_model.tree_.node_count)

        node_depth(node_ind=0, cur_depth=0)

        return depths

    @classmethod
    def ft_leaves_branch(cls, dt_info_table: np.ndarray,
                         tree_depth: np.ndarray) -> np.ndarray:
        """Size of branches, which consists in the level of all leaves of the
        DT model.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        tree_depth : :obj:`np.ndarray`
            Tree depth from ``tree_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            Size of branches.
        """

        return tree_depth[dt_info_table[:, 1] == 1]

    @classmethod
    def ft_leaves_corrob(cls, N: np.ndarray,
                         dt_info_table: np.ndarray) -> np.ndarray:
        """Leaves corroboration, which is the proportion of examples that
        belong to each leaf of the DT model.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        Returns
        -------
        :obj:`np.ndarray`
            Leaves corroboration.
        """
        return dt_info_table[:, 2][dt_info_table[:, 1] == 1] / N.shape[0]

    @classmethod
    def ft_tree_shape(cls, dt_info_table: np.ndarray,
                      tree_depth: np.ndarray) -> np.ndarray:
        """Calculate the Tree shape.

        The tree shape is the probability of arrive in each leaf given a
        random walk. We call this as the ``structural shape of the DT model.``

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        tree_depth : :obj:`np.ndarray`
            Tree depth from ``tree_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            The tree shape.
        """
        aux = tree_depth[dt_info_table[:, 1] == 1]  # type: np.ndarray
        return -(1.0 / 2**aux) * np.log2(1.0 / 2**aux)

    @classmethod
    def ft_leaves_homo(cls, dt_info_table: np.ndarray,
                       tree_depth: np.ndarray) -> np.ndarray:
        """Homogeneity, which is the number of leaves divided by the structural
        shape of the DT model.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        tree_depth : :obj:`np.ndarray`
            Tree depth from ``tree_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            The homogeneity.
        """
        num_leaves = MFEModelBased.ft_leaves(dt_info_table)  # type: int

        tree_shape = MFEModelBased.ft_tree_shape(
            dt_info_table, tree_depth)  # type: np.ndarray

        return num_leaves / tree_shape

    @classmethod
    def ft_leaves_per_class(cls, dt_info_table: np.ndarray) -> np.ndarray:
        """Leaves per class, which is the proportion of leaves of the DT model
        associated with each class.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        Returns
        -------
        :obj:`np.ndarray`
            Leaves per class.
        """
        aux = np.array(list(Counter(
            dt_info_table[:, 3]).values()))  # np.ndarray

        aux = aux[1:] / MFEModelBased.ft_leaves(dt_info_table)

        return aux

    @classmethod
    def ft_nodes(cls, dt_info_table: np.ndarray) -> int:
        """Number of nodes of the DT model.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        Returns
        -------
        :obj:`np.ndarray`
            Number of nodes.
        """
        return np.sum(dt_info_table[:, 1] != 1)

    @classmethod
    def ft_nodes_per_attr(cls, N: np.ndarray,
                          dt_info_table: np.ndarray) -> float:
        """Ratio of the number of nodes of the DT model per the number of
        attributes.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        Returns
        -------
        :obj:`np.ndarray`
            Ratio of the number of nodes.
        """
        num_nodes = MFEModelBased.ft_nodes(dt_info_table)  # type: int
        num_attr = N.shape[1]  # type: float

        return num_nodes / num_attr

    @classmethod
    def ft_nodes_per_inst(cls, N: np.ndarray,
                          dt_info_table: np.ndarray) -> float:
        """Ratio of the number of nodes of the DT model per the number of
        instances.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        Returns
        -------
        :obj:`np.ndarray`
            Ratio of the number of nodes per instances.
        """
        num_nodes = MFEModelBased.ft_nodes(dt_info_table)  # type: int
        num_inst = N.shape[0]  # type: float

        return num_nodes / num_inst

    @classmethod
    def ft_nodes_per_level(cls, dt_info_table: np.ndarray,
                           tree_depth: np.ndarray) -> float:
        """Number of nodes of the DT model per level.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        tree_depth : :obj:`np.ndarray`
            Tree depth from ``tree_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            Number of nodes per level.
        """
        non_leaf_depths = tree_depth[dt_info_table[:, 1] == 0]

        _, node_num_per_level = np.unique(non_leaf_depths, return_counts=True)

        return node_num_per_level

    @classmethod
    def ft_nodes_repeated(cls, dt_info_table: np.ndarray) -> np.ndarray:
        """Counts the number of repeated nodes.

        The number of repeated nodes is the number of repeated attributes
        that appear in the DT model.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        Returns
        -------
        :obj:`np.ndarray`
            Repeated nodes.
        """
        nodes_attr_ids = dt_info_table[dt_info_table[:, 1] == 0, 0]

        _, attr_counts = np.unique(nodes_attr_ids, return_counts=True)

        return attr_counts

    @classmethod
    def ft_var_importance(cls, dt_model: DecisionTreeClassifier) -> np.ndarray:
        """Get each features importance of the DT model.

        It is calculated using the Gini index to estimate the amount of
        information used in the DT model.

        Parameters
        ----------
        dt_model : :obj:`DecisionTreeClassifier`
            The DT model.

        Return:
        :obj:`np.ndarray`
            Features importance.
        """
        importance = dt_model.tree_.compute_feature_importances()  # np.ndarray
        return importance

    @classmethod
    def ft_tree_imbalance(cls, dt_info_table: np.ndarray,
                          tree_depth: np.ndarray) -> np.ndarray:
        """Tree imbalance.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            Decision tree properties table.

        tree_depth : :obj:`np.ndarray`
            Tree depth from ``tree_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            Tree imbalance values.
        """
        leaves_depth = tree_depth[dt_info_table[:, 1] == 1]  # np.ndarray
        aux = 1.0 / 2**leaves_depth
        tmp = 1.0 / 2**np.multiply(*np.unique(
            aux, return_counts=True))  # np.ndarray
        return -tmp * np.log2(tmp)
