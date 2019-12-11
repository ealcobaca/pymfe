"""Module dedicated to extraction of model-based metafeatures."""

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
    def precompute_model_based_class(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            random_state: t.Optional[int] = None,
            hypparam_model_dt: t.Optional[t.Dict[str, t.Any]] = None,
            **kwargs) -> t.Dict[str, t.Any]:
        """Precompute the DT Model and some information related to it.

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
        :obj:`dict`
            With following precomputed items:
                - ``dt_model`` (:obj:`DecisionTreeClassifier`): decision tree
                  classifier.
                - ``dt_info_table`` (:obj:`np.ndarray`): some tree properties
                  table.
                - ``dt_nodes_depth`` (:obj:`np.ndarray`): the depth of each
                  tree node ordered by node (e.g., index one contain the node
                  one depth, the index two the node two depth and so on.)
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if (N is not None and y is not None and not {
                "dt_model", "dt_info_table", "dt_nodes_depth", "leaf_nodes",
                "non_leaf_nodes"
        }.issubset(kwargs)):
            if hypparam_model_dt is None:
                hypparam_model_dt = {}

            dt_model = MFEModelBased._fit_dt_model(
                N=N, y=y, random_state=random_state, **hypparam_model_dt)

            leaf_nodes = dt_model.tree_.feature < 0

            dt_info_table = MFEModelBased._extract_table(
                dt_model=dt_model, leaf_nodes=leaf_nodes)
            dt_nodes_depth = MFEModelBased._calc_dt_node_depths(dt_model)

            precomp_vals["leaf_nodes"] = np.flatnonzero(leaf_nodes)
            precomp_vals["non_leaf_nodes"] = np.flatnonzero(~leaf_nodes)
            precomp_vals["dt_model"] = dt_model
            precomp_vals["dt_info_table"] = dt_info_table
            precomp_vals["dt_nodes_depth"] = dt_nodes_depth

        return precomp_vals

    @classmethod
    def _fit_dt_model(cls,
                      N: np.ndarray,
                      y: np.ndarray,
                      random_state: t.Optional[int] = None,
                      **kwargs) -> DecisionTreeClassifier:
        """Build a Decision Tree Classifier model."""
        dt_model = DecisionTreeClassifier(random_state=random_state, **kwargs)
        return dt_model.fit(X=N, y=y)

    @classmethod
    def _extract_table(cls, dt_model: DecisionTreeClassifier,
                       leaf_nodes: np.ndarray) -> np.ndarray:
        """Bookkeep some information table from the ``dt_model`` into an array.

        Parameters
        ----------
        dt_model : :obj:`DecisionTreeClassifier`
            The DT model.

        Returns
        -------
        :obj:`np.ndarray`
            DT model properties table.
                - Each line represents a node.
                - Column 0: It is the id of the attribute splitted in that
                  node.
                - Columns 1: It is the number of examples that fall on that
                  node.
                - Columns 2: It is 0 if the node is not a leaf, otherwise is
                  the class number represented by that leaf node.
        """
        dt_info_table = np.zeros((dt_model.tree_.node_count, 3),
                                 dtype=int)  # type: np.ndarray

        dt_info_table[:, 0] = dt_model.tree_.feature
        dt_info_table[:, 1] = dt_model.tree_.n_node_samples

        dt_info_table[leaf_nodes, 2] = np.argmax(
            dt_model.tree_.value[leaf_nodes], axis=2).ravel() + 1

        return dt_info_table

    @classmethod
    def _calc_dt_node_depths(cls,
                             dt_model: DecisionTreeClassifier) -> np.ndarray:
        """Compute the depth of each node in the DT model.

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

        depths = np.zeros(dt_model.tree_.node_count, dtype=int)

        node_depth(node_ind=0, cur_depth=0)

        return depths

    @classmethod
    def ft_leaves(cls, dt_model: DecisionTreeClassifier) -> int:
        """Number of leaf nodes in the DT model.

        Parameters
        ----------
        dt_model : :obj:`DecisionTreeClassifier`
            The DT model.

        Returns
        -------
        :obj:`int`
            Number of leaf nodes in the DT model.

        References
        ----------
        .. [1] Yonghong Peng, PA Flach, Pavel Brazdil, and Carlos Soares.
           Decision tree-based data characterization for meta-learning.
           In 2nd ECML/PKDD International Workshop on Integration and
           Collaboration Aspects of Data Mining, Decision Support and
           Meta-Learning(IDDM), pages 111 – 122, 2002a.
        """
        return dt_model.tree_.n_leaves

    @classmethod
    def ft_tree_depth(cls, dt_nodes_depth: np.ndarray) -> np.ndarray:
        """Calculate the depth of every node in the DT model.

        Parameters
        ----------
        dt_nodes_depth : :obj:`np.ndarray`
            Depth of each node in the DT model.

        Returns
        -------
        :obj:`np.ndarray`
            Depth of every node in the DT model.

        References
        ----------
        .. [1] Yonghong Peng, PA Flach, Pavel Brazdil, and Carlos Soares.
           Decision tree-based data characterization for meta-learning.
           In 2nd ECML/PKDD International Workshop on Integration and
           Collaboration Aspects of Data Mining, Decision Support and
           Meta-Learning(IDDM), pages 111 – 122, 2002a.
        """
        return dt_nodes_depth

    @classmethod
    def ft_leaves_branch(cls, leaf_nodes: np.ndarray,
                         dt_nodes_depth: np.ndarray) -> np.ndarray:
        """Compute the size of branches in the DT model.

        The size of branches consists in the depth of all leaves of the
        DT model.

        Parameters
        ----------
        leaf_nodes : :obj:`np.ndarray`
            Array referencing the leaf nodes of the DT model.

        dt_nodes_depth : :obj:`np.ndarray`
            Tree depth from ``dt_nodes_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            Size of branches of the DT model.

        References
        ----------
        .. [1] Yonghong Peng, PA Flach, Pavel Brazdil, and Carlos Soares.
           Decision tree-based data characterization for meta-learning.
           In 2nd ECML/PKDD International Workshop on Integration and
           Collaboration Aspects of Data Mining, Decision Support and
           Meta-Learning(IDDM), pages 111 – 122, 2002a.
        """
        return dt_nodes_depth[leaf_nodes]

    @classmethod
    def ft_leaves_corrob(cls, leaf_nodes: np.ndarray,
                         dt_info_table: np.ndarray) -> np.ndarray:
        """Calculate the leaves corroboration of the DT model.

        The Leaves corroboration is the proportion of examples that
        belong to each leaf of the DT model.

        Parameters
        ----------
        leaf_nodes : :obj:`np.ndarray`
            Array referencing the leaf nodes of the DT model.

        dt_info_table : :obj:`np.ndarray`
            DT model properties table.

        Returns
        -------
        :obj:`np.ndarray`
            Leaves corroboration for every leaf node.

        References
        ----------
        .. [1] Hilan Bensusan, Christophe Giraud-Carrier, and Claire Kennedy.
            A higher-order approachto meta-learning. In 10th International
            Conference Inductive Logic Programming (ILP), pages 33 – 42, 2000.
        """
        num_samples_leaves = dt_info_table[leaf_nodes, 1]  # type: np.ndarray

        # Note: the 0th node is the tree root and, therefore,
        # contains all training samples
        num_samples_total = dt_info_table[0, 1]  # type: int

        return num_samples_leaves / num_samples_total

    @classmethod
    def ft_tree_shape(cls, leaf_nodes: np.ndarray,
                      dt_nodes_depth: np.ndarray) -> np.ndarray:
        """Calculate the tree shape for every leaf node.

        The tree shape is the probability of arrive in each leaf given a
        random walk. We call this as the ``structural shape of the DT model.``

        Parameters
        ----------
        leaf_nodes : :obj:`np.ndarray`
            Array referencing the leaf nodes of the DT model.

        dt_nodes_depth : :obj:`np.ndarray`
            Tree depth from ``dt_nodes_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            The tree shape for every leaf node.

        References
        ----------
        .. [1] Hilan Bensusan, Christophe Giraud-Carrier, and Claire Kennedy.
            A higher-order approachto meta-learning. In 10th International
            Conference Inductive Logic Programming (ILP), pages 33 – 42, 2000.
        """
        leaf_depths = dt_nodes_depth[leaf_nodes]
        prob_random_arrival = np.power(2.0, -leaf_depths)
        return -prob_random_arrival * np.log2(prob_random_arrival)

    @classmethod
    def ft_leaves_homo(cls, leaf_nodes: np.ndarray, dt_nodes_depth: np.ndarray,
                       dt_model: DecisionTreeClassifier) -> np.ndarray:
        """Calculate the DT model Homogeneity for every leaf node.

        The DT model homogeneity is calculated by the number of leaves
        divided by the ``structural shape`` (which is calculated by the
        ``ft_tree_shape`` method) of the DT model.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            DT model properties table.

        dt_nodes_depth : :obj:`np.ndarray`
            Tree depth from ``dt_nodes_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            The DT model homogeneity for every leaf node.

        References
        ----------
        .. [1] Hilan Bensusan, Christophe Giraud-Carrier, and Claire Kennedy.
            A higher-order approachto meta-learning. In 10th International
            Conference Inductive Logic Programming (ILP), pages 33 – 42, 2000.
        """
        num_leaves = MFEModelBased.ft_leaves(dt_model)

        tree_shape = MFEModelBased.ft_tree_shape(
            leaf_nodes, dt_nodes_depth)  # type: np.ndarray

        return num_leaves / tree_shape

    @classmethod
    def ft_leaves_per_class(cls, dt_info_table: np.ndarray,
                            dt_model: DecisionTreeClassifier) -> np.ndarray:
        """Computer the proportion of leaves per class in the DT model.

        This quantity is computed by the proportion of leaves of the DT model
        associated with each class.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            DT model properties table.

        Returns
        -------
        :obj:`np.ndarray`
            Leaves per class.

        References
        ----------
        .. [1] Andray Filchenkov and Arseniy Pendryak. Datasets meta-feature
           description for recom-mending feature selection algorithm. In
           Artificial Intelligence and Natural Language and Information
           Extraction, Social Media and Web Search FRUCT Conference
           (AINL-ISMWFRUCT), pages 11 – 18, 2015.
        """
        node_class_ids = dt_info_table[:, 2]

        _, class_id_freqs = np.unique(node_class_ids, return_counts=True)

        # Note: the id == 0 is not associated to any class.
        return class_id_freqs[1:] / MFEModelBased.ft_leaves(dt_model)

    @classmethod
    def ft_nodes(cls, dt_model: DecisionTreeClassifier) -> int:
        """Number of non-leaf nodes of the DT model.

        Parameters
        ----------
        dt_model : :obj:`DecisionTreeClassifier`
            The DT model.

        Returns
        -------
        :obj:`int`
            Number of non-leaf nodes.

        References
        ----------
        .. [1] Yonghong Peng, PA Flach, Pavel Brazdil, and Carlos Soares.
           Decision tree-based data characterization for meta-learning.
           In 2nd ECML/PKDD International Workshop on Integration and
           Collaboration Aspects of Data Mining, Decision Support and
           Meta-Learning(IDDM), pages 111 – 122, 2002a.
        """
        return dt_model.tree_.node_count - dt_model.tree_.n_leaves

    @classmethod
    def ft_nodes_per_attr(cls, dt_model: DecisionTreeClassifier) -> float:
        """Ratio of the DT model number of nodes per number of attributes.

        Parameters
        ----------
        dt_model : :obj:`DecisionTreeClassifier`
            The DT model.

        Returns
        -------
        :obj:`float`
            Ratio of the number of non-leaf nodes per number of attributes.

        References
        ----------
        .. [1] Hilan Bensusan, Christophe Giraud-Carrier, and Claire Kennedy.
            A higher-order approachto meta-learning. In 10th International
            Conference Inductive Logic Programming (ILP), pages 33 – 42, 2000.
        """
        num_non_leaf_nodes = MFEModelBased.ft_nodes(dt_model)

        return num_non_leaf_nodes / dt_model.tree_.n_features

    @classmethod
    def ft_nodes_per_inst(cls, dt_model: DecisionTreeClassifier) -> float:
        """Ratio of the number of non-leaf nodes per the number of instances.

        Parameters
        ----------
        dt_model : :obj:`DecisionTreeClassifier`
            The DT model.

        Returns
        -------
        :obj:`float`
            Ratio of the number of non-leaf nodes per instances.

        References
        ----------
        .. [1] Hilan Bensusan, Christophe Giraud-Carrier, and Claire Kennedy.
            A higher-order approachto meta-learning. In 10th International
            Conference Inductive Logic Programming (ILP), pages 33 – 42, 2000.
        """
        num_non_leaf_nodes = MFEModelBased.ft_nodes(dt_model)
        num_inst = dt_model.tree_.n_node_samples[0]

        return num_non_leaf_nodes / num_inst

    @classmethod
    def ft_nodes_per_level(cls, dt_nodes_depth: np.ndarray,
                           non_leaf_nodes: np.ndarray) -> np.ndarray:
        """Number of nodes of the DT model per tree level.

        Parameters
        ----------
        dt_nodes_depth : :obj:`np.ndarray`
            Tree depth from ``dt_nodes_depth`` method.

        non_leaf_nodes : :obj:`np.ndarray`
            Array referencing the non-leaf nodes of the DT model.

        Returns
        -------
        :obj:`np.ndarray`
            Number of nodes per level.

        References
        ----------
        .. [1] Yonghong Peng, PA Flach, Pavel Brazdil, and Carlos Soares.
           Decision tree-based data characterization for meta-learning.
           In 2nd ECML/PKDD International Workshop on Integration and
           Collaboration Aspects of Data Mining, Decision Support and
           Meta-Learning(IDDM), pages 111 – 122, 2002a.
        """
        non_leaf_depths = dt_nodes_depth[non_leaf_nodes]

        _, node_num_per_level = np.unique(non_leaf_depths, return_counts=True)

        return node_num_per_level

    @classmethod
    def ft_nodes_repeated(cls, dt_info_table: np.ndarray,
                          non_leaf_nodes: np.ndarray) -> np.ndarray:
        """Counts the number of repeated nodes.

        The number of repeated nodes is the number of repeated attributes
        that appear in the DT model.

        Parameters
        ----------
        dt_info_table : :obj:`np.ndarray`
            DT model properties table.

        non_leaf_nodes : :obj:`np.ndarray`
            Array referencing the non-leaf nodes of the DT model.

        Returns
        -------
        :obj:`np.ndarray`
            Absolute frequency of each repeated node.

        References
        ----------
        .. [1] Hilan Bensusan, Christophe Giraud-Carrier, and Claire Kennedy.
            A higher-order approachto meta-learning. In 10th International
            Conference Inductive Logic Programming (ILP), pages 33 – 42, 2000.
        """
        nodes_attr_ids = dt_info_table[non_leaf_nodes, 0]

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

        Returns
        -------
        :obj:`np.ndarray`
            Features importance given by the DT model.

        References
        ----------
        .. [1] Hilan Bensusan, Christophe Giraud-Carrier, and Claire Kennedy.
            A higher-order approachto meta-learning. In 10th International
            Conference Inductive Logic Programming (ILP), pages 33 – 42, 2000.
        """
        return dt_model.feature_importances_

    @classmethod
    def ft_tree_imbalance(cls, leaf_nodes: np.ndarray,
                          dt_nodes_depth: np.ndarray) -> np.ndarray:
        """Calculate the Tree imbalance for each leaf node.

        Parameters
        ----------
        leaf_nodes : :obj:`np.ndarray`
            Array referencing the leaf nodes of the DT model.

        dt_nodes_depth : :obj:`np.ndarray`
            Tree depth from ``dt_nodes_depth`` method.

        Returns
        -------
        :obj:`np.ndarray`
            Tree imbalance values for every leaf node.

        References
        ----------
        .. [1] Hilan Bensusan, Christophe Giraud-Carrier, and Claire Kennedy.
            A higher-order approachto meta-learning. In 10th International
            Conference Inductive Logic Programming (ILP), pages 33 – 42, 2000.
        """
        leaf_depths = dt_nodes_depth[leaf_nodes]
        prob_random_arrival = np.power(2.0, -leaf_depths)
        aux = np.power(2.0, -np.multiply(
            *np.unique(prob_random_arrival, return_counts=True)))  # np.ndarray
        return -aux * np.log2(aux)
