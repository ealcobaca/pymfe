"""Module dedicated to extraction of Landmarking and Relative Landmarking
Metafeatures."""

import typing as t

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class MFELandmarking:
    """Keep methods for metafeatures of ``landmarking`` group.

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
       value or a generic Sequence (preferably a :obj:`np.ndarray`) type with
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
    def postprocess_landmarking_relative(
            cls,
            results: t.Tuple[t.List[str], t.List[float], t.List[float]],
            groups: t.Tuple["str", ...],
            **kwargs  # ignore: W0613
    ) -> t.Optional[t.Tuple[t.List[str], t.List[float], t.List[float]]]:
        # pylint: disable=W0613
        """Structure to implement relative landmarking metafeatures."""

        if "relative" not in groups:
            return None

        mtf_rel_names = []  # type: t.List[str]
        mtf_rel_vals = []  # type: t.List[float]
        mtf_rel_time = []  # type: t.List[float]

        if "landmarking" not in groups:
            # If ``landmarking`` not in groups, then just change
            # the metafeatures in-place and return None
            return None

        return mtf_rel_names, mtf_rel_vals, mtf_rel_time

    @classmethod
    def precompute_landmarking_class(cls,
                                     N: np.ndarray,
                                     sample_size: float,
                                     folds: int,
                                     random_state: t.Optional[int] = None,
                                     **kwargs) -> t.Dict[str, t.Any]:
        """Precompute k-fold cross validation strategy.

        Parameters
        ----------
        N : :obj:`np.ndarray`, optional
            Attributes from fitted data.

        sample_size : :obj: `float`
            The percentage of examples subsampled. Value different from default
            will generate the subsampling-based relative landmarking
            metafeatures.

        folds : :obj: `int`
            Number of folds to k-fold cross validation.

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
                - ``skf`` (:obj:`StratifiedKFold`): Stratified K-Folds
                  cross-validator. Provides train/test indices to split data in
                  train/test sets.
        """

        prepcomp_vals = {}

        if N is not None and "sample_indexes" not in kwargs:
            if sample_size != 1:
                num_inst, _ = N.shape

                prepcomp_vals["sample_indexes"] = (
                    MFELandmarking._get_sample_indexes(
                        num_inst=num_inst,
                        sample_size=sample_size,
                        random_state=random_state))

        if "skf" not in kwargs:
            prepcomp_vals["skf"] = StratifiedKFold(
                n_splits=folds,
                random_state=random_state)

        return prepcomp_vals

    @classmethod
    def _get_sample_indexes(cls, num_inst: int, sample_size: float,
                            random_state: t.Optional[int]) -> np.ndarray:
        """Sample indexes to calculate subsampling landmarking metafeatures."""
        if random_state is not None:
            np.random.seed(random_state)

        sample_indexes = np.random.choice(
            a=num_inst, size=int(sample_size * num_inst), replace=False)

        return sample_indexes

    @classmethod
    def _sample_data(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            sample_size: float,
            random_state: t.Optional[int] = None,
            sample_indexes: t.Optional[np.ndarray] = None,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Select ``sample_size`` percent of data points in ``N`` and ``y``."""
        if sample_size >= 1.0 and sample_indexes is None:
            return N, y

        if sample_indexes is None:
            num_inst = y.size

            sample_indexes = MFELandmarking._get_sample_indexes(
                num_inst=num_inst,
                sample_size=sample_size,
                random_state=random_state)

        return N[sample_indexes, :], y[sample_indexes]

    @classmethod
    def _importance(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            sample_size: float = 1.0,
            sample_indexes: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Gini index of a decision tree.

        It is used the ``DecisionTreeClassifier`` implementation
        from ``sklearn`` package.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        sample_size : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_indexes`` is None.

        sample_indexes : :obj:`np.ndarray`, optional
            Array of indexes of instances to be effectively used while
            extracting this metafeature. If None, then ``sample_size``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            Return the decision tree features importance.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            sample_size=sample_size,
            random_state=random_state,
            sample_indexes=sample_indexes)

        clf = DecisionTreeClassifier(random_state=random_state).fit(N, y)
        return np.argsort(clf.feature_importances_)

    @classmethod
    def ft_best_node(cls,
                     N: np.ndarray,
                     y: np.ndarray,
                     score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
                     skf: t.Optional[StratifiedKFold] = None,
                     folds: int = 10,
                     sample_size: float = 1.0,
                     sample_indexes: t.Optional[np.ndarray] = None,
                     random_state: t.Optional[int] = None) -> np.ndarray:
        """Performance of a the best single decision tree node.

        Construct a single decision tree node model induced by the most
        informative attribute to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        folds : :obj: `int`, optional
            Number of folds to k-fold cross validation. Used only if ``skf``
            is None.

        sample_size : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_indexes`` is None.

        sample_indexes : :obj:`np.ndarray`, optional
            Array of indexes of instances to be effectively used while
            extracting this metafeature. If None, then ``sample_size``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            sample_size=sample_size,
            random_state=random_state,
            sample_indexes=sample_indexes)

        if skf is None:
            skf = StratifiedKFold(n_splits=folds, random_state=random_state)

        result = []
        for train_index, test_index in skf.split(N, y):
            model = DecisionTreeClassifier(
                max_depth=1, random_state=random_state)
            X_train = N[train_index, :]
            X_test = N[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.array(result)

    @classmethod
    def ft_random_node(cls,
                       N: np.ndarray,
                       y: np.ndarray,
                       score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
                       skf: t.Optional[StratifiedKFold] = None,
                       folds: int = 10,
                       sample_size: float = 1.0,
                       sample_indexes: t.Optional[np.ndarray] = None,
                       random_state: t.Optional[int] = None) -> np.ndarray:
        """Single decision tree node model induced by a random attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        folds : :obj: `int`, optional
            Number of folds to k-fold cross validation. Used only if ``skf``
            is None.

        sample_size : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_indexes`` is None.

        sample_indexes : :obj:`np.ndarray`, optional
            Array of indexes of instances to be effectively used while
            extracting this metafeature. If None, then ``sample_size``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            sample_size=sample_size,
            random_state=random_state,
            sample_indexes=sample_indexes)

        if skf is None:
            skf = StratifiedKFold(n_splits=folds, random_state=random_state)

        result = []
        for train_index, test_index in skf.split(N, y):
            if isinstance(random_state, int):
                np.random.seed(random_state)

            attr = np.random.randint(0, N.shape[1], size=(1, ))
            model = DecisionTreeClassifier(
                max_depth=1, random_state=random_state)
            X_train = N[train_index, :][:, attr]
            X_test = N[test_index, :][:, attr]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.array(result)

    @classmethod
    def ft_worst_node(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            folds: int = 10,
            sample_size: float = 1.0,
            sample_indexes: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Single decision tree node model induced by the worst informative attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        score : obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        folds : :obj: `int`, optional
            Number of folds to k-fold cross validation. Used only if ``skf``
            is None.

        sample_size : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_indexes`` is None.

        sample_indexes : :obj:`np.ndarray`, optional
            Array of indexes of instances to be effectively used while
            extracting this metafeature. If None, then ``sample_size``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            sample_size=sample_size,
            random_state=random_state,
            sample_indexes=sample_indexes)

        if skf is None:
            skf = StratifiedKFold(n_splits=folds, random_state=random_state)

        result = []
        for train_index, test_index in skf.split(N, y):
            importance = MFELandmarking._importance(
                N=N[train_index], y=y[train_index], random_state=random_state)

            model = DecisionTreeClassifier(
                max_depth=1, random_state=random_state)
            X_train = N[train_index, :][:, [importance[0]]]
            X_test = N[test_index, :][:, [importance[0]]]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.array(result)

    @classmethod
    def ft_linear_discr(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            folds: int = 10,
            sample_size: float = 1.0,
            sample_indexes: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Apply the Linear Discriminant classifier.

        The Linear Discriminant Classifier is used to construct a linear split
        (non parallel axis) in the data to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        folds : :obj: `int`, optional
            Number of folds to k-fold cross validation. Used only if ``skf``
            is None.

        sample_size : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_indexes`` is None.

        sample_indexes : :obj:`np.ndarray`, optional
            Array of indexes of instances to be effectively used while
            extracting this metafeature. If None, then ``sample_size``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            sample_size=sample_size,
            random_state=random_state,
            sample_indexes=sample_indexes)

        if skf is None:
            skf = StratifiedKFold(n_splits=folds, random_state=random_state)

        result = []
        for train_index, test_index in skf.split(N, y):
            model = LinearDiscriminantAnalysis()
            X_train = N[train_index, :]
            X_test = N[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.array(result)

    @classmethod
    def ft_naive_bayes(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            folds: int = 10,
            sample_size: float = 1.0,
            sample_indexes: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Evaluate the performance of the Naive Bayes classifier.

        It assumes that the attributes are independent and each example
        belongs to a certain class based on the Bayes probability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        folds : :obj: `int`, optional
            Number of folds to k-fold cross validation. Used only if ``skf``
            is None.

        sample_size : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_indexes`` is None.

        sample_indexes : :obj:`np.ndarray`, optional
            Array of indexes of instances to be effectively used while
            extracting this metafeature. If None, then ``sample_size``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            sample_size=sample_size,
            random_state=random_state,
            sample_indexes=sample_indexes)

        if skf is None:
            skf = StratifiedKFold(n_splits=folds, random_state=random_state)

        result = []
        for train_index, test_index in skf.split(N, y):
            model = GaussianNB()
            X_train = N[train_index, :]
            X_test = N[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.array(result)

    @classmethod
    def ft_one_nn(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            folds: int = 10,
            sample_size: float = 1.0,
            sample_indexes: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Evaluate the performance of the 1-nearest neighbor classifier.

        It uses the euclidean distance of the nearest neighbor to determine
        how noisy is the data.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y :obj:`np.ndarray`
            Target attribute from fitted data.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        folds : :obj: `int`, optional
            Number of folds to k-fold cross validation. Used only if ``skf``
            is None.

        sample_size : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_indexes`` is None.

        sample_indexes : :obj:`np.ndarray`, optional
            Array of indexes of instances to be effectively used while
            extracting this metafeature. If None, then ``sample_size``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            sample_size=sample_size,
            random_state=random_state,
            sample_indexes=sample_indexes)

        if skf is None:
            skf = StratifiedKFold(n_splits=folds, random_state=random_state)

        result = []
        for train_index, test_index in skf.split(N, y):
            model = KNeighborsClassifier(n_neighbors=1)
            X_train = N[train_index, :]
            X_test = N[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.array(result)

    @classmethod
    def ft_elite_nn(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            folds: int = 10,
            sample_size: float = 1.0,
            sample_indexes: t.Optional[np.ndarray] = None,
            random_state: t.Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Elite Nearest Neighbor score of dataset.

        Elite nearest neighbor uses the most informative attribute in the
        dataset to induce the 1-nearest neighbor. With the subset of informati-
        ve attributes is expected that the models should be noise tolerant.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        skf : :obj:`StratifiedKFold`, optional
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        folds : :obj: `int`, optional
            Number of folds to k-fold cross validation. Used only if ``skf``
            is None.

        sample_size : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_indexes`` is None.

        sample_indexes : :obj:`np.ndarray`, optional
            Array of indexes of instances to be effectively used while
            extracting this metafeature. If None, then ``sample_size``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            sample_size=sample_size,
            random_state=random_state,
            sample_indexes=sample_indexes)

        if skf is None:
            skf = StratifiedKFold(n_splits=folds, random_state=random_state)

        result = []
        for train_index, test_index in skf.split(N, y):
            importance = MFELandmarking._importance(
                N=N[train_index], y=y[train_index], random_state=random_state)

            model = KNeighborsClassifier(n_neighbors=1)
            X_train = N[train_index, :][:, [importance[-1]]]
            X_test = N[test_index, :][:, [importance[-1]]]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.array(result)
