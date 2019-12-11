"""Module dedicated to extraction of landmarking metafeatures."""

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
    def precompute_landmarking_class(cls,
                                     N: np.ndarray,
                                     lm_sample_frac: float,
                                     num_cv_folds: int,
                                     shuffle_cv_folds: bool,
                                     random_state: t.Optional[int] = None,
                                     **kwargs) -> t.Dict[str, t.Any]:
        """Precompute k-fold cross validation strategy.

        Parameters
        ----------
        N : :obj:`np.ndarray`, optional
            Attributes from fitted data.

        lm_sample_frac : :obj: `float`
            The percentage of examples subsampled. Value different from default
            will generate the subsampling-based relative landmarking
            metafeatures.

        num_cv_folds : :obj: `int`
            Number of num_cv_folds to k-fold cross validation.

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
        precomp_vals = {}

        if N is not None and "sample_inds" not in kwargs:
            if lm_sample_frac < 1.0:
                num_inst, _ = N.shape

                precomp_vals["sample_inds"] = (MFELandmarking._get_sample_inds(
                    num_inst=num_inst,
                    lm_sample_frac=lm_sample_frac,
                    random_state=random_state))

        if "skf" not in kwargs:
            precomp_vals["skf"] = StratifiedKFold(
                n_splits=num_cv_folds,
                shuffle=shuffle_cv_folds,
                random_state=random_state if shuffle_cv_folds else None)

        return precomp_vals

    @classmethod
    def _get_sample_inds(cls, num_inst: int, lm_sample_frac: float,
                         random_state: t.Optional[int]) -> np.ndarray:
        """Sample indices to calculate subsampling landmarking metafeatures."""
        if random_state is not None:
            np.random.seed(random_state)

        sample_inds = np.random.choice(
            a=num_inst, size=int(lm_sample_frac * num_inst), replace=False)

        return sample_inds

    @classmethod
    def _sample_data(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            lm_sample_frac: float,
            random_state: t.Optional[int] = None,
            sample_inds: t.Optional[np.ndarray] = None,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Select ``lm_sample_frac`` percent of data from ``N`` and ``y``."""
        if lm_sample_frac >= 1.0 and sample_inds is None:
            return N, y

        if sample_inds is None:
            num_inst = y.size

            sample_inds = MFELandmarking._get_sample_inds(
                num_inst=num_inst,
                lm_sample_frac=lm_sample_frac,
                random_state=random_state)

        return N[sample_inds, :], y[sample_inds]

    @classmethod
    def _importance(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            lm_sample_frac: float = 1.0,
            sample_inds: t.Optional[np.ndarray] = None,
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

        lm_sample_frac : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
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
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds)

        clf = DecisionTreeClassifier(random_state=random_state).fit(N, y)
        return np.argsort(clf.feature_importances_)

    @classmethod
    def ft_best_node(cls,
                     N: np.ndarray,
                     y: np.ndarray,
                     score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
                     skf: t.Optional[StratifiedKFold] = None,
                     num_cv_folds: int = 10,
                     shuffle_cv_folds: bool = False,
                     lm_sample_frac: float = 1.0,
                     sample_inds: t.Optional[np.ndarray] = None,
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

        num_cv_folds : :obj: `int`, optional
            Number of num_cv_folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : :obj:`bool`, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.

        References
        ----------
        .. [1] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds)

        if skf is None:
            skf = StratifiedKFold(
                n_splits=num_cv_folds,
                shuffle=shuffle_cv_folds,
                random_state=random_state if shuffle_cv_folds else None)

        result = np.zeros(skf.n_splits, dtype=float)
        for fold_ind, split_inds in enumerate(skf.split(N, y)):
            train_inds, test_inds = split_inds
            model = DecisionTreeClassifier(
                max_depth=1, random_state=random_state)
            X_train = N[train_inds, :]
            X_test = N[test_inds, :]
            y_train, y_test = y[train_inds], y[test_inds]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result[fold_ind] = score(y_test, pred)

        return result

    @classmethod
    def ft_random_node(cls,
                       N: np.ndarray,
                       y: np.ndarray,
                       score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
                       skf: t.Optional[StratifiedKFold] = None,
                       num_cv_folds: int = 10,
                       shuffle_cv_folds: bool = False,
                       lm_sample_frac: float = 1.0,
                       sample_inds: t.Optional[np.ndarray] = None,
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

        num_cv_folds : :obj: `int`, optional
            Number of num_cv_folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : :obj:`bool`, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.

        References
        ----------
        .. [1] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds)

        if skf is None:
            skf = StratifiedKFold(
                n_splits=num_cv_folds,
                shuffle=shuffle_cv_folds,
                random_state=random_state if shuffle_cv_folds else None)

        if random_state is not None:
            np.random.seed(random_state)

        rand_attr_ind = np.random.randint(0, N.shape[1], size=1)

        result = np.zeros(skf.n_splits, dtype=float)
        for fold_ind, split_inds in enumerate(skf.split(N, y)):
            train_inds, test_inds = split_inds

            model = DecisionTreeClassifier(
                max_depth=1, random_state=random_state)
            X_train = N[train_inds, :][:, rand_attr_ind]
            X_test = N[test_inds, :][:, rand_attr_ind]
            y_train, y_test = y[train_inds], y[test_inds]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result[fold_ind] = score(y_test, pred)

        return result

    @classmethod
    def ft_worst_node(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            num_cv_folds: int = 10,
            shuffle_cv_folds: bool = False,
            lm_sample_frac: float = 1.0,
            sample_inds: t.Optional[np.ndarray] = None,
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

        num_cv_folds : :obj: `int`, optional
            Number of num_cv_folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : :obj:`bool`, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.

        References
        ----------
        .. [1] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds)

        if skf is None:
            skf = StratifiedKFold(
                n_splits=num_cv_folds,
                shuffle=shuffle_cv_folds,
                random_state=random_state if shuffle_cv_folds else None)

        result = np.zeros(skf.n_splits, dtype=float)
        for fold_ind, split_inds in enumerate(skf.split(N, y)):
            train_inds, test_inds = split_inds
            importance = MFELandmarking._importance(
                N=N[train_inds], y=y[train_inds], random_state=random_state)

            model = DecisionTreeClassifier(
                max_depth=1, random_state=random_state)
            X_train = N[train_inds, :][:, [importance[0]]]
            X_test = N[test_inds, :][:, [importance[0]]]
            y_train, y_test = y[train_inds], y[test_inds]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result[fold_ind] = score(y_test, pred)

        return result

    @classmethod
    def ft_linear_discr(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            num_cv_folds: int = 10,
            shuffle_cv_folds: bool = False,
            lm_sample_frac: float = 1.0,
            sample_inds: t.Optional[np.ndarray] = None,
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

        num_cv_folds : :obj: `int`, optional
            Number of num_cv_folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : :obj:`bool`, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.

        References
        ----------
        .. [1] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds)

        if skf is None:
            skf = StratifiedKFold(
                n_splits=num_cv_folds,
                shuffle=shuffle_cv_folds,
                random_state=random_state if shuffle_cv_folds else None)

        result = np.zeros(skf.n_splits, dtype=float)
        for fold_ind, split_inds in enumerate(skf.split(N, y)):
            train_inds, test_inds = split_inds
            model = LinearDiscriminantAnalysis()
            X_train = N[train_inds, :]
            X_test = N[test_inds, :]
            y_train, y_test = y[train_inds], y[test_inds]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result[fold_ind] = score(y_test, pred)

        return result

    @classmethod
    def ft_naive_bayes(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            num_cv_folds: int = 10,
            shuffle_cv_folds: bool = False,
            lm_sample_frac: float = 1.0,
            sample_inds: t.Optional[np.ndarray] = None,
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

        num_cv_folds : :obj: `int`, optional
            Number of num_cv_folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : :obj:`bool`, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.

        References
        ----------
        .. [1] Johannes Furnkranz and Johann Petrak. An evaluation of
           landmarking variants. In 1st ECML/PKDD International Workshop
           on Integration and Collaboration Aspects of Data Mining,
           Decision Support and Meta-Learning (IDDM), pages 57 – 68, 2001.
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds)

        if skf is None:
            skf = StratifiedKFold(
                n_splits=num_cv_folds,
                shuffle=shuffle_cv_folds,
                random_state=random_state if shuffle_cv_folds else None)

        result = np.zeros(skf.n_splits, dtype=float)
        for fold_ind, split_inds in enumerate(skf.split(N, y)):
            train_inds, test_inds = split_inds
            model = GaussianNB()
            X_train = N[train_inds, :]
            X_test = N[test_inds, :]
            y_train, y_test = y[train_inds], y[test_inds]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result[fold_ind] = score(y_test, pred)

        return result

    @classmethod
    def ft_one_nn(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            num_cv_folds: int = 10,
            shuffle_cv_folds: bool = False,
            lm_sample_frac: float = 1.0,
            sample_inds: t.Optional[np.ndarray] = None,
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

        num_cv_folds : :obj: `int`, optional
            Number of num_cv_folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : :obj:`bool`, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.

        References
        ----------
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds)

        if skf is None:
            skf = StratifiedKFold(
                n_splits=num_cv_folds,
                shuffle=shuffle_cv_folds,
                random_state=random_state if shuffle_cv_folds else None)

        result = np.zeros(skf.n_splits, dtype=float)
        for fold_ind, split_inds in enumerate(skf.split(N, y)):
            train_inds, test_inds = split_inds
            model = KNeighborsClassifier(
                n_neighbors=1,
                algorithm="auto",
                weights="uniform",
                p=2,
                metric="minkowski")
            X_train = N[train_inds, :]
            X_test = N[test_inds, :]
            y_train, y_test = y[train_inds], y[test_inds]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result[fold_ind] = score(y_test, pred)

        return result

    @classmethod
    def ft_elite_nn(
            cls,
            N: np.ndarray,
            y: np.ndarray,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
            skf: t.Optional[StratifiedKFold] = None,
            num_cv_folds: int = 10,
            shuffle_cv_folds: bool = False,
            lm_sample_frac: float = 1.0,
            sample_inds: t.Optional[np.ndarray] = None,
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

        num_cv_folds : :obj: `int`, optional
            Number of num_cv_folds to k-fold cross validation. Used only if
            ``skf`` is None.

        shuffle_cv_folds : :obj:`bool`, optional
            If True, shuffle the data before splitting into the k-fold cross
            validation folds. The random seed used for this process is the
            ``random_state`` argument.

        lm_sample_frac : :obj:`float`, optional
            Proportion of instances to be sampled before extracting the
            metafeature. Used only if ``sample_inds`` is None.

        sample_inds : :obj:`np.ndarray`, optional
            Array of indices of instances to be effectively used while
            extracting this metafeature. If None, then ``lm_sample_frac``
            is taken into account. Argument used to exploit precomputations.

        random_state : :obj`int`, optional
            If int, random_state is the seed used by the random number
            generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.

        References
        ----------
        """
        N, y = MFELandmarking._sample_data(
            N=N,
            y=y,
            lm_sample_frac=lm_sample_frac,
            random_state=random_state,
            sample_inds=sample_inds)

        if skf is None:
            skf = StratifiedKFold(
                n_splits=num_cv_folds,
                shuffle=shuffle_cv_folds,
                random_state=random_state if shuffle_cv_folds else None)

        result = np.zeros(skf.n_splits, dtype=float)
        for fold_ind, split_inds in enumerate(skf.split(N, y)):
            train_inds, test_inds = split_inds
            importance = MFELandmarking._importance(
                N=N[train_inds], y=y[train_inds], random_state=random_state)

            model = KNeighborsClassifier(n_neighbors=1)
            X_train = N[train_inds, :][:, [importance[-1]]]
            X_test = N[test_inds, :][:, [importance[-1]]]
            y_train, y_test = y[train_inds], y[test_inds]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result[fold_ind] = score(y_test, pred)

        return result
