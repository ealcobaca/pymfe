"""Module dedicated to extraction of Landmarking Metafeatures.
"""

import typing as t
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


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
            groups: t.Tuple["str", ...],
            **kwargs
    ) -> t.Optional[t.Tuple[t.List[str], t.List[float], t.List[float]]]:
        """Structure to implement relative landmarking metafeatures."""

        if "relative" not in groups:
            return None

        mtf_rel_names = []
        mtf_rel_vals = []
        mtf_rel_time = []

        # To be implemented soon.

        return mtf_rel_names, mtf_rel_vals, mtf_rel_time

    @classmethod
    def precompute_landmarking_class(cls, N: np.ndarray, y: np.ndarray,
                                     folds: int, random_state: t.Optional[int],
                                     **kwargs) -> t.Dict[str, t.Any]:
        """Precompute k-fold cross validation strategy.

        Parameters
        ----------
        N : :obj:`np.ndarray`, optional
            Attributes from fitted data.

        y : :obj:`np.ndarray`, optional
            Target attribute from fitted data.

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

        if N is not None and y is not None\
           and not {"skf"}.issubset(kwargs):
            skf = StratifiedKFold(n_splits=folds, random_state=random_state)
            prepcomp_vals["skf"] = skf

        return prepcomp_vals

    @classmethod
    def importance(cls, N: np.ndarray, y: np.ndarray,
                   random_state: t.Optional[int]) -> np.ndarray:
        """Compute de gini index importance of a decision tree building using
        ``X`` and ``y``. We use sklearn ``DecisionTreeClassifier`` implementa-
        tion.

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

        Returns
        -------
        :obj:`np.ndarray`
            Return the decision tree features importance.
        """

        clf = DecisionTreeClassifier(random_state=random_state).fit(N, y)
        return np.argsort(clf.feature_importances_)

    @classmethod
    def ft_best_node(cls, N: np.ndarray, y: np.ndarray, skf: StratifiedKFold,
                     score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
                     random_state: t.Optional[int]) -> np.ndarray:
        """Construct a single decision tree node model induced by the most
        informative attribute to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`, optional
            Attributes from fitted data.

        y :obj:`np.ndarray`, optional
            Target attribute from fitted data.

        skf :obj:`StratifiedKFold`
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

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
            The performance of each fold.
        """
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
    def ft_random_node(cls, N: np.ndarray, y: np.ndarray, skf: StratifiedKFold,
                       score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
                       random_state: t.Optional[int]) -> np.ndarray:
        """Construct a single decision tree node model induced by a random
        attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        skf : :obj:`StratifiedKFold`
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        random_state : :obj`int`, optional
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
            The performance of each fold.
        """
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
    def ft_worst_node(cls, N: np.ndarray, y: np.ndarray, skf: StratifiedKFold,
                      score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
                      random_state: t.Optional[int]) -> np.ndarray:
        """Construct a single decision tree node model induced by the worst
        informative attribute.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        skf : :obj:`StratifiedKFold`
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        score : obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        random_state : :obj`int`, optional
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
            The performance of each fold.
        """
        result = []
        for train_index, test_index in skf.split(N, y):
            importance = MFELandmarking.importance(
                N[train_index], y[train_index], random_state)
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
    def ft_linear_discr(cls, N: np.ndarray, y: np.ndarray,
                        skf: StratifiedKFold,
                        score: t.Callable[[np.ndarray, np.ndarray], np.ndarray]
                        ) -> np.ndarray:
        """Apply the Linear Discriminant classifier to construct a linear split
        (non parallel axis) in the data to establish the linear separability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        skf : :obj:`StratifiedKFold`
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """

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
            skf: StratifiedKFold,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Evaluate the performance of the Naive Bayes classifier. It assumes
        that the attributes are independent and each example belongs to a cer-
        tain class based on the Bayes probability.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        skf : :obj:`StratifiedKFold`
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """

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
            skf: StratifiedKFold,
            score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Evaluate the performance of the 1-nearest neighbor classifier. It
        uses the euclidean distance of the nearest neighbor to determine how
        noisy is the data.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y :obj:`np.ndarray`
            Target attribute from fitted data.

        skf :obj:`StratifiedKFold`
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        kwargs:
            Additional arguments. May have previously precomputed before this
            method from other precomputed methods, so they can help speed up
            this precomputation.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """

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
    def ft_elite_nn(cls, N: np.ndarray, y: np.ndarray, skf: StratifiedKFold,
                    score: t.Callable[[np.ndarray, np.ndarray], np.ndarray],
                    random_state: t.Optional[int]) -> np.ndarray:
        """Elite nearest neighbor uses the most informative attribute in the
        dataset to induce the 1-nearest neighbor. With the subset of informati-
        ve attributes is expected that the models should be noise tolerant.

        Parameters
        ----------
        N : :obj:`np.ndarray`
            Attributes from fitted data.

        y : :obj:`np.ndarray`
            Target attribute from fitted data.

        skf : :obj:`StratifiedKFold`
            Stratified K-Folds cross-validator. Provides train/test indices to
            split data in train/test sets.

        score : :obj:`callable`
            Function to compute score of the K-fold evaluations. Possible
            functions are described in `scoring.py` module.

        random_state : :obj:`int`, optional
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

            kwargs:
                Additional arguments. May have previously precomputed before
                this method from other precomputed methods, so they can help
                speed up this precomputation.

        Returns
        -------
        :obj:`np.ndarray`
            The performance of each fold.
        """
        result = []
        for train_index, test_index in skf.split(N, y):
            importance = MFELandmarking.importance(
                N[train_index], y[train_index], random_state)
            model = KNeighborsClassifier(n_neighbors=1)
            X_train = N[train_index, :][:, [importance[-1]]]
            X_test = N[test_index, :][:, [importance[-1]]]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result.append(score(y_test, pred))

        return np.array(result)
