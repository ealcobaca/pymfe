"""Main module for extracting metafeatures from datasets.
"""
import typing as t
import collections
import shutil
import time

import texttable
import numpy as np

import pymfe._internal as _internal

_TypeSeqExt = t.Sequence[t.Tuple[str, t.Callable, t.Tuple[str, ...],
                                 t.Tuple[str, ...]]]
"""Type annotation for a sequence of TypeExtMtdTuple objects."""


class MFE:
    """Core class for metafeature extraction.

    Attributes
    ----------
    X : :obj:`Sequence`
        Independent attributes of the dataset.

    y : :obj:`Sequence`
        Target attributes of the dataset.

    groups : :obj:`tuple` of :obj:`str`
        Tuple object containing fitted meta-feature groups loaded in the model
        at instantiation.

    features : :obj:`tuple` of :obj:`str`
        Contains loaded meta-feature extraction method names available for
        meta-feature extraction, from selected metafeatures groups and features
        listed at instantiation.

        summary : :obj:`tuple` of :obj:`str`
            Tuple object which contains summary functions names for features
            summarization.
    """
    groups_alias = [('default', _internal.DEFAULT_GROUP)]

    def __init__(self,
                 groups: t.Union[str, t.Iterable[str]] = "default",
                 features: t.Union[str, t.Iterable[str]] = "all",
                 summary: t.Union[str, t.Iterable[str]] = ("mean", "sd"),
                 measure_time: t.Optional[str] = None,
                 wildcard: str = "all",
                 score: str = "accuracy",
                 num_cv_folds: int = 10,
                 shuffle_cv_folds: bool = False,
                 lm_sample_frac: float = 1.0,
                 hypparam_model_dt: t.Optional[t.Dict[str, t.Any]] = None,
                 suppress_warnings: bool = False,
                 random_state: t.Optional[int] = None) -> None:
        """Provides easy access for metafeature extraction from datasets.

        It expected that user first calls ``fit`` method after instantiation
        and then ``extract`` for effectively extract the selected metafeatures.
        Check reference [1]_ for more information.

        Parameters
        ----------
        groups : :obj:`Iterable` of :obj:`str` or :obj:`str`
            A collection or a single metafeature group name representing the
            desired group of metafeatures for extraction. Use the method
            ``valid_groups`` to get a list of all available groups.

            Setting with ``all`` enables all available groups.

            Setting with ``default`` enables ``general``, ``info-theory``,
            ``statistical``, ``model-based`` and ``landmarking``. It is the
            default value.

            The value provided by the argument ``wildcard`` can be used to
            select all metafeature groups rapidly.

        features : :obj:`Iterable` of :obj:`str` or :obj:`str`, optional
            A collection or a single metafeature name desired for extraction.
            Keep in mind that the extraction only gathers features also in the
            selected ``groups``. Check this class ``features`` attribute to get
            a list of available metafeatures from selected groups, or use the
            method ``valid_metafeatures`` to get a list of all available
            metafeatures filtered by group. Alternatively, you can use the
            method ``metafeature_description`` to get or print a table with
            all metafeatures with its respectives groups and descriptions.

            The value provided by the argument ``wildcard`` can be used to
            select all features from all selected groups rapidly.

        summary : :obj:`Iterable` of :obj:`str` or :obj:`str`, optional
            A collection or a single summary function to summarize a group of
            metafeature measures into a fixed-length group of value, typically
            a single value. The values must be one of the following:

                1. ``mean``: Average of the values.
                2. ``sd``: Standard deviation of the values.
                3. ``count``: Computes the cardinality of the measure. Suitable
                   for variable cardinality.
                4. ``histogram``: Describes the distribution of the measured
                   values. Suitable for high cardinality.
                5. ``iq_range``: Computes the interquartile range of the
                   measured values.
                6. ``kurtosis``: Describes the shape of the measures values
                   distribution.
                7. ``max``: Results in the maximum value of the measure.
                8. ``median``: Results in the central value of the measure.
                9. ``min``: Results in the minimum value of the measure.
                10. ``quantiles``: Results in the minimum, first quartile,
                    median, third quartile and maximum of the measured values.
                11. ``range``: Computes the range of the measured values.
                12. ``skewness``: Describes the shape of the measure values
                    distribution in terms of symmetry.

            You can concatenate `nan` with the desired summary function name
            to use an alternative version of the same summary which ignores
            `nan` values. For instance, `nanmean` is the `mean` summary
            function which ignores all `nan` values, while 'naniq_range`
            is the interquartile range calculated only with valid (non-`nan`)
            values.

            If more than one summary function is selected, then all multivalued
            extracted metafeatures are summarized with each summary function.

            The particular value provided by the argument ``wildcard`` can be
            used to select all summary functions rapidly.

            Use the method ``valid_summary`` to get a list of all available
            summary functions.

        measure_time : :obj:`str`, optional
            Options for measuring the time elapsed during metafeature
            extraction. If this argument value is :obj:`NoneType`, no time
            elapsed is measured. Otherwise, this argument must be a :obj:`str`
            valued as one of the options below:

                1. ``avg``: average time for each metafeature (total time
                   divided by the feature cardinality, i.e., number of features
                   extracted by a single feature-extraction related method),
                   without summarization time.
                2. ``avg_summ``: average time for each metafeature (total time
                   of extraction divided by feature cardinality) including
                   required time for summarization.
                3. ``total``: total time for each metafeature, without
                   summarization time.
                4. ``total_summ``: total time for each metafeature including
                   the required time for summarization.

            The ``cardinality`` of the feature is the number of values
            extracted by a single calculation method.

            For example, ``mean`` feature has cardinality equal to the number
            of numeric features in the dataset, where ``cor`` (from
            ``correlation``) has cardinality equals to (N - 1)/2, where N is
            the number of numeric features in the dataset.

            The cardinality is used to divide the total execution time of that
            method if an option starting with ``avg`` is selected.

            If a summary method has cardinality higher than one (more than one
            value returned after summarization and, thus, creating more than
            one entry in the result lists) like, for example, ``histogram``
            summary method, then the corresponding time of this summary will be
            inserted only in the first correspondent element of the time list.
            The remaining entries are all filled with 0 value, to keep
            consistency between the size of all lists returned and index
            correspondence between they.

        wildcard : :obj:`str`, optional
            Value used as ``select all`` for ``groups``, ``features`` and
            ``summary`` arguments.

        score : :obj:`str`, optional
            Score metric used to extract ``landmarking`` metafeatures.

        num_cv_folds : :obj:`int`, optional
            Number of folds to create a Stratified K-Fold cross
            validation to extract the ``landmarking`` metafeatures.

        shuffle_cv_folds : :obj:`bool`, optional
            If True, then the fitted data will be shuffled before splitted in
            the Stratified K-Fold Cross Validation of ``landmarking`` features.
            The shuffle random seed is the ``random_state`` argument.

        lm_sample_frac : :obj:`float`, optional
            Sample proportion used to produce the ``landmarking`` metafeatures.
            This argument must be in 0.5 and 1.0 (both inclusive) interval.

        hypparam_model_dt : :obj:`dict`, optional
            Dictionary providing extra hyperparameters for the Decision Tree
            algorithm for building the Decision Tree model, used to extract the
            model-based metafeatures. The class used to fit the model is the
            ``sklearn.tree.DecisionTreeClassifier`` (sklearn library). Using
            this argument, it is possible to provide extra arguments in the
            DecisionTreeClassifier class initialization (e.g., ``max_depth``
            and ``min_samples_split``.) In order to use this argument, provide
            the DecisionTreeClassifier init argument name as the dictionary
            keys and the corresponding custom values, as the dictionary values.
            Example:
            {"min_samples_split": 10, "criterion": "entropy"}

        suppress_warnings : :obj:`bool`, optional
            If True, then ignore all warnings invoked at the instantiation
            time.

        random_state : :obj:`int`, optional
            Random seed used to control random events. Keeps the experiments
            reproducible.

        Notes
        -----
            .. [1] Rivolli et al. "Towards Reproducible Empirical
               Research in Meta-Learning,".
               Rivolli et al. URL: https://arxiv.org/abs/1808.10406

        Examples
        --------

        Load a dataset

        >>> from sklearn.datasets import load_iris
        >>> from pymfe.mfe import MFE

        >>> data = load_iris()
        >>> y = data.target
        >>> X = data.data

        Extract all measures

        >>> mfe = MFE()
        >>> mfe.fit(X, y)
        >>> ft = mfe.extract()
        >>> print(ft)

        Extract general, statistical and information-theoretic measures

        >>> mfe = MFE(groups=["general", "statistical", "info-theory"])
        >>> mfe.fit(X, y)
        >>> ft = mfe.extract()
        >>> print(ft)

        """
        self.groups = _internal.process_generic_set(
            values=groups, group_name="groups",
            groups_alias=MFE.groups_alias,
            wildcard=wildcard)  # type: t.Tuple[str, ...]

        self.groups, self.inserted_group_dep = (
            _internal.solve_group_dependencies(
                groups=self.groups))

        proc_feat = _internal.process_features(
            features=features,
            groups=self.groups,
            suppress_warnings=suppress_warnings,
            wildcard=wildcard,
        )  # type: t.Tuple[t.Tuple[str, ...], _TypeSeqExt, t.Tuple[str, ...]]

        self.features, self._metadata_mtd_ft, self.groups = proc_feat
        del proc_feat

        self.summary, self._metadata_mtd_sm = _internal.process_summary(
            summary,
            wildcard=wildcard)  # type: t.Tuple[t.Tuple[str, ...], _TypeSeqExt]

        self.timeopt = _internal.process_generic_option(
            value=measure_time, group_name="timeopt",
            allow_none=True)  # type: t.Optional[str]

        self.X = None  # type: t.Optional[np.ndarray]
        self.y = None  # type: t.Optional[np.ndarray]

        self._custom_args_ft = None  # type: t.Optional[t.Dict[str, t.Any]]
        """User-independent arguments for ft. methods (e.g. ``X`` and ``y``)"""

        self._custom_args_sum = None  # type: t.Optional[t.Dict[str, t.Any]]
        """User-independent arguments for summary functions methods."""

        self._attr_indexes_num = None  # type: t.Optional[t.Tuple[int, ...]]
        """Numeric column indexes from ``X`` (independent attributes)."""

        self._attr_indexes_cat = None  # type: t.Optional[t.Tuple[int, ...]]
        """Categoric column indexes from ``X`` (independent attributes)."""

        self._precomp_args_ft = None  # type: t.Optional[t.Dict[str, t.Any]]
        """Precomputed common feature-extraction method arguments."""

        self._postprocess_args_ft = {}  # type: t.Dict[str, t.Any]
        """User-independent arguments for post-processing methods."""

        if random_state is None or isinstance(random_state, int):
            self.random_state = random_state
            np.random.seed(random_state)

        else:
            raise ValueError(
                'Invalid "random_state" argument ({0}). '
                'Expecting None or an integer.'.format(random_state))

        self.shuffle_cv_folds = shuffle_cv_folds

        if isinstance(num_cv_folds, int):
            self.num_cv_folds = num_cv_folds

        else:
            raise ValueError('Invalid "num_cv_folds" argument ({0}). '
                             'Expecting an integer.'.format(random_state))

        if isinstance(lm_sample_frac, int):
            lm_sample_frac = float(lm_sample_frac)

        if isinstance(lm_sample_frac, float)\
           and 0.5 <= lm_sample_frac <= 1.0:
            self.lm_sample_frac = lm_sample_frac

        else:
            raise ValueError('Invalid "lm_sample_frac" argument ({0}). '
                             'Expecting an float [0.5, 1].'
                             .format(random_state))

        self.score = _internal.check_score(score, self.groups)
        self.hypparam_model_dt = (hypparam_model_dt.copy()
                                  if hypparam_model_dt else None)

        self.time_precomp = -1.0
        """Total time elapsed for precomputations."""

        self.time_extract = -1.0
        """Total time elapsed for metafeature extraction."""

        self.time_total = -1.0
        """Total time elapsed in total (precomp + extract.)"""

    def _call_summary_methods(
            self,
            feature_values: t.Sequence[_internal.TypeNumeric],
            feature_name: str,
            remove_nan: bool = True,
            verbose: int = 0,
            suppress_warnings: bool = False,
            **kwargs
    ) -> t.Tuple[t.List[str], t.List[t.Union[float, t.Sequence]], t.
                 List[float]]:
        """Invoke summary functions loaded in the model on given feature
        values.

        Parameters
        ----------
        feature_values : :obj:`sequence` of numerics
            Sequence containing values from feature-extraction methods.

        feature_name : :obj:`str`
            Name of the feature method used for produce the ``feature_value.``

        remove_nan : :obj:`bool`, optional
            If True, all non-numeric values are removed from ``feature_values``
            before calling each summary method. Note that the summary method
            itself may still remove non-numeric values and, in this case, the
            user must suppress these warnings using some built-in argument of
            the summary method using the kwargs argument, if possible.

        verbose : :obj:`int`, optional
            Select the verbosity level of the summarization process.
            If == 1, then print just the ending message, without a line break.
            If >= 2, then messages about the summarization process may be
            printed. Note that there is no relation between this argument and
            warnings (see ``suppress_warnings`` argument below).

        suppress_warnings : :obj:`bool`, optional
            If True, ignore all warnings invoked before and after summary
            method calls. Note that, as seen in the ``remove_nan`` argument,
            the summary callables may still invoke warnings by itself and the
            user need to ignore them, if possible, via kwargs.

        kwargs:
            User-defined arguments for the summary callables.

        Returns
        -------
        :obj:`tuple`(:obj:`list`, :obj:`list`, :obj:`list`)
            A tuple containing three lists.

            The first field is the identifiers of each summarized value in the
            form ``feature_name.summary_mtd_name`` (i.e., the feature
            extraction name concatenated by the summary method name, separated
            by a dot). If the summary function return more than one value
            (cardinality greater than 1), then each value name have an extra
            concatenated id starting from 0 to differ between values (i.e.
            ``feature_name.summary_mtd_name.id``).

            The second field is the summarized values. Both lists have a 1-1
            correspondence by the index of each element (i.e., the value at
            index ``i`` in the second list has its identifier at the same index
            in the first list and vice-versa).

            The third field is a list with measured time wasted by each summary
            function. If the cardinality of the summary function is greater
            than 1, then the correspondent measured time is kept only in the
            first correspondent field, and the extra fields are filled with 0
            to keep the consistency of the size between all lists.

                Example:
                    ([``attr_ent.mean``, ``attr_ent.sd``], [0.98346, 0.34436])
                    is the return value for the feature `attr_end` summarized
                    by both ``mean`` and ``sd`` (standard deviation), giving
                    the values ``0.98347`` and ``0.34436``, respectively.
        """
        metafeat_vals = []  # type: t.List[t.Union[int, float, t.Sequence]]
        metafeat_names = []  # type: t.List[str]
        metafeat_times = []  # type: t.List[float]

        for cur_metadata in self._metadata_mtd_sm:
            sm_mtd_name, sm_mtd_callable, sm_mtd_args, _ = cur_metadata

            if verbose >= 2:
                print(
                    " {} Summarizing '{}' feature with '{}' summary "
                    "function...".format(
                        _internal.VERBOSE_BLOCK_MID_SYMBOL,
                        feature_name,
                        sm_mtd_name),
                    end=" ")

            sm_mtd_args_pack = _internal.build_mtd_kwargs(
                mtd_name=sm_mtd_name,
                mtd_args=sm_mtd_args,
                mtd_mandatory=set(),
                user_custom_args=kwargs.get(sm_mtd_name),
                inner_custom_args=self._custom_args_sum,
                suppress_warnings=suppress_warnings)

            summarized_val, time_sm = _internal.timeit(
                _internal.summarize, feature_values, sm_mtd_callable,
                sm_mtd_args_pack, remove_nan)

            if not suppress_warnings:
                _internal.check_summary_warnings(
                    value=summarized_val,
                    name_feature=feature_name,
                    name_summary=sm_mtd_name)

            if isinstance(summarized_val, np.ndarray):
                summarized_val = summarized_val.flatten().tolist()

            if (isinstance(summarized_val, collections.Sequence)
                    and not isinstance(summarized_val, str)):
                metafeat_vals += summarized_val
                metafeat_names += [
                    ".".join((feature_name, sm_mtd_name, str(i)))
                    for i in range(len(summarized_val))
                ]
                metafeat_times += ([time_sm] + (
                    (len(summarized_val) - 1) * [0.0]))

            else:
                metafeat_vals.append(summarized_val)
                metafeat_names.append(".".join((feature_name, sm_mtd_name)))
                metafeat_times.append(time_sm)

            if verbose >= 2:
                print("Done.")

        if verbose >= 2:
            print(" {} Done summarizing '{}' feature.".format(
                _internal.VERBOSE_BLOCK_END_SYMBOL,
                feature_name))

        return metafeat_names, metafeat_vals, metafeat_times

    def _call_feature_methods(
            self,
            remove_nan: bool = True,
            verbose: int = 0,
            # enable_parallel: bool = False,
            suppress_warnings: bool = False,
            **kwargs) -> t.Tuple[t.List[str],
                                 t.List[t.Union[int, float, t.Sequence]],
                                 t.List[float]]:
        """Invoke feature methods loaded in the model and gather results.

        The returned values are already summarized if needed.

        For more information, check ``extract`` method documentation for
        in-depth information about arguments and return value.
        """
        metafeat_vals = []  # type: t.List[t.Union[int, float, t.Sequence]]
        metafeat_names = []  # type: t.List[str]
        metafeat_times = []  # type: t.List[float]

        skipped_count = 0
        for ind, cur_metadata in enumerate(self._metadata_mtd_ft, 1):
            (ft_mtd_name, ft_mtd_callable,
             ft_mtd_args, ft_mandatory) = cur_metadata

            ft_name_without_prefix = _internal.remove_prefix(
                value=ft_mtd_name, prefix=_internal.MTF_PREFIX)

            try:
                ft_mtd_args_pack = _internal.build_mtd_kwargs(
                    mtd_name=ft_name_without_prefix,
                    mtd_args=ft_mtd_args,
                    mtd_mandatory=ft_mandatory,
                    user_custom_args=kwargs.get(ft_name_without_prefix),
                    inner_custom_args=self._custom_args_ft,
                    precomp_args=self._precomp_args_ft,
                    suppress_warnings=suppress_warnings)

            except RuntimeError:
                # Not all method's mandatory arguments were satisfied.
                # Skip the current method.
                if verbose >= 2:
                    print("\nSkipped '{}' ({} of {}).".format(
                        ft_mtd_name, ind, len(self._metadata_mtd_ft)))

                skipped_count += 1
                continue

            if verbose >= 2:
                print("\nExtracting '{}' feature ({} of {})..."
                      .format(ft_mtd_name, ind, len(self._metadata_mtd_ft)))

            features, time_ft = _internal.timeit(
                _internal.get_feat_value, ft_mtd_name, ft_mtd_args_pack,
                ft_mtd_callable, suppress_warnings)

            ft_has_length = isinstance(features,
                                       (np.ndarray, collections.Sequence))

            if ft_has_length and self._timeopt_type_is_avg():
                time_ft /= len(features)

            if self._metadata_mtd_sm and ft_has_length:
                sm_ret = self._call_summary_methods(
                    feature_values=features,
                    feature_name=ft_name_without_prefix,
                    remove_nan=remove_nan,
                    verbose=verbose,
                    suppress_warnings=suppress_warnings,
                    **kwargs)

                summarized_names, summarized_vals, times_sm = sm_ret

                metafeat_vals += summarized_vals
                metafeat_names += summarized_names
                metafeat_times += self._combine_time(time_ft, times_sm)

            else:
                metafeat_vals.append(features)
                metafeat_names.append(ft_name_without_prefix)
                metafeat_times.append(time_ft)

            if verbose > 0:
                _internal.print_verbose_progress(
                    cur_progress=100 * ind / len(self._metadata_mtd_ft),
                    cur_mtf_name=ft_mtd_name,
                    item_type="feature",
                    verbose=verbose)

        if verbose == 1:
            _t_num_cols, _ = shutil.get_terminal_size()
            print("\r{:<{fill}}".format(
                "Process of metafeature extraction finished.",
                fill=_t_num_cols))

        if verbose >= 2 and skipped_count > 0:
            print("\nNote: skipped a total of {} metafeatures, "
                  "out of {} ({:.2f}%).".format(
                      skipped_count,
                      len(self._metadata_mtd_ft),
                      100 * skipped_count / len(self._metadata_mtd_ft)))

        return metafeat_names, metafeat_vals, metafeat_times

    def _fill_col_ind_by_type(
            self,
            cat_cols: t.Optional[t.Union[str, t.Iterable[int]]] = "auto",
            check_bool: bool = True) -> None:
        """Select ``X`` column indexes based in its data type.

        The indexes for numerical and categorical attributes are kept,
        respectively, at ``_attr_indexes_num`` and ``_attr_indexes_cat``
        instance attributes.

        Parameters
        ----------
        cat_cols : :obj:`str` or :obj:`iterable` of :obj:`int`, optional
            Iterable of indexes identifying categorical columns. If special
            keyword ``auto`` is given, then an automatic verification is done
            in the fitted attributes.

        check_bool : :obj:`bool`, optional
            Check ``fit`` method corresponding argument for more information.

        Raises
        ------
        TypeError
            If ``X`` attribute is :obj:`NoneType`.
        ValueError
            If ``cat_cols`` is neither ``auto`` or a valid integer iterable.
        """

        if self.X is None:
            raise TypeError("X can't be 'None'.")

        categorical_cols = None  # type: np.ndarray[bool]

        if not cat_cols:
            categorical_cols = np.array([False] * self.X.shape[1])

        elif isinstance(cat_cols, str) and cat_cols.lower() == "auto":
            categorical_cols = np.logical_not(
                np.apply_along_axis(
                    _internal.isnumeric,
                    axis=0,
                    arr=self.X,
                    check_subtype=True,
                ))

            if check_bool:
                categorical_cols |= np.apply_along_axis(
                    func1d=lambda col: len(np.unique(col)) == 2,
                    axis=0,
                    arr=self.X,
                )

        elif (isinstance(cat_cols, (np.ndarray, collections.Iterable))
              and not isinstance(cat_cols, str)):
            # and all(isinstance(x, int) for x in cat_cols)):
            categorical_cols = [i in cat_cols for i in range(self.X.shape[1])]

        else:
            raise ValueError(
                'Invalid "cat_cols" argument ({0}). '
                'Expecting "auto" or an integer Iterable.'.format(cat_cols))

        categorical_cols = np.array(categorical_cols)

        self._attr_indexes_num = tuple(
            np.where(np.logical_not(categorical_cols))[0])
        self._attr_indexes_cat = tuple(np.where(categorical_cols)[0])

    def _timeopt_type_is_avg(self) -> bool:
        """Checks if user selected time option is an ``average`` type."""
        return (isinstance(self.timeopt, str)
                and self.timeopt.startswith(_internal.TIMEOPT_AVG_PREFIX))

    def _timeopt_include_summary(self) -> bool:
        """Checks if user selected time option includes ``summary`` time."""
        return (isinstance(self.timeopt, str)
                and self.timeopt.endswith(_internal.TIMEOPT_SUMMARY_SUFFIX))

    def _combine_time(self, time_ft: float,
                      times_sm: t.List[float]) -> t.List[float]:
        """Treat time from feature extraction and summarization based in
        ``timeopt``.

        Parameters
        ----------
        time_ft : :obj:`float`
            Time necessary to extract some feature.

        times_sm : :obj:`list` of :obj:`float`
            List of values to summarize the metafeature value with each summary
            function.

        Returns
        -------
        :obj:`list`
            If ``timeopt`` attribute considers ``summary`` time (i.e., selected
            option ends with ``summ``), then these returned list values are the
            combination of times gathered in feature extraction and
            summarization methods. Otherwise, the list values are the value of
            ``time_ft`` copied ``len(times_sm)`` times, to keep consistency
            with the correspondence between the values of all lists returned by
            ``extract`` method.
        """
        total_time = np.array([time_ft] * len(times_sm))

        if self._timeopt_include_summary():
            total_time += times_sm

        # As seen in ``_call_summary_methods`` method documentation, zero-
        # valued elements are created to fill the time list to keep its size
        # consistent with another feature extraction related lists. In this
        # case, here they're kept zero-valued.
        total_time[np.array(times_sm) == 0.0] = 0.0

        return total_time.tolist()

    def _set_data_categoric(self, transform_num: bool,
                            num_bins: bool = None) -> np.ndarray:
        """Returns categorical data from the fitted dataset.

        Parameters
        ----------
        transform_num : :obj:`bool`
            If True, then all numeric-type data are discretized using an
            equal-frequency histogram. Otherwise, this method ignores these
            attributes.

        num_bins : :obj:`bool`, optional
            Number of bins of the discretization histogram. This argument is
            used only if ``transform_num`` is True. If this argument value is
            :obj:`NoneType`, then it is set to min(2, c), where ``c`` is the
            cubic root of the number of instances of the fitted dataset.

        Returns
        -------
        :obj:`np.ndarray`
            Processed categorical data. If no need for changes from the
            original dataset, then this method does not create a copy of it to
            prevent unnecessary memory usage. Otherwise, this method returns a
            modified version of the original categorical data, thus consuming
            more memory.

        Raises
        ------
        TypeError:
            If either ``X`` or ``_attr_indexes_cat`` instance attributes are
            :obj:`NoneType`. This can be avoided passing valid data to fit and
            first calling ``_fill_col_ind_by_type`` instance method before this
            method.
        """
        if self.X is None:
            raise TypeError("It is necessary to fit valid data into the "
                            'model before setting up categoric data. ("X" '
                            'attribute is "NoneType").')

        if self._attr_indexes_cat is None:
            raise TypeError("No information about indexes of categoric "
                            "attributes. Please be sure to call method "
                            '"_fill_col_ind_by_type" before this method.')

        data_cat = self.X[:, self._attr_indexes_cat]

        if transform_num:
            data_num_discretized = _internal.transform_num(
                self.X[:, self._attr_indexes_num], num_bins=num_bins)

            if data_num_discretized is not None:
                data_cat = np.concatenate((data_cat, data_num_discretized),
                                          axis=1)

        return data_cat

    def _set_data_numeric(
            self,
            transform_cat: str,
            rescale: t.Optional[str] = None,
            rescale_args: t.Optional[t.Dict[str, t.Any]] = None) -> np.ndarray:
        """Returns numeric data from the fitted dataset.

        Parameters
        ----------
        transform_cat: :obj:`bool`
            If `gray`, then all categoric-type data will be binarized with a
            model matrix strategy. If `one-hot`, then all categoric-type
            data will be transformed using the one-hot encoding strategy.
            If None, then categorical attributes are not transformed.

        rescale : :obj:`str`, optional
            Check ``fit`` documentation for more information about this
            parameter.

        rescale_args : :obj:`dict`, optional
            Check ``fit`` documentation for more information about this
            parameter.

        Returns
        -------
        :obj:`np.ndarray`
            Processed numerical data. If no need for changes from the original
            dataset, then this method does not create a copy of it to prevent
            unnecessary memory usage. Otherwise, this method returns a modified
            version of the original numerical data, thus consuming more memory.

        Raises
        ------
        TypeError
            If ``X`` or ``_attr_indexes_num`` instance attributes are
            :obj:`NoneType`. This can be avoided passing valid data to fit and
            first calling ``_fill_col_ind_by_type`` instance method before
            this method.

        ValueError
            If `transform_cat` is neither None nor a value among `one-hot` and
            `gray`.
        """
        if self.X is None:
            raise TypeError("It is necessary to fit valid data into the "
                            'model before setting up numeric data. ("X" '
                            'attribute is "NoneType").')

        if self._attr_indexes_num is None:
            raise TypeError("No information about indexes of numeric "
                            "attributes. Please be sure to call method "
                            '"_fill_col_ind_by_type" before this method.')

        if (transform_cat is not None and
                transform_cat not in _internal.VALID_TRANSFORM_CAT):
            raise ValueError("Invalid 'transform_cat' value ('{}'). Must be "
                             "a value in {}.".format(
                                 transform_cat, _internal.VALID_TRANSFORM_CAT))

        data_num = self.X[:, self._attr_indexes_num]

        if transform_cat:
            if transform_cat == "gray":
                categorical_dummies = _internal.transform_cat_gray(
                    self.X[:, self._attr_indexes_cat])

            else:
                categorical_dummies = _internal.transform_cat_onehot(
                    self.X[:, self._attr_indexes_cat])

            if categorical_dummies is not None:
                data_num = np.concatenate((data_num, categorical_dummies),
                                          axis=1).astype(float)

        if rescale:
            data_num = _internal.rescale_data(
                data=data_num, option=rescale, args=rescale_args)

        if data_num.dtype != float:
            data_num = data_num.astype(float)

        return data_num

    def fit(self,
            X: t.Sequence,
            y: t.Optional[t.Sequence] = None,
            transform_num: bool = True,
            transform_cat: str = "gray",
            rescale: t.Optional[str] = None,
            rescale_args: t.Optional[t.Dict[str, t.Any]] = None,
            cat_cols: t.Optional[t.Union[str, t.Iterable[int]]] = "auto",
            check_bool: bool = False,
            precomp_groups: t.Optional[str] = "all",
            wildcard: str = "all",
            suppress_warnings: bool = False,
            verbose: int = 0,
            ) -> "MFE":
        """Fits dataset into an MFE model.

        Parameters
        ----------
        X : :obj:`Sequence`
            Predictive attributes of the dataset.

        y : :obj:`Sequence`, optional
            Target attributes of the dataset, assuming that it is a supervised
            task.

        transform_num : :obj:`bool`, optional
            If True, numeric attributes are discretized using equal-frequency
            histogram technique to use alongside categorical data when
            extracting categoric-only metafeatures. Note that numeric-only
            features still uses the original numeric values, not the
            discretized ones. If False, then numeric attributes are ignored for
            categorical-only meta-features.

        transform_cat : :obj:`str`, optional
            Transform categorical data to use alongside numerical data while
            extracting numeric-only metafeatures. Note that categoric-only
            features still uses the original categoric values, and not the
            binarized ones.

            If `one-hot`, categorical attributes are binarized using one-hot
            encoding.

            If `gray`, categorical attributes are binarized using a model
            matrix.

            The formula used for this transformation is just the union (+) of
            all categoric attributes using formula language from ``patsy``
            package API, removing the intercept terms:
            ``~ 0 + A_1 + ... + A_n``, where ``n`` is the number of attributes
            and A_i is the ith categoric attribute, 1 <= i <= n.

            If None, then categorical attributes are not transformed.

        rescale : :obj:`str`, optional
            If :obj:`NoneType`, the model keeps all numeric data with its
            original values. Otherwise, this argument can assume one of the
            string options below to rescale all numeric values:

                1. ``standard``: set numeric data to zero mean, unit variance.
                   Also known as ``z-score`` normalization. Check the
                   documentation of ``sklearn.preprocessing.StandardScaler``
                   for in-depth information.

                2. `'min-max``: set numeric data to interval [a, b], a < b. It
                   is possible to define values to ``a`` and ``b`` using
                   argument ``rescale_args``. The default values are a = 0.0
                   and b = 1.0. Check ``sklearn.preprocessing.MinMaxScaler``
                   documentation for more information.

                3. ``robust``: rescale data using statistics robust to the
                   presence of outliers. For in-depth information, check
                   documentation of ``sklearn.preprocessing.RobustScaler``.

        rescale_args : :obj:`dict`, optional
            Dictionary containing parameters for rescaling data. Used only if
            ``rescale`` argument is not :obj:`NoneType`. These dictionary keys
            are the parameter names as strings and the values, the
            corresponding parameter value.

        cat_cols :obj:`Sequence` of :obj:`int` or :obj:`str`, optional
            Categorical columns of dataset. If given :obj:`NoneType` or an
            empty sequence, assume all columns as numeric. If given value
            ``auto``, then an attempt of automatic detection is performed while
            fitting the dataset.

        check_bool : :obj:`bool`, optional
            If `cat_cols` is ``auto``, and this flag is True, assume that all
            columns with precisely two different values is also a categorical
            (boolean) column, independently of its data type. Otherwise, these
            columns may be considered numeric depending on their data type.

        missing_data : :obj:`str`, optional
            Defines the strategy to handle missing values in data. Still not
            implemented.

        precomp_groups : :obj:`str`, optional
            Defines which metafeature groups common values should be cached to
            share among various meta-feature extraction related methods (e.g.
            ``classes``, or ``covariance``). This argument may speed up
            meta-feature extraction but also consumes more memory, so it may
            not be suitable for huge datasets.

        wildcard : :obj:`str`, optional
            Value used as ``select all`` for ``precomp_groups``.

        suppress_warnings : :obj:`bool`, optional
            If True, ignore all warnings invoked while fitting dataset.

        verbose : :obj:`int`, optional
            Defines the level of verbosity for the fit method. If `1`, then
            print a progress bar related to the precomputations. If `2` or
            higher, then log every step of the fitted data transformations and
            the precomputation steps.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of rows of X and y length does not match.
        TypeError
            If X or y (or both) is neither a :obj:`list` or a :obj:`np.ndarray`
            object.

        """
        if verbose >= 2:
            print("Fitting data into model... ", end="")

        self.X, self.y = _internal.check_data(X, y)

        if verbose >= 2:
            print("Done.")

        rescale = _internal.process_generic_option(
            value=rescale, group_name="rescale", allow_none=True)

        self._fill_col_ind_by_type(cat_cols=cat_cols, check_bool=check_bool)

        if verbose >= 2:
            print("Started data transformation process.",
                  " {} Encoding numerical data into discrete values... "
                  .format(_internal.VERBOSE_BLOCK_END_SYMBOL),
                  sep="\n", end="")

        data_cat = self._set_data_categoric(transform_num=transform_num)

        if verbose >= 2:
            print("Done.",
                  " {} Enconding categorical data into numerical values... "
                  .format(_internal.VERBOSE_BLOCK_END_SYMBOL),
                  sep="\n", end="")

        data_num = self._set_data_numeric(
            transform_cat=transform_cat,
            rescale=rescale,
            rescale_args=rescale_args)

        if verbose >= 2:
            print("Done.",
                  "Finished data transformation process.",
                  sep="\n")

        # Custom arguments for metafeature extraction methods
        self._custom_args_ft = {
            "X": self.X,
            "N": data_num,
            "C": data_cat,
            "num_cv_folds": self.num_cv_folds,
            "shuffle_cv_folds": self.shuffle_cv_folds,
            "lm_sample_frac": self.lm_sample_frac,
            "score": self.score,
            "random_state": self.random_state,
            "cat_cols": self._attr_indexes_cat,
            "hypparam_model_dt": self.hypparam_model_dt,
        }

        if self.y is not None:
            self._custom_args_ft["y"] = self.y

        if verbose >= 2:
            print("Started precomputation process.")

        _time_start = time.time()

        # Custom arguments from preprocessing methods
        self._precomp_args_ft = _internal.process_precomp_groups(
            precomp_groups=precomp_groups,
            groups=self.groups,
            wildcard=wildcard,
            suppress_warnings=suppress_warnings,
            verbose=verbose,
            **self._custom_args_ft)

        self.time_precomp = time.time() - _time_start

        if verbose >= 2:
            print("\nFinished precomputation process.",
                  " {} Total time elapsed: {:.8f} seconds".format(
                      _internal.VERBOSE_BLOCK_MID_SYMBOL,
                      self.time_precomp),
                  " {} Got a total of {} precomputed values.".format(
                      _internal.VERBOSE_BLOCK_END_SYMBOL,
                      len(self._precomp_args_ft)),
                  sep="\n")

        # Custom arguments for postprocessing methods
        self._postprocess_args_ft = {
            "inserted_group_dep": self.inserted_group_dep,
        }

        # Custom arguments for summarization methods
        self._custom_args_sum = {
            "ddof": 1,
        }

        return self

    def extract(
            self,
            remove_nan: bool = True,
            verbose: int = 0,
            enable_parallel: bool = False,
            suppress_warnings: bool = False,
            **kwargs) -> t.Tuple[t.Sequence, ...]:
        """Extracts metafeatures from the previously fitted dataset.

        Parameters
        ----------
        remove_nan : :obj:`bool`, optional
            If True, remove any non-numeric values features before summarizing
            values from all feature extraction methods. Note that the summary
            methods may still remove non-numeric values by itself. In this
            case, the user must modify this behavior using built-in summary
            method arguments via kwargs, if possible.

        verbose : :obj:`int`, optional
            Defines the verbosity level related to the metafeature extraction.
            If == 1, show just the current progress, without line breaks.
            If >= 2, print all messages related to the metafeature extraction
            process.

            Note that warning messages are not affected by this option (see
            ``suppress_warnings`` argument below).

        enable_parallel : :obj:`bool`, optional
            If True, then the meta-feature extraction is done with
            multi-processes. Currently, this argument has no effect by now
            (to be implemented).

        suppress_warnings : :obj:`bool`, optional
            If True, do not show warnings about unknown user custom parameters
            for feature extraction and summary methods passed via kwargs. Note
            that both feature extraction and summary methods may still raise
            warnings by itself. In this case, just like the ``remove_nan``
            situation, the user must suppress them by built-in args from these
            methods via kwargs, if possible.

        kwargs:
            Used to pass custom arguments for both feature-extraction and
            summary methods. The expected format is the following:

            {``mtd_name``: {``arg_name``: arg_value, ...}, ...}

            In words, the key values of ``**kwargs`` should be the target
            methods which receives the custom arguments, and each method has
            another dictionary containing customs method argument names as keys
            and their correspondent values, as values. See ``Examples``
            subsection for a clearer explanation.

            For more information see Examples.

        Returns
        -------
        :obj:`tuple`(:obj:`list`, :obj:`list`)
            A tuple containing two lists.

            The first field is the identifiers of each summarized value in the
            form ``feature_name.summary_mtd_name`` (i.e., the feature
            extraction name concatenated by the summary method name, separated
            by a dot).

            The second field is the summarized values.

            Both lists have a 1-1 correspondence by the index of each element
            (i.e., the value at index ``i`` in the second list has its
            identifier at the same index in the first list and vice-versa).

            Example:
                ([``attr_ent.mean``, ``attr_ent.sd``], [``0.983``, ``0.344``])
                is the return value for the feature ``attr_end`` summarized by
                both ``mean`` and ``sd`` (standard deviation), giving the valu-
                es ``0.983`` and ``0.344``, respectively.

        Raises
        ------
        TypeError
            If calling ``extract`` method before ``fit`` method.

        Examples
        --------
        Using kwargs. Option 1 to pass ft. extraction custom arguments:

        >>> args = {
        >>> 'sd': {'ddof': 2},
        >>> '1NN': {'metric': 'minkowski', 'p': 2},
        >>> 'leaves': {'max_depth': 4},
        >>> }

        >>> model = MFE().fit(X=data, y=labels)
        >>> result = model.extract(**args)

        Option 2 (note: metafeatures with name starting with numbers are not
        allowed!):

        >>> model = MFE().fit(X=data, y=labels)
        >>> res = extract(sd={'ddof': 2}, leaves={'max_depth': 4})

        """
        if self.X is None:
            raise TypeError("Fitted data not found. Call "
                            '"fit" method before "extract".')

        if (not isinstance(self.X, np.ndarray)
                or not isinstance(self.y, np.ndarray)):
            self.X, self.y = _internal.check_data(self.X, self.y)

        if verbose >= 2:
            print("Started the metafeature extraction process.")

        _time_start = time.time()

        results = self._call_feature_methods(
            remove_nan=remove_nan,
            verbose=verbose,
            enable_parallel=enable_parallel,
            suppress_warnings=suppress_warnings,
            **kwargs)  # type: t.Tuple[t.List, ...]

        _internal.post_processing(
            results=results,
            groups=self.groups,
            suppress_warnings=suppress_warnings,
            **self._postprocess_args_ft,
            **kwargs)

        self.time_extract = time.time() - _time_start
        self.time_total = self.time_extract + self.time_precomp

        if results and results[0]:
            # Sort results by metafeature name
            results = tuple(
                map(list, zip(*sorted(zip(*results),
                                      key=lambda item: item[0]))))

        res_names, res_vals, res_times = results

        if verbose >= 2:
            _ext_t_pct = 100 * self.time_extract / self.time_total
            print(
                "\nMetafeature extraction process done.",
                " {} Time elapsed in total (precomputations + extraction): "
                "{:.8f} seconds.".format(
                    _internal.VERBOSE_BLOCK_MID_SYMBOL, self.time_total),
                " {} Time elapsed for extractions: {:.8f} seconds ({:.2f}% "
                "from the total).".format(
                    _internal.VERBOSE_BLOCK_MID_SYMBOL,
                    self.time_extract,
                    _ext_t_pct),
                " {} Time elapsed for precomputations: {:.8f} seconds "
                "({:.2f}% from the total).".format(
                    _internal.VERBOSE_BLOCK_MID_SYMBOL,
                    self.time_precomp, 100 - _ext_t_pct),
                " {} Total of {} values obtained.".format(
                    _internal.VERBOSE_BLOCK_END_SYMBOL, len(res_vals)),
                sep="\n")

        if self.timeopt:
            return res_names, res_vals, res_times

        return res_names, res_vals

    @classmethod
    def valid_groups(cls) -> t.Tuple[str, ...]:
        """Return a tuple of valid metafeature groups.

        Notes
        -----
        The returned ``groups`` are not related to the groups fitted in
        the model in the model instantation. The returned groups are all
        available metafeature groups in the ``Pymfe`` package. Check the
        ``MFE`` documentation for deeper information.
        """
        return _internal.VALID_GROUPS

    @classmethod
    def valid_summary(cls) -> t.Tuple[str, ...]:
        """Return a tuple of valid summary functions.

        Notes
        -----
        The returned ``summaries`` are not related to the summaries fitted
        in the model in the model instantation. The returned summaries are
        all available in the ``Pymfe`` package. Check the documentation of
        ``MFE`` for deeper information.
        """
        return _internal.VALID_SUMMARY

    @classmethod
    def _check_groups_type(cls,
                           groups: t.Optional[t.Union[str, t.Iterable[str]]]
                           ) -> t.Set[str]:
        """Cast ``groups`` to a tuple of valid metafeature group names."""
        if groups is None:
            return set(_internal.VALID_GROUPS)

        groups = _internal.convert_alias(MFE.groups_alias, groups)

        return set(groups)

    @classmethod
    def _filter_groups(cls,
                       groups: t.Set[str]
                       ) -> t.Set[str]:
        """Filter given groups by the available metafeature group names."""
        filtered_group_set = {
            group for group in groups
            if group in _internal.VALID_GROUPS
        }
        return filtered_group_set

    @classmethod
    def valid_metafeatures(
            cls,
            groups: t.Optional[t.Union[str, t.Iterable[str]]] = None,
    ) -> t.Tuple[str, ...]:
        """Return a tuple with all metafeatures related to given ``groups``.

        Parameters
        ----------
        groups : :obj:`Sequence` of :obj:`str` or :obj:`str`, optional:
            Can be a string such value is a name of a specific metafeature
            group (see ``valid_groups`` method for more information) or a
            sequence of metafeature group names. It can be also None, which
            in that case all available metafeature names will be returned.

        Returns
        -------
        :obj:`tuple` of :obj:`str`
            Tuple with all available metafeature names of the given ``groups``.

        Notes
        -----
        The returned ``metafeatures`` are not related to the groups or to the
        metafeatures fitted in the model in the model instantation. All the
        returned metafeatures are available in the ``Pymfe`` package. Check
        the ``MFE`` documentation for deeper information.
        """
        groups = MFE._check_groups_type(groups)
        groups = MFE._filter_groups(groups)

        deps = _internal.check_group_dependencies(groups)

        mtf_names = []  # type: t.List
        for group in groups.union(deps):
            class_ind = _internal.VALID_GROUPS.index(group)

            mtf_names += (
                _internal.get_prefixed_mtds_from_class(
                    class_obj=_internal.VALID_MFECLASSES[class_ind],
                    prefix=_internal.MTF_PREFIX,
                    only_name=True,
                    prefix_removal=True))

        return tuple(mtf_names)

    @classmethod
    def parse_by_group(
            cls,
            groups: t.Union[t.Sequence[str], str],
            extracted_results: t.Tuple[t.Sequence, ...],
    ) -> t.Tuple[t.List, ...]:
        """Parse the result of ``extract`` for given metafeature ``groups``.

        Can be used to easily separate the results of each metafeature
        group.

        Parameters
        ----------
        groups : :obj:`Sequence` of :obj:`str` or :obj:`str`
            Metafeature group names which the results should be parsed
            relative to. Use ``valid_groups`` method to check the available
            metafeature groups.

        extracted_results : :obj:`tuple` of :obj:`t.Sequence`
            Output of ``extract`` method. Should contain all outputed lists
            (metafeature names, values and elapsed time for extraction, if
            present.)

        Returns
        -------
        :obj:`tuple` of :obj:`str`
            Slices of lists of ``extracted_results``, selected based on
            given ``groups``.

        Notes
        -----
        The given ``groups`` are not related to the groups fitted in the
        model in the model instantation. Check ``valid_groups`` method to
        get a list of all available groups from the ``Pymfe`` package.
        Check the ``MFE`` documentation for deeper information about all
        these groups.
        """
        selected_indexes = _internal.select_results_by_classes(
            mtf_names=extracted_results[0],
            class_names=groups,
            include_dependencies=True)

        filtered_res = (
            [seq[ind] for ind in selected_indexes]
            for seq in extracted_results
        )

        return tuple(filtered_res)

    @staticmethod
    def _parse_description(docstring: str,
                           include_references: bool = False
                           ) -> t.Tuple[str, str]:
        """Parse the docstring to get initial description and reference.

        Parameters
        ----------
        docstring : str
            An numpy docstring as ``str``.

        include_references : bool
            If True include a column with article reference.

        Returns
        -------
        tuple of str
            The initial docstring description in the first position and the
            reference in the second.

        """
        initial_description = ""  # type: str
        reference_description = ""  # type: str

        # get initial description
        split = docstring.split("\n\n")
        if split:
            initial_description = " ".join(split[0].split())

        # get reference description
        if include_references:
            aux = docstring.split("References\n        ----------\n")
            if len(aux) >= 2:
                split = aux[1].split(".. [")
                if len(split) >= 2:
                    del split[0]
                    for spl in split:
                        reference_description += "[" + " ".join(
                            spl.split()) + "\n"

        return (initial_description, reference_description)

    @classmethod
    def metafeature_description(
            cls,
            groups: t.Optional[t.Union[str, t.Iterable[str]]] = None,
            sort_by_group: bool = False,
            sort_by_mtf: bool = False,
            print_table: bool = True,
            include_references: bool = False
    ) -> t.Optional[t.Tuple[t.List[t.List[str]], str]]:
        """Print a table with groups, metafeatures and description.

        Parameters
        ----------
        groups : sequence of str or str, optional:
            Can be a string such value is a name of a specific metafeature
            group (see ``valid_groups`` method for more information) or a
            sequence of metafeature group names. It can be also None, which
            in that case all available metafeature names will be returned.

        sort_by_group: bool
            Sort table by meta-feature group name.

        sort_by_mtf: bool
            Sort table by meta-feature name.

        print_table : bool
            If True a table will be printed with the description, otherwise the
            table will be send by return.

        print_table : bool
            If True sort the table by metafeature name.

        include_references : bool
            If True include a column with article reference.

        Returns
        -------
        list of list
            A table with the metafeature descriptions or None.

        Notes
        -----
        The returned ``metafeatures`` are not related to the groups or to the
        metafeatures fitted in the model instantation. All the
        returned metafeatures are available in the ``Pymfe`` package. Check
        the ``MFE`` documentation for deeper information.
        """

        groups = MFE._check_groups_type(groups)
        groups = MFE._filter_groups(groups)

        deps = _internal.check_group_dependencies(groups)

        if not isinstance(sort_by_group, bool):
            raise TypeError("The parameter sort_by_group should be bool.")

        if not isinstance(sort_by_mtf, bool):
            raise TypeError("The parameter sort_by_mtf should be bool.")

        if not isinstance(print_table, bool):
            raise TypeError("The parameter print_table should be bool.")

        mtf_names = []  # type: t.List[str]
        mtf_desc = [["Group", "Meta-feature name", "Description"]]
        if include_references:
            mtf_desc[0].append("Reference")

        for group in groups.union(deps):
            class_ind = _internal.VALID_GROUPS.index(group)

            mtf_names = (  # type: ignore
                _internal.get_prefixed_mtds_from_class(  # type: ignore
                    class_obj=_internal.VALID_MFECLASSES[class_ind],
                    prefix=_internal.MTF_PREFIX,
                    only_name=False,
                    prefix_removal=True))

            for name, method in mtf_names:
                ini_desc, ref_desc = MFE._parse_description(
                    str(method.__doc__), include_references)
                mtf_desc_line = [group, name, ini_desc]
                mtf_desc.append(mtf_desc_line)

                if include_references:
                    mtf_desc_line.append(ref_desc)

        if sort_by_mtf:
            mtf_desc.sort(key=lambda i: i[1])

        if sort_by_group:
            mtf_desc.sort(key=lambda i: i[0])

        draw = texttable.Texttable().add_rows(mtf_desc).draw()
        if print_table:
            print(draw)
            return None
        return mtf_desc, draw
