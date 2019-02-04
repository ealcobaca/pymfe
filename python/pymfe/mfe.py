"""Main module for extracting metafeatures from datasets.

Todo:
    * Precomputation options, to avoid frequent recalculations.
    * Implement parallel computing.
    * Handle missing data.
    * By-class feature extraction.
    * Support for multiclass, regression and unsupervised tasks.
    * Remove program driver for quick tests.
"""
import typing as t
import collections
import copy

import numpy as np

import _internal

_TypeSeqExt = t.Sequence[t.Tuple[str, t.Callable, t.Sequence]]
"""Type annotation for a sequence of TypeExtMtdTuple objects."""


class MFE:
    """Core class for metafeature extraction.

    Attributes:
        X (:obj:`Sequence`): independent attributes of the dataset.

        y (:obj:`Sequence`): target attributes of the dataset.

        splits (:obj:`Iterable`): Iterable/generator for K-Fold Cross
            Validation train and test indexes splits.

        groups (:obj:`tuple` of :obj:`str`): tuple object containing fitted
            metafeature groups loaded in model at instantiation.

        features (:obj:`tuple` of :obj:`str`): contains loaded metafeature-
            extraction method names available for metafeature extraction, from
            selected metafeatures groups and features listed at instantiation

        summary (:obj:`tuple` of :obj:`str`): tuple object which contains sum-
            mary functions names for features summarization.
    """

    def __init__(self,
                 groups: t.Union[str, t.Iterable[str]] = "all",
                 features: t.Union[str, t.Iterable[str]] = "all",
                 summary: t.Union[str, t.Iterable[str]] = ("mean", "sd"),
                 measure_time: t.Optional[str] = None,
                 wildcard: str = "all",
                 suppress_warnings: bool = False) -> None:
        """
        Provides easy access for metafeature extraction from structured
        datasets. It expected that user first calls `fit` method after
        instantiation and then `extract` for effectively extract the se-
        lected metafeatures. Check reference `Rivolli et al.`_ for more
        information.

        Attributes:
            groups (:obj:`Iterable` of :obj:`str` or `str`): a collection
                or a single metafeature group name representing the desired
                group of metafeatures for extraction. The supported groups
                are:

                    1. `general`: general/simples metafeatures.
                    2. `statistical`: statistical metafeatures.
                    3. `info-theory`: information-theoretic type of metafea-
                        ture.
                    4. `model-based`: metafeatures based on machine learning
                        model characteristics.
                    5. `landmarking`: metafeatures representing performance
                        metrics from simple machine learning models or machi-
                        ne learning models induced with sampled data.

                The special value provided by the argument `wildcard` can be
                used to rapidly select all metafeature groups.

            features (:obj:`Iterable` of :obj:`str` or `str`, optional): a col-
                lection or a single metafeature name desired for extraction.
                Keep in mind that only features in the selected `groups` will
                be used. Check `feature` attribute in order to get a list of
                available metafeatures from the selected groups.

                The special value provided by the argument `wildcard` can be
                used to rapidly select all features from the selected groups.

            summary (:obj:`Iterable` of :obj:`str` or `str`, optional): a
                collection or a single summary function to summarize a group
                of metafeature measures into a fixed-length group of value,
                typically a single value. The values must be one of the follo-
                wing:

                    1. `mean`: Average of the values.
                    2. `sd`: Standard deviation of the values.
                    3. `count`: Computes the cardinality of the measure.
                        Suitable for variable cardinality.
                    4. `histogram`: Describes the distribution of the mea-
                        sure values. Suitable for high cardinality.
                    5. `iq_range`: Computes the interquartile range of the
                        measure values.
                    6. `kurtosis`: Describes the shape of the measures values
                        distribution.
                    7. `max`: Resilts in the maximum vlaues of the measure.
                    8. `median`: Results in the central value of the measure.
                    9. `min`: Results in the minimum value of the measure.
                    10. `quantiles`: Results in the minimum, first quartile,
                        median, third quartile and maximum of the measure
                        values.
                    11. `range`: Computes the range of the measure values.
                    12. `skewness`: Describes the shape of the measure values
                        distribution in terms of symmetry.

                If more than one summary function is selected, then all multi-
                valued metafeatures extracted will be summarized with each
                summary function.

                The special value provided by the argument `wildcard` can be
                used to rapidly select all summary functions.

            measure_time (:obj:`str`, optional): options for measuring time
                elapsed during metafeature extraction. If :obj:`None`, no time
                elapsed will be measured. Otherwise, this argument must be a
                :obj:`str` valued as one of the options below:

                    1. `avg`: average time for each metafeature (total time di-
                        vided by the feature cardinality, i.e., number of feat-
                        ures extracted by a single feature-extraction related
                        method), without summarization time.
                    2. `avg_summ`: average time for each metafeature (total ti-
                        me of extraction divided by feature cardinality) inclu-
                        ding required time for summarization.
                    3. `total`: total time for each metafeature, without sum-
                        marization time.
                    4. `total_summ`: total time for each metafeature including
                        required time for summarization.

                The ``cardinality`` of the feature, used to divide the total
                time measured by a single feature extraction method if an op-
                tion starting with ``avg`` is selected, is the number of values
                that can be extracted with a single method in the fitted data-
                set. For example, ``mean`` feature has cardinality equal to the
                number of numeric features in the dataset, where ``cor`` (from
                ``correlation``) has cardinality (N - 1)/2, where N is the num-
                ber of numeric features in the dataset.

                If a summary method has cardinality higher than one (more than
                one value returned after summarization and, thus, creating more
                than one entry in the result lists) like, for example, ``histo-
                gram`` summary method, then the corresponding time of this sum-
                mary will be inserted only in the first correspondent element
                of the time list. The remaining entries are all filled with 0.0
                value. This is done to keep consistency between the size of all
                returned lists and index correct correspondation between all
                lists.

            wildcard (:obj:`str`, optional): value used as `select all` for
                `groups`, `features` and `summary` arguments.

            suppress_warnings (:obj:`bool`, optional): if True, than all warn-
                ings invoked at the instantiation time will be ignored.


        References:
            .. _Rivolli et al.:
                "Towards Reproducible Empirical Research in Meta-Learning",
                Rivolli et al. URL: https://arxiv.org/abs/1808.10406
        """
        self.groups = _internal.process_generic_set(
            values=groups,
            group_name="groups")  # type: t.Sequence[str]

        self.features, self._metadata_mtd_ft = _internal.process_features(
            features=features,
            groups=self.groups,
            wildcard=wildcard,
            suppress_warnings=suppress_warnings)  \
            # type: t.Tuple[t.Tuple[str, ...], _TypeSeqExt]

        self.summary, self._metadata_mtd_sm = _internal.process_summary(
            summary)  # type: t.Tuple[t.Tuple[str, ...], _TypeSeqExt]

        self.timeopt = _internal.process_generic_option(
            value=measure_time,
            group_name="timeopt",
            allow_none=True)  # type: t.Optional[str]

        self.X = None  # type: t.Optional[np.ndarray]
        self.y = None  # type: t.Optional[np.ndarray]

        self.splits = None  # type: t.Optional[t.Iterable[int]]

        self._custom_args_ft = None  # type: t.Optional[t.Dict[str, t.Any]]
        """User-independent custom arguments for features (e.g. `X` and `y`)"""

        self._custom_args_sum = None  # type: t.Optional[t.Dict[str, t.Any]]
        """User-independent custom arguments for summary functions."""

        self._attr_indexes_num = None  # type: t.Optional[t.Tuple[int, ...]]
        """Numerical column indexes from fitted X (independent attributes)."""

        self._attr_indexes_cat = None  # type: t.Optional[t.Tuple[int, ...]]
        """Categorical column indexes from fitted X (indep. attributes)."""

    def _call_summary_methods(
            self,
            feature_values: t.Sequence[_internal.TypeNumeric],
            feature_name: str,
            remove_nan: bool = True,
            verbose: bool = False,
            suppress_warnings: bool = False,
            **kwargs
    ) -> t.Tuple[t.List[str], t.List[t.Union[float, t.Sequence]], t.
                 List[float]]:
        """Invoke summary functions loaded in model on given feature values.

        Args:
            feature_values (:obj:`Sequence` of numerics): sequence containing
                values from feature-extraction methods.

            feature_name (:obj:`str`): name of the feature method used for
                produce the `feature_value`.

            remove_nan (:obj:`bool`, optional): if True, all non-numeric values
                will be removed from `feature_values` before calling each sum-
                mary method. Note that the summary method itself may still re-
                move non-numeric values and, in this case, user must suppress
                this using a built-in argument of the summary method using the
                **kwargs argument.

            verbose (:obj:`bool`, optional): if True, then messages about the
                summarization process may be printed. Note that warnings are
                not related with this argument (see ``suppress_warnings`` arg-
                ument below).

            suppress_warnings (:obj:`bool`, optional): if True, ignore all
                warnings invoked before and after summary method calls. Note
                that, as the `remove_nan` parameter, the summary callables may
                still invoke warnings by itself and the user need to ignore
                then, if possible, via **kwargs.

            **kwargs: user-defined arguments for the summary callables.

        Returns:
            tuple(list, list, list): a tuple containing three lists. The first
                field is the identifiers of each summarized value in the form
                `feature_name.summary_mtd_name` (i.e. the feature-extrac-
                tion name concatenated by the summary method name, separated
                by a dot). If the summary function return more than one value
                (cardinality greater than 1), then each value name will have
                an extra concatenated id starting from 0 to differ between
                values (i.e. `feature_name.summary_mtd_name.id`). The second
                field is the summarized values. Both lists has a 1-1 corres-
                pondence by the index of each element (i.e. the value at in-
                dex `i` in the second list has its identifier at the same
                index in the first list and vice-versa). The third field is
                a list with measured time wasted by each summary function. If
                the cardinality of the summary function is greater than 1,
                then the correspondent measured time will be kept only in the
                first correspondent field and the extra fields will be filled
                with 0.0. This is necessary to keep consistency of the size
                between all lists.

            Example:
                ([`attr_ent.mean`, `attr_ent.sd`], [0.983459, 0.344361]) is
                the return value for the feature `attr_end` summarized by
                both `mean` and `sd` (standard deviation), giving the values
                `0.983469` and `0.344361`, respectively.
        """
        metafeat_vals = []  # type: t.List[t.Union[int, float, t.Sequence]]
        metafeat_names = []  # type: t.List[str]
        metafeat_times = []  # type: t.List[float]

        for sm_mtd_name, sm_mtd_callable, sm_mtd_args in self._metadata_mtd_sm:

            if verbose:
                print(
                    "  Summarizing {0} feature with {1} summary"
                    " function...".format(feature_name, sm_mtd_name),
                    end=" ")

            sm_mtd_args_pack = _internal.build_mtd_kwargs(
                mtd_name=sm_mtd_name,
                mtd_args=sm_mtd_args,
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

            if verbose:
                print("Done.")

        return metafeat_names, metafeat_vals, metafeat_times

    def _call_feature_methods(self,
                              remove_nan: bool = True,
                              verbose: bool = False,
                              enable_parallel: bool = False,
                              suppress_warnings: bool = False,
                              **kwargs) -> t.Tuple[t.List, ...]:
        """Invoke feature methods/functions loaded in model and gather results.

        The returned values are already summarized, if needed.

        For more information, check ``extract`` method documentation for
        in-depth information about arguments and return value.
        """
        metafeat_vals = []  # type: t.List[t.Union[int, float, t.Sequence]]
        metafeat_names = []  # type: t.List[str]
        metafeat_times = []  # type: t.List[float]

        for ft_mtd_name, ft_mtd_callable, ft_mtd_args in self._metadata_mtd_ft:

            if verbose:
                print("Extracting {} feature...".format(ft_mtd_name))

            ft_name_without_prefix = _internal.remove_mtd_prefix(ft_mtd_name)

            ft_mtd_args_pack = _internal.build_mtd_kwargs(
                mtd_name=ft_name_without_prefix,
                mtd_args=ft_mtd_args,
                user_custom_args=kwargs.get(ft_name_without_prefix),
                inner_custom_args=self._custom_args_ft,
                suppress_warnings=suppress_warnings)

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

            if verbose:
                print("Done with {} feature.".format(ft_mtd_name))

        return metafeat_names, metafeat_vals, metafeat_times

    def _fill_col_ind_by_type(
            self,
            cat_cols: t.Optional[t.Union[str, t.Iterable[int]]] = "auto",
            check_bool: bool = True) -> None:
        """Get X column indexes by its data type.

        The indexes for numerical and categorical attributes are kept,
        respectively, at ``_attr_indexes_num`` and ``_attr_indexes_cat``
        instance attributes.

        Args:
            cat_cols (:obj:`str` or :obj:`iterable` of :obj:`int`, optional):
                Iterable of indexes identifying categorical columns. If spe-
                cial keyword ``auto`` is given, then an automatic verification
                will be done in the fitted attributes.

            check_bool (:obj:`bool`, optional): check ``fit`` method corres-
                ponding argument for more information.

        Raises:
            TypeError: if ``X`` attribute is :obj:`NoneType`.
            ValueError: if ``cat_cols`` is neither ``auto`` or a valid
                integer Iterable.
        """

        if self.X is None:
            raise TypeError("X can't be 'None'.")

        categorical_cols = None  # type: np.ndarray[bool]

        if not cat_cols:
            categorical_cols = np.array([False] * self.X.shape[1])

        elif isinstance(cat_cols, str) and cat_cols.lower() == "auto":
            categorical_cols = np.logical_not(np.apply_along_axis(
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
              and not isinstance(cat_cols, str)
              and all(isinstance(x, int) for x in cat_cols)):
            categorical_cols = (i in cat_cols for i in range(self.X.shape[1]))

        else:
            raise ValueError(
                'Invalid "cat_cols" argument ({0}). '
                'Expecting "auto" or a integer Iterable.'.format(cat_cols))

        categorical_cols = np.array(categorical_cols)

        self._attr_indexes_num = tuple(
            np.where(np.logical_not(categorical_cols))[0])
        self._attr_indexes_cat = tuple(np.where(categorical_cols)[0])

    def _timeopt_type_is_avg(self) -> bool:
        """Checks if user selected time option is an ``average`` type."""
        return (isinstance(self.timeopt, str)
                and self.timeopt.startswith(_internal.TIMEOPT_AVG_PREFIX))

    def _timeopt_include_summary(self) -> bool:
        """Checks if user selected time option include ``summary`` time."""
        return (isinstance(self.timeopt, str)
                and self.timeopt.endswith(_internal.TIMEOPT_SUMMARY_SUFIX))

    def _combine_time(self, time_ft: float,
                      times_sm: t.List[float]) -> t.List[float]:
        """Treat ft. extraction and summarization time based in ``timeopt``.

        Args:
            time_ft (:obj:`float`): time necessary to extract some feature.

            times_sm (:obj:`list` of :obj:`float`): list of values to summarize
                the metafeature value with each summary function.

        Returns:
            list[float]: if ``timeopt`` attribute considers ``summary`` time
            (i.e. selected option ends with ``summ``), then this returned list
            values are the combination of times gathered in feature extraction
            and summarization methods. Otherwise, the list values are the value
            of ``time_ft`` copied ``len(times_sm)`` times to keep consistency
            with the correspondence between the values of all lists returned by
            ``extract`` method.
        """
        total_time = np.array([time_ft] * len(times_sm))

        if self._timeopt_include_summary():
            total_time += times_sm

        # As seen in ``_call_summary_methods`` method documentation,
        # zero-valued elements are created to fill the time list in order
        # to keep its size consistent with another feature extraction re-
        # lated lists. In this case, here they're kept zero-valued.
        total_time[np.array(times_sm) == 0.0] = 0.0

        return total_time.tolist()

    def _set_data_categoric(self, transform_num: bool,
                            num_bins: bool = None) -> np.ndarray:
        """Returns categorical data from fitted dataset.

        Args:
            transform_num (:obj:`bool`): if True, then all numeric-type
                data will be discretized using an equal-frequency histo-
                gram. Otherwise, these attributes will be ignored.

            num_bins (:obj:`bool`, optional): number of bins of the dis-
                cretization histogram. Used only if ``transform_num`` is
                True. If this argument is :obj:`NoneType`, the default
                value is min(2, c), where ``c`` is the cubic root of the
                number of instances of the fitted dataset.

        Returns:
            np.ndarray: processed categorical data. If no changes are ne-
                eded from the original dataset, then this method will not
                create a copy of the original data to prevent unnecessary
                memory usage. Otherwise, this method will return a modifi-
                ed version of the original categorical data, thus consum-
                ing more memory.

        Raises:
            TypeError: if ``X`` or ``_attr_indexes_cat`` instance attribu-
                tes are :obj:`NoneType`. This can be avoided passing valid
                data to fit and first calling ``_fill_col_ind_by_type`` ins-
                tance method before this method.
        """
        if self.X is None:
            raise TypeError("It is necessary to fit valid data into"
                            "model before setting up categoric data."
                            '("X" attribute is "NoneType").')

        if self._attr_indexes_cat is None:
            raise TypeError("No information about indexes of categoric"
                            "attributes. Please be sure to call method"
                            '"_fill_col_ind_by_type" before this.')

        data_cat = self.X[:, self._attr_indexes_cat]

        if transform_num:
            data_num_discretized = _internal.transform_num(
                self.X[:, self._attr_indexes_num], num_bins=num_bins)

            if data_num_discretized is not None:
                data_cat = np.concatenate((data_cat, data_num_discretized),
                                          axis=1)

        return data_cat

    def _set_data_numeric(self, transform_cat: bool,
                          rescale: t.Optional[str] = None,
                          rescale_args: t.Optional[t.Dict[str, t.Any]] = None
                          ) -> np.ndarray:
        """Returns numeric data from fitted dataset.

        Args:
            transform_cat (:obj:`bool`): if True, then all categoric-type
                data will be binarized with One Hot Encoding strategy.

            rescale (:obj:`str`, optional): check ``fit`` documentation for
                more information about this parameter.

            rescale_args (:obj:`dict`, optional): check ``fit`` documentation
                for more information about this parameter.

        Returns:
            np.ndarray: processed numerical data. If no changes are needed
                from the original dataset, then this method will not create
                a copy of the original data to prevent unnecessary memory
                usage. Otherwise, this method will return a modified versi-
                on of the original numerical data, thus consuming more me-
                mory.

        Raises:
            TypeError: if ``X`` or ``_attr_indexes_num`` instance attributes
                are :obj:`NoneType`. This can be avoided passing valid data
                to fit and first calling ``_fill_col_ind_by_type`` instance
                method before this method.
        """
        if self.X is None:
            raise TypeError("It is necessary to fit valid data into"
                            "model before setting up numeric data."
                            '("X" attribute is "NoneType").')

        if self._attr_indexes_num is None:
            raise TypeError("No information about indexes of numeric"
                            "attributes. Please be sure to call method"
                            '"_fill_col_ind_by_type" before this.')

        data_num = self.X[:, self._attr_indexes_num]

        if transform_cat:
            categorical_dummies = _internal.transform_cat(
                self.X[:, self._attr_indexes_cat])

            if categorical_dummies is not None:
                data_num = np.concatenate((data_num, categorical_dummies),
                                          axis=1).astype(float)

        if rescale:
            data_num = _internal.rescale_data(data=data_num,
                                              option=rescale,
                                              args=rescale_args)

        return data_num

    def fit(self,
            X: t.Sequence,
            y: t.Sequence,
            splits: t.Optional[t.Iterable[int]] = None,
            transform_num: bool = False,
            transform_cat: bool = False,
            rescale: t.Optional[str] = None,
            rescale_args: t.Optional[t.Dict[str, t.Any]] = None,
            cat_cols: t.Optional[t.Union[str, t.Iterable[int]]] = "auto",
            check_bool: bool = False,
            missing_data: str = "ignore",
            precompute: str = "all",
            ) -> "MFE":
        """Fits dataset into the a MFE model.

        Args:
            X (:obj:`Sequence`): predictive attributes of the dataset.

            y (:obj:`Sequence`): target attributes of the dataset, assuming
                that it's a supervised task.

            splits (:obj:`Iterable`, optional): iterable which contains K-Fold
                Cross Validation index splits to use mainly in landmarking
                metafeatures. If not given, each metafeature will be extra-
                cted a single time, which may give poor results.

            transform_num (:obj:`bool`, optional): if True, numeric attributes
                will be discretized using equal-frequency histogram technique
                to use when extracting categorical data only metafeatures. Note
                that numeric features will still use the original numeric valu-
                es. If False, then numeric attributes will just be ignored for
                categorical-only metafeatures.

            transform_cat (:obj:`bool`, optional): if True, categorical attri-
                butes will be binarized using One Hot Encoding technique to use
                when extracting categorical data only metafeatures. Note that
                categoric features will still use the original categorical va-
                lues. If False, then categorical attributes will just be igno-
                red for numeric-only metafeatures.

            rescale (:obj:`str`, optional): if :obj:`NoneType`, all numeric da-
                ta will be kept with its original values. Otherwise, this argu-
                ment can assume one of the string options below in order to re-
                scale all numeric values:

                    1. ``standard``: set numeric data to zero mean, unit varia-
                        nce. Also known as ``z-score`` normalization. Check do-
                        cumentation of ``sklearn.preprocessing.StandardScaler``
                        for in-depth information.

                    2. `'min-max``: set numeric data to interval [a, b], a < b.
                        You can define values to ``a`` and ``b`` using argument
                        ``rescale_args``. The default values are a = 0.0  and
                        b = 1.0. Check ``sklearn.preprocessing.MinMaxScaler``
                        documentation for more information.

                    3. ``robust``: rescale data using statististics robust to
                        outliers. Check ``sklearn.preprocessing.RobustScaler``
                        documentation for in-depth information.

            rescale_args (:obj:`dict`, optional): dictionary containing parame-
                ters for rescaling data. Used only if ``rescale`` argument is
                not :obj:`NoneType`. This dict keys are the parameter names as
                strings and the values, the corresponding parameter value.

            cat_cols (:obj:`Sequence` of :obj:`int` or :obj:`str`, optional):
                categorical columns of dataset. If :obj:`NoneType` or empty se-
                quence is given, all columns are assumed as numeric. If ``au-
                to`` value is given, then an attempt of automatic detection is
                performed while fitting the dataset.

            check_bool (:obj:`bool`, optional): if `cat_cols` is ``auto``,
                and this flag is True, assume that all columns with exactly
                two different values is also a categorical (boolean) column,
                independently of its data type. Otherwise, these columns may
                be considered Numeric depending on its data type.

            missing_data (:obj:`str`, optional): strategy to handle missing
                values in data. Not implemented yet.

            precompute (:obj:`str`, optional): which computed common values
                should be cached to share among various metafeature-extrac-
                tion related methods (e.g. ``distinct classes``, ``covariance
                matrix``, etc). This argument may speed up metafeature extrac-
                tion but also consumes more memory, so it may not be suitable
                for very large datasets.

        Raises:
            ValueError: if number of rows of X and y length does not match.
            TypeError: if X or y (or both) is neither a :obj:`list` or
                a :obj:`np.array` object.

        Returns:
            MFE: the instance itself, to allow inline instantiation-and-fit
                code `model = MFE(...).fit(...)` or inline fit-and-extraction
                `result = MFE(...).fit(...).extract(...)`.
        """
        self.X, self.y = _internal.check_data(X, y)

        rescale = _internal.process_generic_option(
            value=rescale,
            group_name="rescale",
            allow_none=True)

        if (splits is not None
                and (not isinstance(splits, collections.Iterable)
                     or isinstance(splits, str))):
            raise TypeError('"splits" argument must be a iterable.')

        self.splits = copy.deepcopy(splits)

        self._fill_col_ind_by_type(cat_cols=cat_cols, check_bool=check_bool)

        data_cat = self._set_data_categoric(transform_num=transform_num)
        data_num = self._set_data_numeric(transform_cat=transform_cat,
                                          rescale=rescale,
                                          rescale_args=rescale_args)

        self._custom_args_ft = {
            "X": self.X,
            "N": data_num,
            "C": data_cat,
            "y": self.y,
            "splits": self.splits,
        }

        self._custom_args_sum = {
            "ddof": 1,
        }

        return self

    def extract(self,
                remove_nan: bool = True,
                verbose: bool = False,
                enable_parallel: bool = False,
                by_class: bool = False,
                suppress_warnings: bool = False,
                **kwargs) -> t.Tuple[t.List, ...]:
        """Extracts metafeatures from previously fitted dataset.

        Args:
            remove_nan (:obj:`bool`, optional): if True, remove any non-numeric
                values features before summarizing values from feature-extrac-
                tion methods. Note that the summary methods may still remove
                non-numeric values by itself. In this case, the user will need
                to modify this behavior using built-in summary method arguments
                via this method **kwargs, if possible.

            verbose (:obj:`bool`, optional): if True, messages related with the
                metafeature extraction process can be printed. Note that warn-
                ing messages is not affected by this option (see ``suppress_-
                warnings`` argument below).

            enable_parallel (:obj:`bool`, optional): if True, then the meta-
                feature extraction will be done with multiprocesses. This
                argument has no effect for now (to be implemented).

            by_class (:obj:`bool, optional): not implemented yet.

            suppress_warnings (:obj:`bool`, optional): if True, do not show
                warnings about unknown user custom parameters for feature-
                extraction and summary methods passed via **kwargs. Note that
                both feature-extraction and summary methods may still raise
                warnings by itself. In this case, just like the `remove_nan`
                situation, user will need to suppress they by built-in args
                from these methods via **kwargs, if possible.

            **kwargs: used to pass custom arguments for both feature-extrac-
                tion and summary methods. The expected format is the follow-
                ing:

                    {`mtd_name`: {`arg_name`: value, ...}, ...}

                In words, the key values of `**kwargs` should be the target
                methods to pass the custom arguments, and each method has
                another dict containing each method argument to be modified
                as keys and their correspondent values. See ``Examples`` sub-
                section for a clearer explanation.

                Example:
                    args = {
                        'sd': {'ddof': 2},
                        '1NN': {'metric': 'minkowski', 'p': 2},
                        'leaves': {'max_depth': 4},
                    }

                    res = MFE().fit(X=data, y=labels).extract(**args)

        Returns:
            tuple(list, list): a tuple containing two lists. The first field
                is the identifiers of each summarized value in the form
                `feature_name.summary_mtd_name` (i.e. the feature-extrac-
                tion name concatenated by the summary method name, separated
                by a dot). The second field is the summarized values. Both
                lists has a 1-1 correspondence by the index of each element
                (i.e. the value at index `i` in the second list has its iden-
                tifier at the same index in the first list and vice-versa).

            Example:
                ([`attr_ent.mean`, `attr_ent.sd`], [0.983459, 0.344361]) is
                the return value for the feature `attr_end` summarized by
                both `mean` and `sd` (standard deviation), giving the values
                `0.983469` and `0.344361`, respectively.

        Raises:
            TypeError: if calling `extract(...)` method before `fit(...)`
                method.
        """
        if self.X is None or self.y is None:
            raise TypeError("Fitted data not found. Call "
                            '"fit" method before "extract".')

        if (not isinstance(self.X, np.ndarray)
                or not isinstance(self.y, np.ndarray)):
            self.X, self.y = _internal.check_data(self.X, self.y)

        if verbose:
            print("Started metafeature extraction process...")

        results = self._call_feature_methods(
            remove_nan=remove_nan,
            verbose=verbose,
            enable_parallel=enable_parallel,
            suppress_warnings=suppress_warnings,
            **kwargs)

        res_names, res_vals, res_times = results

        if verbose:
            if self._timeopt_type_is_avg():
                time_type = "average"
            else:
                time_type = "total"

            print(
                "Done with metafeature extraction process.",
                "Total of {0} values obtained. Time elapsed "
                "({1}) = {2:.8f}.".format(
                    len(res_vals), time_type, sum(res_times)),
                sep="\n")

        if self.timeopt:
            return res_names, res_vals, res_times

        return res_names, res_vals


if __name__ == "__main__":
    attr = np.array([
        [1, -.2, '1', 4],
        [0, .0, 'a', -1],
        [1, 2.2, '0', -1.2],
        [1, -1, 'b', .12],
    ],
                    dtype=object)

    labels = np.array([1, 1, 0, 0])

    MODEL = MFE(
        groups="all",
        features=["cat_to_num", "mean", "nr_inst"],
        summary=["histogram", "mean", "sd", "kurtosis"],
        measure_time="avg_summ")
    MODEL.fit(rescale="robust", rescale_args=None,
              X=attr, y=labels, transform_num=True, transform_cat=True)

    print(MODEL._custom_args_ft["X"])
    print(MODEL._custom_args_ft["N"])
    print(MODEL._custom_args_ft["N"].mean())
    print(MODEL._custom_args_ft["N"].var())
    print(MODEL._custom_args_ft["N"].max())
    print(MODEL._custom_args_ft["N"].min())
    print(MODEL._custom_args_ft["C"])

    names, vals, times = MODEL.extract(
        suppress_warnings=False, remove_nan=True,
        verbose=True, kurtosis={"method": 1})

    for n, v, i in zip(names, vals, times):
        print(n, v, i)
