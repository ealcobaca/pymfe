"""Main module for extracting metafeatures from datasets.

Todo:
    * Implement parallel computing.
    * Implement time measurement for metafeature extraction.
"""
import typing as t
import collections

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

    # Disable limits for instance var and method args num
    # pylint: disable=R0902, R0913

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
                    2. `avg_summ`: average time for each metafeature including
                        required time for summarization.
                    3. `total`: total time for each metafeature, without sum-
                        marization time.
                    4. `total_summ`: total time for each metafeature including
                        required time for summarization.

            wildcard (:obj:`str`, optional): value used as `select all` for
                `groups`, `features` and `summary` arguments.

            suppress_warnings (:obj:`bool`, optional): if True, than all warn-
                ings invoked at the instantiation time will be ignored.


        References:
            .. _Rivolli et al.:
                "Towards Reproducible Empirical Research in Meta-Learning",
                Rivolli et al. URL: https://arxiv.org/abs/1808.10406
        """
        # pylint: disable=R0913

        self.groups = _internal.process_groups(groups)  # type: t.Sequence[str]

        self.features, self._metadata_mtd_ft = _internal.process_features(
            features=features,
            groups=self.groups,
            wildcard=wildcard,
            suppress_warnings=suppress_warnings)  \
            # type: t.Tuple[t.Tuple[str, ...], _TypeSeqExt]

        self.summary, self._metadata_mtd_sm = _internal.process_summary(
            summary)  # type: t.Tuple[t.Tuple[str, ...], _TypeSeqExt]

        self.timeopt = _internal.process_timeopt(
            measure_time)  # type: t.Optional[str]

        self.X = None  # type: t.Optional[np.array]
        self.y = None  # type: t.Optional[np.array]

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
    ) -> t.Tuple[t.List[str], t.List[t.Union[float, t.Sequence]]]:
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
            tuple(list, list): a tuple containing two lists. The first field
                is the identifiers of each summarized value in the form
                `feature_name.summary_mtd_name` (i.e. the feature-extrac-
                tion name concatenated by the summary method name, separated
                by a dot). If the summary function return more than one value
                (cardinality greater than 1), then each value name will have
                an extra concatenated id starting from 0 to differ between
                values (i.e. `feature_name.summary_mtd_name.id`). The second
                field is the summarized values. Both lists has a 1-1 corres-
                pondence by the index of each element (i.e. the value at in-
                dex `i` in the second list has its identifier at the same
                index in the first list and vice-versa).

            Example:
                ([`attr_ent.mean`, `attr_ent.sd`], [0.983459, 0.344361]) is
                the return value for the feature `attr_end` summarized by
                both `mean` and `sd` (standard deviation), giving the values
                `0.983469` and `0.344361`, respectively.
        """
        metafeat_vals = []  # type: t.List[t.Union[int, float, t.Sequence]]
        metafeat_names = []  # type: t.List[str]

        for sm_mtd_name, sm_mtd_callable, sm_mtd_args in self._metadata_mtd_sm:

            if verbose:
                print("  Summarizing {0} feature with {1} summary"
                      " function...".format(feature_name,
                                            sm_mtd_name), end=" ")

            sm_mtd_args_pack = _internal.build_mtd_kwargs(
                mtd_name=sm_mtd_name,
                mtd_args=sm_mtd_args,
                user_custom_args=kwargs.get(sm_mtd_name),
                inner_custom_args=self._custom_args_sum,
                suppress_warnings=suppress_warnings)

            summarized_val = _internal.summarize(
                features=feature_values,
                callable_sum=sm_mtd_callable,
                callable_args=sm_mtd_args_pack,
                remove_nan=remove_nan)

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
                    "{0}.{1}.{2}".format(feature_name, sm_mtd_name, i)
                    for i in range(len(summarized_val))
                ]

            else:
                metafeat_vals.append(summarized_val)
                metafeat_names.append(
                    "{0}.{1}".format(feature_name, sm_mtd_name))

            if verbose:
                print("Done.")

        return metafeat_names, metafeat_vals

    def _fill_col_ind_by_type(
            self,
            cat_cols: t.Optional[t.Union[str, t.Iterable[int]]] = "auto",
            check_bool: bool = True
    ) -> None:
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
            categorical_cols = ~np.apply_along_axis(
                _internal.isnumeric,
                axis=0,
                arr=self.X,
                **{"check_subtype": True},
            )

            if check_bool:
                categorical_cols |= np.apply_along_axis(
                    func1d=lambda col: len(np.unique(col)) == 2,
                    axis=0,
                    arr=self.X,
                )

        elif (isinstance(cat_cols, (np.ndarray, collections.Iterable))
              and not isinstance(cat_cols, str)
              and all(isinstance(x, int) for x in cat_cols)):
            categorical_cols = (
                i in cat_cols for i in range(self.X.shape[1])
            )

        else:
            raise ValueError(
                'Invalid "cat_cols" argument ({0}). '
                'Expecting "auto" or a integer Iterable.'.format(cat_cols))

        categorical_cols = np.array(categorical_cols)

        self._attr_indexes_num = tuple(np.where(~categorical_cols)[0])
        self._attr_indexes_cat = tuple(np.where(categorical_cols)[0])

    def fit(self,
            X: t.Sequence,
            y: t.Sequence,
            splits: t.Optional[t.Iterable[int]] = None,
            cat_cols: t.Optional[t.Union[str, t.Iterable[int]]] = "auto",
            check_bool: bool = True
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

        if (splits is not None
                and (not isinstance(splits, collections.Iterable)
                     or isinstance(splits, str))):
            raise TypeError('"splits" argument must be a iterable.')

        self.splits = splits

        self._fill_col_ind_by_type(cat_cols=cat_cols, check_bool=check_bool)

        data_num = self.X[:, self._attr_indexes_num]
        data_cat = self.X[:, self._attr_indexes_cat]

        self._custom_args_ft = {
            "X": self.X,
            "N": data_num,
            "C": data_cat,
            "y": self.y,
            "splits": self.splits,
        }

        self._custom_args_sum = {
            "sd": {"ddof": 1},
        }

        return self

    def extract(self,
                remove_nan: bool = True,
                verbose: bool = False,
                enable_parallel: bool = False,
                suppress_warnings: bool = False,
                **kwargs
                ) -> t.Tuple[t.List[str], t.List[t.Union[float, t.Sequence]]]:
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

        metafeat_vals = []  # type: t.List[t.Union[int, float, t.Sequence]]
        metafeat_names = []  # type: t.List[str]

        if verbose:
            print("Started metafeature extraction process...")

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

            features = _internal.get_feat_value(ft_mtd_name, ft_mtd_args_pack,
                                                ft_mtd_callable,
                                                suppress_warnings)

            if (self._metadata_mtd_sm is not None
                    and isinstance(features, (np.ndarray,
                                              collections.Sequence))):
                summarized_names, summarized_vals = self._call_summary_methods(
                    feature_values=features,
                    feature_name=ft_name_without_prefix,
                    remove_nan=remove_nan,
                    verbose=verbose,
                    suppress_warnings=suppress_warnings,
                    **kwargs)

                metafeat_vals += summarized_vals
                metafeat_names += summarized_names

            else:
                metafeat_vals.append(features)
                metafeat_names.append(ft_name_without_prefix)

            if verbose:
                print("Done with {} feature.".format(ft_mtd_name))

        if verbose:
            print("Done with metafeature extraction process.",
                  "Total of {} values obtained.".format(len(metafeat_vals)))

        return metafeat_names, metafeat_vals


if __name__ == "__main__":
    attr = np.array([
        [1, -.2, '1', 4],
        [0, .0, 'a', -1],
        [1, 2.2, '0', -1.2],
        [1, -1, 'b', .12],
    ], dtype=object)

    labels = np.array([1, 1, 0, 0])

    MODEL = MFE(groups="all", features="all",
                summary="all", measure_time="avg_summ")
    MODEL.fit(X=attr, y=labels)

    names, vals = MODEL.extract(
        suppress_warnings=False,
        remove_nan=True,
        **{
            "sd": {"ddof": 0},
        })

    for n, v in zip(names, vals):
        print(n, v)
