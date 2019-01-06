"""Main module for extracting metafeatures from datasets."""
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

    # pylint: disable=R0902

    def __init__(self,
                 groups: t.Union[str, t.Iterable[str]] = "all",
                 features: t.Union[str, t.Iterable[str]] = "all",
                 summary: t.Union[str, t.Iterable[str]] = ("mean", "sd"),
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
                    10. `quartiles`: Results in the minimum, first quartile,
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

        self.X = None  # type: t.Optional[np.array]
        self.y = None  # type: t.Optional[np.array]

        self.splits = None  # type: t.Optional[t.Iterable[int]]

        self._custom_args_ft = None  # type: t.Optional[t.Dict[str, t.Any]]
        """User-independent custom arguments for features (e.g. `X` and `y`)"""

        self._custom_args_sum = None  # type: t.Optional[t.Dict[str, t.Any]]
        """User-independent custom arguments for summary functions."""

    def _call_summary_methods(
            self,
            feature_values: t.Sequence[_internal.TypeNumeric],
            feature_name: str,
            remove_nan: bool = True,
            suppress_warnings: bool = False,
            **kwargs
    ) -> t.Tuple[t.List[str], t.List[t.Union[float, t.Sequence]]]:
        """Invoke summary functions loaded in model on given feature values.

        Args:
            feature_values (:obj:`Sequence` of numerics): sequence containing
                values from feature-extraction methods.

            feature_name (:obj:`str`): name of the feature method used for
                produce the `feature_value`.

            remove_nan (:obj:`bool`): if True, all non-numeric values will
                be removed from `feature_values` before calling each summary
                method. Note that the summary method itself may still remove
                non-numeric values and, in this case, user must suppress this
                using a built-in argument of the summary method via **kwargs.

            suppress_warnings (:obj:`bool`): if True, ignore all warnings in-
                voked before and after summary method calls. Note that, as
                the `remove_nan` parameter, the summary callables may still
                invoke warnings by itself and the user need to ignore then,
                if possible, via **kwargs.

            **kwargs: user-defined arguments for the summary callables.

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
                0.983469 and 0.344361, respectively.
        """
        metafeat_vals = []  # type: t.List[t.Union[int, float, t.Sequence]]
        metafeat_names = []  # type: t.List[str]

        for sm_mtd_name, sm_mtd_callable, sm_mtd_args in self._metadata_mtd_sm:

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

            metafeat_vals.append(summarized_val)
            metafeat_names.append("{0}.{1}".format(feature_name, sm_mtd_name))

        return metafeat_names, metafeat_vals

    def fit(self,
            X: t.Sequence,
            y: t.Sequence,
            splits: t.Optional[t.Iterable[int]] = None) -> "MFE":
        """Fits dataset into the a MFE model.

        Args:
            X (:obj:`Sequence`): predictive attributes of the dataset.

            y (:obj:`Sequence`): target attributes of the dataset, assuming
                that it's a supervised task.

            splits (:obj:`Iterable`, optional): iterable which contains K-Fold
                Cross Validation index splits to use mainly in landmarking
                metafeatures. If not given, each metafeature will be extra-
                cted a single time, which may give poor results.

        Raises:
            ValueError: if number of rows of X and y length does not match.
            TypeError: if X or y (or both) is neither a :obj:`list` or
                a :obj:`np.array` object.

        Returns:
            MFE: the instance itself, to permit inline instantiation-and-fit
                code like MFE(...).fit(...).
        """

        self.X, self.y = _internal.check_data(X, y)

        if (splits is not None
                and not isinstance(splits, collections.Iterable)):
            raise TypeError('"splits" argument must be a iterable.')

        self.splits = splits

        self._custom_args_ft = {
            "X": self.X,
            "y": self.y,
            "splits": self.splits,
        }

        self._custom_args_sum = None

        return self

    def extract(self,
                remove_nan: bool = True,
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

        for ft_mtd_name, ft_mtd_callable, ft_mtd_args in self._metadata_mtd_ft:

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

            if isinstance(features, (np.ndarray, collections.Sequence)):
                summarized_names, summarized_vals = self._call_summary_methods(
                    feature_values=features,
                    feature_name=ft_name_without_prefix,
                    remove_nan=remove_nan,
                    suppress_warnings=suppress_warnings,
                    **kwargs)

                metafeat_vals += summarized_vals
                metafeat_names += summarized_names

            else:
                metafeat_vals.append(features)
                metafeat_names.append(ft_name_without_prefix)

        return metafeat_names, metafeat_vals


if __name__ == "__main__":
    attr = np.array([
        [1, 2, 3, 4],
        [0, 0, 1, 1],
        [1, 2, 1, 1],
        [1, 1, 1, 1],
    ])

    labels = np.array([1, 1, 0, 0])

    MODEL = MFE(groups="all", features="all")
    print(MODEL.features)
    print(MODEL.summary)
    MODEL.fit(X=attr, y=labels)

    names, vals = MODEL.extract(
        suppress_warnings=False,
        remove_nan=True,
        cache_kwargs=False,
        **{
            "sd": {"ddof": 1},
        })

    for n, v in zip(names, vals):
        print(n, v)
