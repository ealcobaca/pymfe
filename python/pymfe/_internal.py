"""Provides useful functions for MFE package.

Attributes:
    VALID_GROUPS (:obj:`tuple` of :obj:`str`): Supported groups of
        metafeatures of pymfe.

    VALID_SUMMARY (:obj:`tuple` of :obj:`str`): Supported summary
        functions to combine metafeature values.

    VALID_MFECLASSES (:obj:`tuple` of Classes): Metafeature ex-
        tractors predefined classes, where metafeature-extraction
        methods will be searched.

    MTF_PREFIX (:obj:`str`): prefix of metafeature-extraction me-
        thod names. For example, the metafeature called `inst_nr`
        is implemented in the method named `[MTF_PREFIX]_inst_nr`.
"""
import typing as t
import inspect
import collections
import operator
import warnings
import time

import numpy as np

import _summary
import general
import statistical
import info_theory
import landmarking
import model_based

VALID_GROUPS = (
    "landmarking",
    "general",
    "statistical",
    "model-based",
    "info-theory",
)  # type: t.Tuple[str, ...]

VALID_SUMMARY = (*_summary.SUMMARY_METHODS, )  # type: t.Tuple[str, ...]

VALID_MFECLASSES = (
    landmarking.MFELandmarking,
    general.MFEGeneral,
    statistical.MFEStatistical,
    model_based.MFEModelBased,
    info_theory.MFEInfoTheory,
)  # type: t.Tuple

VALID_TIMEOPT = (
    "avg",
    "avg_summ",
    "total",
    "total_summ",
)

MTF_PREFIX = "ft_"
"""Prefix which is that metafeat. extraction related methods starts with."""

TypeMtdTuple = t.Tuple[str, t.Callable[[], t.Any]]
"""Type annotation which describes the a metafeature method tuple."""

TypeExtMtdTuple = t.Tuple[str, t.Callable[[], t.Any], t.Sequence]
"""Type annotation which extends TypeMtdTuple with extra field (for 'Args')"""

_TYPE_NUMERIC = (
    int,
    float,
    np.number,
)

TypeNumeric = t.TypeVar(
    "TypeNumeric",
    int,
    float,
    np.number,
)
"""Typing alias for both numeric types."""


def _check_value_in_group(value: t.Union[str, t.Iterable[str]],
                          group: t.Iterable[str],
                          wildcard: str = "all"
                          ) -> t.Tuple[t.Tuple[str, ...], t.Tuple[str, ...]]:
    """Checks if a value is in a set or a set of values is a subset of a set.

    Args:
        value (:obj:`Iterable` of :obj:`str` or :obj:`str): value(s) to be
            checked if are in the given group of strings.

        group (:obj:`Iterable` of :obj:`str`): a group of strings represen-
            ting the values such that `value` will be verified against.

        wildcard (:obj:`str`, optional): a value which represent 'all values'.
            The case is ignored, so, for example, both values 'all', 'ALL' and
            any mix of cases are considered to be the same wildcard token.

    Returns:
        tuple(tuple, tuple): A pair of tuples containing, respectively, values
        that are in the given group and those that are not. If no value is in
        either group, then this group will be :obj:`None`.

    Raises:
        TypeError: if `value` is not a Iterable type or some of its elements
            are not a :obj:`str` type.
    """

    if not isinstance(value, collections.Iterable):
        raise TypeError("Parameter type is not "
                        "consistent ({0}).".format(type(value)))

    in_group = tuple()  # type: t.Tuple[str, ...]
    not_in_group = tuple()  # type: t.Tuple[str, ...]

    if isinstance(value, str):
        value = value.lower()
        if value == wildcard.lower():
            in_group = tuple(group)

        elif value in group:
            in_group = (value, )

        else:
            not_in_group = (value, )

    else:
        value_set = set(map(str.lower, value))

        if wildcard.lower() in value_set:
            in_group = tuple(group)

        else:
            in_group = tuple(value_set.intersection(group))
            not_in_group = tuple(value_set.difference(group))

    return in_group, not_in_group


def _get_feat_mtds_from_class(class_obj: t.Callable) -> t.List[TypeMtdTuple]:
    """Get feature-extraction related methods from a given Class.

    Is assumed that methods related with feature extraction are prefixed
    with :obj:`MTF_PREFIX` value.

    Args:
        class_obj (:obj:`Class`): Class from which the feature methods
            should be extracted.

    Returns:
        list(tuple): a list of tuples in the form (`mtd_name`,
        `mtd_address`) which contains all methods associated with
        feature extraction (prefixed with :obj:`MTF_PREFIX`).
    """
    feat_mtd_list = inspect.getmembers(
        class_obj, predicate=inspect.ismethod)  # type: t.List[TypeMtdTuple]

    # It is assumed that all feature-extraction related methods
    # name are all prefixed with "MTF_PREFIX".
    feat_mtd_list = [
        ft_method for ft_method in feat_mtd_list
        if ft_method[0].startswith(MTF_PREFIX)
    ]

    return feat_mtd_list


def _get_all_ft_mtds() -> t.Dict[str, t.List[TypeMtdTuple]]:
    """Get all feature-extraction related methods in prefefined Classes.

    Feature-extraction methods are prefixed with :obj:`MTF_PREFIX` from all
    Classes predefined in :obj:`VALID_MFECLASSES` tuple.

    Returns:
        dict: in the form {`group_name`: [(`mtd_name`, `mtd_address`)]},
        i.e. the keys are the names of feature groups (e.g. `general` or
        `landmarking`) and values are lists of tuples which first entry are
        feature-extraction related method names and the second entry are its
        correspondent address. For example:

            {
                `general`: [
                    (`ft_nr_num`, <mtd_address>),
                    (`ft_nr_inst`, <mtd_address>),
                    ...
                ],

                `statistical`: [
                    (`ft_mean`, <mtd_address>),
                    (`ft_max`, <mtd_address>),
                    ...
                ],

                ...
            }
    """
    feat_mtd_dict = {
        ft_type_id: _get_feat_mtds_from_class(mfe_class)
        for ft_type_id, mfe_class in zip(VALID_GROUPS, VALID_MFECLASSES)
    }  # type: t.Dict[str, t.List[TypeMtdTuple]]

    return feat_mtd_dict


def _filter_mtd_dict(
        ft_mtds_dict: t.Dict[str, t.List[TypeMtdTuple]],
        groups: t.Optional[t.Tuple[str, ...]]) -> t.Tuple[TypeMtdTuple, ...]:
    """Filter return of `_get_all_ft_mtds(...)` function based on given `groups`.

    This is an auxiliary function for `process_features(...)` function.

    Args:
        ft_mtds_dict (:obj:`Dict`): return from `_get_all_ft_mtds(...)`
            function.

        groups (:obj:`Tuple` of :obj:`str`): a tuple of feature group names. It
        can assume value :obj:`None`, which is interpreted as ``no filter``
        (i.e. all features of all groups will be returned).

    Returns:
        tuple(str): containing only values of input `ft_mtds_dict` related
        to the given `groups`.
    """

    if groups:
        groups = tuple(set(groups).intersection(ft_mtds_dict.keys()))

        ft_mtds_filtered = operator.itemgetter(*groups)(ft_mtds_dict)

        if len(groups) == 1:
            ft_mtds_filtered = (ft_mtds_filtered, )

    else:
        ft_mtds_filtered = tuple(ft_mtds_dict.values())

    ft_mtds_filtered = tuple(
        mtd_tuple for ft_group in ft_mtds_filtered for mtd_tuple in ft_group)

    return ft_mtds_filtered


def _preprocess_ft_arg(features: t.Union[str, t.Iterable[str]]) -> t.List[str]:
    """Process `features` to a canonical form.

    Remove repeated elements from a collection of features and cast all values
    to lower-case.

    Args:
        features (:obj:`Iterable` of :obj:`str` or :obj:`str`): feature names
            or a collection of to be processed into a lower-case form.

    Returns:
        list(str): `features` values as iterable. The values within strings
            all lower-cased.
    """
    if isinstance(features, str):
        features = {features}

    return list(map(str.lower, set(features)))


def _extract_mtd_args(ft_mtd_callable: t.Callable) -> t.Tuple[str, ...]:
    """Extracts arguments from given method.

    Args:
        ft_mtd_callable (:obj:`Callable`): a callable related to a feature
            extraction method.

    Returns:
        list(str): containing the name of arguments of `ft_mtd_callable`.

    Raises:
        TypeError: if 'ft_mtd_callable' is not a valid Callable.
    """
    ft_mtd_signature = inspect.signature(ft_mtd_callable)
    mtd_callable_args = tuple(ft_mtd_signature.parameters.keys())
    return mtd_callable_args


def summarize(
        features: t.Union[np.ndarray, t.Sequence],
        callable_sum: t.Callable,
        callable_args: t.Optional[t.Dict[str, t.Any]] = None,
        remove_nan: bool = True,
        ) -> t.Union[t.Sequence, TypeNumeric]:
    """Returns feature summarized by `callable_sum`.

    Args:
        features (:obj:`Sequence` of numerics): Sequence containing values
            to summarize.

        callable_sum (:obj:`Callable`): Callable of the method which im-
            plements the desired summary function.

        callable_args (:obj:`Dict`, optional): arguments to the summary
            function. The expected dictionary format is the following:
            {`argument_name`: value}. In order to know the summary func-
            tion arguments you need to check out the documentation of
            the method which implements it.

        remove_nan (:obj:`bool`, optional): check and remove all elements
            in `features` which are not numeric. Note that :obj:`np.inf`
            is still considered numeric (:obj:`float` type).

    Returns:
        float: value of summarized feature values if possible. May
        return :obj:`np.nan` if summary function call invokes TypeError.

    Raises:
        AttributeError: if `callable_sum` is invalid.
        TypeError: if `features`  is not a sequence.
    """
    processed_feat = np.array(features)

    if remove_nan:
        numeric_vals = list(map(isnumeric, features))
        processed_feat = processed_feat[numeric_vals]
        processed_feat = processed_feat.astype(np.float32)

    if callable_args is None:
        callable_args = {}

    try:
        metafeature = callable_sum(processed_feat, **callable_args)

    except TypeError:
        metafeature = np.nan

    return metafeature


def get_feat_value(
        mtd_name: str,
        mtd_args: t.Dict[str, t.Any],
        mtd_callable: t.Callable,
        suppress_warnings: bool = False) -> t.Union[TypeNumeric, np.ndarray]:
    """Extract feat. from `mtd_callable` with `mtd_args` as args.

    Args:
        mtd_name (:obj:`str`): name of the feature-extraction method
            to be invoked.

        mtd_args (:obj:`Dic`): arguments of method to be invoked. The
            expected format of the arguments is {`argument_name`: value}.
            In order to know the method arguments available, you need to
            check its documentation.

        mtd_callable(:obj:`Callable`): callable of the feature-extra-
            ction method.

        suppress_warnings(:obj:`bool`): if True, all warnings invoked whi-
            before invoking the method (or after) will be ignored. The me-
            thod itself may still invoke warnings.

    Returns:
        numeric or array: return value of the feature-extraction method.

    Raises:
        AttributeError: if `mtd_callable` is not valid.
    """

    try:
        features = mtd_callable(**mtd_args)

    except (TypeError, ValueError) as type_e:
        if not suppress_warnings:
            warnings.warn(
                "Error extracting {0}: \n{1}.\nWill set it "
                "as 'np.nan' for all summary functions.".format(
                    mtd_name, repr(type_e)), RuntimeWarning)

        features = np.nan

    return features


def build_mtd_kwargs(mtd_name: str,
                     mtd_args: t.Iterable[str],
                     inner_custom_args: t.Optional[t.Dict[str, t.Any]] = None,
                     user_custom_args: t.Optional[t.Dict[str, t.Any]] = None,
                     suppress_warnings: bool = False) -> t.Dict[str, t.Any]:
    """Build a `kwargs` (:obj:`Dict`) for a feature-extraction :obj:`Callable`.

    Args:
        mtd_name (:obj:`str`): name of the method.

        mtd_args (:obj:`Iterable` of :obj:`str`): Iterable containing
            the name of all arguments of the callable.

        inner_custom_args (:obj:`Dict`, optional): custom arguments for
            inner usage, for example, to pass ``X``, ``y`` or other user-
            independent arguments necessary for the callable. The expected
            format of this dict is {`argument_name`: value}.

        user_custom_args (:obj:`Dict`, optional): assumes the same model
            as the dict above, but this one is dedicated to keep user-dep-
            endent arguments for method callable, for example, number of
            bins of a histogram-like metafeature or degrees of freedom of
            a standard deviation-related metafeature. The name of the ar-
            guments must be verified in its correspondent method documen-
            tation.

        suppress_warnings(:obj:`bool`, optional): if True, will not show
            any warnings about unknown callable parameters.

    Returns:
        dict: a ready-to-use `kwargs` for the correspondent callable. The
            format is {`argument_name`: value}.
    """

    if user_custom_args is None:
        user_custom_args = {}

    if inner_custom_args is None:
        inner_custom_args = {}

    combined_args = {
        **user_custom_args,
        **inner_custom_args,
    }

    callable_args = {
        custom_arg: combined_args[custom_arg]
        for custom_arg in combined_args if custom_arg in mtd_args
    }

    if not suppress_warnings:
        unknown_arg_set = (unknown_arg
                           for unknown_arg in user_custom_args.keys()
                           if unknown_arg not in mtd_args
                           )  # type: t.Generator[str, None, None]

        for unknown_arg in unknown_arg_set:
            warnings.warn(
                'Unknown argument "{0}" for method "{1}".'.format(
                    unknown_arg, mtd_name), UserWarning)

    return callable_args


def check_summary_warnings(value: t.Union[TypeNumeric, t.Sequence, np.ndarray],
                           name_feature: str, name_summary: str) -> None:
    """Check if there is :obj:`np.nan` within summarized values.

    Args:
        value (numeric or :obj:`Sequence`): summarized values.

        name_feature (:obj:`str`): name of the feature-extraction
            method used to generate the values which was summarized.

        name_summary (:obj:`str`): name of the summary method
            used to produce `value`.
    """

    if not isinstance(value, collections.Iterable):
        value = [value]

    if any(np.isnan(value)):
        warnings.warn(
            "Failed to summarize {0} with {1}. "
            "(generated NaN).".format(name_feature, name_summary),
            RuntimeWarning)


def process_groups(
        groups: t.Union[t.Iterable[str], str],
        wildcard: str = "all") -> t.Tuple[str, ...]:
    """Process `groups` argument from MFE.__init__ to generate internal metadata.

    Args:
        groups (:obj:`str` or :obj:`t.Iterable` of :obj:`str`): a single
            string or a iterable with group identifiers to be processed.
            Check out ``MFE`` Class documentation for more information.

        wildcard (:obj:`str`): value to be used as ``select all`` value.

    Returns:
        tuple(str): containing all valid group lower-cased identifiers.

    Raises:
        TypeError: if `groups` is neither a string `all` nor a Iterable
            containing valid group identifiers as strings.

        ValueError: if `groups` is None or is a empty Iterable or if a unknown
            group identifier is given.
    """
    if not groups:
        raise ValueError('"Groups" can not be None nor empty.')

    in_group, not_in_group = _check_value_in_group(
        value=groups,
        group=VALID_GROUPS,
        wildcard=wildcard)

    if not_in_group:
        raise ValueError("Unknown groups: {0}. "
                         "Please select values in {1}.".format(
                             not_in_group, VALID_GROUPS))

    return in_group


def process_summary(
        summary: t.Union[str, t.Iterable[str]],
        wildcard: str = "all"
        ) -> t.Tuple[t.Tuple[str, ...], t.Tuple[TypeExtMtdTuple, ...]]:
    """Process `summary` argument from MFE.__init__ to generate internal metadata.

    Args:
        summary (:obj:`t.Iterable` of :obj:`str` or a :obj:`str`): a
            summary function or a list of these, which are used to
            combine different calculations of the same metafeature. Check
            ``MFE`` Class documentation for more information about this
            parameter.

        wildcard (:obj:`str`): value to be used as ``select all`` value.

    Returns:
        tuple(tuple, tuple): the first field contains all valid lower-cased
            summary function names, where the last field contains internal
            metadata about methods which implements each summary function.
            This last tuple model is:

                (
                    `summary_mtd_name`,
                    `summary_mtd_callable`,
                    `summary_mtd_args`,
                )

    Raises:
        TypeError: if `summary` is not :obj:`None`, empty, a valid string
            nor a Iterable containing valid group identifiers as strings.
    """
    if not summary:
        return tuple(), tuple()

    in_group, not_in_group = _check_value_in_group(
        value=summary,
        group=VALID_SUMMARY,
        wildcard=wildcard)

    if not_in_group:
        raise ValueError("Unknown summary: {0}. "
                         "Please select values in {1}.".format(
                             not_in_group, VALID_SUMMARY))

    summary_methods = []  # type: t.List[TypeExtMtdTuple]
    available_sum_methods = []  # type: t.List[str]

    for summary_func in in_group:
        summary_mtd_callable = _summary.SUMMARY_METHODS.get(summary_func)

        if not summary_mtd_callable:
            warnings.warn("Missing summary function "
                          "{0} at _summary module.".format(summary_func),
                          RuntimeWarning)
        else:
            try:
                summary_mtd_args = _extract_mtd_args(summary_mtd_callable)

            except ValueError:
                summary_mtd_args = tuple()

            summary_mtd_pack = (
                summary_func,
                summary_mtd_callable,
                summary_mtd_args,
            )

            summary_methods.append(summary_mtd_pack)
            available_sum_methods.append(summary_func)

    return tuple(available_sum_methods), tuple(summary_methods)


def process_features(
        features: t.Union[str, t.Iterable[str]],
        groups: t.Optional[t.Tuple[str, ...]] = None,
        wildcard: str = "all",
        suppress_warnings: bool = False
        ) -> t.Tuple[t.Tuple[str, ...], t.Tuple[TypeExtMtdTuple, ...]]:
    """Process `features` argument from MFE.__init__ to generate internal metadata.

    This function is expected to be used after `process_groups` function,
    as `groups` parameter is expected to be in a canonical form (lower-cased
    values inside a tuple).

    Args:
        features (:obj:`Iterable` of `str` or `str`): Iterable containing a
            collection of features or a string describing a single feature. No-
            te that only features that are in the given `groups` will be retur-
            ned.

        groups (:obj:`Tuple` of :obj:`str`, optional): collection containing
            one or more group identifiers. Check out ``MFE`` class documenta-
            tion for more information.

        wildcard (:obj:`str`): value to be used as `select all` for `features`
            argument.

    Returns:
        tuple(tuple, tuple): A pair of tuples. The first Tuple is all feature
        names extracted from this method, in order to give to the user an easy
        access to available features in model. The second field is a tuple for
        internal usage, containing metadata in the form of tuples in the follo-
        wing format: (`mtd_name`, `mtd_callable`, `mtd_args`), i.e., the first
        tuple item field is a string containing the name of a feature-extracti-
        on related method, and the second field is a callable object for the
        corresponding method, and the third is the method arguments.

    Raises:
        ValueError: if features is :obj:`None` or is empty.
    """

    if not features:
        raise ValueError('"features" can not be None nor empty.')

    processed_ft = _preprocess_ft_arg(features)  # type: t.List[str]

    ft_mtds_filtered = _filter_mtd_dict(
        _get_all_ft_mtds(), groups)  # type: t.Tuple[TypeMtdTuple, ...]

    if wildcard in processed_ft:
        processed_ft = [
            remove_mtd_prefix(mtd_name) for mtd_name, _ in ft_mtds_filtered
        ]

    available_feat_names = []  # type: t.List[str]
    ft_mtd_processed = []  # type: t.List[TypeExtMtdTuple]

    for ft_mtd_tuple in ft_mtds_filtered:
        ft_mtd_name, ft_mtd_callable = ft_mtd_tuple

        mtd_name_without_prefix = remove_mtd_prefix(ft_mtd_name)

        if mtd_name_without_prefix in processed_ft:
            mtd_callable_args = _extract_mtd_args(ft_mtd_callable)

            extended_item = (*ft_mtd_tuple,
                             mtd_callable_args)  # type: TypeExtMtdTuple

            ft_mtd_processed.append(extended_item)
            available_feat_names.append(mtd_name_without_prefix)
            processed_ft.remove(mtd_name_without_prefix)

    if not suppress_warnings:
        for unknown_ft in processed_ft:
            warnings.warn('Unknown feature "{0}"'.format(unknown_ft),
                          UserWarning)

    return tuple(available_feat_names), tuple(ft_mtd_processed)


def process_timeopt(timeopt: t.Optional[str]):

    if timeopt is None:
        return None

    if not isinstance(timeopt, str):
        raise TypeError("Time argument must be "
                        "string type or None (got {}).".format(type(timeopt)))

    timeopt = timeopt.lower()

    if timeopt not in VALID_TIMEOPT:
        raise ValueError('Invalid time option "{0}". Please choose one '
                         "amongst {1} or None.".format(timeopt, VALID_TIMEOPT))

    return timeopt


def check_data(X: t.Union[np.ndarray, list], y: t.Union[np.ndarray, list]
               ) -> t.Tuple[np.ndarray, np.ndarray]:
    """Checks received `X` and `y` data type and shape.

    Args:
        Check `mfe.fit` method for more information.

    Raises:
        TypeError: if `X` or `y` is neither a np.ndarray nor a list-
        type object.

    Returns:
        tuple(np.ndarray, np.ndarray): X and y possibly reshaped and
        casted to np.ndarray type.
    """
    if not isinstance(X, (np.ndarray, list)):
        raise TypeError('"X" is neither "list" nor "np.array".')

    if not isinstance(y, (np.ndarray, list)):
        raise TypeError('"y" is neither "list" nor "np.array".')

    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    y = y.flatten()

    if len(X.shape) == 1:
        X = X.reshape(*X.shape, -1)

    if X.shape[0] != y.shape[0]:
        raise ValueError('"X" number of rows and "y" '
                         "length shapes do not match.")

    return X, y


def isnumeric(
        value: t.Any,
        check_subtype: bool = True) -> bool:
    """Checks if `value` is a Numeric Type or a collection of Numerics.

    The ``Numeric Type`` is assumed to be one of the following:
        1. :obj:`int`
        2. :obj:`float`
        3. :obj:`np.number`

    Args:
        value (:obj:`Any`): any object to be checked as numeric or a
            collection of numerics.

        check_subtype (:obj:`bool`, optional): if True, check elements of
            ``value`` if it is a Iterable object. Otherwise, only checks
            ``value`` type ignoring the fact that it can be a Iterable ob-
            ject.

    Returns:
        bool: True if `value` is a numeric type object or a collection
            of numeric-only elements. False otherwise.
    """
    if (check_subtype
            and isinstance(value, (collections.Iterable, np.ndarray))
            and not isinstance(value, str)):

        value = np.array(value)

        if value.size == 0:
            return False

        return all(isinstance(x, _TYPE_NUMERIC) for x in value)

    return isinstance(value, _TYPE_NUMERIC)


def remove_mtd_prefix(mtd_name: str) -> str:
    """Remove feature-extraction method prefix from its name.

    The predefined prefix is stored in :obj:`MTF_PREFIX`.

    Args:
        mtd_name (:obj:`str`): method name prefixed with value
            stored in :obj:`MTF_PREFIX`.

    Returns:
        str: method name without prefix.

    Raises:
        TypeError: if `mtd_name` is not a string.
    """
    if mtd_name.startswith(MTF_PREFIX):
        return mtd_name[len(MTF_PREFIX):]

    return mtd_name


def timeit(func: t.Callable, **kwargs) -> t.Tuple[t.Any, float]:
    """Measure how much time is for calling ``func`` with ``args``.

    Args:
        func (:obj:`Callable`): a callable which invokation time will be
            measured from.

        **kwargs: arguments for ``func``.

    Return:
        tuple[any, float]: the first element is the return value from
            ``func``. The second argument is the time necessary for a
            complement invokation of ``func``.

    Raises:
        Any exception raised by ``func`` with arguments ``args`` is not
        catched by this method.
    """
    t_start = time.time()
    ret_val = func(**kwargs)
    time_total = time.time() - t_start
    return ret_val, time_total
