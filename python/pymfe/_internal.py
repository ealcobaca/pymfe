"""Provides useful functions for MFE package.

Attributes:
    VALID_VALUE_PREFIX (:obj:`str`): Prefix which all tuples that
        keeps valid values for custom user options must use in its
        name. This is used to enable automatic detectation of these
        groups.

    VALID_GROUPS (:obj:`tuple` of :obj:`str`): Supported groups of
        metafeatures of pymfe.

    VALID_SUMMARY (:obj:`tuple` of :obj:`str`): Supported summary
        functions to combine metafeature values.

    VALID_MFECLASSES (:obj:`tuple` of Classes): Metafeature ex-
        tractors predefined classes, where metafeature-extraction
        methods will be searched.

    VALID_TIMEOPT (:obj:`tuple` of :obj:`str`): valid options for
        time measurements while extracting metafeatures.

    VALID_RESCALE (:obj:`tuple` of :obj:`str`): valid options for
        rescaling numeric data while fitting dataset.

    MTF_PREFIX (:obj:`str`): prefix of metafeature-extraction me-
        thod names for classes in ``VALID_MFECLASSES``. For exam-
        ple, the metafeature called ``inst_nr`` is implemented in
        the method named `[MTF_PREFIX]_inst_nr`. This is used to
        enable automatic detection of these methods.

    PRECOMPUTE_PREFIX (:obj:`str`): prefix for precomputation me-
        thod names. If a method of a class in ``VALID_MFECLASSES``
        starts with this prefix, it will be automatically executed
        to gather values that this class frequently uses. These
        values will be shared between all feature-extraction related
        methods of all ``VALID_MFECLASSES`` classes to avoid redun-
        dant computation.

    TIMEOPT_AVG_PREFIX (:obj:`str`): prefix for time options ba-
        sed on average of gathered metrics. It means necessarily
        that, if an option is prefixed with this constant value,
        then it is supposed that the gathered time elapsed metri-
        cs must be divided by the cardinality of the features ex-
        tracted (``cardinality`` means ``number of``).

    TIMEOPT_SUMMARY_SUFIX (:obj:`str`): sufix for time options
        which include summarization time alongside the time ne-
        cessary for the extraction of the feature. It means that,
        if an time option is sufixed with this constant value,
        then the time metrics must include the time necessary
        for the summarization of each value with cardinality gre-
        ater than one.
"""
import typing as t
import inspect
import collections
import operator
import warnings
import time
import sys

import numpy as np
import sklearn.preprocessing
import patsy

import _summary
import general
import statistical
import info_theory
import landmarking
import model_based

VALID_VALUE_PREFIX = "VALID_"

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

_RESCALE_SCALERS = {
    "standard": sklearn.preprocessing.StandardScaler,
    "min-max": sklearn.preprocessing.MinMaxScaler,
    "robust": sklearn.preprocessing.RobustScaler,
}

VALID_RESCALE = (*_RESCALE_SCALERS, )

TIMEOPT_AVG_PREFIX = "avg"

TIMEOPT_SUMMARY_SUFIX = "summ"

MTF_PREFIX = "ft_"

PRECOMPUTE_PREFIX = "precompute"

TypeMtdTuple = t.Tuple[str, t.Callable[[], t.Any]]
"""Type annotation which describes the a metafeature method tuple."""

TypeExtMtdTuple = t.Tuple[str, t.Callable[[], t.Any], t.Sequence]
"""Type annotation which extends TypeMtdTuple with extra field (for 'Args')"""

_TYPE_NUMERIC = (
    int,
    float,
    np.number,
)
"""Tuple with generic numeric types."""

TypeNumeric = t.TypeVar(
    "TypeNumeric",
    int,
    float,
    np.number,
)
"""Typing alias for both numeric types."""


def _check_values_in_group(value: t.Union[str, t.Iterable[str]],
                           valid_group: t.Iterable[str],
                           wildcard: t.Optional[str] = "all"
                           ) -> t.Tuple[t.Tuple[str, ...], t.Tuple[str, ...]]:
    """Checks if a value is in a set or a set of values is a subset of a set.

    Args:
        value (:obj:`Iterable` of :obj:`str` or :obj:`str): value(s) to be
            checked if are in the given valid_group of strings.

        valid_group (:obj:`Iterable` of :obj:`str`): a valid_group of strings
            representing the values such that `value` will be verified against.

        wildcard (:obj:`str`, optional): a value which represent 'all values'.
            The case is ignored, so, for example, both values 'all', 'ALL' and
            any mix of cases are considered to be the same wildcard token.

    Returns:
        tuple(tuple, tuple): A pair of tuples containing, respectively, values
        that are in the given valid_group and those that are not. If no value
        is in either valid_group, then this valid_group will be :obj:`None`.

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
        if wildcard and value == wildcard.lower():
            in_group = tuple(valid_group)

        elif value in valid_group:
            in_group = (value, )

        else:
            not_in_group = (value, )

    else:
        value_set = set(map(str.lower, value))

        if wildcard and wildcard.lower() in value_set:
            in_group = tuple(valid_group)

        else:
            in_group = tuple(value_set.intersection(valid_group))
            not_in_group = tuple(value_set.difference(valid_group))

    return in_group, not_in_group


def _get_prefixed_mtds_from_class(class_obj: t.Any,
                                  prefix: str) -> t.List[TypeMtdTuple]:
    """Get all class methods from ``class_obj`` prefixed with ``prefix``.

    Args:
        class_obj (:obj:`Class`): Class from which the methods should be
            extracted.

        prefix (:obj:`str`): prefix which method names must have in order
            to it be gathered.

    Returns:
        list(tuple): a list of tuples in the form (`mtd_name`, `mtd_address`)
            of all class methods from ``class_obj`` prefixed with ``prefix``.
    """
    feat_mtd_list = inspect.getmembers(
        class_obj, predicate=inspect.ismethod)  # type: t.List[TypeMtdTuple]

    # It is assumed that all feature-extraction related methods
    # name are all prefixed with "MTF_PREFIX".
    feat_mtd_list = [
        ft_method for ft_method in feat_mtd_list
        if ft_method[0].startswith(prefix)
    ]

    return feat_mtd_list


def _filter_mtd_dict(
        ft_mtds_dict: t.Dict[str, t.List[TypeMtdTuple]],
        groups: t.Optional[t.Tuple[str, ...]]) -> t.Tuple[TypeMtdTuple, ...]:
    """Filter return of `_get_all_prefixed_mtds` function based on given `groups`.

    This is an auxiliary function for ``process_features`` function.

    Args:
        ft_mtds_dict (:obj:`Dict`): return from ``_get_all_prefixed_mtds``
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


def _get_all_prefixed_mtds(
        prefix: str, groups: str) -> t.Dict[str, t.List[TypeMtdTuple]]:
    """Get all feature-extraction related methods in predefined Classes.

    Feature-extraction methods are prefixed with ``prefix`` from all Clas-
    ses predefined in :obj:`VALID_MFECLASSES` tuple.

    Args:
        prefix (:obj:`str`): prefix which method names must have in order
            to it be gathered.

        groups (:obj:`Tuple` of :obj:`str`): a tuple of feature group names.
        It can assume value :obj:`NoneType`, which is interpreted as ``no
        filter`` (i.e. all features of all groups will be returned).

    Returns:
        dict: in the form {`group_name`: [(`mtd_name`, `mtd_address`)]},
        i.e. the keys are the names of feature groups (e.g. `general` or
        `landmarking`) and values are lists of tuples which first entry are
        feature-extraction related method names. The second entries are
        their correspondent addresses. For example:

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
    groups = tuple(set(VALID_GROUPS).intersection(groups))

    if not groups:
        return {}

    feat_mtd_dict = {
        ft_type_id: _get_prefixed_mtds_from_class(
            class_obj=mfe_class,
            prefix=prefix)
        for ft_type_id, mfe_class in zip(groups, VALID_MFECLASSES)
    }  # type: t.Dict[str, t.List[TypeMtdTuple]]

    return _filter_mtd_dict(ft_mtds_dict=feat_mtd_dict, groups=groups)


def _preprocess_iterable_arg(
        features: t.Union[str, t.Iterable[str]]) -> t.List[str]:
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

    except (TypeError, ValueError, ZeroDivisionError):
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

    except (TypeError, ValueError, ZeroDivisionError) as type_e:
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
                     precomp_args: t.Optional[t.Dict[str, t.Any]] = None,
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

        precomp_args (:obj:`Dict`, optional): precomputed cached arguments
            which may be used for the feature-extraction method to speed
            up its calculations.

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

    if precomp_args is None:
        precomp_args = {}

    combined_args = {
        **user_custom_args,
        **inner_custom_args,
        **precomp_args,
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


def process_generic_set(
        values: t.Optional[t.Union[t.Iterable[str], str]],
        group_name: str,
        wildcard: t.Optional[str] = "all",
        allow_none: bool = False,
        allow_empty: bool = False,
        ) -> t.Tuple[str, ...]:
    """Check if given ``values`` are in an internal valid set named ``group_name``.

    Args:
        wildcard (:obj:`str`, optional): special value to ``accept any value``.

        group_name (:obj:`str`, optional): name of which internal group ``va-
            lues`` should be searched inside. Please check this module Attri-
            bute documentation in order to verify which groups are available
            for valid options. They are always prefixed with ``VALID_GROUPS_-
            PREFIX``, and this parameter must be the name of the group without
            its prefix. For example, to select ``VALID_CLASSES`` group for
            ``values`` reference, then group_names must be just ``classes``.

        allow_none (:obj:`bool`, optional): if True, then :obj:`NoneType` is
            a accepted as ``values``. Note that, if ``values`` is an Iterable,
            it does not mean that :obj:`NoneType` will become a valid value wi-
            thin, but ``values`` can assume value :obj:`NoneType`.

        allow_empty (:obj:`bool`, optional): if True, then ``values`` can be an
            zero-length iterable.

    Return:
        tuple(str): lower-cased tuple with all valid values.

    Raises:
        TypeError: if ``group_name`` is :obj:`NoneType`.
        ValueError: These are the conditions for raising this exception:
            - Some element in ``values`` is a valid value (not in the
                selected valid values based in ``group_name`` argument).

            - ``values`` is None and ``allow_none`` is False.

            - ``values`` is a empty sequence and ``allow_empty`` is False.

            - ``group_name`` is ``summary`` or ``features``, as both of
                these groups have their own special function to process
                user custom arguments (check ``process_features`` and
                ``process_summary`` for more information).

            - ``group_names`` is not a valid group for ``values`` reference.
    """
    if not group_name:
        raise TypeError('"group_name" can not be empty or None.')

    if values is None:
        if allow_none:
            return tuple()

        raise ValueError('"Values" can not be None. (while checking '
                         'group "{}").'.format(group_name))

    if values is not None and not values:
        if allow_empty:
            return tuple()

        raise ValueError('"Values" can not be empty. (while checking '
                         'group "{}")'.format(group_name))

    if group_name.upper() in ("SUMMARY", "FEATURES"):
        raise ValueError('Forbidden "group_name" option ({}). There is a '
                         "specify processing method for it".format(group_name))

    _module_name = sys.modules[__name__]

    try:
        valid_values = inspect.getattr_static(
            _module_name, "{0}{1}".format(VALID_VALUE_PREFIX,
                                          group_name.upper()))
    except AttributeError:
        raise ValueError('Invalid "group_name" "{}". Check _internal '
                         "module documentation to verify which ones "
                         "are available for use.".format(group_name))

    in_valid_set, not_in_valid_set = _check_values_in_group(
        value=values,
        valid_group=valid_values,
        wildcard=wildcard)

    if not_in_valid_set:
        raise ValueError("Unknown values: {0}. "
                         "Please select values in {1}.".format(
                             not_in_valid_set, valid_values))

    return in_valid_set


def process_generic_option(
        value: t.Optional[str],
        group_name: str,
        allow_none: bool = False,
        allow_empty: bool = False,
        ) -> t.Optional[str]:
    """Check if given ``value`` is in an internal reference group of values.

    This function is essentially a wrapper for the ``process_generic_set``
    function, with some differences:

        - Only string-typed values are accepted, with the exception that
            it can also assume :obj:`NoneType` if ``allow_none`` is True.

        - The return value is not a tuple, but instead a lower-cased ver-
            sion of ``value``.

    Check ``process_generic_set`` for more precise information about this
    process.

    Return:
        str: lower-cased version of ``value``.

    Raises:
        TypeError: if value is neither :obj:`NoneType` (and ``allow_none`` is
            also True) nor a :obj:`str` type object.

        All exceptions from ``process_generic_set`` are also raised, with the
        same conditions as described in that function documentation.
    """

    if value is not None and not isinstance(value, str):
        raise TypeError('"value" (group name {}) must be a string-'
                        "type object (got {}).".format(group_name,
                                                       type(value)))

    processed_value = process_generic_set(
        values=value,
        group_name=group_name,
        wildcard=None,
        allow_none=allow_none,
        allow_empty=allow_empty)

    canonical_value = None

    if processed_value:
        canonical_value = processed_value[0]

        if not isinstance(canonical_value, str):
            canonical_value = None

    return canonical_value


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

    in_group, not_in_group = _check_values_in_group(
        value=summary,
        valid_group=VALID_SUMMARY,
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

    ft_mtds_filtered = _get_all_prefixed_mtds(
        prefix=MTF_PREFIX,
        groups=groups)  # type: t.Tuple[TypeMtdTuple, ...]

    processed_ft = _preprocess_iterable_arg(features)  # type: t.List[str]

    if wildcard in processed_ft:
        processed_ft = [
            remove_prefix(value=mtd_name, prefix=MTF_PREFIX)
            for mtd_name, _ in ft_mtds_filtered
        ]

    available_feat_names = []  # type: t.List[str]
    ft_mtd_processed = []  # type: t.List[TypeExtMtdTuple]

    for ft_mtd_tuple in ft_mtds_filtered:
        ft_mtd_name, ft_mtd_callable = ft_mtd_tuple

        mtd_name_without_prefix = remove_prefix(
            value=ft_mtd_name,
            prefix=MTF_PREFIX)

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


def process_precomp_groups(
        precomp_groups: t.Union[str, t.Iterable[str]],
        groups: t.Optional[t.Tuple[str, ...]] = None,
        wildcard: str = "all",
        suppress_warnings: bool = False,
        **kwargs
        ) -> t.Dict[str, t.Any]:
    """Process `precomp_groups` argument while fitting into a MFE model.

    This function is expected to be used after `process_groups` function,
    as `groups` parameter is expected to be in a canonical form (lower-cased
    values inside a tuple).

    Args:
        precomp_groups (:obj:`Iterable` of `str` or `str`): a single or a se-
            quence of metafeature group names whose precomputation methods
            should be taken. Note that any group not in ``groups`` (see argu-
            ment below) is completely ignored.

        groups (:obj:`Tuple` of :obj:`str`, optional): collection containing
            one or more group identifiers. Check out ``MFE`` class documenta-
            tion for more information.

        wildcard (:obj:`str`, optional): value to be used as ``select all``
            for ``precompute`` argument.

        suppress_warnings (:obj:`bool`, optional): if True, suppress warnings
            invoked while processing precomputation option.

        **kwargs: used to pass extra custom arguments to precomputation metho-
            ds.

    Returns:
        dict:
    """
    if not precomp_groups:
        return {}

    processed_precomp_groups = _preprocess_iterable_arg(
        precomp_groups)  # type: t.List[str]

    if wildcard in processed_precomp_groups:
        processed_precomp_groups = groups

    else:
        processed_precomp_groups = tuple(
            set(processed_precomp_groups).intersection(groups))

        if not suppress_warnings:
            unknown_groups = processed_precomp_groups.difference(groups)

            for unknown_precomp in unknown_groups:
                warnings.warn('Unknown precomp_groups "{0}"'.format(
                        unknown_precomp), UserWarning)

    precomp_mtds_filtered = _get_all_prefixed_mtds(
        prefix=PRECOMPUTE_PREFIX,
        groups=processed_precomp_groups)  # type: t.Tuple[TypeMtdTuple, ...]

    precomp_items = {}

    for precomp_mtd_tuple in precomp_mtds_filtered:
        _, precomp_mtd_callable = precomp_mtd_tuple

        new_precomp_vals = precomp_mtd_callable(**kwargs)

        if new_precomp_vals:
            precomp_items = {
                **precomp_items,
                **new_precomp_vals,
            }

    return precomp_items


def check_data(X: t.Union[np.ndarray, list],
               y: t.Union[np.ndarray, list]
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

    return np.copy(X), np.copy(y)


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


def remove_prefix(value: str, prefix: str) -> str:
    """Remove ``prefix`` from ``value``.

    The predefined prefix is stored in :obj:`prefix`.

    Args:
        value (:obj:`str`): method name prefixed with value stored in
            ``prefix``.

    Returns:
        str: method name without prefix.

    Raises:
        TypeError: if `value` is not a string.
    """
    if value.startswith(prefix):
        return value[len(prefix):]

    return value


def timeit(func: t.Callable, *args) -> t.Tuple[t.Any, float]:
    """Measure how much time is for calling ``func`` with ``args``.

    Args:
        func (:obj:`Callable`): a callable which invokation time will be
            measured from.

        *args: arguments for ``func``.

    Return:
        tuple[any, float]: the first element is the return value from
            ``func``. The second argument is the time necessary for a
            complement invokation of ``func``.

    Raises:
        Any exception raised by ``func`` with arguments ``args`` is not
        catched by this method.
    """
    t_start = time.time()
    ret_val = func(*args)
    time_total = time.time() - t_start
    return ret_val, time_total


def _unused_transform_cat(
        data_categoric: np.ndarray
        ) -> t.Optional[np.ndarray]:
    """One Hot Encoding (Binarize) given categorical data.

    Currently unused.
    """
    if data_categoric.size == 0:
        return None

    label_enc = sklearn.preprocessing.LabelEncoder()
    hot_enc = sklearn.preprocessing.OneHotEncoder(sparse=False)

    data_numeric = np.apply_along_axis(
        func1d=label_enc.fit_transform,
        axis=0,
        arr=data_categoric)

    num_row, _ = data_categoric.shape

    dummies_vars = np.empty((num_row, 0), float)
    for column in data_numeric.T:
        new_dummies = hot_enc.fit_transform(column.reshape(-1, 1))
        dummies_vars = np.concatenate((dummies_vars, new_dummies), axis=1)

    return dummies_vars


def transform_cat(data_categoric: np.ndarray) -> t.Optional[np.ndarray]:
    """To do."""
    if data_categoric.size == 0:
        return None

    _, num_col = data_categoric.shape

    dummy_attr_names = [
        "C{}".format(i) for i in range(num_col)
    ]

    named_data = {
        attr_name: data_categoric[:, attr_index]
        for attr_index, attr_name in enumerate(dummy_attr_names)
    }

    formula = "~ 0 + {}".format(" + ".join(dummy_attr_names))

    return np.asarray(patsy.dmatrix(formula, named_data))


def _equal_freq_discretization(data: np.ndarray, num_bins: int) -> np.ndarray:
    """Discretize a 1-D numeric array into a equal-frequency histogram."""
    perc_interval = int(100.0 / num_bins)
    perc_range = range(perc_interval, 100, perc_interval)
    hist_divs = np.percentile(data, perc_range)

    if hist_divs.size == 0:
        hist_divs = [np.median(data)]

    return np.digitize(data, hist_divs, right=True)


def transform_num(data_numeric: np.ndarray,
                  num_bins: t.Optional[int] = None) -> t.Optional[np.ndarray]:
    """Discretize numeric data with a equal-frequency histogram.

    The numeric values will be overwritten by the index of the his-
    togram bin which each value will fall into.

    Args:
        data_numeric (:obj:`np.ndarray`): 2-D numpy array of numeric-
            only data to be discretized.

        num_bins (:obj:`int`, optional): number of bins of the equal-
            frequency histogram used to discretize the data. If no
            value is given, then the default value is min(2, c), where
            ``c`` is the cubic root of number of instances rounded down.

    Returns:
        np.ndarray: discretized version of ``data_numeric``.

    Raises:
        TypeError: if num_bins isn't :obj:`int`.
        ValueError: if num_bins is a non-positive value.
    """
    if data_numeric.size == 0:
        return None

    if num_bins is not None:
        if not isinstance(num_bins, int):
            raise TypeError('"num_bins" must be integer or NoneType.')

        if num_bins <= 0:
            raise ValueError('"num_bins" must be a positive'
                             "integer or NoneType.")

    num_inst, _ = data_numeric.shape

    if not num_bins:
        num_bins = int(num_inst**(1/3))

    data_numeric = data_numeric.astype(float)

    digitalized_data = np.apply_along_axis(
        func1d=_equal_freq_discretization,
        axis=0,
        arr=data_numeric,
        num_bins=num_bins)

    return digitalized_data


def rescale_data(data: np.ndarray,
                 option: str,
                 args: t.Optional[t.Dict[str, t.Any]] = None) -> np.ndarray:
    """Rescale numeric fitted data accordingly to user select option.

    Args:
        data (:obj:`np.ndarray`): data to be rescaled.

        option (:obj:`str`): rescaling strategy. Must be one in
            ``VALID_RESCALE`` attribute.

        args (:obj:`dict`, optional): extra arguments for scaler. All
            scaler used are from ``sklearn`` package, so you should
            consult they documentation for a complete list of available
            arguments to user costumization. The used scalers for each
            available ``option`` are:

                ``min-max``: ``sklearn.preprocessing.MinMaxScaler``
                ``standard``: ``sklearn.preprocessing.StandardScale``
                ``robust``: ``sklearn.preprocessing.RobustScaler``

    Returns:
        np.ndarray: scaled ``data`` based in ``option`` correspondent
            strategy.

    Raises:
        ValueError: if ``option`` is not in ``VALID_RESCALE``.

        Any exception caused by arguments from ``args`` into the
        scaler model is also raised by this function.
    """
    if option not in VALID_RESCALE:
        raise ValueError('Unknown option "{0}". Please choose one '
                         "between {1}".format(option, VALID_RESCALE))

    if not args:
        args = {}

    scaler_model = _RESCALE_SCALERS.get(option, "min-max")(**args)

    return scaler_model.fit_transform(data)
