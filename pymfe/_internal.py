"""This module provides useful functions for the MFE package.

Attributes:
    VALID_VALUE_PREFIX (:obj:``str``): Prefix which all tuples that
        keep valid values for custom user options must use in its name.
        This prefix is used to enable the automatic detection of these
        groups.

    VALID_GROUPS (:obj:``tuple`` of :obj:``str``): Supported groups of
        metafeatures of pymfe.

    VALID_SUMMARY (:obj:``tuple`` of :obj:``str``): Supported summary
        functions to combine metafeature values.

    VALID_MFECLASSES (:obj:``tuple`` of Classes): Metafeature extractors
        predefined classes, where to perform the search of metafeature-ex-
        traction methods.

    VALID_TIMEOPT (:obj:``tuple`` of :obj:``str``): valid options for time
        measurements while extracting metafeatures.

    VALID_RESCALE (:obj:``tuple`` of :obj:``str``): valid options for res-
        caling numeric data while fitting dataset.

    MTF_PREFIX (:obj:``str``): prefix of metafeature-extraction method
        names for classes in ``VALID_MFECLASSES``. For example, the metafeature
        called ``inst_nr`` is implemented in the method named ``[MTF_PREFIX]_-
        inst_nr.`` Prefixation is used to enable the automatic detection of
        these methods.

    PRECOMPUTE_PREFIX (:obj:``str``): prefix for precomputation method names.
        If a method of a class in ``VALID_MFECLASSES`` starts with this prefix,
        it is automatically executed to gather values that this class frequen-
        tly uses. These values are shared between all feature-extraction rela-
        ted methods of all ``VALID_MFECLASSES`` classes to avoid redundant com-
        putation.

    TIMEOPT_AVG_PREFIX (:obj:``str``): prefix for time options based on the
        average of gathered metrics. It means necessarily that; if this cons-
        tant value prefixes an option, then this option is supposed to divide
        the gathered time elapsed metrics by the cardinality of the features
        extracted (``cardinality`` means ``number of``).

    TIMEOPT_SUMMARY_SUFFIX (:obj:``str``): suffix for time options which in-
        clude summarization time alongside the time necessary for the extracti-
        on of the feature. It means that, if this constant value suffixes a ti-
        me option, then the time metrics must include the time necessary for
        the summarization of each value with cardinality greater than one
        (``cardinality`` means ``number of values``).
"""
import typing as t
import inspect
import collections
import warnings
import time
import sys

import numpy as np
import sklearn.preprocessing
import patsy

import pymfe._summary as _summary
import pymfe.general as general
import pymfe.statistical as statistical
import pymfe.info_theory as info_theory
import pymfe.landmarking as landmarking
import pymfe.model_based as model_based
import pymfe.scoring as scoring

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

TIMEOPT_SUMMARY_SUFFIX = "summ"

MTF_PREFIX = "ft_"

PRECOMPUTE_PREFIX = "precompute_"

TypeMtdTuple = t.Tuple[str, t.Callable[[], t.Any]]
"""Type annotation which describes the a metafeature method tuple."""

TypeExtMtdTuple = t.Tuple[str, t.Callable[[], t.Any], t.Sequence]
"""Type annotation which extends TypeMtdTuple with extra field (``Args``)"""

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
"""Typing alias of generic numeric types for static code checking."""


def warning_format(message: str,
                   category: t.Type[Warning],
                   filename: str,
                   lineno: int,
                   line: str = None) -> str:
    """Change warnings format to a simpler one.

    Args:
        message (:obj:`str`): warning message to print.

        category: not used. Just to maintain consistency with warnings API.

        filename: not used. Just to maintain consistency with warnings API.

        lineno: not used. Just to maintain consistency with warnings API.

        line: not used. Just to maintain consistency with warnings API.

    Return:
        str: formated warning message.
    """
    # pylint: disable=W0613
    return "Warning: {}\n".format(message)


warnings.formatwarning = warning_format


def _check_values_in_group(value: t.Union[str, t.Iterable[str]],
                           valid_group: t.Iterable[str],
                           wildcard: t.Optional[str] = "all"
                           ) -> t.Tuple[t.Tuple[str, ...], t.Tuple[str, ...]]:
    """Checks if a value is in a set or a set of values is a subset of a set.

    Args:
        value (:obj:`iterable` of :obj:`str` or :obj:`str): value(s) to check
            if is (are) in the given valid_group of strings.

        valid_group (:obj:`iterable` of :obj:`str`): a valid_group of strings
            representing the valid tokens which  ``value`` is verified against
            it.

        wildcard (:obj:`str`, optional): a value which represents ``all valu-
            es``, ignoring capital letters, so, for example, values ``all``,
            ``ALL`` and any mix of upper and lower case are all considered to
            be the same wildcard token.

    Returns:
        tuple(tuple, tuple): A pair of tuples containing, respectively, values
            that are in the given valid_group and those that are not.

    Raises:
        TypeError: if ``value`` is not an iterable type or some of its elements
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

    return tuple(in_group), tuple(not_in_group)


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
    # name are all prefixed with "MTF_PREFIX" and all precomputa-
    # tion methos, prefixed with "PRECOMPUTE_PREFIX".
    feat_mtd_list = [
        ft_method for ft_method in feat_mtd_list
        if ft_method[0].startswith(prefix)
    ]

    return feat_mtd_list


def _get_all_prefixed_mtds(
        prefix: str,
        groups: t.Tuple[str, ...],
        update_groups_by: t.Optional[t.Union[t.FrozenSet[str],
                                             t.Set[str]]] = None,
        ) -> t.Dict[str, t.Tuple]:
    """Get all methods prefixed with ``prefix`` in predefined feature ``groups``.

    The predefined metafeature groups are inside ``VALID_GROUPS`` attribute.

    Args:
        prefix (:obj:`str`): gather methods prefixed with this value.

        groups (:obj:`Tuple` of :obj:`str`): a tuple of feature group names.
            It can assume value :obj:`NoneType`, which is interpreted as ``no
            filter`` (i.e. all features of all groups will be returned).

        return_groups (:obj:`bool`, optional): if True, then the returned value
            will be a :obj:`dict` (instead of a :obj:`tuple`) which maps each
            group (as keys) with its correspondent values (as :obj:`tuple`s).

        update_groups_by (:obj:`set` of :obj:`str`, optional): values to filter
            ``groups``. This function also returns a new version of ``groups``
            with all its elements that do not contribute with any new method
            for the final output. It other words, it is removed any group which
            do not contribute to the output of this function. This is particu-
            larly useful for precomputations, as it helps avoiding unecessary
            precomputation methods from feature groups not related with user
            selected features.

    Returns:
        If ``filter_groups_by`` argument is :obj:`NoneType` or empty:
            tuple: with all filtered methods by ``group``.

        Else:
            tuple(tuple, tuple): the first field is the output described above,
                the second field is a new version of ``groups``, with all ele-
                ments that do not contribute with any element listed in the set
                ``update_groups_by`` removed.
    """
    groups = tuple(set(VALID_GROUPS).intersection(groups))

    if not groups:
        return {"methods": tuple(), "groups": tuple()}

    methods_by_group = {
        ft_type_id: _get_prefixed_mtds_from_class(
            class_obj=mfe_class,
            prefix=prefix)

        for ft_type_id, mfe_class in zip(VALID_GROUPS, VALID_MFECLASSES)
        if ft_type_id in groups
    }

    gathered_methods = []  # type: t.List[TypeMtdTuple]
    new_groups = []  # type: t.List[str]

    for group_name in methods_by_group:
        group_mtds = methods_by_group[group_name]
        gathered_methods += group_mtds

        if update_groups_by:
            group_mtds_names = {
                remove_prefix(mtd_name, prefix=MTF_PREFIX)
                for mtd_name, _ in group_mtds
            }

            if not update_groups_by.isdisjoint(group_mtds_names):
                new_groups.append(group_name)

    ret_val = {
        "methods": tuple(gathered_methods),
    }  # type: t.Dict[str, t.Tuple]

    if update_groups_by:
        ret_val["groups"] = tuple(new_groups)

    return ret_val


def _preprocess_iterable_arg(
        values: t.Union[str, t.Iterable[str]]) -> t.List[str]:
    """Process ``values`` to a canonical form.

    This canonical form consists in removing repeated elements from ``values``,
    and cast all elements to lower-case.

    Args:
        values (:obj:`iterable` of :obj:`str` or :obj:`str`): feature names or
            a collection of to be processed into a canonical form.

    Returns:
        list: ``values`` values as iterable. The values within strings all low-
            er-cased.
    """
    if isinstance(values, str):
        values = {values}

    return list(map(str.lower, set(values)))


def _extract_mtd_args(ft_mtd_callable: t.Callable) -> t.Tuple[str, ...]:
    """Extracts arguments from given method.

    Args:
        ft_mtd_callable (:obj:`callable`): a callable related to a feature
            extraction method.

    Returns:
        list: containing the name of arguments of ``ft_mtd_callable``.

    Raises:
        TypeError: if ``ft_mtd_callable`` is not a valid callable.
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
    """Returns ``feature`` values summarized by ``callable_sum``.

    Args:
        features (:obj:`Sequence` of numerics): Sequence containing values
            to summarize.

        callable_sum (:obj:`callable`): callable of the method which im-
            plements the desired summary function.

        callable_args (:obj:`dict`, optional): arguments to the summary fun-
            ction. The expected dictionary format is the following: {`argu-
            ment_name`: value}. To know the summary function arguments, you
            need to check out the documentation of the method which implemen-
            ts it.

        remove_nan (:obj:`bool`, optional): check and remove all elements
            in `features` which are not numeric. Note that :obj:`np.inf`
            is still considered numeric (:obj:`float` type).

    Returns:
        float: value of summarized feature values, if possible. May return
            :obj:`np.nan` if summary function call invokes TypeError, Value-
            Error or ZeroDivisionError.

    Raises:
        AttributeError: if ``callable_sum`` is invalid.
        TypeError: if ``features``  is not a sequence.
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
    """Extract features from ``mtd_callable`` with ``mtd_args`` as args.

    Args:
        mtd_name (:obj:`str`): name of the feature-extraction method
            to be invoked.

        mtd_args (:obj:`dict`): arguments of method to be invoked. The
            expected format of the arguments is {`argument_name`: value}.
            In order to know the method arguments available, you need to
            check its documentation.

        mtd_callable (:obj:`callable`): callable of the feature-extraction
            method.

        suppress_warnings (:obj:`bool`, optional): if True, all warnings
            invoked before invoking the method (or after) will be ignored.
            The method (from ``mtd_callable``) itself may still invoke war-
            nings.

    Returns:
        numeric or array: return value of the feature-extraction method.

    Raises:
        AttributeError: if ``mtd_callable`` is not valid.
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
    """Build a ``kwargs`` (:obj:`dict`) for a feature-extraction :obj:`callable`.

    Args:
        mtd_name (:obj:`str`): name of the method.

        mtd_args (:obj:`iterable` of :obj:`str`): iterable containing the name
            of all arguments of the callable.

        inner_custom_args (:obj:`dict`, optional): custom arguments for inner
            usage, for example, to pass ``X``, ``y`` or other user-independent
            arguments necessary for the callable. The expected format of this
            dict is {`argument_name`: value}.

        user_custom_args (:obj:`dict`, optional): assumes the same model as the
            dict above, but this one keeps user-dependent arguments for method
            callable, for example, number of bins of a histogram-like metafea-
            ture or degrees of freedom of a standard deviation-related metafe-
            ature. The name of the arguments must be verified in its correspon-
            dent method documentation.

        precomp_args (:obj:`dict`, optional): precomputed cached arguments whi-
            ch may be used for the feature-extraction method to speed up its
            calculations.

        suppress_warnings(:obj:`bool`, optional): if True, do not show any war-
            nings about unknown callable parameters.

    Returns:
        dict: a ready-to-use ``kwargs`` for the correspondent callable. The
            format is {``argument_name``: value}.
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
        values (:obj:`iterable` of :obj:`str` or :obj:`str`): a group os values
            or a single value to process.

        group_name (:obj:`str`, optional): name of which internal group ``valu-
            es`` should be searched inside. Please check this module Attribute
            documentation to verify which groups are available for valid opti-
            ons. The constant ``VALID_GROUPS_PREFIX`` always should prefix gro-
            up options, and this parameter must be the name of the group with-
            out its prefix. For example, to select ``VALID_CLASSES`` group for
            ``values`` reference, then group_names must be just ``classes``.

        wildcard (:obj:`str`, optional): special value to ``accept any value``.

        allow_none (:obj:`bool`, optional): if True, then :obj:`NoneType` is
            a accepted as ``values``. Note that, if ``values`` is an iterable,
            it does not mean that :obj:`NoneType` will become a valid value wi-
            thin, but ``values`` can assume value :obj:`NoneType`.

        allow_empty (:obj:`bool`, optional): if True, then ``values`` can be an
            zero-length iterable.

    Return:
        tuple: lower-cased tuple with all valid values.

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
    """Generate metadata from ``summary`` MFE instantiation argument.

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
            metadata about methods which implement each summary function.
            This last tuple model is:

                (
                    `summary_mtd_name`,
                    `summary_mtd_callable`,
                    `summary_mtd_args`,
                )

    Raises:
        TypeError: if `summary` is not :obj:`NoneType`, empty, a valid string
            nor an iterable containing valid group identifiers as strings.
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
        groups: t.Tuple[str, ...],
        wildcard: str = "all",
        suppress_warnings: bool = False,
        ) -> t.Tuple[t.Tuple[str, ...],
                     t.Tuple[TypeExtMtdTuple, ...],
                     t.Tuple[str, ...]]:
    """Generate metadata from ``features`` MFE instantiation argument.

    The use of this function to happen after ``process_groups`` function, as
    ``groups`` parameter is expected to be in a canonical form (lower-cased
    values inside a tuple).

    Args:
        features (:obj:`iterable` of `str` or `str`): iterable containing a
            collection of features or a string describing a single feature. No-
            te that only features that are in the given `groups` will be retur-
            ned.

        groups (:obj:`Tuple` of :obj:`str`, optional): collection containing
            one or more group identifiers. Check out ``MFE`` class documenta-
            tion for more information.

        wildcard (:obj:`str`, optional): value to be used as ``select all`` for
            ``features`` argument.

        suppress_warnings (:obj:`bool`, optional): if True, hide all warnings
            raised during this method processing.

    Returns:
        tuple(tuple, tuple): A pair of tuples. The first Tuple is all feature
            names extracted from this method, to give the user easy access to
            available features in the model. The second field is a tuple for
            internal usage, containing metadata in the form of tuples in the
            following format: (`mtd_name`, `mtd_callable`, `mtd_args`), i.e.,
            the first tuple item field is a string containing the name of a
            feature-extraction related method, and the second field is a cal-
            lable object for the corresponding method, and the third is the
            method arguments.

    Raises:
        ValueError: if features is :obj:`NoneType` or is empty.
    """
    if not features:
        raise ValueError('"features" can not be None nor empty.')

    if groups is None:
        groups = tuple()

    processed_ft = _preprocess_iterable_arg(features)  # type: t.List[str]

    reference_values = None
    if wildcard not in processed_ft:
        reference_values = frozenset(processed_ft)

    mtds_metadata = _get_all_prefixed_mtds(
        prefix=MTF_PREFIX,
        update_groups_by=reference_values,
        groups=groups,
    )  # type: t.Dict[str, t.Tuple]

    ft_mtds_filtered = mtds_metadata.get(
        "methods", tuple())  # type: t.Tuple[TypeMtdTuple, ...]

    groups = mtds_metadata.get("groups", groups)

    del mtds_metadata

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
            warnings.warn('Unknown feature "{}"'.format(unknown_ft),
                          UserWarning)

    return tuple(available_feat_names), tuple(ft_mtd_processed), groups


def _patch_precomp_groups(
        precomp_groups: t.Union[str, t.Iterable[str]],
        groups: t.Optional[t.Tuple[str, ...]] = None,
        ) -> t.Union[str, t.Iterable[str]]:
    """Enforce precomputation in landmarking and model-based metafeatures."""
    if not precomp_groups:
        precomp_groups = set()

    # Enforce precomputation from landmarking and model-based metafeature group
    # due to strong dependencies of machine learning models.
    if groups and not isinstance(precomp_groups, str):
        if "landmarking" in groups and "landmarking" not in precomp_groups:
            precomp_groups = set(precomp_groups).union({"landmarking"})

        if "model-based" in groups and "model-based" not in precomp_groups:
            precomp_groups = set(precomp_groups).union({"model-based"})

    return precomp_groups


def process_precomp_groups(
        precomp_groups: t.Union[str, t.Iterable[str]],
        groups: t.Optional[t.Tuple[str, ...]] = None,
        wildcard: str = "all",
        suppress_warnings: bool = False,
        **kwargs
        ) -> t.Dict[str, t.Any]:
    """Process ``precomp_groups`` argument while fitting into a MFE model.

    This function is expected to be used after ``process_groups`` function,
    as ``groups`` parameter is expected to be in a canonical form (lower-cased
    values inside a tuple).

    Args:
        precomp_groups (:obj:`iterable` of `str` or `str`): a single or a se-
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
        dict: precomputed values given by ``kwargs`` using convenient methods
            based in valid selected metafeature groups.
    """
    if groups is None:
        groups = tuple()

    precomp_groups = _patch_precomp_groups(precomp_groups, groups)

    if not precomp_groups:
        return {}

    processed_precomp_groups = _preprocess_iterable_arg(
        precomp_groups)  # type: t.Sequence[str]

    if wildcard in processed_precomp_groups:
        processed_precomp_groups = groups

    else:
        if not suppress_warnings:
            unknown_groups = set(processed_precomp_groups).difference(groups)

            for unknown_precomp in unknown_groups:
                warnings.warn(
                    'Unknown precomp_groups "{0}"'.format(unknown_precomp),
                    UserWarning)

        processed_precomp_groups = tuple(
            set(processed_precomp_groups).intersection(groups))

    mtds_metadata = _get_all_prefixed_mtds(
        prefix=PRECOMPUTE_PREFIX,
        groups=processed_precomp_groups,
    )  # type: t.Dict[str, t.Tuple]

    precomp_mtds_filtered = mtds_metadata.get(
        "methods", tuple())  # type: t.Tuple[TypeMtdTuple, ...]

    del mtds_metadata

    precomp_items = {}  # type: t.Dict[str, t.Any]

    for precomp_mtd_tuple in precomp_mtds_filtered:
        precomp_mtd_name, precomp_mtd_callable = precomp_mtd_tuple

        try:
            new_precomp_vals = precomp_mtd_callable(**kwargs)  # type: ignore

        except (AttributeError, TypeError, ValueError) as type_err:
            new_precomp_vals = {}

            if not suppress_warnings:
                warnings.warn("Something went wrong while "
                              'precomputing "{0}". Will ignore '
                              "this method. Error message:\n"
                              "{1}.".format(precomp_mtd_name, repr(type_err)))

        if new_precomp_vals:
            precomp_items = {
                **precomp_items,
                **new_precomp_vals,
            }

            # Update kwargs to avoid recalculations iteratively
            kwargs = {
                **kwargs,
                **new_precomp_vals,
            }

    return precomp_items


def check_data(X: t.Union[np.ndarray, list],
               y: t.Union[np.ndarray, list]
               ) -> t.Tuple[np.ndarray, np.ndarray]:
    """Checks ``X`` and ``y`` data type and shape and transform it if necessary.

    Args:
        Check ``mfe.fit`` method for more information.

    Raises:
        TypeError: if ``X`` or ``y`` is neither a np.ndarray nor a list-
            type object.

        ValueError: if ``X`` is empty or number of rows between X and Y
            mismatch.

    Returns:
        tuple(np.ndarray, np.ndarray): ``X`` and ``y`` possibly reshaped and
            casted to :obj:`np.ndarray` type.
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

    if len(X.shape) == 1 and X.shape[0]:
        X = X.reshape(*X.shape, -1)

    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError('Neither "X" nor "y" can be empty.')

    if X.shape[0] != y.shape[0]:
        raise ValueError('"X" number of rows and "y" '
                         "length shapes do not match.")

    return np.copy(X), np.copy(y)


def isnumeric(
        value: t.Any,
        check_subtype: bool = True) -> bool:
    """Checks if ``value`` is a numeric type or a collection of numerics.

    The ``Numeric Type`` is assumed to be one of the following:
        1. :obj:`int`
        2. :obj:`float`
        3. :obj:`np.number`

    Args:
        value (:obj:`Any`): any object to be checked as numeric or a collec-
            tion of numerics.

        check_subtype (:obj:`bool`, optional): if True, check elements of
            ``value`` if it is an iterable object. Otherwise, only checks
            ``value`` type, ignoring the fact that it can be an iterable ob-
            ject.

    Returns:
        bool: True if `value` is a numeric type object or a collection of nume-
            ric-only elements. False otherwise.
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

    Args:
        value (:obj:`str`): method name prefixed with value stored in
            ``prefix``.

    Returns:
        str: ``value`` without ``prefix``.

    Raises:
        TypeError: if ``value`` is not a string.
    """
    if value.startswith(prefix):
        return value[len(prefix):]

    return value


def timeit(func: t.Callable, *args) -> t.Tuple[t.Any, float]:
    """Measure how much time is necessary for calling ``func`` with ``args``.

    Args:
        func (:obj:`callable`): a callable which invokation time will be
            measured from.

        *args: additional arguments for ``func``.

    Return:
        tuple[any, float]: the first element is the return value from ``func``.
            The second argument is the time necessary for the invokation of
            ``func``.

    Raises:
        This method raises all exceptions from ``func``.
    """
    t_start = time.time()
    ret_val = func(*args)
    time_total = time.time() - t_start
    return ret_val, time_total


def transform_cat(data_categoric: np.ndarray) -> t.Optional[np.ndarray]:
    """Transform categorical data using a model matrix.

    The formula used for this transformation is just the union (+) of all cat-
    egoric attributes using formula language from ``patsy`` package API, re-
    moving the intercept terms: ``~ 0 + A_1 + ... + A_n``, where ``n`` is the
    number of attributes and A_i is the ith categoric attribute, 1 <= i <= n.
    """
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
    """Discretize a 1-D numeric array into an equal-frequency histogram."""
    perc_interval = 100.0 / num_bins
    perc_range = np.arange(perc_interval, 100, perc_interval)
    hist_divs = np.percentile(data, perc_range)

    if hist_divs.size == 0:
        hist_divs = [np.median(data)]

    return np.digitize(data, hist_divs, right=True)


def transform_num(data_numeric: np.ndarray,
                  num_bins: t.Optional[int] = None) -> t.Optional[np.ndarray]:
    """Discretize numeric data with an equal-frequency histogram.

    The index of the histogram bin overwrites its correspondent numeric
    values.

    Args:
        data_numeric (:obj:`np.ndarray`): 2-D numpy array of numeric-
            only data to discretize.

        num_bins (:obj:`int`, optional): number of bins of the equal-frequen-
            cy histogram used to discretize the data. If no value is given,
            then the default value is min(2, c), where ``c`` is the cubic root
            of the number of instances rounded down.

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
        data (:obj:`np.ndarray`): data to rescale.

        option (:obj:`str`): rescaling strategy. Must be one in ``VALID_RESCA-
            LE`` attribute.

        args (:obj:`dict`, optional): additional arguments for the scaler. All
            scaler used are from ``sklearn`` package, so you should consult
            their documentation for a complete list of available arguments to
            user customization. The used scalers for each  available ``option``
            are:
                    - ``min-max``: ``sklearn.preprocessing.MinMaxScaler``
                    - ``standard``: ``sklearn.preprocessing.StandardScale``
                    - ``robust``: ``sklearn.preprocessing.RobustScaler``

    Returns:
        np.ndarray: scaled ``data`` based in ``option`` correspondent strategy.

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

    return scaler_model.fit_transform(data.astype(float))


def check_score(score: str, groups: t.Tuple[str, ...]):
    """Checks if a given score is valid.

    Args:
        score (:obj: `str`): the score metrics name.

        groups (:obj:`Tuple` of :obj:`str`): a tuple of feature group names.

    Returns:
        None

    Raises:
        ValueError: if ``score`` is not None or ``str``.
        ValueError: if ``score`` is not valid.

    """
    valid_scoring = {
        "accuracy": scoring.accuracy,
        "balanced-accuracy": scoring.balanced_accuracy,
        "f1": scoring.f1,
        "kappa": scoring.kappa,
        "auc": scoring.auc,
    }  # type: t.Dict[str, t.Callable[[np.ndarray, np.ndarray], float]]

    if score is not None and not isinstance(score, str):
        raise ValueError('"score" is not None or str but "{0}" was passed.'
                         'The valid values are {1}'.format(
                             score, list(valid_scoring.keys())))

    if "landmarking" in groups:
        if score is None:
            raise ValueError(
                'Landmarking metafeatures need a score metric.'
                'One of the following "score" values is required:'
                '{0}'.format(list(valid_scoring.keys())))
        if score not in valid_scoring:
            raise ValueError(
                'One of the following "score" values is required:'
                '{0}'.format(list(valid_scoring.keys())))
        return valid_scoring[score]

    return None
