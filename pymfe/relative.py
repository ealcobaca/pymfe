"""Module dedicated to extraction of relative landmarking metafeatures."""

import typing as t
import time
import re

import scipy.stats


class MFERelativeLandmarking:
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
       value or a generic List (preferably a :obj:`np.ndarray`) type with
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
        mtf_names: t.List[str],
        mtf_vals: t.List[float],
        mtf_time: t.List[float],
        class_indexes: t.List[int],
        groups: t.Tuple[str, ...],
        inserted_group_dep: t.FrozenSet[str],
        **kwargs
    ) -> t.Optional[t.Tuple[t.List[str], t.List[float], t.List[float]]]:
        """Generate Relative Landmarking from Landmarking metafeatures.

        Parameters
        ----------
        mtf_names : str
            Name of each generated metafeature (after extraction and
            summarization).

        mtf_vals : str
            Value of each generated metafeature (after extraction and
            summarization).

        mtf_time : str
            Time elapsed to generate each metafeature (after extraction
            and summarization).

        class_indexes : :obj:`list` of int
            List of indexes corresponding to metafeatures associated to
            metafeature groups present in this postprocessing method name
            (``Landmarking`` and ``Relative``.)

        groups : :obj:`tuple` of str
            User-selected and automatic inserted (due to group dependencies)
            groups of metafeatures.

        inserted_group_dep : :obj:`tuple` of str
            Tuple with all automatic inserted metafeature groups due to
            dependency between groups.

        **kwargs: to keep consistency with the framework postprocessing
            signature. Not used in this method.

        Returns
        -------
        If either ``landmarking`` or ``relative`` is not selected by the user
        as a metafeature group:
            Returns None.

        Else:
            Returns three lists for generated relative landmarking metafeature
            names, values and time elapsed (in this order.)
        """
        # pylint: disable=W0613

        if "relative" not in groups:
            return None

        mtf_rel_names = []  # type: t.List[str]
        mtf_rel_vals = []  # type: t.List[float]
        mtf_rel_time = []  # type: t.List[float]

        mtf_by_summ, mtf_orig_indexes = cls.group_mtf_by_summary(
            mtf_names=mtf_names, mtf_vals=mtf_vals, class_indexes=class_indexes
        )

        avg_time = time.time()

        mtf_by_summ = {
            summary: scipy.stats.rankdata(
                a=mtf_by_summ[summary], method="average"
            )
            for summary in mtf_by_summ
        }

        avg_time = (time.time() - avg_time) / (
            len(mtf_by_summ) if mtf_by_summ else 1.0
        )

        mtf_rel_vals, original_indexes = cls._flatten_dictionaries(
            mtf_by_summ, mtf_orig_indexes
        )

        for cur_orig_index in original_indexes:
            mtf_rel_names.append(
                "{}.relative".format(mtf_names[cur_orig_index])
            )
            mtf_rel_time.append(mtf_time[cur_orig_index] + avg_time)

        change_in_place = (
            "landmarking" not in groups or "landmarking" in inserted_group_dep
        )

        if change_in_place:
            for cur_index, cur_orig_index in enumerate(original_indexes):
                mtf_names[cur_orig_index] = mtf_rel_names[cur_index]
                mtf_vals[cur_orig_index] = mtf_rel_vals[cur_index]
                mtf_time[cur_orig_index] = mtf_rel_time[cur_index]

            return None

        return mtf_rel_names, mtf_rel_vals, mtf_rel_time

    @classmethod
    def group_mtf_by_summary(
        cls,
        mtf_names: t.List[str],
        mtf_vals: t.List[float],
        class_indexes: t.List[int],
    ) -> t.Tuple[t.Dict[str, t.List[float]], t.Dict[str, t.List[int]]]:
        """Group metafeatures by its correspondent summary method.

        It is assumed that every distinct suffix after the first
        separator ``.`` in the metafeature name corresponds to a
        different summary method, even if it is, for example, due
        to different bins of a histogram summarization.
        """
        re_get_summ = re.compile(
            r"""[^\.]+\.  # Feature name with the first separator
                (.*)      # Summary name (can have more than one suffix)
            """,
            re.VERBOSE,
        )

        mtf_by_summ = {}  # type: t.Dict[str, t.List[float]]
        mtf_orig_indexes = {}  # type: t.Dict[str, t.List[int]]

        for mtf_index, cur_mtf_name in enumerate(mtf_names):
            if mtf_index in class_indexes:
                re_match = re_get_summ.match(cur_mtf_name)
                if re_match:
                    summary_suffixes = re_match.group(1)

                    if summary_suffixes not in mtf_by_summ:
                        mtf_by_summ[summary_suffixes] = []
                        mtf_orig_indexes[summary_suffixes] = []

                    mtf_by_summ[summary_suffixes].append(mtf_vals[mtf_index])
                    mtf_orig_indexes[summary_suffixes].append(mtf_index)

        return mtf_by_summ, mtf_orig_indexes

    @classmethod
    def _flatten_dictionaries(
        cls,
        mtf_by_summ: t.Dict[str, t.List[float]],
        mtf_orig_indexes: t.Dict[str, t.List[int]],
    ) -> t.Tuple[t.List[float], t.List[int]]:
        """Flatten dictionary values to two lists with correspondence."""
        ranked_values = []  # type: t.List[float]
        orig_indexes = []  # type: t.List[int]

        while mtf_by_summ:
            summary, cur_ranks = mtf_by_summ.popitem()
            cur_orig_indexes = mtf_orig_indexes.pop(summary)
            ranked_values += list(cur_ranks)
            orig_indexes += cur_orig_indexes

        return ranked_values, orig_indexes
