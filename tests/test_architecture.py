"""Module dedicated to framework testing."""
import pytest
import typing as t

import numpy as np

from pymfe import _internal
from pymfe.mfe import MFE
from tests.utils import load_xy

GNAME = "framework-testing"


def summary_exception(values: np.ndarray,
                      raise_exception: bool = False) -> int:
    if raise_exception:
        raise ValueError("Summary exception raised.")

    return len(values)


def summary_memory_error(values: np.ndarray,
                         raise_mem_err: bool = False) -> int:
    if raise_mem_err:
        aux = np.zeros(int(1e+20), dtype=np.float64)

    return len(values)


class MFETestClass:
    """Some generic methods for testing the MFE Framework."""

    @classmethod
    def postprocess_return_none(cls, **kwargs) -> None:
        """Postprocess: return None."""
        return None

    @classmethod
    def postprocess_return_new_feature(cls, number_of_lists: int = 3, **kwargs
                                       ) -> t.Tuple[t.List, t.List, t.List]:
        """Postprocess: return Tuple of lists."""
        return tuple(["test_value"] for _ in range(number_of_lists))

    @classmethod
    def postprocess_raise_exception(cls,
                                    raise_exception: bool = False,
                                    **kwargs) -> None:
        """Posprocess: raise exception."""
        if raise_exception:
            raise ValueError("Expected exception (postprocess).")

        return None

    @classmethod
    def postprocess_memory_error(cls, raise_mem_err: bool = False,
                                 **kwargs) -> t.Optional[np.ndarray]:
        """Posprocess: memory error."""
        if raise_mem_err:
            return np.zeros(int(1e+20), dtype=np.float64)

    @classmethod
    def precompute_return_empty(cls, **kwargs) -> t.Dict[str, t.Any]:
        """Precompute: return empty dictionary."""
        precomp_vals = {}

        return precomp_vals

    @classmethod
    def precompute_return_something(cls, **kwargs) -> t.Dict[str, t.Any]:
        """Precompute: return empty dictionary."""
        precomp_vals = {
            "test_param_1": 0,
            "test_param_2": "euclidean",
            "test_param_3": list,
            "test_param_4": abs,
        }

        return precomp_vals

    @classmethod
    def precompute_raise_exception(cls,
                                   raise_exception: bool = False,
                                   **kwargs) -> t.Dict[str, t.Any]:
        """Precompute: raise exception."""
        precomp_vals = {}

        if raise_exception:
            raise ValueError("Expected exception (precompute).")

        return precomp_vals

    @classmethod
    def precompute_memory_error(cls, raise_mem_err: bool = False,
                                **kwargs) -> None:
        """Precompute: memory error."""
        precomp_vals = {}

        if raise_mem_err:
            precomp_vals["huge_array"] = np.zeros(int(1e+20), dtype=np.float64)

        return precomp_vals

    @classmethod
    def ft_valid_number(cls, X: np.ndarray, y: np.ndarray) -> float:
        """Metafeature: float type."""
        return 0.0

    @classmethod
    def ft_valid_array(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Metafeature: float type."""
        return np.zeros(5)

    @classmethod
    def ft_raise_exception(cls, X: np.ndarray, y: np.ndarray,
                           raise_exception: False) -> float:
        """Metafeature: float type."""
        if raise_exception:
            raise ValueError("Expected exception (feature).")

        return -1.0

    @classmethod
    def ft_memory_error(cls, raise_mem_err: bool = False,
                        **kwargs) -> np.ndarray:
        """Metafeature: memory error."""
        if raise_mem_err:
            return np.zeros(int(1e+20), dtype=np.float64)

        return np.array([1, 2, 3])


class TestArchitecture:
    """Tests for the framework architecture."""

    def test_summary_valid1(self):
        vals = np.arange(5)

        res = _internal.summarize(
            features=vals, callable_sum=summary_exception)

        assert res == len(vals)

    def test_summary_valid2(self):
        vals = np.arange(5)

        res = _internal.summarize(
            features=vals, callable_sum=summary_memory_error)

        assert res == len(vals)

    def test_summary_invalid1(self):
        res = _internal.summarize(
            features=np.arange(5),
            callable_sum=summary_exception,
            callable_args={
                "raise_exception": True
            })

        assert np.isnan(res)

    def test_summary_invalid2(self):
        res = _internal.summarize(
            features=np.arange(5),
            callable_sum=summary_memory_error,
            callable_args={
                "raise_mem_err": True
            })

        assert np.isnan(res)

    def test_postprocessing_valid(self):
        """Test valid postprocessing and its automatic detection."""
        results = [], [], []

        _internal.post_processing(
            results=results, groups=tuple(), custom_class_=MFETestClass)

        assert all(map(lambda l: len(l) > 0, results))

    def test_postprocessing_invalid_1(self):
        """Test exception handling in invalid postprocessing."""
        results = [], [], []

        with pytest.warns(UserWarning):
            _internal.post_processing(
                results=results,
                groups=tuple(),
                custom_class_=MFETestClass,
                raise_exception=True)

    def test_postprocessing_invalid_2(self):
        """Test incorrect return value in postprocessing methods."""
        results = [], [], []

        with pytest.warns(UserWarning):
            _internal.post_processing(
                results=results,
                groups=tuple(),
                custom_class_=MFETestClass,
                number_of_lists=2)

    def test_preprocessing_valid(self):
        """Test valid precomputation and its automatic detection."""
        precomp_args = _internal.process_precomp_groups(
            precomp_groups=tuple(), groups=tuple(), custom_class_=MFETestClass)

        assert len(precomp_args) > 0

    def test_preprocessing_invalid(self):
        """Test exception handling of precomputation."""
        with pytest.warns(UserWarning):
            _internal.process_precomp_groups(
                precomp_groups=tuple(),
                groups=tuple(),
                custom_class_=MFETestClass,
                raise_exception=True)

    def test_feature_detection(self):
        """Test automatic dectection of metafeature extraction method."""
        name, mtd, groups = _internal.process_features(
            features="all",
            groups=tuple(),
            suppress_warnings=True,
            custom_class_=MFETestClass)

        assert len(name) == 4 and len(mtd) == 4 and len(groups) == 1

    def test_feature_warning1(self):
        """Test exception handling of feature extraction."""
        name, mtd, groups = map(np.asarray,
                                _internal.process_features(
                                    features="raise_exception",
                                    groups=tuple(),
                                    suppress_warnings=True,
                                    custom_class_=MFETestClass))

        with pytest.warns(RuntimeWarning):
            _internal.get_feat_value(
                mtd_name=name[0],
                mtd_args={
                    "X": np.array([]),
                    "y": np.ndarray([]),
                    "raise_exception": True
                },
                mtd_callable=mtd[0][1],
                suppress_warnings=False)

    def test_feature_warning2(self):
        """Test memory error handling of feature extraction."""
        name, mtd, groups = map(np.asarray,
                                _internal.process_features(
                                    features="memory_error",
                                    groups=tuple(),
                                    suppress_warnings=True,
                                    custom_class_=MFETestClass))

        with pytest.warns(RuntimeWarning):
            _internal.get_feat_value(
                mtd_name=name[0],
                mtd_args={
                    "X": np.array([]),
                    "y": np.ndarray([]),
                    "raise_mem_err": True
                },
                mtd_callable=mtd[0][1],
                suppress_warnings=False)

    def test_get_groups(self):
        model = MFE()
        res = model.valid_groups()
        assert (len(res) == len(_internal.VALID_GROUPS)
                and not set(res).symmetric_difference(_internal.VALID_GROUPS))

    def test_metafeature_description(self):
        desc, _ = MFE.metafeature_description(print_table=False)
        groups = [d[0] for d in desc]
        assert len(set(groups)) == len(_internal.VALID_GROUPS)

        desc, _ = MFE.metafeature_description(
            sort_by_group=True,
            sort_by_mtf=True,
            print_table=False,
            include_references=True)
        mtf = [d[1] for d in desc]
        assert mtf[1][0] < mtf[-1][0]

        desc = MFE.metafeature_description()
        assert desc is None

    def test_metafeature_description_exceptions(self):
        """Test metafeature description exceptions"""
        with pytest.raises(TypeError):
            MFE.metafeature_description(print_table="False")

        with pytest.raises(TypeError):
            MFE.metafeature_description(sort_by_mtf=1)

        with pytest.raises(TypeError):
            MFE.metafeature_description(sort_by_group=[True])

    def test_default_alias_groups(self):
        model = MFE(groups="default")
        res = model.valid_groups()
        assert (len(res) == len(_internal.VALID_GROUPS)
                and not set(res).symmetric_difference(_internal.VALID_GROUPS))

        model = MFE(groups=["default"])
        res = model.valid_groups()
        assert (len(res) == len(_internal.VALID_GROUPS)
                and not set(res).symmetric_difference(_internal.VALID_GROUPS))

        model = MFE(groups=["general", "default"])
        res = model.valid_groups()
        assert (len(res) == len(_internal.VALID_GROUPS)
                and not set(res).symmetric_difference(_internal.VALID_GROUPS))

    @pytest.mark.parametrize("groups", [
        "statistical",
        "general",
        "landmarking",
        "relative",
        "model-based",
        "info-theory",
        ("statistical", "landmarking"),
        ("landmarking", "relative"),
        ("general", "model-based", "statistical"),
        ("statistical", "statistical"),
    ])
    def test_parse_valid_metafeatures(self, groups):
        """Check the length of valid metafeatures per group."""
        X, y = load_xy(0)

        mfe = MFE(
            groups="all", summary=None, lm_sample_frac=0.5, random_state=1234)

        mfe.fit(X.values, y.values)

        res = mfe.extract()

        target_mtf = mfe.valid_metafeatures(groups=groups)
        names, _ = mfe.parse_by_group(groups, res)

        assert not set(names).symmetric_difference(target_mtf)

    def test_no_cat_transformation(self):
        X, y = load_xy(1)
        mfe = MFE()
        mfe.fit(X.values, y.values, transform_cat=None)
        assert mfe._custom_args_ft["N"].size == 0

    def test_one_hot_encoding_01(self):
        X, y = load_xy(1)
        mfe = MFE()
        mfe.fit(X.values, y.values, transform_cat="one-hot")

        exp_value = np.sum([np.unique(attr).size for attr in X.values.T])

        assert mfe._custom_args_ft["N"].shape[1] == exp_value

    def test_one_hot_encoding_02(self):
        X, y = load_xy(2)
        mfe = MFE()
        mfe.fit(X.values, y.values, transform_cat="one-hot")

        exp_value = X.values.shape[1]

        assert mfe._custom_args_ft["N"].shape[1] == exp_value


class TestMemoryError:
    """Test memory error related methods."""

    def test_mem_err_precompute(self):
        with pytest.warns(UserWarning):
            _internal.process_precomp_groups(
                precomp_groups=tuple(),
                groups=tuple(),
                custom_class_=MFETestClass,
                raise_mem_err=True)

    def test_mem_err_postprocess(self):
        """Test memory error in postprocessing methods."""
        results = [], [], []

        with pytest.warns(UserWarning):
            _internal.post_processing(
                results=results,
                groups=tuple(),
                custom_class_=MFETestClass,
                raise_mem_err=True)
