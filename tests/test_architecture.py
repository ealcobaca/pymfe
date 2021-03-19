"""Module dedicated to framework testing."""
import pytest
import typing as t

import numpy as np
import sklearn.tree

from pymfe import _internal
from pymfe.mfe import MFE
from . import utils

GNAME = "framework-testing"


def summary_exception(
    values: np.ndarray, raise_exception: bool = False
) -> int:
    """Returns the length of ``values`` or raise a ValueError exception."""
    if raise_exception:
        raise ValueError("Summary exception raised.")

    return len(values)


def summary_memory_error(
    values: np.ndarray, raise_mem_err: bool = False
) -> int:
    """Returns the length of ``values`` or raise a MemoryError exception."""
    if raise_mem_err:
        utils.raise_memory_error()

    return len(values)


class MFETestClass:
    """Some generic methods for testing the MFE Framework."""

    @classmethod
    def postprocess_return_none(cls, **kwargs) -> None:
        """Postprocess: return None."""
        return None

    @classmethod
    def postprocess_return_new_feature(
        cls, number_of_lists: int = 3, **kwargs
    ) -> t.Tuple[t.List, t.List, t.List]:
        """Postprocess: return Tuple of lists."""
        return tuple(["test_value"] for _ in range(number_of_lists))

    @classmethod
    def postprocess_raise_exception(
        cls, raise_exception: bool = False, **kwargs
    ) -> None:
        """Posprocess: raise exception."""
        if raise_exception:
            raise ValueError("Expected exception (postprocess).")

        return None

    @classmethod
    def postprocess_memory_error(
        cls, raise_mem_err: bool = False, **kwargs
    ) -> t.Optional[np.ndarray]:
        """Posprocess: memory error."""
        if raise_mem_err:
            return utils.raise_memory_error()

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
    def precompute_raise_exception(
        cls, raise_exception: bool = False, **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precompute: raise exception."""
        precomp_vals = {}

        if raise_exception:
            raise ValueError("Expected exception (precompute).")

        return precomp_vals

    @classmethod
    def precompute_memory_error(
        cls, raise_mem_err: bool = False, **kwargs
    ) -> None:
        """Precompute: memory error."""
        precomp_vals = {}

        if raise_mem_err:
            precomp_vals["huge_array"] = utils.raise_memory_error()

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
    def ft_raise_exception(
        cls, X: np.ndarray, y: np.ndarray, raise_exception: False
    ) -> float:
        """Metafeature: float type."""
        if raise_exception:
            raise ValueError("Expected exception (feature).")

        return -1.0

    @classmethod
    def ft_memory_error(
        cls, raise_mem_err: bool = False, **kwargs
    ) -> np.ndarray:
        """Metafeature: memory error."""
        if raise_mem_err:
            return utils.raise_memory_error()

        return np.array([1, 2, 3])


class TestArchitecture:
    """Tests for the framework architecture."""

    def test_summary_valid1(self):
        vals = np.arange(5)

        res = _internal.summarize(
            features=vals, callable_sum=summary_exception
        )

        assert res == len(vals)

    def test_summary_valid2(self):
        vals = np.arange(5)

        res = _internal.summarize(
            features=vals, callable_sum=summary_memory_error
        )

        assert res == len(vals)

    def test_summary_invalid1(self):
        res = _internal.summarize(
            features=np.arange(5),
            callable_sum=summary_exception,
            callable_args={"raise_exception": True},
        )

        assert np.isnan(res)

    def test_summary_invalid2(self):
        res = _internal.summarize(
            features=np.arange(5),
            callable_sum=summary_memory_error,
            callable_args={"raise_mem_err": True},
        )

        assert np.isnan(res)

    def test_postprocessing_valid(self):
        """Test valid postprocessing and its automatic detection."""
        results = [], [], []

        _internal.post_processing(
            results=results, groups=tuple(), custom_class_=MFETestClass
        )

        assert all(map(lambda l: len(l) > 0, results))

    def test_preprocessing_valid(self):
        """Test valid precomputation and its automatic detection."""
        precomp_args = _internal.process_precomp_groups(
            precomp_groups=tuple(), groups=tuple(), custom_class_=MFETestClass
        )

        assert len(precomp_args) > 0

    def test_feature_detection(self):
        """Test automatic dectection of metafeature extraction method."""
        name, mtd, groups = _internal.process_features(
            features="all",
            groups=tuple(),
            suppress_warnings=True,
            custom_class_=MFETestClass,
        )

        assert len(name) == 4 and len(mtd) == 4 and len(groups) == 1

    def test_get_groups(self):
        model = MFE()
        res = model.valid_groups()
        assert len(res) == len(_internal.VALID_GROUPS) and not set(
            res
        ).symmetric_difference(_internal.VALID_GROUPS)

    def test_metafeature_description(self):
        desc, _ = MFE.metafeature_description(print_table=False)
        groups = [d[0] for d in desc]
        assert len(set(groups)) == len(_internal.VALID_GROUPS)

        desc, _ = MFE.metafeature_description(
            sort_by_group=True,
            sort_by_mtf=True,
            print_table=False,
            include_references=True,
        )
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
        assert len(res) == len(_internal.VALID_GROUPS) and not set(
            res
        ).symmetric_difference(_internal.VALID_GROUPS)

        model = MFE(groups=["default"])
        res = model.valid_groups()
        assert len(res) == len(_internal.VALID_GROUPS) and not set(
            res
        ).symmetric_difference(_internal.VALID_GROUPS)

        model = MFE(groups=["general", "default"])
        res = model.valid_groups()
        assert len(res) == len(_internal.VALID_GROUPS) and not set(
            res
        ).symmetric_difference(_internal.VALID_GROUPS)

    @pytest.mark.parametrize(
        "groups, summary",
        [
            ("statistical", "all"),
            ("general", "all"),
            ("landmarking", "all"),
            ("relative", "all"),
            ("model-based", "all"),
            ("info-theory", "all"),
            ("statistical", ("mean", "sd")),
            ("general", ("mean", "sd")),
            ("landmarking", ("mean", "sd")),
            ("model-based", ("mean", "sd")),
            ("general", ("mean", "histogram")),
            ("landmarking", ("mean", "histogram")),
            ("model-based", ("mean", "histogram")),
            ("general", ("quantiles", "histogram")),
            ("landmarking", ("quantiles", "histogram")),
            ("model-based", ("quantiles", "histogram")),
            (["general", "relative"], ("mean", "sd")),
            (["general", "relative"], ("quantiles", "histogram")),
            (["landmarking", "relative"], ("mean", "sd")),
            (["landmarking", "relative"], ("quantiles", "histogram")),
            (["statistical", "landmarking", "relative"], ("mean", "sd")),
            ("all", "all"),
        ],
    )
    def test_extract_metafeature_names_supervised(self, groups, summary):
        """Test .extract_metafeature_names method."""
        X, y = utils.load_xy(0)

        mfe = MFE(groups=groups, summary=summary)

        mtf_names_1 = mfe.extract_metafeature_names(supervised=True)
        mtf_names_2 = mfe.fit(X.values, y.values).extract(
            suppress_warnings=True
        )[0]

        assert mtf_names_1 == tuple(mtf_names_2)

    @pytest.mark.parametrize(
        "groups, summary",
        [
            ("statistical", "all"),
            ("general", "all"),
            ("landmarking", "all"),
            ("relative", "all"),
            ("model-based", "all"),
            ("info-theory", "all"),
            ("statistical", ("mean", "sd")),
            ("general", ("mean", "sd")),
            ("landmarking", ("mean", "sd")),
            ("model-based", ("mean", "sd")),
            ("general", ("mean", "histogram")),
            ("landmarking", ("mean", "histogram")),
            ("model-based", ("mean", "histogram")),
            ("general", ("quantiles", "histogram")),
            ("landmarking", ("quantiles", "histogram")),
            ("model-based", ("quantiles", "histogram")),
            (["general", "relative"], ("mean", "sd")),
            (["general", "relative"], ("quantiles", "histogram")),
            (["landmarking", "relative"], ("mean", "sd")),
            (["landmarking", "relative"], ("quantiles", "histogram")),
            (["statistical", "landmarking", "relative"], ("mean", "sd")),
            ("all", "all"),
        ],
    )
    def test_extract_metafeature_names_unsupervised_01(self, groups, summary):
        """Test .extract_metafeature_names method."""
        X, _ = utils.load_xy(0)

        mfe = MFE(groups=groups, summary=summary)

        mtf_names_1 = mfe.extract_metafeature_names(supervised=False)
        mtf_names_2 = mfe.fit(X.values).extract(suppress_warnings=True)[0]

        assert mtf_names_1 == tuple(mtf_names_2)

    @pytest.mark.parametrize(
        "groups, summary",
        [
            ("general", "all"),
            ("statistical", ("mean", "sd")),
            (["general", "relative"], ("mean", "sd")),
            (["general", "relative"], ("quantiles", "histogram")),
            (["landmarking", "relative"], ("mean", "sd")),
            (["landmarking", "relative"], ("quantiles", "histogram")),
            (["statistical", "landmarking", "relative"], ("mean", "sd")),
            ("all", "all"),
        ],
    )
    def test_extract_metafeature_names_unsupervised_02(self, groups, summary):
        """Test .extract_metafeature_names method."""
        X, _ = utils.load_xy(0)

        mfe = MFE(groups=groups, summary=summary)

        mtf_names_1 = mfe.fit(X.values).extract(suppress_warnings=True)[0]
        # Note: by default, .extract_metafeature_names should check wether
        # 'y' was fitted or not if .fit was called before. Therefore, here,
        # supervised=True is expected to be ignored and behave like
        # supervised=False.
        mtf_names_2 = mfe.extract_metafeature_names(supervised=True)
        mtf_names_3 = mfe.extract_metafeature_names(supervised=False)

        assert tuple(mtf_names_1) == mtf_names_2 == mtf_names_3

    @pytest.mark.parametrize(
        "groups",
        [
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
        ],
    )
    def test_parse_valid_metafeatures(self, groups):
        """Check the length of valid metafeatures per group."""
        X, y = utils.load_xy(0)

        mfe = MFE(
            groups="all", summary=None, lm_sample_frac=0.5, random_state=1234
        )

        mfe.fit(X.values, y.values)

        res = mfe.extract()

        target_mtf = mfe.valid_metafeatures(groups=groups)
        names, _ = mfe.parse_by_group(groups, res)

        assert not set(names).symmetric_difference(target_mtf)

    def test_no_cat_transformation(self):
        X, y = utils.load_xy(1)
        mfe = MFE()
        mfe.fit(X.values, y.values, transform_cat=None)
        assert mfe._custom_args_ft["N"].size == 0

    def test_gray_encoding_missing_value(self):
        X, y = utils.load_xy(1)
        mfe = MFE()

        X = np.copy(X.values)
        y = y.values

        X[5, 0] = np.nan

        with pytest.raises(ValueError):
            mfe.fit(X, y, transform_cat="gray")

    def test_one_hot_encoding_01(self):
        X, y = utils.load_xy(1)
        mfe = MFE()
        mfe.fit(X.values, y.values, transform_cat="one-hot")

        exp_value = np.sum([np.unique(attr).size - 1 for attr in X.values.T])

        assert mfe._custom_args_ft["N"].shape[1] == exp_value

    def test_one_hot_encoding_02(self):
        X, y = utils.load_xy(1)
        mfe = MFE()
        mfe.fit(X.values, y.values, transform_cat="one-hot-full")

        exp_value = np.sum([np.unique(attr).size for attr in X.values.T])

        assert mfe._custom_args_ft["N"].shape[1] == exp_value

    def test_one_hot_encoding_03(self):
        X, y = utils.load_xy(2)
        mfe = MFE()
        mfe.fit(X.values, y.values, transform_cat="one-hot")

        exp_value = X.values.shape[1]

        assert mfe._custom_args_ft["N"].shape[1] == exp_value

    def test_one_hot_encoding_04(self):
        X, y = utils.load_xy(2)
        mfe = MFE()

        X = np.hstack((X.values, np.ones((y.size, 1), dtype=str)))
        y = y.values

        with pytest.raises(ValueError):
            mfe.fit(X=X, y=y, transform_cat="one-hot")

    @pytest.mark.parametrize("confidence", (0.95, 0.99))
    def test_extract_with_confidence(self, confidence):
        X, y = utils.load_xy(2)

        mtf_names, mtf_vals, mtf_conf_int = (
            MFE(
                groups="all",
                features=["mean", "best_node", "sil"],
                random_state=1234,
            )
            .fit(X=X.values, y=y.values, precomp_groups=None)
            .extract_with_confidence(
                sample_num=64, confidence=confidence, verbose=0
            )
        )

        in_range = np.zeros(len(mtf_names), dtype=bool)

        for mtf_ind, cur_mtf_vals in enumerate(mtf_vals):
            int_low, int_high = mtf_conf_int[mtf_ind, :]
            in_range[mtf_ind] = np.logical_and(
                int_low <= cur_mtf_vals, cur_mtf_vals <= int_high
            )

        assert np.all(in_range)

    def test_extract_with_confidence_invalid1(self):
        with pytest.raises(TypeError):
            MFE().extract_with_confidence()

    def test_extract_with_confidence_invalid2(self):
        X, y = utils.load_xy(2)

        with pytest.raises(ValueError):
            MFE().fit(X.values, y.values).extract_with_confidence(
                confidence=-0.0001
            )

    def test_extract_with_confidence_invalid3(self):
        X, y = utils.load_xy(2)

        with pytest.raises(ValueError):
            MFE().fit(X.values, y.values).extract_with_confidence(
                confidence=1.0001
            )

    def test_extract_with_confidence_time(self):
        X, y = utils.load_xy(2)

        res = (
            MFE(features=["mean", "nr_inst", "unknown"], measure_time="avg")
            .fit(X=X.values, y=y.values)
            .extract_with_confidence(sample_num=3)
        )

        mtf_names, mtf_vals, mtf_time, mtf_conf_int = res

        assert (
            len(mtf_names)
            == len(mtf_vals)
            == len(mtf_time)
            == len(mtf_conf_int)
        )

    def test_extract_with_confidence_multiple_conf_level(self):
        X, y = utils.load_xy(2)

        confidence = [0.8, 0.9, 0.7]

        mtf_conf_int = (
            MFE(features=["mean", "nr_inst", "unknown"])
            .fit(X=X.values, y=y.values)
            .extract_with_confidence(sample_num=2, confidence=confidence)[2]
        )

        assert 2 * len(confidence) == mtf_conf_int.shape[1]

    def test_extract_with_confidence_random_state1(self):
        X, y = utils.load_xy(2)

        _, mtf_vals_1, mtf_conf_int_1 = (
            MFE(features=["mean", "sd"], random_state=16)
            .fit(X=X.values, y=y.values)
            .extract_with_confidence(sample_num=3)
        )

        _, mtf_vals_2, mtf_conf_int_2 = (
            MFE(features=["mean", "sd"], random_state=16)
            .fit(X=X.values, y=y.values)
            .extract_with_confidence(sample_num=3)
        )

        assert np.allclose(mtf_vals_1, mtf_vals_2) and np.allclose(
            mtf_conf_int_1, mtf_conf_int_2
        )

    def test_extract_with_confidence_random_state2(self):
        X, y = utils.load_xy(2)

        _, mtf_vals_1, mtf_conf_int_1 = (
            MFE(features=["mean", "sd"], random_state=16)
            .fit(X=X.values, y=y.values)
            .extract_with_confidence(sample_num=3)
        )

        _, mtf_vals_2, mtf_conf_int_2 = (
            MFE(features=["mean", "sd"], random_state=17)
            .fit(X=X.values, y=y.values)
            .extract_with_confidence(sample_num=3)
        )

        assert np.allclose(mtf_vals_1, mtf_vals_2) and np.any(
            ~np.isclose(mtf_conf_int_1, mtf_conf_int_2)
        )

    def test_extract_with_confidence_random_state3(self):
        X, y = utils.load_xy(2)

        np.random.seed(1234)
        _, mtf_vals_1, mtf_conf_int_1 = (
            MFE(features=["mean", "sd"])
            .fit(X=X.values, y=y.values)
            .extract_with_confidence(sample_num=3)
        )

        np.random.seed(1234)
        _, mtf_vals_2, mtf_conf_int_2 = (
            MFE(features=["mean", "sd"])
            .fit(X=X.values, y=y.values)
            .extract_with_confidence(sample_num=3)
        )

        assert np.allclose(mtf_vals_1, mtf_vals_2) and np.any(
            ~np.isclose(mtf_conf_int_1, mtf_conf_int_2)
        )

    def test_extract_from_model(self):
        X, y = utils.load_xy(2)

        model = sklearn.tree.DecisionTreeClassifier(random_state=1234).fit(
            X.values, y.values
        )

        mtf_name, mtf_vals = MFE(random_state=1234).extract_from_model(model)

        extractor = MFE(groups="model-based", random_state=1234)
        extractor.fit(X=X.values, y=y.values, transform_num=False)
        mtf_name2, mtf_vals2 = extractor.extract()

        assert np.all(mtf_name == mtf_name2) and np.allclose(
            mtf_vals, mtf_vals2
        )

    def test_extract_from_model_invalid1(self):
        X, y = utils.load_xy(2)

        model = sklearn.tree.DecisionTreeRegressor().fit(X.values, y.values)

        with pytest.raises(TypeError):
            MFE().extract_from_model(model)

    def test_extract_from_model_invalid2(self):
        X, y = utils.load_xy(2)

        model = sklearn.tree.DecisionTreeClassifier(random_state=1234).fit(
            X.values, y.values
        )

        with pytest.raises(KeyError):
            MFE().extract_from_model(model, arguments_fit={"dt_model": model})

    def test_extract_from_model_invalid3(self):
        model = sklearn.tree.DecisionTreeClassifier()

        with pytest.raises(RuntimeError):
            MFE().extract_from_model(model)

    def test_extract_from_model_invalid4(self):
        X, y = utils.load_xy(2)

        model = sklearn.tree.DecisionTreeClassifier().fit(X, y)

        with pytest.raises(ValueError):
            MFE(groups="general").extract_from_model(model)


class TestArchitectureWarnings:
    def test_feature_warning1(self):
        """Test exception handling of feature extraction."""
        name, mtd, groups = map(
            np.asarray,
            _internal.process_features(
                features="raise_exception",
                groups=tuple(),
                suppress_warnings=True,
                custom_class_=MFETestClass,
            ),
        )

        with pytest.warns(RuntimeWarning):
            _internal.get_feat_value(
                mtd_name=name[0],
                mtd_args={
                    "X": np.array([]),
                    "y": np.ndarray([]),
                    "raise_exception": True,
                },
                mtd_callable=mtd[0][1],
                suppress_warnings=False,
            )

    def test_feature_warning2(self):
        """Test memory error handling of feature extraction."""
        name, mtd, groups = map(
            np.asarray,
            _internal.process_features(
                features="memory_error",
                groups=tuple(),
                suppress_warnings=True,
                custom_class_=MFETestClass,
            ),
        )

        with pytest.warns(RuntimeWarning):
            _internal.get_feat_value(
                mtd_name=name[0],
                mtd_args={
                    "X": np.array([]),
                    "y": np.ndarray([]),
                    "raise_mem_err": True,
                },
                mtd_callable=mtd[0][1],
                suppress_warnings=False,
            )

    def test_mem_err_precompute(self):
        with pytest.warns(UserWarning):
            _internal.process_precomp_groups(
                precomp_groups=tuple(),
                groups=tuple(),
                custom_class_=MFETestClass,
                raise_mem_err=True,
            )

    def test_mem_err_postprocess(self):
        """Test memory error in postprocessing methods."""
        results = [], [], []

        with pytest.warns(UserWarning):
            _internal.post_processing(
                results=results,
                groups=tuple(),
                custom_class_=MFETestClass,
                raise_mem_err=True,
            )

    def test_postprocessing_invalid1(self):
        """Test exception handling in invalid postprocessing."""
        results = [], [], []

        with pytest.warns(UserWarning):
            _internal.post_processing(
                results=results,
                groups=tuple(),
                custom_class_=MFETestClass,
                raise_exception=True,
            )

    def test_postprocessing_invalid2(self):
        """Test incorrect return value in postprocessing methods."""
        results = [], [], []

        with pytest.warns(UserWarning):
            _internal.post_processing(
                results=results,
                groups=tuple(),
                custom_class_=MFETestClass,
                number_of_lists=2,
            )

    def test_preprocessing_invalid(self):
        """Test exception handling of precomputation."""
        with pytest.warns(UserWarning):
            _internal.process_precomp_groups(
                precomp_groups=tuple(),
                groups=tuple(),
                custom_class_=MFETestClass,
                raise_exception=True,
            )
