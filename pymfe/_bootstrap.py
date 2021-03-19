"""TODO."""
import typing as t

import numpy as np
import pandas as pd

import pymfe._internal as _internal


class BootstrapExtractor:
    """TODO."""

    def __init__(
        self,
        X: np.ndarray,
        y: t.Optional[np.ndarray],
        extractor,
        sample_num: int = 256,
        confidence: t.Union[float, t.Sequence[float]] = 0.95,
        arguments_fit: t.Optional[t.Dict[str, t.Any]] = None,
        arguments_extract: t.Optional[t.Dict[str, t.Any]] = None,
        verbose: int = 0,
        random_state: t.Optional[int] = None,
    ):
        """TODO."""
        _confidence = np.asfarray(confidence)

        if np.any(np.logical_or(_confidence <= 0.0, _confidence >= 1.0)):
            raise ValueError(
                "'confidence' must be in (0.0, 1.0) range (got {}.)".format(
                    confidence
                )
            )

        self.X = X
        self.y = y
        self.sample_num = sample_num
        self.verbose = verbose
        self.random_state = random_state

        self._extractor = extractor
        self._arguments_fit = arguments_fit if arguments_fit else {}
        self._arguments_extract = (
            arguments_extract if arguments_extract else {}
        )

        _half_sig_level = 0.5 * (1.0 - _confidence)
        self._crit_points_inds = np.hstack(
            (1.0 - _half_sig_level, _half_sig_level)
        )

        self._handle_output = {
            tuple: lambda args: args,
            dict: lambda args: (
                args["mtf_names"],
                args["mtf_vals"],
                args["mtf_time"],
            )
            if self._extractor.timeopt
            else (
                args["mtf_names"],
                args["mtf_vals"],
            ),
            pd.DataFrame: lambda args: (
                list(args.columns),
                args.values[0],
                args.values[1],
            )
            if self._extractor.timeopt
            else (
                list(args.columns),
                args.values[0],
            ),
        }

    def _handle_extract_ret(
        self,
        mtf_vals: np.ndarray,
        mtf_time: np.ndarray,
        args: t.Union[t.Tuple[t.List, ...], t.Dict[str, t.Any]],
        it_num: int,
    ) -> t.Tuple[np.ndarray, ...]:
        """Handle each .extraction method return value."""

        if self._extractor.timeopt:
            _, cur_mtf_vals, cur_mtf_time = self._handle_output[type(args)](
                args
            )
            mtf_time += cur_mtf_time

        else:
            _, cur_mtf_vals = self._handle_output[type(args)](args)

        mtf_vals[:, it_num] = cur_mtf_vals

        return mtf_vals, mtf_time

    def _extract_with_bootstrap(
        self, mtf_num: int
    ) -> t.Tuple[np.ndarray, ...]:
        """Extract metafeatures using bootstrapping."""
        if self.X is None:
            raise TypeError(
                "Fitted data not found. Please call 'fit' method first."
            )

        mtf_vals = np.zeros((mtf_num, self.sample_num), dtype=float)
        mtf_time = np.empty(0)

        if self._extractor.timeopt:
            mtf_time = np.zeros(mtf_num, dtype=float)

        if self.random_state is None:
            # Enforce pseudo-random behaviour to avoid previously set
            # random seeds out of this context
            np.random.seed()

        bootstrap_random_state = (
            self.random_state
            if self.random_state is not None
            else np.random.randint(2 ** 20 - 1)
        )

        for it_num in np.arange(self.sample_num):
            if self.verbose > 0:
                print(
                    "Extracting from sample dataset {} of {} ({:.2f}%)..."
                    "".format(
                        1 + it_num,
                        self.sample_num,
                        100.0 * (1 + it_num) / self.sample_num,
                    )
                )

            # Note: setting random state to prevent same sample indices due
            # to random states set during fit/extraction
            np.random.seed(bootstrap_random_state)
            bootstrap_random_state += 1

            sample_inds = np.random.randint(
                self.X.shape[0], size=self.X.shape[0]
            )

            X_sample = self.X[sample_inds, :]
            y_sample = self.y[sample_inds] if self.y is not None else None

            self._extractor.fit(X_sample, y_sample, **self._arguments_fit)

            mtf_vals, mtf_time = self._handle_extract_ret(
                mtf_vals=mtf_vals,
                mtf_time=mtf_time,
                args=self._extractor.extract(**self._arguments_extract),
                it_num=it_num,
            )

            if self.verbose > 0:
                print(
                    "Done extracting from sample dataset {}.\n".format(
                        1 + it_num
                    )
                )

        return mtf_vals, mtf_time

    def calc_conf_intervals(
        self, mtf_vals: np.ndarray, bootstrap_vals: np.ndarray
    ) -> np.ndarray:
        """TODO."""
        mtf_vals = np.expand_dims(mtf_vals, axis=1)

        diff_conf_int = np.quantile(
            a=bootstrap_vals - mtf_vals, q=self._crit_points_inds, axis=1
        ).T

        mtf_conf_int = -diff_conf_int + mtf_vals

        return mtf_conf_int

    def extract_original_metafeatures(
        self,
    ) -> t.Tuple[t.Sequence[str], t.Sequence[float]]:
        """TODO."""
        self._extractor.fit(self.X, self.y, **self._arguments_fit)
        ret_type = self._arguments_extract.get("out_type")
        self._arguments_extract["out_type"] = tuple
        mtf_names, mtf_vals = self._extractor.extract(
            **self._arguments_extract
        )[
            :2
        ]  # type: ignore

        if ret_type is not None:
            self._arguments_extract["out_type"] = ret_type

        else:
            self._arguments_extract.pop("out_type")

        return mtf_names, mtf_vals

    def extract_with_confidence(
        self,
    ) -> t.Tuple[
        t.Sequence[str], t.Sequence[float], t.Sequence[float], np.ndarray
    ]:
        """TODO."""

        if self.verbose > 0:
            print("Started metafeature extract with confidence interval.")
            print("Random seed:")
            print(
                " {} For extractor model: {}".format(
                    _internal.VERBOSE_BLOCK_END_SYMBOL,
                    self._extractor.random_state,
                )
            )

            print(
                " {} For bootstrapping: {}".format(
                    _internal.VERBOSE_BLOCK_END_SYMBOL, self.random_state
                )
            )

        mtf_names, mtf_vals = self.extract_original_metafeatures()

        bootstrap_vals, mtf_time = self._extract_with_bootstrap(
            mtf_num=len(mtf_names)
        )

        if self.verbose > 0:
            print("Finished metafeature extract with confidence interval.")
            print("Now getting confidence intervals...", end=" ")

        mtf_conf_int = self.calc_conf_intervals(mtf_vals, bootstrap_vals)

        if self.verbose > 0:
            print("Done.")

        return mtf_names, mtf_vals, mtf_time, mtf_conf_int
