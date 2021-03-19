"""TODO."""
import typing as t

import numpy as np
import pandas as pd

import pymfe._internal as _internal


class BootstrapExtractor:
    """TODO."""

    def __init__(
        self,
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

        self.sample_num = sample_num
        self.verbose = verbose
        self.random_state = random_state

        self._extractor = extractor
        self._arguments_fit = arguments_fit if arguments_fit else {}
        self._arguments_extract = (
            arguments_extract if arguments_extract else {}
        )

        self.X = np.empty(0)
        self.y = np.empty(0)
        self._fit = False
        self._mtf_names = []  # type: t.List[str]
        self._mtf_vals = []  # type: t.List[float]
        self._mtf_time = []  # type: t.List[float]

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

    def _extract_with_bootstrap(self, mtf_num: int) -> np.ndarray:
        """Extract metafeatures using bootstrapping."""
        mtf_vals = np.zeros((mtf_num, self.sample_num), dtype=float)

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

            args = self._extractor.extract(**self._arguments_extract)
            cur_mtf_vals = self._handle_output[type(args)](args)[1]
            mtf_vals[:, it_num] = cur_mtf_vals

            if self.verbose > 0:
                print(
                    "Done extracting from sample dataset {}.\n".format(
                        1 + it_num
                    )
                )

        return mtf_vals

    def fit(
        self, X: np.ndarray, y: t.Optional[np.ndarray] = None
    ) -> "BootstrapExtractor":
        """TODO."""
        self._extractor.fit(X, y, **self._arguments_fit)
        ret_type = self._arguments_extract.get("out_type")
        self._arguments_extract["out_type"] = tuple
        res = self._extractor.extract(**self._arguments_extract)
        mtf_names, mtf_vals = res[:2]  # type: ignore

        if ret_type is not None:
            self._arguments_extract["out_type"] = ret_type

        else:
            self._arguments_extract.pop("out_type")

        self._mtf_names = mtf_names
        self._mtf_vals = mtf_vals

        if len(res) >= 3:
            self._mtf_time = res[2]

        self.X = self._extractor.X
        self.y = self._extractor.y

        self._fit = True

        return self

    def calc_conf_intervals(self, bootstrap_vals: np.ndarray) -> np.ndarray:
        """TODO."""
        mtf_vals = np.expand_dims(self._mtf_vals, axis=1)

        diff_conf_int = np.quantile(
            a=bootstrap_vals - mtf_vals, q=self._crit_points_inds, axis=1
        ).T

        mtf_conf_int = -diff_conf_int + mtf_vals

        return mtf_conf_int

    def extract_with_confidence(
        self,
    ) -> t.Tuple[
        t.Sequence[str], t.Sequence[float], t.Sequence[float], np.ndarray
    ]:
        """TODO."""
        if not self._fit:
            raise TypeError(
                "Please call BootstrapExtractor.fit() method before "
                "BootstrapExtractor.extract_with_confidence()"
            )

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

        bootstrap_vals = self._extract_with_bootstrap(
            mtf_num=len(self._mtf_names)
        )

        if self.verbose > 0:
            print("Finished metafeature extract with confidence interval.")
            print("Now getting confidence intervals...", end=" ")

        mtf_conf_int = self.calc_conf_intervals(bootstrap_vals)

        if self.verbose > 0:
            print("Done.")

        return self._mtf_names, self._mtf_vals, self._mtf_time, mtf_conf_int