"""Metafeature extraction with empirical bootstrap confidence intervals."""
import typing as t

import numpy as np
import pandas as pd
import tqdm

import pymfe._internal as _internal


class BootstrapExtractor:
    """Extract metafeatures with empirical bootstrap confidence intervals."""

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
        """Extract metafeatures with confidence intervals.

        The method used is the Empirical Bootstrap.

        Please refer to 'MFE.extract_with_confidence' documentation for
        more details.
        """
        _confidence = np.asfarray(confidence)

        if np.any(np.logical_or(_confidence <= 0.0, _confidence >= 1.0)):
            raise ValueError(
                "'confidence' must be in (0.0, 1.0) range (got {}.)".format(
                    confidence
                )
            )

        if _confidence.ndim == 0:
            _confidence = np.expand_dims(_confidence, axis=0)

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

        self.confidence = _confidence
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
            else np.random.randint(np.iinfo(np.uint32).max + 1)
        )

        if self.verbose >= 1:
            print("Now extracting metafeatures from resampled data.")

        for it_num in tqdm.auto.tqdm(
            np.arange(self.sample_num), disable=self.verbose <= 0
        ):
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

        if self.verbose >= 1:
            print("Done extracting metafeatures from resampled data.")

        return mtf_vals

    def fit(
        self, X: np.ndarray, y: t.Optional[np.ndarray] = None
    ) -> "BootstrapExtractor":
        """Fit data into the model."""

        if self.verbose >= 1:
            print(
                "Began the metafeature extraction with confidence intervals "
                "process."
            )
            print("Now extracting metafeatures from original sample.")

        self._extractor.fit(X, y, **self._arguments_fit)
        ret_type = self._arguments_extract.get("out_type")
        self._arguments_extract["out_type"] = tuple
        res = self._extractor.extract(**self._arguments_extract)
        mtf_names, mtf_vals = res[:2]  # type: ignore

        if ret_type is not None:
            self._arguments_extract["out_type"] = ret_type

        else:
            self._arguments_extract.pop("out_type")

        if self.verbose >= 1:
            print(
                "Done extracting metafeatures from original sample "
                "(total of {} metafeatures).".format(len(mtf_names))
            )

        self._mtf_names = mtf_names
        self._mtf_vals = mtf_vals

        if len(res) >= 3:
            self._mtf_time = res[2]

        self.X = self._extractor.X
        self.y = self._extractor.y

        self._fit = True

        return self

    def calc_conf_intervals(self, bootstrap_vals: np.ndarray) -> np.ndarray:
        """Calculate bootstrap confidence intervals.

        Parameters
        ----------
        bootstrap_vals : :obj:`np.ndarray`
            Metafeatures extracted from bootstrap resampling (check
            `BootstrapExtractor.extract_with_confidence` method
            documentation for more information). Must have shape
            (metafeature_num, resample_num).

        Returns
        -------
        :obj:`np.ndarray`
            Confidence interval for the data sample metafeatures. Will
            have shape (metafeature_num, 2 * C), where C is the number
            of distinct confidence levels fitted in this model. For each
            column, the even indices (starting from 0) are the confidence
            intervals lower bounds, while the odd indices are confidence
            intervals upper bounds.
        """
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
        """Extract metafeatures with empirical bootstrap confidence intervals.

        Returns
        -------
        tuple of sequences
            The sequences of values are organized as follows:
            Metafeature names, metafeature values, metafeature extraction time
            (if provided by the fitted MFE extractor. Otherwise, return an
            empty list) and the confidence intervals.
            Check the `Bootstrap.calc_conf_intervals` method documentation for
            a more detailed explanation about the confidence intervals data
            shape.
        """
        if not self._fit:
            raise TypeError(
                "Please call BootstrapExtractor.fit() method before "
                "BootstrapExtractor.extract_with_confidence()"
            )

        if self.verbose > 0:
            print(
                "Started data resampling with bootstrap with the following "
                "configurations:"
            )

            print(
                "{} Total data resamples: {}".format(
                    _internal.VERBOSE_BLOCK_MID_SYMBOL, self.sample_num
                )
            )
            print(
                "{} Confidence levels used: {} (total of {}).".format(
                    _internal.VERBOSE_BLOCK_MID_SYMBOL,
                    self.confidence,
                    len(self.confidence),
                )
            )
            print(
                "{} Random seeds:".format(_internal.VERBOSE_BLOCK_END_SYMBOL)
            )
            print(
                "   {} For extractor model: {}".format(
                    _internal.VERBOSE_BLOCK_MID_SYMBOL,
                    self._extractor.random_state,
                )
            )

            print(
                "   {} For bootstrapping: {}".format(
                    _internal.VERBOSE_BLOCK_END_SYMBOL, self.random_state
                )
            )

        bootstrap_vals = self._extract_with_bootstrap(
            mtf_num=len(self._mtf_names)
        )

        if self.verbose > 0:
            print("Finished data resampling with bootstrap.")
            print("Now calculating confidence intervals...", end=" ")

        mtf_conf_int = self.calc_conf_intervals(bootstrap_vals)

        if self.verbose > 0:
            print("Done.")

        return self._mtf_names, self._mtf_vals, self._mtf_time, mtf_conf_int
