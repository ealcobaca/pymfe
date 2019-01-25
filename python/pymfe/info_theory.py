"""Module dedicated to extraction of Information Theory Metafeatures."""
import typing as t
import itertools

import numpy as np
import scipy
import sklearn.metrics


class MFEInfoTheory:
    """To do this documentation."""

    @classmethod
    def _entropy(cls, values: t.Union[np.ndarray, t.List]) -> float:
        """Calculate entropy within array 'values'."""
        _, counts = np.unique(values, return_counts=True)
        return scipy.stats.entropy(counts, base=2)

    @classmethod
    def _joint_prob_mat(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute joint probability matrix P(a, b), a in x and b in y."""
        x_vals = np.unique(x)
        y_vals = np.unique(y)

        joint_prob_mat = np.array([
            sum((x == x_val) & (y == y_val))
            for y_val, x_val in itertools.product(y_vals, x_vals)
        ]).reshape((y_vals.size, x_vals.size)) / x.size

        return joint_prob_mat

    @classmethod
    def _conc(cls, x: np.ndarray, y: np.ndarray,
              epsilon: float = 1.0e-8) -> float:
        """Compute concentration coefficient between two arrays ``x`` and ``y``.

        Used for methods ``ft_class_conc`` and ``ft_attr_conc``.

        Args:
            epsilon (:obj:`float`, optional): small numeric value to
                avoid division by zero.
        """
        pij = MFEInfoTheory._joint_prob_mat(x, y)

        isum = pij.sum(axis=1)
        jsum2 = sum(pij.sum(axis=0)**2.0)

        conc = ((((pij**2.0).T / isum).sum().sum() - jsum2) /
                (1.0 - jsum2 + epsilon))

        return conc

    @classmethod
    def ft_attr_conc(cls, C: np.ndarray) -> np.ndarray:
        """Compute concentration coef. of each pair of distinct attributes."""

        _, num_col = C.shape

        col_permutations = itertools.permutations(range(num_col), 2)

        attr_conc = np.array([
            MFEInfoTheory._conc(C[:, col_a], C[:, col_b])
            for col_a, col_b in col_permutations
        ])

        return attr_conc

    @classmethod
    def ft_attr_ent(cls, C: np.ndarray) -> t.Union[np.ndarray, np.float]:
        """Calculates Shannon entropy for each attribute."""
        try:
            return np.apply_along_axis(
                func1d=MFEInfoTheory._entropy, axis=0, arr=C)

        except ValueError:
            return np.nan

    @classmethod
    def ft_class_conc(cls, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute concentration coefficient between each attr. and class."""
        return np.apply_along_axis(
            func1d=MFEInfoTheory._conc, axis=0, arr=C, y=y)

    @classmethod
    def ft_class_ent(cls, y: np.ndarray) -> t.Union[np.ndarray, np.float]:
        """Calculates class Shannon entropy."""
        return MFEInfoTheory._entropy(y)

    @classmethod
    def ft_eq_num_attr(cls, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Number of attributes equivalent for a predictive task."""
        ent_class = MFEInfoTheory._entropy(y)
        mutual_info = MFEInfoTheory.ft_mut_inf(C, y)

        _, num_col = C.shape

        return num_col * (ent_class / sum(mutual_info))

    @classmethod
    def ft_joint_entropy(cls, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate Joint entropy between each attribute and class."""

        def joint_entropy(x: np.ndarray,
                          y: np.ndarray,
                          epsilon: float = 1.0e-8) -> float:
            joint_prob_mat = MFEInfoTheory._joint_prob_mat(x, y) + epsilon

            joint_entropy = np.multiply(joint_prob_mat,
                                        np.log2(joint_prob_mat)).sum().sum()

            return -1.0 * joint_entropy

        joint_entropy = np.apply_along_axis(
            func1d=joint_entropy, axis=0, arr=C, y=y)

        return joint_entropy

    @classmethod
    def ft_mut_inf(cls, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mutual information between each attribute and class."""
        return np.apply_along_axis(
            func1d=sklearn.metrics.mutual_info_score,
            axis=0,
            arr=C,
            labels_pred=y)

    @classmethod
    def ft_ns_ratio(cls, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute noisiness of attributes."""
        ent_attr = sum(MFEInfoTheory.ft_attr_ent(C))
        mutual_info = sum(MFEInfoTheory.ft_mut_inf(C, y))

        return (ent_attr - mutual_info) / mutual_info


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()

    res = MFEInfoTheory.ft_joint_entropy(iris.data, iris.target)

    print(res)
