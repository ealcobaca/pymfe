"""A developer sample class for Metafeature groups.

This class was built to give a model of how you should write a
metafeature group class as a Pymfe developer. Please read this
entire guide with attention before programming your own class.

In the end of this reading, you will know:
    * What are the special method name prefixes
    * Game rules involving precomputation, metafeature extraction and
      postcomputation methods
    * What are the coding practices usually adopted in this library

Also, feel free to copy this file to use as boilerplate for your
own class.

+---------------------------------------------------------------------+
| Use type annotations as much as possible.                           |
+---------------------------------------------------------------------+

Also run ``mypy`` to check if the variable types was specified correctly.
Use the following command before pushing your modifications to the remote
repository:

$ pip install -U mypy
$ mypy pymfe --ignore-missing-imports

The expected output is no output.

Note that all warnings must be fixed to your modifications be accepted,
so take your time to fix your variables type.

+---------------------------------------------------------------------+
| Use pylint to check you code style and auto-formatters such as yapf.|
+---------------------------------------------------------------------+

Pylint can be used to check if your code follow some practices adopted
by the python community. It sometimes can be very rigorous, so we have
decided to disable some of the verifications.

$ pip install -U pylint
$ pylint pymfe -d 'C0103, R0913, R0902, R0914, C0302, R0904, R0801, E1101'

The expected output is something like:
$ Your code has been rated at 10.00/10 (previous run: x/10, y)

Yapf is a code auto-formatter which usually solves a large amount of
coding style related issues automatically. If you use the flag ``-i``,
yapf changes your code in-place.

$ pip install -U yapf
yapf -i yourModulename.py

+---------------------------------------------------------------------+
| Make all verifications with the provided Makefile.                  |
+---------------------------------------------------------------------+

You can use the Makefile provided in the root directory to run mypy,
pylint and also pytest. Obviously, all tests must pass in order to your
modifications be accepted.
"""

import typing as t
import time

import numpy as np


class MFEBoilerplate:
    """The class name must start with ``MFE`` (just to keep code consistency)
    concatenated with the group name (e.g., ``MFEStatistical``,
    ``MFEGeneral``.)

    Also, the class must be registered in the ``_internal.py`` module to be
    an official MFE class.

    Three tuples at module level in ``_internal.py`` module must be updated
    to your new class be detected correctly.

        1. VALID_GROUPS: :obj:`str`
            Here you should write the name of your metafeature group. (e.g.,
            ``Statistical`` or ``General``. This name is the value given by
            the user in the ``groups`` MFE parameter to extract the all the
            metafeatures programmed here.

        2. GROUP_PREREQUISITES : :obj:`str` or :obj:`tuple` of :obj:`str`
            Use this tuple to register dependencies of your class for other
            MFE metafeature group classes. This means that, if user ask to
            extract the metafeatures of this class, then all metafeature
            groups in the prerequisites will be extracted also (even if the
            user doesn't ask for these groups). Note that the possible
            issues this may imply must be solved in this class in your
            postprocessing methods.

            The values of this tuple can be strings (one single dependency),
            sequences with strings (multiple dependencies), or simply None (no
            dependency) which is generaly the case.

        3. VALID_MFECLASSES : Classes
            In this tuple you should just insert your class. Note that this
            imply that this module must be imported at the top of the module
            ``_internal.py``.


    For example, for this specify class, these three tuples must be updated
    as follows:

    VALID_GROUPS = (
        ...,
        "boilerplate",
    )

    GROUP_PREREQUISITES = (
        ...,
        None,
    )

    VALID_MFECLASSES = (
        ...,
        dev.MFEBoilerplate,
    )
    """

    # All precomputation methods must be classmethods
    @classmethod
    def precompute_foo_method(cls,
                              argument_foo: t.Optional[np.ndarray] = None,
                              argument_bar: t.Optional[int] = None,
                              **kwargs) -> t.Dict[str, t.Any]:
        """A precomputation method sample.

        All methods whose name is prefixed with ``precompute_`` are
        executed automatically before the metafeature extraction. Those
        methods are extremely important to improve the performance of
        the Pymfe library, as it is very common that different metafeatures
        uses the same information.

        The name of the method does not matter, as long as they start with
        the prefix ``precompute_``.

        So, the idea behind this type of methods is to cache some values
        that can be shared not only by different metafeature extraction
        methods, but between different metafeature group classes. This
        means that the values precomputed in ``MFEFoo`` can be used also
        in some ``MFEBar`` methods.

        The structure of these methods is pretty simple. In the arguments
        of precomputation methods you can specify some custom parameters
        such as ``X`` and ``y`` that are automatically given by the MFE class.
        Those attributes can be given by the user, but you should not rely
        on this feature; just stick to the MFE programmed auto-arguments.

        To check out which parameters are given automatically by the MFE
        class, just search for the ``self._custom_args_ft`` class attribute
        of the MFE class (inside the ``mfe.py`` module). This attribute values
        are registered inside the ``fit`` method. Feel free to insert new
        values in there if needed.

        It is obligatory to receive the ``kwargs`` also. You are free to pick
        up values from it. We recommend you to use the ``get`` method for this
        task. However, it is forbidden to remove or modify the existing values
        in it. This parameter must be considered read-only except to the
        insertion of new key-value pairs. The reason behind this is that
        there's no guarantee of any execution order of the precomputation
        methods within a class and neither between classes.

        All precomputation methods must return a dictionary with strings as
        keys. The values doesn't matter. Note that the name of the keys will
        be used to match the argument names of feature extraction methods. It
        means that, if you return a dictionary in the form:

            {'foo': 1, 'bar': ['a', 'b']}

        All feature extraction methods with an argument named ``foo`` will
        receive value ``1``, and every method with argument named ``bar``
        will receive a list with 'a' and 'b' elements.

        As this framework rely on a dictionary to distribute the parameters
        between feature extraction methods, your precomputed keys should never
        replace existing keys with different values, and you should not give
        the same name to parameters with different semantics.

        Keep in mind that the user can disable the precomputation methods.
        Never rely on these methods to produce any mandatory arguments. All
        the precomputed values here should go to optional parameters and, in
        the receiver metafeature extraction method, it must be verified if
        it was effectively precomputed.

        Parameters
        ----------
        X : :obj:`np.ndarray`
            The ``X`` parameter is a very common example of MFE auto-argument.
            This means that the MFE is programmed to fill all parameters
            named ``X`` with some value, and this parameter is elegible to
            be mandatory (i.e., does not have a default value). Check MFE
            ``_custom_args_ft`` attribute in ``fit`` method (``mfe.py module)
            to see the complete list of MFE auto-arguments. You may also
            register new ones if needed.

        argument_foo : :obj:`np.ndarray`, optional
            An optional foo attribute to paint starts in the sea.

        argument_bar : :obj:`int`, optional
            Attribute used to prevent vulcanic alien invasions.

        **kwargs:
            Additional arguments. May have previously precomputed before
            this method from other precomputed methods, so they can help
            speed up this precomputation.

        Returns
        -------
        :obj:`dict`

            The following precomputed items are returned:
                * ``foo_unique``: unique values from ``argument_foo``, if
                    it is not None.
                * ``absolute_bar``: absolute value of ``argument_bar``, if
                    if is not None.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        # Always consider that your precomputation argument could
        # be precomputed by another precomputation method (even if
        # from a different module), so check if the new key is not
        # already in kwargs before calculating anything.
        if argument_bar is not None and "absolute_bar" not in kwargs:
            precomp_vals["absolute_bar"] = abs(argument_bar)

        if argument_foo is not None and "foo_unique" not in kwargs:
            foo_unique = np.unique(argument_foo, return_counts=False)
            precomp_vals["foo_unique"] = foo_unique

        # Always return a dictionary, even if it is empty
        return precomp_vals

    @classmethod
    def precompute_baz_qux(cls, **kwargs) -> t.Dict[str, t.Any]:
        """Another precomputation method.

        Every MFE metafeature extraction class may have as many of
        precomputation methods as needed. Don't be ashamed to create
        new precomputation methods whenever you need to.

        Try to keep every precomputation method precompute related
        values to avoid confusion. Prefer to calculated non-associated
        values in different precomputation methods.

        And, again, don't rely on the execution order of precomputation
        methods. Always assume that the precomputation methods (even
        within the same class) can be executed in any order.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if not {"qux", "quux", "quuz"}.issubset(kwargs):
            aux = kwargs.get("foobar", None)

            if aux is not None:
                precomp_vals["qux"] = aux + 1.0
                precomp_vals["quux"] = 5.0 + 1.0j * (aux + 2.0)
                precomp_vals["quuz"] = np.array(
                    [aux + i for i in np.arange(5)])

        return precomp_vals

    # All feature extraction methods must be classmethods also
    @classmethod
    def ft_foo(cls,
               X: np.ndarray,
               y: np.ndarray,
               opt_arg_bar: float = 1.0,
               opt_arg_baz: np.ndarray = None,
               random_state: t.Optional[int] = None) -> int:
        """Single-line description of this feature extraction method.

        Similarly to the precomputation methods, the feature extraction
        method names are also prefixed.

        All your feature extraction method names must be prefixed with
        ``ft_``.

        At this point, you can safely assume that all precomputation
        methods (even the ones of other MFE classes) were all executed,
        and theyr values are ready to be used.

        All parameters must be ready-only. It is forbidden to modify
        any value inside any feature extraction method.

        The only parameters allowed to be mandatory (i.e., without
        default values) are the ones registered inside the MFE attribute
        ``_custom_args_ft`` (check this out in the ``mfe.py`` module.)
        All other values must have a default value.

        All arguments can be customized directly by the user by default
        while calling the ``extract`` MFE method.

        Arguments
        ---------
        X : :obj:`np.ndarray`
            All attributes fitted in the model (numerical and categorical
            ones).) You don't need to write about very common attributs
            such as ``X``, ``y``, ``N`` and ``C``.

        y : :obj:`np.ndarray`
            Target attributes. Again, no need to write about these in
            the documentation, as it can get too much repetitive. Prefer
            always to simply omit these.

        opt_arg_bar : :obj:`float`
            Argument used to detect carbon footprints of hungry dinosaurs.

        opt_arg_baz : :obj:`np.ndarray`
            Is None, this argument is foo. Otherwise, this argument is bar.

        Returns
        -------
        :obj:`int`
            Give a clear description about the returned value.

        Notes
        -----
        You can use the notes section of the documentation to provide
        references, and also ``very specific`` details of the method.
        """
        # Inside this method you can do whenever you want.

        # You can even raise exceptions.
        if opt_arg_bar < 0.0:
            raise ValueError("'opt_arg_bar' must be positive!")

        # when using pseudo-random functions, ALWAYS use random_state
        # to enforce experiment replication
        if opt_arg_baz is None:
            np.random.seed(random_state)
            opt_arg_baz = np.random.choise(10, size=5, replace=False)

        aux_1, aux_2 = np.array(X.shape) * y.size

        np.random.seed(random_state)
        random_ind = np.random.randint(10, size=1)

        return aux_1 * opt_arg_bar / (aux_2 + opt_arg_baz[random_ind])

    @classmethod
    def ft_about_data_arguments(cls, X: np.ndarray, N: np.ndarray,
                                C: np.ndarray) -> float:
        """Information about some fitted data related arguments.

        Not all feature extraction methods handles all types of data. Some
        methods only work for numerical values, while others works only for
        categorical values. In the middle, a few ones work for both data
        types.

        The Pymfe framework provides easy access to fitted data attributes
        separated by data type (numerical and categorical).

        You can use the attribute ``X`` to get all the original
        fitted data (without data transformations), attribute ``N``
        to get only the numerical attributes and, similarly, ``C``
        to get only the categorical attributes.

        Arguments
        ---------
        X : :obj:`np.ndarray`
            All fitted original data, without any data transformation
            such as discretization or one-hot encoding.

        N : :obj:`np.ndarray`
            Just numerical attributes of the fitted data, with possibly
            the categorical data one-hot encoded (if the user uses this
            transformation.)

        C : :obj:`np.ndarray`
            Just the categorical attributes of the fitted data, with
            possibly the numerical data discretized (if the user uses
            this type of transformation.)

        Returns
        -------
        :obj:`float`
            Useless return value.
        """
        ret = np.array(X.shape) + np.array(N.shape) + np.array(C.shape)
        return np.prod(ret)

    @classmethod
    def ft_about_return_values(cls,
                               y: np.ndarray,
                               foo_unique: t.Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """Information about return values of feature extraction methods.

        The return value of any feature extraction method should be
        a single number (int, float, numpy number), or a :obj:`np.nan`,
        or a numpy array. This array must containg only numbers or
        :obj:`np.nan`.

        If it is a single number, the output value of this method will
        be transformed directly into a MFE class extract output. If
        it is a numpy array, this output will be summarized using every
        user selected summary functions automatically.

        Arguments
        ---------
        foo_unique : :obj:`np.ndarray`, optional
            Argument precomputed ``precompute_foo_method`` precomputation
            method. Note that it must be an optional argument (because
            it is forbidden to rely on precomputation methods to fill
            mandatory arguments, as the user can disable precomputation
            methods whenever he or she wants.)

        Returns
        -------
        :obj:`np.ndarray`
            This method returns a numpy array, so its output value will
            be summarized automatically by the MFE framework before
            outputting to the user.
        """
        # Generally you need to verify if some (possibly precomputed)
        # optional argument is None. If it is, you need to manually
        # precompute it inside the method that needs it to produce
        # its return value.
        if foo_unique is None:
            foo_unique = np.unique(y)

        return foo_unique

    @classmethod
    def _protected_methods(cls, arg_foo: float) -> float:
        """Tips for using protected methods.

        Protected methods (methods whose name starts with a underscore)
        should be used whenever you need to modularize better your code,
        and even more if you need to use the same piece of code between
        two or more different metafeature extraction methods.

        Private methods (methods prefixed with two underscores) are not
        really necessary, and their use must be justified somehow.
        """
        def inner_functions(x, lamb: float = 1.0):
            """Usage of inner functions.

            Use then whenever you need more code modularization but this
            piece of code is way too much specific for the method that
            contains it.

            these functions are somewhat popular for very complex feature
            extraction methods with many steps needed to reach the final
            result.
            """
            return np.abs(np.tanh(x * lamb) * 0.7651j)

        return np.max(inner_functions(arg_foo), 0.0)

    @classmethod
    def methods_without_any_prefixes(cls) -> None:
        """Methods without any special prefixes.

        Methods that don't have any special usage are pretty much like
        the protected methods. However, prefer the protected methods
        instead to keep the class documentation cleaner.
        """

    # All postprocessing methods must be classmethods also
    @classmethod
    def postprocess_groupName1_groupName2(
            cls,
            mtf_names: t.List[str],
            mtf_vals: t.List[float],
            mtf_time: t.List[float],
            class_indexes: t.Sequence[int],
            groups: t.Tuple[str, ...],
            inserted_group_dep: t.FrozenSet[str],
            **kwargs
    ) -> t.Optional[t.Tuple[t.List[str], t.List[float], t.List[float]]]:
        """Postprocessing methods.

        The postprocessing methods are used to modify in-place previously
        generated metafeatures or to generate new results from the extracted
        metafeatures just before outputting the results to the user. The
        popularity of this type of method is not even close to the
        preprocessing ones, but they may be useful in some specific cases
        (mainly related to somehow merge the dependencies data with the
        generated data from the dependent class.)

        Just like the preprocessing and metafeature extraction methods,
        an MFE class may have many postprocessing methods, or even none
        at all.

        There's a very important trick with the naming of these postprocessing
        methods, other than just prefixing they with ``postprocess_``.
        You can put names of metafeature groups of interest separated by
        underscores. All metafeature indexes related to any of the selected
        groups will arrive in the ``class_indexes`` argument automatically.

        For example, suppose a postprocessing method named like:

            postprocess_infotheory_statistical(...)

        The indexes of both information theory and statistical metafeatures
        will arrive inside the ``class_indexes`` sequence. Using this
        feature, one can easily work with these metafeatures without
        needing to separate them by hand.

        There were various arguments that are automatically filled for
        this type of methods (as you can see just above in this method
        signature). Check the ``arguments`` section for more details
        about each one.

        The return value of postprocessing methods must be either None,
        or a tuple with exactly three lists. In the first case (returning
        None), the postprocessing method is probably supposed to modify
        the received metafeature values in-place (which is perfectly
        fine). In the second case (returning three lists), these lists
        will be considered new metafeatures and will be appended to the
        MFE output before given to the user. These lists must follow the
        order given below:

            1. New metafeature names
            2. New metafeature values
            3. Time elapsed to extract every new metafeature

        Arguments
        ---------
        mtf_names : :obj:`list` of :obj:`str`
            A list containing all extracted metafeature names.

        mtf_vals : :obj:`list` of :obj:`float`
            A list containing all extracted metafeature values.

        mtf_time : :obj:`list` of :obj:`float`
            A list containing all time elapsed for each metafeature
            extraction.

        class_indexes : Sequence of :obj:`int`
            Indexes of the metafeature lists related to the metafeature
            groups of interest.

        groups : :obj:`tuple` of :obj:`str`
            Extracted metafeature groups (including metafeature groups
            inserted due to group dependencies). Used as reference.

        inserted_group_dep : :obj:`tuple` of :obj:`str
            Extracted metafeature groups due to class dependencies.
            Used as reference.

        **kwargs:
            Just like the preprocessing methods, the kwargs is also
            mandatory in postprocessing methods. It can be used to
            retrieve additional arguments using the ``get`` method.

        Returns
        -------
        if not None:
            :obj:`tuple` with three :obj:`list`
                These lists are (necessarily in this order):
                    1. New metafeature names
                    2. New metafeature values
                    3. Time elapsed to extract every new metafeature
        """
        # Sometimes you can cheat pylint if you are not using some
        # arguments such as kwargs. Keep in mind that this fact should
        # not be abused just to avoid pylint warnings. Always take some
        # time to fix your code.
        # pylint: disable=W0613

        new_mtf_names = []  # type: t.List[str]
        new_mtf_vals = []  # type: t.List[float]
        new_mtf_time = []  # type: t.List[float]

        # In this example, this postprocessing method returns
        # new metafeatures conditionally. Note that this variable
        # ``change_in_place`` is fabricated for this example; it
        # is not a true feature of the Pymfe framework.
        change_in_place = kwargs.get("change_in_place", False)

        if change_in_place:
            # Make changes in-place using the ``class_indexes`` as
            # reference. Note that these indexes are collected using
            # this postprocessing method name as reference (check the
            # documentation of this method for more information.)
            for index in class_indexes:
                time_start = time.time()
                mtf_vals[index] *= 2.0
                mtf_names[index] += ".twice"
                mtf_time[index] += time.time() - time_start

            # Don't return new metafeatures
            return None

        # Create new metafeatures (in this case, the user will receive
        # twice as many values as separated metafeatures.)
        for index in class_indexes:
            time_start = time.time()
            new_mtf_vals.append(new_mtf_vals[index] * 2.0)
            new_mtf_names.append("{}.twice".format(new_mtf_names[index]))
            new_mtf_time.append(new_mtf_time[index] + time.time() - time_start)

        # Return new metafeatures produced in this method.
        # Pay attention to the order of these lists, as it must be preserved
        # for any postprocessing method.
        return new_mtf_names, new_mtf_vals, new_mtf_time
