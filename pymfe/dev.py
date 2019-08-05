"""A developer sample class for Metafeature groups.

This class was built to give a model of how you should write a
metafeature group class as a Pymfe developer. Please read this
entire guide with attention before programming your own class.

In the end of this reading, you will know:
    * How to register your class as a valid MFE metafeature class
    * What are the special method name prefixes and how to use them
      properly
    * What are the game rules involving precomputation, metafeature
      extraction and post-computation methods
    * What are the coding practices usually adopted in this library

Also, feel free to copy this file to use as template for your own
class.

First, some tips and tricks which may help you follow the code
standards stabilished in this library.

1. Use type annotations as much as possible.

Always run ``mypy`` to check if the variable types was specified correctly.
You can install it with pip using the following code:

$ pip install -U mypy

Use the following command before pushing your modifications to the remote
repository:

$ mypy pymfe --ignore-missing-imports

The expected output for this command is no output.

Note that all warnings must be fixed to your modifications be accepted in
the master branch, so take your time to fix your variable types carefully.


2. Use Pylint to check your code style and auto-formatters such as Yapf

Pylint can be used to check if your code follow some coding practices
adopted by the python community. You can install with with pip using the
following command:

$ pip install -U pylint

It can be harsh sometimes, so we have decided to disable some of the
verifications. You can use the following command to check if your code
met the standards stabilished in this library,

$ pylint pymfe -d 'C0103, R0913, R0902, R0914, C0302, R0904, R0801, E1101'

The expected output is something like:
$ Your code has been rated at 10.00/10 (previous run: x/10, y)

Your code will not be accepted in the master branch unless it gets the
maximum pylint score.

Yapf is a code auto-formatter which usually solves a large amount of
coding style related issues automatically.

$ pip install -U yapf

If you use the flag ``-i``, Yapf changes your code in-place.

$ yapf -i yourModulename.py


3. Make all verifications with the provided Makefile.

You can use the Makefile provided in the root directory to run mypy,
pylint, and also pytest. Obviously, all tests (both for coding style
and programming logic) must pass in order to your modifications be
accepted.

You can use the tag 'test-cov' for make test and get the coverage:

$ make test-cov

You can use the tag 'test' for make only tests:

$ make test

You can use the tag 'code-check' for chack the mypy, pylint and pep8:

$ make code-check

Note:
    You should not forget to create tests for all new functionalities that you
    implemented. The test can be found in './tests/' fold.
"""

import typing as t
import time

import numpy as np


class MFEBoilerplate:
    """The class name must start with ``MFE`` (just to keep code consistency)
    concatenated with the corresponding metafeature group name (e.g.,
    ``MFEStatistical`` or ``MFEGeneral``.)

    Also, the class must be registered in the ``_internal.py`` module to be
    an official MFE class. Three tuples at module level in ``_internal.py``
    module must be updated to your new class be detected correctly:

        1. VALID_GROUPS: :obj:`str`
            Here you should write the name of your metafeature group. (e.g.,
            ``Statistical`` or ``General``. This name is the value that will
            be given by the user in the ``groups`` MFE parameter to extract
            the all the metafeatures programmed here.

        2. GROUP_PREREQUISITES : :obj:`str` or :obj:`tuple` of :obj:`str`
            Use this tuple to register other MFE metafeature group classes
            as dependencies of your class. This means that, if the user ask
            to extract the metafeatures of this class, then all metafeature
            groups in the prerequisites will also be extracted also (even if
            the user doesn't ask explicity for these groups). Note that the
            possible consequences this may imply must be solved in this class
            in your postprocessing methods (these methods will be explained
            later in this same guide.)

            The values of this tuple can be strings (which means one single
            dependency), sequences with strings (which means your class has
            multiple dependencies), or simply None (which means your class
            has no dependencies). Generally your class will not have any
            dependency, so stick to the last option if you are not sure so
            far.

        3. VALID_MFECLASSES : MFE Classes
            In this tuple you should just insert your class. Note that this
            imply that this module must be imported at the top of the module
            ``_internal.py``.

        These three tuples have one-to-one correspondence using the indexes,
        so the order of values does matter.

    For example, for this specific templace class, these three tuples must be
    updated as follows:

    # All tuples below can be found in ``_internal.py`` module.

    # First, choose carefully a metafeature group name. This value will be
    # used directly by the user when extracting the metafeatures programmed
    # in this class, so it must be meaningful and as short as possible.
    VALID_GROUPS = (
        ...,
        "boilerplate",
    )

    # Generally your class will not have any dependency, so will should
    # just register ``None`` as prerequisites.
    GROUP_PREREQUISITES = (
        ...,
        None,
    )

    # Just put this class in this tuple. Remember to import this module
    # in the ``_internal.py`` module.
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
        executed automatically before the metafeature extraction. These
        methods are extremely important to improve the performance of
        the Pymfe library, as it is very common that different metafeature
        extraction methods uses the same very information.

        The idea behind this type of methods is to fill up a shared cache
        with all values that can be shared by different metafeature extraction
        methods, and also between different metafeature group classes. This
        means that the values precomputed in ``MFEFoo`` class can also be used
        in the ``MFEBar`` class methods.

        The name of the method does not matter, as long as it starts with
        the prefix ``precompute_``. This prefix is used to tell the Pymfe
        framework that this is a precomputation method. As you will see during
        this guide, the Pymfe rely heavily on prefixes in the method names, so
        it is important that you don't forget them, and use them appropriately.

        The structure of these precomputation methods is pretty simple. In the
        arguments you can specify custom parameters such as ``X`` and ``y``
        that are automatically given by the MFE class. Those attributes can be
        registered in a special attribute in the MFE class, or also given by
        the user, but you should not rely on this feature; just stick to the
        MFE registered arguments, and let all user-customizable attributes
        have a default value.

        To check out which parameters are given automatically by the MFE
        class, just search for the ``_custom_args_ft`` instance attribute of
        the MFE class (inside the ``mfe.py`` module). This attribute value
        is given inside the ``fit`` method. Feel free to insert new values in
        there if needed.

        It is obligatory to receive the ``kwargs`` in every precomputation
        method. You are free to pick up values from it. We recommend you to
        use the ``get`` method for this task. However, it is forbidden to
        remove or modify the existing values in it. This parameter must be
        considered ``read-only`` except to the insertion of new key-value
        pairs. The reason behind this is that there's no guarantee of any
        execution order of the precomputation methods within a class and
        neither between classes, so all precomputation methods must have
        the chance to read the same values.

        All precomputation methods must return a dictionary with strings as
        keys. The value data type can be anything. Note that the name of the
        keys will be used later to match the argument names of feature
        extraction methods. It means that, if you return a dictionary in the
        form: {'foo': 1, 'bar': ('a', 'b')}, then all feature extraction
        methods with an argument named ``foo`` will receive value ``1``, and
        every method with argument named ``bar`` will receive a tuple with 'a'
        and 'b' elements.

        As this framework rely on a dictionary to distribute the parameters
        between feature extraction methods, your precomputed keys should never
        replace existing keys with different values, and you should not give
        the same name to parameters with different semantics.

        Keep in mind that the user can disable the precomputation methods.
        Never rely on these methods to produce any mandatory arguments. All
        the precomputed values here should go to optional parameters and all
        receiver metafeature extraction methods must be verify if all values
        were effectively precomputed. If not, these methods must compute
        themselves these arguments.

        Parameters
        ----------
        argument_foo : :obj:`np.ndarray`, optional
            Always give clear and meaningful description to every argument.
            You can skip obvious arguments such as ``X`` or ``y``.

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
        new precomputation methods whenever you think it will help
        to improve the performance of the package.

        Try to keep every precomputation method precompute only related
        values to avoid confusion. Prefer to calculate non-associated
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
    def ft_foo(
            cls,
            X: np.ndarray,
            y: np.ndarray,
            random_state: t.Optional[int] = None,
            opt_arg_bar: float = 1.0,
            opt_arg_baz: np.ndarray = None,
    ) -> int:
        """Single-line description of this feature extraction method.

        Similarly to the precomputation methods, the feature extraction
        method names are also prefixed.

        All your feature extraction method names must be prefixed with
        ``ft_``.

        At this point, you can safely assume that all precomputation
        methods (even the ones of other MFE classes) were all executed,
        and their values are ready to be used as arguments. The same
        value provided as the key of some precomputed value is used to
        match directly the parameter name. All parameters must be treated
        as read-only values; it is forbidden to modify any value inside
        any feature extraction method.

        The only arguments allowed to be mandatory (i.e., arguments without
        any default value) are the ones registered inside the MFE attribute
        ``_custom_args_ft`` (check this out in the ``mfe.py`` module.)
        All other values must have a default value, with no exception.

        Remember that all arguments can be customized directly by the user
        while calling the ``extract`` MFE method.

        Arguments
        ---------
        X : :obj:`np.ndarray`
            All attributes fitted in the model (numerical and categorical
            ones). You don't need to write about very common arguments
            such as ``X``, ``y``, ``N`` and ``C``. In fact, you are
            encouraged to just omit these.

        y : :obj:`np.ndarray`
            Target attributes. Again, no need to write about these type of
            arguments in the method documentation, as it can get way too
            much repetitive without any information gain.

        random_state : :obj:`int`, optional
            Extremely important argument. This one is a feature from the
            MFE framework. If your method has ANY pseudo-random behaviour,
            you should use specifically this argument to provide the random
            seed. In this case, it would be nice if you write about what
            is the random behaviour of your method to make clear to the
            user why he or she ever needs a random seed in the first place.

        opt_arg_bar : :obj:`float`
            Argument used to detect carbon footprints of hungry dinosaurs.

        opt_arg_baz : :obj:`np.ndarray`
            If None, this argument is foo. Otherwise, this argument is bar.

        Returns
        -------
        :obj:`int`
            Give a clear description about the returned value.

        Notes
        -----
        You can use the notes section of the documentation to provide
        references, and also ``very specific`` details of the method.
        """
        # Inside the feature extraction method you can do whenever you
        # want, just make sure to return a number or a numpy array.

        # You can even raise ValueError and TypeError exceptions.
        if opt_arg_bar < 0.0:
            raise ValueError("'opt_arg_bar' must be positive!")

        # When using pseudo-random functions, ALWAYS use random_state
        # to enforce experiment replication
        if opt_arg_baz is None:
            np.random.seed(random_state)
            opt_arg_baz = np.random.choice(10, size=5, replace=False)

        aux_1, aux_2 = np.array(X.shape) * y.size

        np.random.seed(random_state)
        random_ind = np.random.randint(10, size=1)

        return aux_1 * opt_arg_bar / (aux_2 + opt_arg_baz[random_ind])

    @classmethod
    def ft_about_data_arguments(cls, X: np.ndarray, N: np.ndarray,
                                C: np.ndarray, y: np.ndarray) -> int:
        """Information about some arguments related to fitted data.

        Not all feature extraction methods handles all type of data. Some
        methods only work for numerical values, while others works only for
        categorical values. A few ones work for both data types, but this
        is generally not the case.

        The Pymfe framework provides easy access to the fitted data
        attributes separated by data type (numerical and categorical).

        You can use the attribute ``X`` to get all the original fitted
        data (without any data transformations), attribute ``N`` to get
        only the numerical attributes and, similarly, ``C`` to get only
        the categorical attributes.

        Arguments
        ---------
        X : :obj:`np.ndarray`
            All fitted original data, without any data transformation
            such as discretization or one-hot encoding.

        N : :obj:`np.ndarray`
            Just numerical attributes of the fitted data, with possibly
            categorical data one-hot encoded (if the user uses this
            type of transformation.)

        C : :obj:`np.ndarray`
            Just the categorical attributes of the fitted data, with
            possibly numerical data discretized (if the user uses
            this type of transformation.)

        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`int`
            Useless return value.

        Notes
        -----
        You can even receive more than one of these attributes in the
        same method, but keep in mind that this may cause confusion as
        the user may enable or disable data transformations (encoding
        for categorical values and discretization for numerical values).
        """
        ret = np.array(X.shape) + np.array(N.shape) + np.array(C.shape)
        return np.prod(ret) * y.size

    @classmethod
    def ft_about_return_values(cls,
                               y: np.ndarray,
                               foo_unique: t.Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """Information about return values of feature extraction methods.

        The return value of any feature extraction method should be
        a single number (int, float, numpy number), or a :obj:`np.nan`,
        or a numpy array. This array must contain only numbers or
        :obj:`np.nan`.

        If the return value is a single number, the output value of this
        method will be transformed directly into a MFE class extract output.
        If it is a numpy array, then this output will automatically be
        summarized using every user selected summary functions.

        Arguments
        ---------
        foo_unique : :obj:`np.ndarray`, optional
            Argument precomputed in the ``precompute_foo_method``
            precomputation method. Note that it must be an optional
            argument (because it is forbidden to rely on precomputation
            methods to fill mandatory arguments, as the user can disable
            precomputation methods whenever he or she wants.) Note also
            that the argument name must match exatcly the corresponding
            dictionary key given inside the precomputation method.

        Returns
        -------
        :obj:`np.ndarray`
            This method returns a numpy array, so its output value will
            be summarized automatically by the MFE framework before
            outputting to the user.
        """
        # Generally you need to verify if some (possibly precomputed)
        # optional argument is None. If this is the case, you need to
        # manually compute it inside the method that needs it.
        if foo_unique is None:
            # If ``foo_unique``, it means probably that the user
            # disabled the precomputations, so we need to compute
            # it now as this argument is needed to compute this
            # method output.
            foo_unique = np.unique(y)

        return -1.0 * foo_unique

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

            Use them whenever you need more code modularization but this
            piece of code is way too much specific for the method that
            contains it.

            These functions are kind of useful for very complex feature
            extraction methods with many steps needed to reach the final
            result. In that case, consider creating an separated inner
            function for every step.
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
            cls, mtf_names: t.List[str], mtf_vals: t.List[float],
            mtf_time: t.List[float], class_indexes: t.Sequence[int],
            groups: t.Tuple[str, ...], inserted_group_dep: t.FrozenSet[str],
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

        All postprocessing methods receive all previously extracted
        metafeatures from every MFE class. It will not receive just the
        metafeatures related to the metafeature extraction methods of this
        class. It is very import to keep this in mind.

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
        needing to separate them by hand. Of course, you can give as
        many metafeature group names as needed. If you need them
        all, then simply don't put any metafeature group name, as every
        metafeature is an metafeature of interest in this case.

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
            A list containing all previously extracted metafeature names.

        mtf_vals : :obj:`list` of :obj:`float`
            A list containing all previously extracted metafeature values.

        mtf_time : :obj:`list` of :obj:`float`
            A list containing all time elapsed for each metafeature
            previously extracted.

        class_indexes : Sequence of :obj:`int`
            Indexes of the metafeatures related to this method ``groups of
            interest``. The ``groups of interest`` are the metafeature groups
            whose name are in this method's name after the ``postprocess_``
            prefix, separated with underscores (in this example, they are
            ``groupName1`` and ``groupName2``.)

            If it is not clear for you so far, the metafeatures received
            in this method are all the metafeatures extracted in every MFE
            classes, not just the ones related to this class. Then, this
            argument can be used as reference to target only the metafeatures
            effectively used in this postprocessing method.

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
        # Sometimes you can cheat pylint in case you are not using some
        # arguments, such as kwargs. Keep in mind that this fact should
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
            new_mtf_vals.append(-1.0 * new_mtf_vals[index])
            new_mtf_names.append("{}.negative".format(new_mtf_names[index]))
            new_mtf_time.append(new_mtf_time[index] + time.time() - time_start)

        # Return new metafeatures produced in this method. Pay attention to the
        # order of these lists, as it must be preserved for any postprocessing
        # method.
        return new_mtf_names, new_mtf_vals, new_mtf_time
