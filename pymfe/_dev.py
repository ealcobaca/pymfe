"""A developer sample class for Metafeature groups.
===================================================

This class was built to give a model of how you should write a
metafeature group class as a Pymfe developer. Please read this
entire guide with attention before programming your own class.

At the end of this reading, you will know:
    * How to register your class as a valid MFE metafeature class
    * Which are the special method name prefixes, and how to properly
      use each of them
    * Which are the rules involving precomputation, metafeature
      extraction and post-processing methods
    * Which are the coding practices usually adopted in this library,
      that you should follow in order to get your changes accepted
      in the master branch

Also, feel free to copy this file to use as template for your own
class.

First, some tips and tricks which may help you follow the code
standards stabilished in this library.

1. Use type annotations as much as possible
-------------------------------------------

Always run `mypy` to check if the variable types was specified correctly.
You can install it with pip using the following command line:

>>> pip install -U mypy

Use the following command before pushing your modifications to the remote
repository:

>>> mypy pymfe --ignore-missing-imports

The expected output for this command is no output.

Note that all warnings must be fixed to your modifications be accepted in
the master branch, so take your time to fix your variable types carefully.


2. Use `pylint` to check your code style and auto-formatters such as `yapf`
---------------------------------------------------------------------------

Pylint can be used to check if your code follow some coding practices
adopted by the python community. You can install with with pip using the
following command:

>>> pip install -U pylint

It can be harsh sometimes, so we have decided to disable some of the
verifications. You can use the following command to check if your code
met the standards stabilished in this library,

>>> pylint pymfe -d 'C0103, R0913, R0902, R0914, C0302, R0904, R0801, E1101'

The expected output is something like

>>> Your code has been rated at 10.00/10 (previous run: x/10, y)

Your code will not be accepted in the master branch unless it gets the
maximum pylint score.

Yapf is a code auto-formatter which usually solves a large amount of
coding style related issues automatically.

>>> pip install -U yapf

If you use the flag ``-i``, Yapf changes your code in-place.

>>> yapf -i yourModulename.py


3. Make all verifications with the provided Makefile
----------------------------------------------------

You can use the Makefile provided in the root directory to run mypy,
pylint, and also pytest. Obviously, all tests (both for coding style
and programming logic) must pass in order to your modifications be
accepted.

You can use the tag ``test-cov`` for run tests and get the code coverage:

>>> make test-cov

You can use the tag ``test`` for only run the tests:

>>> make test

You can use the tag ``code-check`` for make all verifications with `mypy`,
`pylint` and `pep8`:

>>> make code-check

Remember that your code must pass all verifications included in both
``code-check`` and ``test``/``test-cov`` to your changes be accepted in
the master branch.

.. note::
    This example shows how to create a new group of meta-features. If you want
    only to add a new meta-feature, you should insert it in the meta-feature
    group file and create an "ft\\_" method to it. The new meta-feature will be
    automatically picked up (as the method "ft_metafeature_name" in this
    example). You should not forget to use the precompute methods to save
    time.


.. note::
    You should not forget to create tests for all new functionalities that
    you implemented. All tests can be found in `./tests/` fold. Please follow
    the existing code style while creating your tests as much as possible.


.. note::
    This class is being updated in GitHub, check this
    `link <https://github.com/ealcobaca/pymfe/blob/master/pymfe/_dev.py>`_
    to see the current version.

"""

import typing as t
import time

import numpy as np


class MFEBoilerplate:
    """The class name must start with ``MFE`` (just to keep code consistency)
    concatenated with the corresponding metafeature group name (e.g.,
    ``MFEStatistical`` or ``MFEGeneral``) in CamelCase format.

    Also, the class must be registered in the ``_internal.py`` module to be
    an official MFE class, because the pymfe framework is supposed to detect
    the metafeature extraction methods automatically, so you must explain
    where it is supposed to look for those methods.

    Three tuples at module level in ``_internal.py`` module must be updated
    to your new class be detected correctly:

        1. VALID_GROUPS: str
            Here you should write the name of your metafeature group. (e.g.,
            ``statistical`` or ``general``. This name is the value that will
            be given by the user in the ``groups`` MFE parameter to extract
            the all the metafeatures programmed here. Please select a
            sufficiently representative name for your metafeature group.

        2. GROUP_PREREQUISITES : str or :obj:`tuple` of str
            Use this tuple to register other MFE metafeature group classes
            as dependencies of your class. This means that, if the user ask
            to extract the metafeatures of your class, then all metafeature
            groups in the prerequisites will also be extracted also (even if
            the user doesn't ask explicity for these groups). Note that the
            possible consequences this may imply must be solved within this
            class post-processing methods (these methods will be explained
            later in this same guide.)

            The values of this tuple can be strings (which means one single
            dependency), sequences with strings (which means your class has
            multiple dependencies), or simply None (which means your class
            has no dependencies). Generally your class will not have any
            dependency, so just stick to the last option if you are not sure
            so far.

        3. VALID_MFECLASSES : MFE Classes
            In this tuple you should just insert a reference to your class.
            Note that this imply that this module must be imported at the top
            of the module ``_internal.py``.

        These three tuples have one-to-one correspondence using the indexes,
        so the order of values does matter. Please insert your class in the
        same index for all three tuples.

    ===================================================================

    For example, if we want to make this specific template class an official
    MFE class, those three tuples should be updated as follows: (Remember that
    all tuples below are found in ``_internal.py`` module.)

    -------------------------------------------------------------------

    # 1. First, choose carefully a metafeature group name. This value will be
    # used directly by the user when extracting the metafeatures programmed
    # in this class, so it must be meaningful and as short as possible.
    VALID_GROUPS = (
        ...,
        "boilerplate",
    )

    -------------------------------------------------------------------

    # 2. Generally your class will not have any dependency, so you should
    # just register ``None`` as prerequisites. Remember that a class can
    # have any number of dependencies (0, 1 or more than 1.)
    GROUP_PREREQUISITES = (
        ...,
        None,
    )

    -------------------------------------------------------------------

    # 3. The last step is to insert your class in this tuple below.
    # Remember to import your module in the ``_internal.py`` module.
    # So, for instance, to register this class, 'MFEBoilerplate', as
    # an official MFE metafeature extractor class, we should make the
    # following modifications:

    import pymfe._dev as _dev

    VALID_MFECLASSES = (
        ...,
        _dev.MFEBoilerplate,
    )

    After this three simple steps, your class is now an official MFE
    metafeature extraction class. From now on you no longer need to
    worry about the ``_internal.py`` module and any other external
    pymfe module.

    ===================================================================

    Now that you know how to handle the issues related to the
    ``_internal.py`` module, let's start with the actual MFE class
    development.

    This tutorial is built to introduce all the different elements
    following the natural order of how a regular MFE Class is usually
    presented.

    Therefore, the order that we shall see the different concepts in
    this guide is:

    1. Precomputation methods (prefixed with ``precompute_``)
    Methods related to this subject:
        1.1 precompute_basic_precomp_method
        1.2 precompute_more_info
        1.3 precompute_random_values

    2. Feature extraction methods (prefixed with ``ft_``)
    Methods related to this subject:
        2.1 ft_metafeature_name
        2.2 ft_fitted_data_arguments
        2.3 ft_using_precomputed_values
        2.4 ft_about_return_values

    3. Regular/auxiliary methods (non-prefixed )
    Methods related to this subject:
        3.1 _protected_methods
        3.2 non_protected_methods_without_any_prefixes

    4. Postprocessing methods (prefixed with ``postprocess_``)
    Methods related to this subject:
        4.1 postprocess_groupName1_groupName2

    So, we shall start looking at a example of a precomputation
    method.
    """

    # Important detail: all methods must be classmethods; there is no class
    # instantiation in the pymfe framework.
    @classmethod
    def precompute_basic_precomp_method(
        cls,
        y: t.Optional[np.ndarray] = None,
        argument_bar: t.Optional[int] = None,
        **kwargs
    ) -> t.Dict[str, t.Any]:
        """A precomputation method example.

        The pydoc of each method must explain cleary what is the purpose of
        that method. This method is supposed to introduce a powerful concept
        of the pymfe framework: precomputation methods.

        1. Why precomputation methods?
        -----------------------------------------------------------------
        All methods whose name is prefixed with ``precompute_`` are
        executed automatically before the metafeature extraction. These
        methods are extremely important to improve the performance of
        the Pymfe library, as it is quite common that different metafeature
        extraction methods uses the very same information.

        The idea behind this type of methods is to fill up a shared cache
        with all values that can be shared by different metafeature extraction
        methods, and also between different metafeature group classes. This
        means that the values precomputed in ``MFEFoo`` class can also be used
        in the ``MFEBar`` class methods.

        2. Naming convention of a precomputation method
        -----------------------------------------------------------------
        The name of the method does not matter, as long as it starts with
        the prefix ``precompute_``. This prefix is used to tell the Pymfe
        framework that this is a precomputation method. As you will see during
        this guide, the Pymfe rely heavily on prefixes in the method names, so
        it is important that you don't forget them, and use them appropriately.

        3. Arguments of a precomputation method
        -----------------------------------------------------------------
        The structure of these precomputation methods is pretty simple. In the
        arguments you can specify custom parameters such as ``X`` and ``y``
        that are automatically given by the MFE class. Those attributes can be
        registered in a special attribute in the MFE class, or also given by
        the user, but you should not rely on this feature; just stick to the
        MFE registered arguments, and let all user-customizable attributes
        have a default value. How exactly those arguments arrive as method
        arguments is not important to develop an MFE metafeature extraction
        class. If you're curious, you should examine the ``mfe.py`` and
        ``_internal.py`` modules by yourself, but it will take some time and
        is not encouraged unless you plan an actual framework redesign.

        It is obligatory to receive the ``kwargs`` in every precomputation
        method. You are free to pick up values from it. We recommend you to
        use the ``get`` method for this task. However, it is forbidden to
        remove or modify the existing values in it. This parameter must be
        considered ``read-only`` except to the insertion of new key-value
        pairs. The reason behind this is that there's no guarantee of any
        execution order of the precomputation methods within a class and
        neither between classes, so all precomputation methods must have
        the chance to read the same values.

        4. Return values of precomputation methods
        -----------------------------------------------------------------
        All precomputation methods must return a dictionary with strings as
        keys. The value data type can be anything. Note that the name of the
        keys will be used later to match the argument names of feature
        extraction methods. It means that, if you return a dictionary in the
        form: {'foo': 1, 'bar': ('a', 'b')}, then all feature extraction
        methods with an argument named ``foo`` will receive value ``1``, and
        every method with argument named ``bar`` will receive a tuple with 'a'
        and 'b' elements. Always choose meaningful key/argument names.

        As this framework rely on a dictionary to distribute the parameters
        between feature extraction methods, your precomputed keys should never
        replace existing keys with different values, and you should not give
        the same name to parameters with different semantics or purposes.
        The rule of thumb for the pymfe lybrary is: 'if two things have the
        same name, then they are the same thing'. Therefore, avoid extremely
        generic argument names such as ``freqs``, ``mean``, ``model`` etc.

        5. The user can disable precomputation methods
        -----------------------------------------------------------------
        Keep in mind that the user can disable the precomputation methods,
        mainly due to memory constraints.

        Never rely on these methods to produce any mandatory arguments. All
        the precomputed values here should go to optional parameters and all
        receptor metafeature extraction methods must be responsible to verify
        if all values were effectively precomputed (i.e., they are not
        ``None``). If this is not the case, unfortunately these methods must
        compute those arguments for themselves. If it is not clear how it
        works for you by now, it will probably be easier to grasp when we
        reach our first actual metafeature extraction method. For now, it is
        just important to keep in mind that: you will need to recompute all
        the stuff precomputed in every precomputations methods inside other
        methods whenever those values are needed for the case when the user
        disable the precomputation methods.

        Parameters
        ----------
        y : :obj:`np.ndarray`, optional
            Always give clear and meaningful description to every argument.

        argument_bar : int, optional
            Some user-given attribute.

        **kwargs:
            Additional arguments. May have previously precomputed before
            this method from other precomputed methods, so they can help
            speed up this precomputation avoiding duplicated work.

        Returns
        -------
        :obj:`dict`
            The following precomputed items are returned:
                * ``y_unique`` (:obj:`np.ndarray`): unique values from
                    ``y``, if it is not None.
                * ``absolute_bar`` (float): absolute value of
                    ``argument_bar``, if it is not None.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        # Always consider that your precomputation argument could
        # be precomputed by another precomputation method (even if
        # from a different module), so check if the new key is not
        # already in kwargs before calculating anything.
        if argument_bar is not None and "absolute_bar" not in kwargs:
            precomp_vals["absolute_bar"] = abs(argument_bar)

        # The number of precomputed values within a single precomputation
        # method vary greatly, from just a single value to a few amount.
        # As long as all values are semantically sufficiently related with
        # each other, you don't need to create new precomputation methods.
        if y is not None and "y_unique" not in kwargs:
            y_unique = np.unique(y, return_counts=False)
            precomp_vals["y_unique"] = y_unique

        # Always return a dictionary, even if it is empty
        return precomp_vals

    @classmethod
    def precompute_more_info(
        cls, argument_bar: t.Optional[int] = None, **kwargs
    ) -> t.Dict[str, t.Any]:
        """Highly relevant information about precomputation methods.

        1. How many precomputation methods per class?
        -----------------------------------------------------------------
        Every MFE metafeature extraction class can have as many of
        precomputation methods as needed. Don't hesitate to create
        new precomputation methods whenever you think it will help
        to improve the performance of the package.

        2. How many precomputed values per precomputation method?
        -----------------------------------------------------------------
        There is no limit of how many values can be precomputed within
        a single precomputation method.

        However, try to keep every precomputation method precompute only
        related values to avoid confusion. Prefer to calculate dissociated
        values in distinct precomputation methods.

        3. Using other precomputed values in a precomputation method
        -----------------------------------------------------------------
        Don't rely on the execution order of precomputation methods. Always
        assume that the precomputation methods (even within the same class)
        can be executed in any order. However, it does not mean that you
        can't at least try to use previously precomputed methods: that's why
        the 'kwargs' is used in all precomputation methods.

        If needed, try to get a value from 'kwargs' using the 'get' method
        (i.e., kwargs.get('argument_name') - remember 'kwargs' is just a
        Python dictionary.) Then, check whether that value was successfully
        gotten (i.e., is not None).

        Parameters
        ----------
        argument_bar : int, optional
            Some user-given attribute. Note that it has the same value as
            in the previous precomputation method, because it is the same
            argument (it has the same name.)

        **kwargs:
            Additional arguments. May have previously precomputed before
            this method from other precomputed methods, so they can help
            speed up this precomputation avoiding duplicated work.

        Returns
        -------
        :obj:`dict`
            The following precomputed items are returned:
                * ``double_absolute_bar`` (int): two times the
                    value of ``absolute_bar``, which may or may not
                    be precomputed in the previous precomputation
                    method. If it is not the case, we precompute
                    ``absolute_bar`` here and also store its value.
                * ``qux`` (float): value is equal to 1.0.
                * ``quux`` (:obj:`complex`) Imaginary value related to
                    ``qux``.
                * ``quuz`` (:obj:`np.ndarray`): an sequence based
                    on ``qux``.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if argument_bar is not None and "double_absolute_bar" not in kwargs:
            # May have been precomputed from another precomputation method
            absolute_bar = kwargs.get("absolute_bar")

            # Wrong! 'absolute_bar' may be None
            # precomp_vals["double_absolute_bar"] = 2 * absolute_bar

            if absolute_bar is None:
                absolute_bar = abs(argument_bar)
                # Because we needed to calculate 'absolute_bar' here, does
                # not hurt also storing this value also, to prevent it
                # being recalculated in 'precompute_basic_precomp_method'.
                precomp_vals["absolute_bar"] = absolute_bar

            # Correct: now 'absolute_bar' is guaranteed to be not None
            precomp_vals["double_absolute_bar"] = 2 * absolute_bar

        if not {"qux", "quux", "quuz"}.issubset(kwargs):
            precomp_vals["qux"] = 1.0
            precomp_vals["quux"] = 5 + 1.0j * (precomp_vals["qux"])
            precomp_vals["quuz"] = np.array(
                [precomp_vals["qux"] + i for i in np.arange(5)]
            )

        return precomp_vals

    @classmethod
    def precompute_random_values(
        cls, random_state: t.Optional[int] = None, **kwargs
    ) -> t.Dict[str, t.Any]:
        """Precomputation method with pseudo-random behavior.

        1. An important pymfe default argument for you: 'random_state'
        -----------------------------------------------------------------
        If you are using anything with pseudo-random properties, you shall
        always get the pymfe framework global random seed using the
        ``random_state`` argument. This seed is user defined. You can get
        it for any precomputation, metafeature extraction or post-processing
        methods.

        2. Important aspects related to pseudo-random behaviour
        -----------------------------------------------------------------
        Uncontrolled pseudo-random behavior is absolutely forbidden in
        this package.

        Also, pseudo-random methods must have related automated tests.
        Therefore, setting up the random seed (as long as the user define
        it) is never optional.

        Parameters
        ----------
        random_state : int, optional
            If given, controls the pseudo-random behavior inside this
            method, so the results will be reproducible.

        **kwargs:
            Additional arguments. May have previously precomputed before
            this method from other precomputed methods, so they can help
            speed up this precomputation avoiding duplicated work.

        Returns
        -------
        :obj:`dict`
            The following precomputed items are returned:
                * ``random_special_num`` (float): a random value
                  that must be controlled by the random seed specified
                  by the user using the ``random_state`` pymfe framework
                  global argument.
        """
        precomp_vals = {}  # type: t.Dict[str, t.Any]

        if "random_special_num" not in kwargs:
            if random_state is not None:
                np.random.seed(random_state)

            aux = np.random.randint(-5, 5, size=10)
            precomp_vals["random_special_num"] = np.random.choice(aux, size=1)

        return precomp_vals

    @classmethod
    def ft_metafeature_name(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        random_state: t.Optional[int] = None,
        opt_arg_bar: float = 1.0,
        opt_arg_baz: np.ndarray = None,
    ) -> int:
        """Single-line description of this feature extraction method.

        The purpose of this method is to introduce the first actual
        metafeature extraction method.

        1. Metafeature extraction methods: the most important ones
        -----------------------------------------------------------------
        Similarly to the precomputation methods, the feature extraction
        method names are also prefixed. All your feature extraction method
        names must be prefixed with ``ft_``.

        2. The pymfe framework provides arguments automatically
        -----------------------------------------------------------------
        As mentioned in the documentation of the very first precomputation
        method, the pymfe framework is responsible to provide to every
        precomputation (those prefixed with ``precompute_``, metafeature
        extraction (those prefixed with ``ft_``) and post-processing (we
        will see those later) methods its arguments. 'How?', you may ask.
        The short answer is dictionary unpacking: the MFE class holds some
        dictionaries that are unpacked while calling those prefixed methods.
        Then, if a method's argument happens to match with a dictionary
        key, that argument will assume the matched key value.

        All precomputed values are packed into one of those dictionaries
        (and it happens automatically; you don't need to worry about it.)
        Therefore, the same value provided as the key of some precomputed
        dictionary is used to match directly the parameter name. All
        parameters must be treated as read-only values; it is forbidden to
        modify any value inside any feature extraction method.

        We will see more about which default parameters are given by the
        pymfe framework soon in the ``ft_fitted_data_arguments`` method
        just below. However, if you want to see with your own eyes the
        actual values, you can check out search for the instance attribute
        ``mfe.MFE._custom_args_ft`` of the MFE class (inside the ``mfe.py``
        module). This attribute is set up inside the ``mfe.MFE.fit`` method.

        If you have a very good reason, feel free to insert new values
        in there if (and only if) they are needed. Note that it is highly
        unlikely.

        2. Mandatory & optional arguments of metafeature extraction methods
        -----------------------------------------------------------------
        The only arguments allowed to be mandatory (i.e., arguments without
        any default value) are the ones registered inside the MFE attribute
        ``_custom_args_ft`` (check this out in the ``mfe.py`` module.)
        All other values must have a default value, without any exception.

        Remember that all arguments can be customized directly by the user
        while calling the ``extract`` MFE method. You usually don't need
        to worry about if the user uses incorrect data types for the
        arguments, as it will most probably raise an TypeError exception.
        However, sometimes you should consider handling incorrect values
        (such as probability arguments with values not within the range
        0 and 1.) Usually, just returning ``np.nan`` (if your metafeature
        is non-summarizable) or ``np.array([np.nan])`` (if your metafeature
        is summarizable)  is one way to go when handling incorrect arguments.

        3. Return values of metafeature extraction methods
        -----------------------------------------------------------------
        We'll see about this soon in the ``ft_about_return_values`` method.

        Arguments
        ---------
        X : :obj:`np.ndarray`
            All attributes fitted in the model (numerical and categorical
            ones). While writing your method documentations, you don't need
            to write about very common arguments such as ``X``, ``y``, ``N``
            and ``C``. In fact, you are encouraged to just omit these.

        y : :obj:`np.ndarray`
            Target attributes. Again, no need to write about these type of
            arguments in the method documentation, as it can get way too
            much repetitive without any information gain.

        random_state : int, optional
            Extremely important argument. This one is a fixed feature from the
            MFE framework. If your method has ANY pseudo-random behaviour,
            you should use specifically this argument to provide the random
            seed. In this case, it would be nice if you write about what
            is the random behaviour of your method to make clear to the
            user why he or she ever needs a random seed in the first place.

        opt_arg_bar : float, optional
            Argument used to detect carbon footprints of hungry dinosaurs.

        opt_arg_baz : :obj:`np.ndarray`, optional
            If None, this argument is foo. Otherwise, this argument is bar.

        Returns
        -------
        int
            Give a clear description about the returned value.

        Notes
        -----
        You can use the notes section of the documentation to provide
        references, and also ``very specific`` details of the method.
        """
        # Inside the feature extraction method you can do whenever you
        # want, just make sure to:
        # 1. Always return a single number, a single np.nan or a numpy
        #    array with numeric values (or np.nan) - no exceptions!
        # 2. Make it run as fast as possible. Metafeatures with high
        #    computational complexity are discouraged.

        # You can raise ValueError, TypeError and LinAlgError exceptions.
        if opt_arg_bar <= 0.0:
            raise ValueError("'opt_arg_bar' must be positive!")

        # When using pseudo-random functions, ALWAYS use random_state
        # to enforce experiment replication. Uncontrolled pseudo-random
        # behavior is absolutely forbidden.
        if opt_arg_baz is None:
            np.random.seed(random_state)
            opt_arg_baz = np.random.choice(10, size=5, replace=False)

        aux_1, aux_2 = np.array(X.shape) * y.size

        np.random.seed(random_state)
        random_ind = np.random.randint(opt_arg_baz.size)

        ret = aux_1 * opt_arg_bar / (aux_2 + opt_arg_baz[random_ind])

        return ret

    @classmethod
    def ft_fitted_data_arguments(
        cls, X: np.ndarray, N: np.ndarray, C: np.ndarray, y: np.ndarray
    ) -> int:
        """Information about some arguments related to fitted data.

        1. Handling Numerical, Categorical and Mixed data types
        -----------------------------------------------------------------
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
        int
            Some important return value.

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
    def ft_using_precomputed_values(
        cls,
        y: np.ndarray,
        # y_unique: np.ndarray,  # Wrong! Need an default value.
        y_unique: t.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Metafeature extraction method using precomputed values.

        1. How to use precomputed arguments
        -----------------------------------------------------------------
        Within any metafeature extraction method, you can safely assume that
        all precomputation methods (even the ones of other MFE classes) were
        all executed (successfully or not!), and their values are hopefully
        ready to be used as arguments. Note that the pymfe framework has a
        huge resilience against exceptions, so the code will most probably
        continue to flow even if a few precomputation methods were not
        successful for some reasons (e.g., math domain errors.)

        To get precomputed values is no different than getting a pymfe
        default automatic argument (such as ``X`` and ``y``): just match
        the argument name with the precomputed dictionary key. For
        instance, the argument ``y_unique`` was precomputed in the
        ``precompute_basic_precomp_method`` and is probably ready to be used in
        this metafeature extraction method, IF the user does not
        disabled the precomputations. As we can't guarantee whether the
        user will or will not disable the precomputations, we need to
        always check if ``y_unique`` is different than ``None`` before
        using it. If, unfortunatelly, it is not the case, then we need
        to compute ``y_unique`` inside this method.

        2. When to use precomputed arguments
        -----------------------------------------------------------------
        Always! :)

        3. The precomputation cache is shared among all pymfe classes
        -----------------------------------------------------------------
        Remember that you can also use precomputed values from other
        pymfe metafeature extraction classes (and, therefore, your
        precomputed values will also be automatically available to the
        other classes aswell.)

        Arguments
        ---------
        y : :obj:`np.ndarray`
            Target attribute.

        y_unique : :obj:`np.ndarray`, optional
            Argument precomputed in the ``precompute_basic_precomp_method``
            precomputation method. Note that it must be an optional
            argument (because it is forbidden to rely on precomputation
            methods to fill mandatory arguments, as the user can disable
            precomputation methods whenever he or she wants.) Note also
            that the argument name must match exatcly the corresponding
            dictionary key given inside the precomputation method.

        Returns
        -------
        :obj:`np.ndarray`
            Describe your return value.
        """
        # res = -1.0 * y_unique  # Wrong! 'y_unique' may be None!

        # You need to verify if precomputed values is None. If this
        # is the case, you need to manually compute it inside the method
        # that needs that value.
        if y_unique is None:
            # If ``y_unique`` is None, it means probably that the user
            # disabled the precomputations (or something went wrong inside
            # the precomputation method,) so we need to compute
            # it now as this argument is needed to compute the
            # method's output.

            # Obviously, the computation inside the metafeature
            # extraction method must be identical to the computation
            # in the precomputation method, as both results must
            # always match. Once again, remember:
            # 'If two things have the same name, then they are the
            # same thing'.
            y_unique = np.unique(y, return_counts=False)

        res = -1.0 * y_unique  # Correct: 'y_unique' is surely not None

        return res

    @classmethod
    def ft_about_return_values(
        cls,
        y: np.ndarray,
    ) -> np.ndarray:
        """Information about return values of feature extraction methods.

        1. You have two return options for metafeature extraction methods
        -----------------------------------------------------------------
        The return value of any feature extraction method should be
        a single value (int, float, numpy number, or a :obj:`np.nan`,)
        or a numpy array. This array must contain only numbers or
        :obj:`np.nan`.

        2. What's the difference?
        -----------------------------------------------------------------
        If the return value is a single number, the output value of this
        method will be transformed directly into a MFE class extract output.
        If it is a numpy array, then this output will automatically be
        summarized using every user selected summary functions.

        3. A more detailed explanation
        -----------------------------------------------------------------
        If you return a single value, your metafeature is said to be
        'non-summarizable'. It means that the value your method return is
        the value the user will get. If you need to return an invalid
        value, always return 'np.nan'.

        If you return an numpy array, then your metafeature is said to be
        'summarizable', and the user will get a few statistics related to
        the values your method returns (instead of the actual values):
        its mean, standard deviation, quantiles, variance etc. It will
        happen automatically, and you should not worry about this. You
        can put 'np.nan' inside your array. If you need to return an
        entire invalid array, consider returning 'np.empty(0)', or simply
        raise and ValueError or TypeError exception.
        DO NOT return a single 'np.nan', as it is reserved for the
        'non-summarizable' metafeature extraction methods.

        Arguments
        ---------
        y : :obj:`np.ndarray`
            Target attribute.

        Returns
        -------
        :obj:`np.ndarray`
            This method returns a numpy array, so its output value will
            be summarized automatically by the MFE framework before
            outputting to the user.
        """
        # Either your method return a single value, or it return an
        # numpy array. You can't mix both within a single metafeature
        # extraction method.

        if np.any(y < 0):
            # My metafeature can't handle negative 'y' values, so I
            # can return an invalid array

            # return np.nan  # Wrong! It is not an array!
            # return np.empty(0)  # Correct, but there's room for improvement.
            raise ValueError("'y' can't have negative values.")  # Better.

        if y.size > 20:
            return np.power(y, 1 / 4) + np.arange(y.size)

        return np.sqrt(y) + np.arange(y.size)

    @classmethod
    def _protected_methods(cls, arg_foo: float) -> float:
        """Tips for using protected methods.

        1. How to use Python's protected methods in pymfe code
        -----------------------------------------------------------------
        Protected methods (methods whose name starts with a underscore)
        should be used whenever you need to modularize better your code,
        and even more if you need to use the same piece of code between
        two or more different metafeature extraction methods.

        2. Using private methods
        -----------------------------------------------------------------
        Private methods (methods prefixed with two underscores) are not
        really necessary, and their use must be justified somehow.

        So far, there is not even a single private method in any pymfe
        code.

        3. Protected method documentation
        -----------------------------------------------------------------
        You don't need to follow the standard documentation format for
        protected methods (method description, argument list, return value
        description etc.) Instead, you can be more technical since the
        documentation will probably be more suitable for other developers
        and maintainers of the package. If you fell more confortable with
        the standard format (just like the public methods), there is no
        harm to follow it in the protected method documentation then.
        """

        def inner_functions(x: float, lamb: float = 1.0) -> float:
            """Usage of inner functions.

            1. When to use inner functions
            ---------------------------------------------------------
            Use them whenever you need modularize a piece of code that
            is way too much specific for the method that contains it.
            Therefore, it is highly unlikely that this same piece of
            code may ever be used from another method.

            2. How many inner functions per method?
            ---------------------------------------------------------
            These functions are quite useful for very complex feature
            extraction methods with many steps needed to reach the final
            result. In that case, consider creating a separated inner
            function for every step.
            """
            return np.abs(np.tanh(x * lamb) * 0.7651j)

        return np.max(inner_functions(arg_foo), 0.0)

    @classmethod
    def non_protected_methods_without_any_prefixes(cls) -> None:
        """Don't use non-protected regular methods.

        The main reason to avoid this type of methods is because
        it will be shown in the package documentation despite the
        fact that it is not of the user's interest.
        """
        raise NotImplementedError(
            "Hide me prefixing my name with a single '_'."
        )

    @classmethod
    def postprocess_groupName1_groupName2(
        cls,
        mtf_names: t.List[str],
        mtf_vals: t.List[float],
        mtf_time: t.List[float],
        class_indexes: t.List[int],
        groups: t.Tuple[str, ...],
        inserted_group_dep: t.FrozenSet[str],
        **kwargs
    ) -> t.Optional[t.Tuple[t.List[str], t.List[float], t.List[float]]]:
        """Introduction to post-processing methods.

        1. What is a post-processing method?
        -----------------------------------------------------------------
        The post-processing methods can be used to either modify in-place
        previously generated metafeatures (not necessarily from the same
        group) or to generate new metafeatures using previously extracted
        metafeatures just before outputting the results to the user. The
        popularity of this type of method is not even close to the
        preprocessing ones, but they may be useful in some specific cases
        (mainly related to `somehow` merge the dependencies data with the
        generated data from the dependent class.)

        For instance, the 'Relative Landmarking' metafeature group is
        entirely based on post-processing methods: that specific group needs
        every 'Landmarking' metafeature results and, therefore, it can be
        computed only after the metafeature extraction process finishes
        (because we have no guarantees of the metafeature extraction order.)

        So, if your MFE class does not have any external dependencies, nor it
        is supposed to somehow merge two or more metafeature values, you
        don't need to read this section, and you are already good to go
        and develop your own MFE class. If it is not your case, then stay
        with us for a couple of extra minutes more.

        2. Structure of a post-processing method
        -----------------------------------------------------------------
        All post-processing methods receive all previously extracted
        metafeatures from every MFE class. It will not receive just the
        metafeatures related to the metafeature extraction methods of this
        class. It is very import to keep this in mind.

        There's a very important trick with the naming of these post-processing
        methods, other than just prefixing they with ``postprocess_``.
        You can put names of metafeature groups of interest separated by
        underscores. All metafeature indexes related to any of the selected
        groups will arrive in the ``class_indexes`` argument automatically.

        For example, suppose a post-processing method named like:

            postprocess_infotheory_statistical(...)

        This implies that the indices of both `information theory` and
        `statistical` metafeature groups will arrive inside the
        ``class_indexes`` sequence. Using this feature, one can easily
        work with these metafeatures without needing to separate them by
        hand. Of course, you can give as many metafeature group names as
        needed. If you need them all, then simply don't put any metafeature
        group name, as every metafeature is an metafeature of interest in
        this case.

        There were various arguments that are automatically filled for
        this type of methods (as you can see just above in this method
        signature). Check the ``arguments`` section for more details
        about each one.

        3. How many post-processing methods are necessary?
        -----------------------------------------------------------------
        Just like the preprocessing and metafeature extraction methods,
        an MFE class may have any number post-processing methods, including
        none. In fact, no post-processing method is by far the common case.

        4. Return value of post-processing methods
        -----------------------------------------------------------------
        The return value of post-processing methods must be either None,
        or a tuple with exactly three lists. In the first case (returning
        None), the post-processing method is probably supposed to modify
        the received metafeature values in-place (which is perfectly
        fine). In the second case (returning three lists), these lists
        will be considered new metafeatures and will be appended to the
        MFE output before given to the user. These lists must follow the
        order given below:

            1. New metafeature names
            2. New metafeature values
            3. Time elapsed to extract every new metafeature

        Now, let's take a quick look at the common post-processing method
        arguments. Note that all the arguments listed below are actual
        arguments from the pymfe framework, and you can use they in your
        post-processing methods.

        Arguments
        ---------
        mtf_names : :obj:`list` of str
            A list containing all previously extracted metafeature names.

        mtf_vals : :obj:`list` of float
            A list containing all previously extracted metafeature values.

        mtf_time : :obj:`list` of float
            A list containing all time elapsed for each metafeature
            previously extracted.

        class_indexes : List of int
            Indexes of the metafeatures related to this method ``groups of
            interest``. The ``groups of interest`` are the metafeature groups
            whose name are in this method's name after the ``postprocess_``
            prefix, separated with underscores (in this example, they are
            ``groupName1`` and ``groupName2``.)

            If it is not clear for you so far, the metafeatures received
            in this method are all the metafeatures extracted in every MFE
            classes, not just the ones related to this class. Then, this
            argument can be used as reference to target only the metafeatures
            effectively used in this post-processing method.

            If you need every single metafeature extracted for your
            post-processing method, then this argument does not matter (nor
            your post-processing method name, as long as it is correctly
            prefixed with ``postprocess_``) as every metafeature is of your
            particular interest, and there is no need for an auxiliary
            list to split the metafeatures.

        groups : :obj:`tuple` of str
            Extracted metafeature groups (including metafeature groups
            inserted due to group dependencies). Can be used as reference
            inside the post-processing method.

        inserted_group_dep : :obj:`tuple` of :obj:`str
            Extracted metafeature groups due to class dependencies. Can be
            used as a reference inside the post-processing method.

        **kwargs:
            Just like the preprocessing methods, the kwargs is also
            mandatory in post-processing methods. It can be used to
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

        # In this example, this post-processing method returns
        # new metafeatures conditionally. Note that this variable
        # ``change_in_place`` is fabricated for this example; it
        # is not a true feature of the Pymfe framework!!! The
        # decision of whether or not to change metafeatures in
        # place depends on your particular context!
        change_in_place = kwargs.get("change_in_place", False)

        if change_in_place:
            # Make changes in-place using the ``class_indexes`` as
            # reference. Note that these indexes are collected using
            # this post-processing method name as reference (check the
            # documentation of this method for a clear explanation.)
            for index in class_indexes:
                time_start = time.time()
                mtf_vals[index] *= 2.0
                mtf_names[index] += ".twice"
                mtf_time[index] += time.time() - time_start

            # Don't return new metafeatures, as the changes made are
            # in-place in this particular situation.
            return None

        # The previous branch was not taken: therefore, the changes
        # are not in-place. This means that new metafeatures will be
        # created and appended to the previously existing ones. Note
        # that whether the new feature values are supposed to be identical
        # to its in-place variants are context dependent. If you have
        # good reasons to do make they different, then you are allowed to.

        # Create new metafeatures (in this case, the user will receive
        # twice as many values as separated metafeatures.) Note that the
        # number of new metafeatures also is context dependent: your
        # post-processing method may return as many as new metafeatures it
        # is supposed to return.
        for index in class_indexes:
            time_start = time.time()
            new_mtf_vals.append(-1.0 * new_mtf_vals[index])
            new_mtf_names.append("{}.negative".format(new_mtf_names[index]))
            new_mtf_time.append(new_mtf_time[index] + time.time() - time_start)

        # Finally:
        # Return new metafeatures produced in this method. Pay attention to the
        # order of these lists, as it must be preserved for any post-processing
        # method.
        return new_mtf_names, new_mtf_vals, new_mtf_time
