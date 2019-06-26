"""A developer sample class for Metafeature groups.

This class was built to give a model of how you should write a
metafeature group class as a Pymfe developer. Please read this
entire guide with attention before programming your own class.

In the end of this reading, you will know:
    * What are the special method name prefixes
    * What are the coding practices usually adopted in this library

Also, feel free to copy this file to use as boilerplate for your
own class.
"""

import typing as t
"""Use type annotations as much as possible.

Also run ``mypy`` to check if the variable types was specified correctly.
Use the following command before pushing your modifications to the remote
repository:

    $ python -m mypy yourModuleName.py --ignore-missing-imports

Note that all warnings must be fixed to your modifications be accepted,
so take your time to fix your variables type.
"""


class MFEBoilerplate:
    """The class name must start with ``MFE`` (just to keep consistency)
    concatenated with the group name (e.g., ``MFEStatistical``, ``MFEGeneral``.)

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
            imply that this module must be imported at the top of ``_internal.py``
            module.


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
    def precompute_foa_method(
            cls,
            X: np.ndarray,
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
        replace existing keys with different values, and you should not give the
        same name to parameters with different semantics.

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

        **kwargs
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
        precomp_vals = {}

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

    def precompute_baz_qux(cls, **kwargs) -> t.Dict[str, t.Any]:
        """Another precomputation method.
        
        Every MFE metafeature extraction class may have as many of
        precomputation methods as needed. Don't be ashamed to create
        new precomputation methods whenever you need to.
        
        Try to keep every precomputation method precompute related
        values to avoid confusion. Prefer to calculated non-associated
        values in different precomputation methods.
        """
        precomp_vals = {}

        return precomp_vals

    # All feature extraction methods must be classmethods
    @classmethod
    def ft_feat_extraction(cls, X: np.ndarray, y: np.ndarray) -> int:
        """Ratio between the number of attributes.

        Arguments
        ---------

        Returns
        -------
        float
            The ration between the number of attributes and instances.

        """
        return X.shape[1] / X.shape[0]

