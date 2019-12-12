"""Auto pages module."""
import os
import pandas as pd
from pymfe.mfe import MFE


AUTO_PAGES_PATH = "auto_pages"
TABLE_CONF = """
.. csv-table:: Meta-feature description
   :file: meta_features_description.csv
   :header-rows: 1
"""

NOTE_REL_SUB = """
.. note::
    Relative and Subsampling Landmarking are subcase of Landmarking. Thus, the
    Landmarking description is the same for Relative and Subsampling groups."""

NOTE_OTHER_INFO = """
.. note::
    More info about implementation can be found in API Documentation.
    See :ref:`sphx_api`.
"""

TITLE = """
Meta-feature Description Table
==============================
    The table shows for each meta-feature the group, a quick description and
    paper reference. See examples of how to compute the meta-feature in
    :ref:`sphx_glr_auto_examples`.
"""


def meta_features_description():
    """Automatically create the meta-feature description file."""
    data, _ = MFE.metafeature_description(sort_by_group=True,
                                          sort_by_mtf=True,
                                          print_table=False,
                                          include_references=True)

    if not os.path.exists(AUTO_PAGES_PATH):
        os.makedirs(AUTO_PAGES_PATH)

    col = data[0]
    del data[0]
    df = pd.DataFrame(data, columns=col)

    df.to_csv(AUTO_PAGES_PATH+"/meta_features_description.csv", index=False)

    notes = NOTE_REL_SUB + "\n" + NOTE_OTHER_INFO

    table_str = TABLE_CONF
    f = open(AUTO_PAGES_PATH+"/meta_features_description.rst", "w")
    f.write(TITLE + '\n' + table_str + '\n' + notes)
    f.close()


def main():
    """Main function of Run script."""
    meta_features_description()


if __name__ == "__main__":
    main()
