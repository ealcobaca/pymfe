"""Test module for General class metafeatures."""
import context
from mfe.general import MFEGeneral


class TestGeneral(object):

    def test_dataset_0_inst_num(self):
        dataset = context.DATASET_LIST[0]
        assert MFEGeneral.inst_num(dataset) == 1000
