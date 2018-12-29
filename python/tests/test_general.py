"""Test module for General class metafeatures."""
import context
from pymfe.general import MFEGeneral


class TestGeneral:
    """TestClass dedicated to test General metafeatures."""

    def test_dataset_0_inst_num(self):
        """Test method for dataset #0 and metafeature "inst_num"."""
        dataset = context.DATASET_LIST[0]
        assert MFEGeneral.inst_num(dataset) == 1000
