import unittest

from anylabeling.views.labeling.advanced.dedup.config import Config


class TestDedupInstanceGate(unittest.TestCase):
    def test_use_instance_gate_defaults_on(self):
        self.assertTrue(Config().strict.use_instance_gate)

    def test_empty_label_filter_disables_instance_from_ui_convention(self):
        cfg = Config()
        cfg.feature.box_label_filter = ()
        cfg.strict.use_instance_gate = False
        self.assertFalse(cfg.strict.use_instance_gate)


if __name__ == "__main__":
    unittest.main()