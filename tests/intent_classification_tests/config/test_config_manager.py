import json
import os
import tempfile
import unittest

from app.core import config_manager as cm

class TestConfigManager(unittest.TestCase):
    def test_switch_variant(self):
        prev = cm.ACTIVE_VARIANT
        try:
            cm.switch_variant('B')
            self.assertEqual(cm.ACTIVE_VARIANT, 'B')
            cm.switch_variant('A')
            self.assertEqual(cm.ACTIVE_VARIANT, 'A')
        finally:
            cm.switch_variant(prev)

    def test_validate_rules_weights(self):
        good = {"rules": {"rule_sets": {"A": {"kw_weight": 0.6, "emb_weight": 0.4}}}}
        bad = {"rules": {"rule_sets": {"A": {"kw_weight": 0.6, "emb_weight": 0.6}}}}
        self.assertTrue(cm.validate_config("rules", good))
        self.assertFalse(cm.validate_config("rules", bad))

    def test_hot_reload_function_call(self):
        # Simulate load_all_configs without filesystem events
        cm.load_all_configs()
        self.assertTrue(isinstance(cm.CONFIG_CACHE, dict))

if __name__ == "__main__":
    unittest.main()
