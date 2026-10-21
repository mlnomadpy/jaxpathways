"""Required shifted and aligned-loss APIs are present on the learner scaffold."""

import importlib.util
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


class StarterContractTests(unittest.TestCase):
    def test_checker_required_shifted_loss_apis_exist_and_fail_at_their_todo(self):
        path = PROJECT / "starter" / "model.py"
        spec = importlib.util.spec_from_file_location("c02_starter_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.shifted_causal_targets))
        self.assertTrue(callable(module.token_loss))
        self.assertTrue(callable(module.aligned_token_loss))
        with self.assertRaisesRegex(NotImplementedError, "TODO: preserve next-token"):
            module.shifted_causal_targets(None, None)
        with self.assertRaisesRegex(NotImplementedError, "TODO: compute shifted"):
            module.token_loss(None, None, None, None)


if __name__ == "__main__":
    unittest.main()
