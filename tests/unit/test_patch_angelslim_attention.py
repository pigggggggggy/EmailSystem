import unittest

from training.patch_angelslim_attention import ORIGINAL, PATCHED, patch_source


class PatchAngelSlimAttentionTests(unittest.TestCase):
    def test_replaces_hardcoded_flash_attention(self) -> None:
        source = f"import os\n{ORIGINAL}\n"
        patched, changed = patch_source(source)
        self.assertTrue(changed)
        self.assertIn(PATCHED, patched)
        self.assertNotIn(ORIGINAL, patched)

    def test_is_idempotent(self) -> None:
        source = f"import os\n{PATCHED}\n"
        patched, changed = patch_source(source)
        self.assertFalse(changed)
        self.assertEqual(patched, source)


if __name__ == "__main__":
    unittest.main()
