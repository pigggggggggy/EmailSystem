import unittest

from training.patch_angelslim_attention import (
    ATTENTION_ORIGINAL,
    ATTENTION_PATCHED,
    DATASET_ORIGINAL,
    DATASET_PATCHED,
    patch_attention_source,
    patch_dataset_source,
)


class PatchAngelSlimTests(unittest.TestCase):
    def test_replaces_hardcoded_flash_attention(self) -> None:
        source = f"import os\n{ATTENTION_ORIGINAL}\n"
        patched, changed = patch_attention_source(source)
        self.assertTrue(changed)
        self.assertIn(ATTENTION_PATCHED, patched)
        self.assertNotIn(ATTENTION_ORIGINAL, patched)

    def test_attention_patch_is_idempotent(self) -> None:
        source = f"import os\n{ATTENTION_PATCHED}\n"
        patched, changed = patch_attention_source(source)
        self.assertFalse(changed)
        self.assertEqual(patched, source)

    def test_maps_llm_dataset_target_type_to_none(self) -> None:
        source = f"first({DATASET_ORIGINAL})\nsecond({DATASET_ORIGINAL})\n"
        patched, changed = patch_dataset_source(source)
        self.assertTrue(changed)
        self.assertEqual(patched.count(DATASET_PATCHED), 2)
        self.assertNotIn(DATASET_ORIGINAL, patched)

    def test_dataset_patch_is_idempotent(self) -> None:
        source = f"first({DATASET_PATCHED})\nsecond({DATASET_PATCHED})\n"
        patched, changed = patch_dataset_source(source)
        self.assertFalse(changed)
        self.assertEqual(patched, source)


if __name__ == "__main__":
    unittest.main()
