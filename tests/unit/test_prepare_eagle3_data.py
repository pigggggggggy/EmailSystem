import unittest

from training.prepare_eagle3_data import prepare_rows


class PrepareEagle3DataTest(unittest.TestCase):
    def test_converts_lora_messages_to_angelslim_conversations(self):
        rows = [
            {
                "email_id": "mail-1",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "email"},
                    {"role": "assistant", "content": '{"category":"spam"}'},
                ],
            }
        ]
        prepared = prepare_rows(rows, limit=10, seed=7)
        self.assertEqual(prepared[0]["id"], "mail-1")
        self.assertEqual(prepared[0]["conversations"], rows[0]["messages"])

    def test_sampling_is_deterministic_and_removes_duplicate_ids(self):
        rows = [
            {
                "email_id": f"mail-{index}",
                "messages": [
                    {"role": "user", "content": "email"},
                    {"role": "assistant", "content": "reply"},
                ],
            }
            for index in range(10)
        ]
        rows.append(rows[0])
        first = prepare_rows(rows, limit=4, seed=11)
        second = prepare_rows(reversed(rows), limit=4, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_rejects_conversation_without_assistant_target(self):
        rows = [{"email_id": "bad", "messages": [{"role": "user", "content": "email"}]}]
        with self.assertRaises(ValueError):
            prepare_rows(rows, limit=1, seed=1)


if __name__ == "__main__":
    unittest.main()
