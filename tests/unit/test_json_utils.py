import unittest

from email_system.skills.json_utils import parse_json_object


class JsonUtilsTest(unittest.TestCase):
    def test_parse_json_from_fenced_output(self):
        self.assertEqual(parse_json_object('```json\n{"category": "support"}\n```'), {"category": "support"})

    def test_parse_first_json_object_from_explanatory_output(self):
        self.assertEqual(parse_json_object('结果如下：{"summary": "已收到"}。'), {"summary": "已收到"})


if __name__ == "__main__":
    unittest.main()
