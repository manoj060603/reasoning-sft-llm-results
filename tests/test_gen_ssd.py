import unittest
from types import SimpleNamespace

from reasoning.gen_ssd import candidate_count, empty_stop_is_terminal, serialized_training_tokens


class GenSSDTests(unittest.TestCase):
    def test_progressive_candidate_schedule(self):
        self.assertEqual(candidate_count(0, 4, 2), 4)
        self.assertEqual(candidate_count(1, 4, 2), 2)
        self.assertEqual(candidate_count(8, 4, 2), 2)

    def test_invalid_schedule_rejected(self):
        with self.assertRaises(ValueError):
            candidate_count(-1, 4, 2)
        with self.assertRaises(ValueError):
            candidate_count(0, 0, 2)

    def test_zero_token_stop_is_terminal(self):
        stop = SimpleNamespace(text="", finish_reason="stop")
        text = SimpleNamespace(text="continued", finish_reason="length")
        self.assertTrue(empty_stop_is_terminal([stop, stop]))
        self.assertFalse(empty_stop_is_terminal([stop, text]))
        self.assertFalse(empty_stop_is_terminal([]))

    def test_serialized_training_tokens_uses_complete_chat_template(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, return_dict, add_generation_prompt):
                self.assertions = (tokenize, return_dict)
                return list(range(10 if add_generation_prompt else 10 + len(messages[-1]["content"])))

        tokenizer = Tokenizer()
        self.assertEqual(serialized_training_tokens(tokenizer, "question", "answer"), 16)
        self.assertEqual(tokenizer.assertions, (True, False))


if __name__ == "__main__":
    unittest.main()
