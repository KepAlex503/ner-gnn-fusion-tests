from __future__ import annotations

import unittest

from mention_graph.encoding import apply_controlled_noise


class ControlledNoiseTests(unittest.TestCase):
    def test_dropout_preserves_mention_offsets_and_text(self) -> None:
        text = "Компанія Орбіта повідомила новину"
        start = text.index("Орбіта")
        end = start + len("Орбіта")
        noisy, noisy_start, noisy_end = apply_controlled_noise(
            text,
            start,
            end,
            kind="context_dropout",
            level=1.0,
            seed=1,
        )
        self.assertEqual((noisy_start, noisy_end), (start, end))
        self.assertEqual(noisy[start:end], "Орбіта")
        self.assertEqual(len(noisy), len(text))

    def test_asr_noise_is_reproducible(self) -> None:
        arguments = {
            "text": "Місто Київ гостинно приймає гостей",
            "mention_start": 6,
            "mention_end": 10,
            "kind": "asr",
            "level": 0.5,
            "seed": 22,
        }
        first = apply_controlled_noise(**arguments)
        second = apply_controlled_noise(**arguments)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

