import re


class RomanUrduPhoneticRules:
    """Heuristic phonetic folding for common Roman Urdu spelling variation."""

    def apply(self, token: str) -> str:
        value = token

        # Repeated letters: pleeease -> please (keep at most 2)
        value = re.sub(r"(.)\1{2,}", r"\1\1", value)

        # Common folds
        value = value.replace("ph", "f")
        value = value.replace("kh", "x")
        value = value.replace("gh", "g")
        value = value.replace("sh", "s")
        value = value.replace("ch", "c")

        # Normalize q/k variation
        value = value.replace("q", "k")

        # Vowel simplification (light)
        value = re.sub(r"aa+", "aa", value)
        value = re.sub(r"ee+", "ee", value)
        value = re.sub(r"ii+", "ii", value)
        value = re.sub(r"oo+", "oo", value)

        # y/i ambiguity in some contexts
        value = re.sub(r"^kiya$", "kia", value)
        value = re.sub(r"^kia$", "kya", value)

        return value
