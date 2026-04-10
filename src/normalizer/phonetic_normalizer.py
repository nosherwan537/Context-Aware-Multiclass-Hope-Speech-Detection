from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set

from rapidfuzz import process, fuzz

from .cleaner import TextCleaner
from .lexicon import CANONICAL_MAP, PROTECTED_TOKENS
from .phonetic_rules import RomanUrduPhoneticRules

DEFAULT_CANONICAL_MAP = CANONICAL_MAP
DEFAULT_PROTECTED_TOKENS = PROTECTED_TOKENS


@dataclass
class PhoneticNormalizer:
    """Roman Urdu phonetic-aware normalizer using cleaner + rules + fuzzy lexicon."""

    canonical_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CANONICAL_MAP))
    protected_tokens: Set[str] = field(default_factory=lambda: set(DEFAULT_PROTECTED_TOKENS))
    similarity_threshold: float = 88.0
    min_token_len: int = 3
    cleaner: Optional[TextCleaner] = None
    rules: Optional[RomanUrduPhoneticRules] = None

    def __post_init__(self) -> None:
        if self.cleaner is None:
            self.cleaner = TextCleaner(lowercase=True)
        if self.rules is None:
            self.rules = RomanUrduPhoneticRules()

    def clean_text(self, text: str) -> str:
        return self.cleaner.clean(text)

    def _normalize_token(self, token: str) -> str:
        if len(token) < self.min_token_len or token in self.protected_tokens:
            return token

        phonetic = self.rules.apply(token)

        if phonetic in self.canonical_map:
            return self.canonical_map[phonetic]
        if token in self.canonical_map:
            return self.canonical_map[token]

        choices = list(self.canonical_map.keys())
        if not choices:
            return token

        match = process.extractOne(phonetic, choices, scorer=fuzz.ratio)
        if match and match[1] >= self.similarity_threshold:
            return self.canonical_map.get(match[0], token)
        return token

    def normalize(self, text: str) -> str:
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        normalized = [self._normalize_token(tok) for tok in tokens]
        return " ".join(normalized)

    def bulk_normalize(self, texts: Iterable[str]) -> Iterable[str]:
        for t in texts:
            yield self.normalize(t)
