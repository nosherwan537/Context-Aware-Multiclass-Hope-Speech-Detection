import re


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")


class TextCleaner:
    """Text cleaning utility for Roman Urdu/code-mixed text."""

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def clean(self, text: str) -> str:
        if text is None:
            return ""

        value = str(text)
        if self.lowercase:
            value = value.lower()

        value = URL_PATTERN.sub(" ", value)
        value = HTML_PATTERN.sub(" ", value)
        value = re.sub(r"[#@]", " ", value)
        value = re.sub(r"[^a-z0-9\s']", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value
