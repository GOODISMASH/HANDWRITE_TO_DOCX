import re

from spellchecker import SpellChecker


CYRILLIC_WORD_PATTERN = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
LETTER_PATTERN = re.compile(r"[А-Яа-яЁёA-Za-z]")
WORD_PATTERN = re.compile(r"[А-Яа-яЁёA-Za-z]{2,}")
MEANINGFUL_WORD_PATTERN = re.compile(r"[А-Яа-яЁёA-Za-z]{4,}")


def is_garbage_ocr(text):
    text = text.strip()

    if len(text) < 4:
        return True

    letters = LETTER_PATTERN.findall(text)
    if len(letters) < 4:
        return True

    if len(letters) / max(len(text), 1) < 0.45:
        return True

    words = WORD_PATTERN.findall(text)
    if not words or not MEANINGFUL_WORD_PATTERN.search(text):
        return True

    return False


class TextPostProcessor:
    """Correct likely Russian spelling errors while preserving whitespace."""

    def __init__(self):
        self.spell_checker = SpellChecker(language="ru", distance=1)

    def correct(self, text):
        if not text or is_garbage_ocr(text):
            return text

        corrected_lines = []
        for line in text.splitlines():
            corrected_lines.append(
                line if is_garbage_ocr(line) else CYRILLIC_WORD_PATTERN.sub(self._correct_token, line)
            )
        return "\n".join(corrected_lines)

    def _correct_token(self, match):
        token = match.group(0)
        corrected_parts = [self._correct_word(part) for part in token.split("-")]
        return "-".join(corrected_parts)

    def _correct_word(self, word):
        normalized = word.lower()

        # Replacing short words automatically introduces too many false positives.
        if len(normalized) < 4 or self.spell_checker.known([normalized]):
            return word

        corrected = self.spell_checker.correction(normalized)
        if not corrected or corrected == normalized:
            return word

        # OCR substitutions are safer than automatically inserting/removing letters.
        if len(corrected) != len(normalized):
            return word

        if word.isupper():
            return corrected.upper()
        if word[0].isupper():
            return corrected.capitalize()
        return corrected
