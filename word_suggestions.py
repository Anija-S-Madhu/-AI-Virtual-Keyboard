"""
Word Suggestion Engine for AI Virtual Keyboard.
Provides contextual prefix-based word suggestions without auto-correcting typed text.
"""

import re
from typing import List

# Top most frequently used English words, ordered roughly by frequency
COMMON_WORDS = [
    # Top function and common words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    
    # Common conversational & practical words
    "hello", "hi", "hey", "please", "thank", "thanks", "welcome", "yes", "sure", "okay",
    "great", "fine", "sorry", "help", "need", "today", "tomorrow", "tonight", "morning", "night",
    "friend", "family", "home", "house", "room", "school", "work", "office", "call", "message",
    "email", "phone", "number", "text", "send", "receive", "meeting", "project", "code", "system",
    "keyboard", "screen", "computer", "device", "camera", "hand", "finger", "gesture", "virtual", "smart",
    
    # Common verbs & adjectives
    "start", "stop", "open", "close", "read", "write", "type", "click", "press", "move",
    "check", "find", "search", "show", "hide", "save", "clear", "delete", "remove", "add",
    "change", "update", "create", "build", "run", "play", "test", "try", "learn", "understand",
    "happy", "nice", "cool", "easy", "hard", "fast", "slow", "high", "low", "big",
    "small", "long", "short", "best", "better", "ready", "done", "fine", "right", "wrong",
    "true", "false", "important", "simple", "clean", "classy", "modern", "beautiful", "smooth", "perfect",
    
    # Question & connecting words
    "where", "why", "again", "always", "never", "sometimes", "often", "usually", "almost", "together",
    "before", "while", "during", "without", "under", "between", "through", "against", "around", "still",
    "already", "enough", "both", "few", "many", "much", "several", "each", "every", "another",
    
    # Common technological and modern terms
    "interface", "application", "software", "hardware", "network", "online", "internet", "browser",
    "button", "window", "display", "color", "theme", "design", "style", "feature", "version", "data",
    "input", "output", "value", "result", "action", "state", "control", "sensor", "model", "tracking",
    "opencv", "python", "mediapipe", "algorithm", "developer", "solution", "quality", "experience"
]

# Ensure uniqueness while preserving order
_seen = set()
UNIQUE_COMMON_WORDS = []
for w in COMMON_WORDS:
    w_lower = w.lower()
    if w_lower not in _seen:
        _seen.add(w_lower)
        UNIQUE_COMMON_WORDS.append(w_lower)

DEFAULT_STARTERS = ["Hello", "Thank", "Please"]


class WordSuggestionEngine:
    """Provides high-relevance prefix-based word suggestions."""

    def __init__(self, vocabulary: List[str] = None):
        if vocabulary is None:
            self.vocabulary = UNIQUE_COMMON_WORDS
        else:
            self.vocabulary = vocabulary

    @staticmethod
    def get_current_word_prefix(text: str) -> str:
        """Extracts the word fragment currently being typed at the end of text."""
        if not text:
            return ""
        # Match trailing alphabetical characters or apostrophes
        match = re.search(r"([A-Za-z0-9'\-]+)$", text)
        if match:
            return match.group(1)
        return ""

    def get_suggestions(self, text: str, max_count: int = 3, is_caps: bool = False) -> List[str]:
        """
        Returns up to `max_count` suggestions based on the last word prefix in `text`.
        Suggestions preserve casing (lowercase, Title Case, or UPPERCASE).
        """
        prefix = self.get_current_word_prefix(text)

        if not prefix:
            # Suggest starter words
            if is_caps:
                return [w.upper() for w in DEFAULT_STARTERS[:max_count]]
            return DEFAULT_STARTERS[:max_count]

        prefix_lower = prefix.lower()
        candidates = []

        # Find matching words that start with prefix
        for word in self.vocabulary:
            if word.startswith(prefix_lower) and word != prefix_lower:
                candidates.append(word)
                if len(candidates) >= max_count * 2:
                    break

        # If not enough exact prefix matches, try finding words containing prefix if prefix length >= 3
        if len(candidates) < max_count and len(prefix_lower) >= 3:
            for word in self.vocabulary:
                if prefix_lower in word and word not in candidates and word != prefix_lower:
                    candidates.append(word)
                    if len(candidates) >= max_count:
                        break

        results = candidates[:max_count]

        # If still empty, return sensible fallback
        if not results:
            if is_caps:
                return [w.upper() for w in DEFAULT_STARTERS[:max_count]]
            return DEFAULT_STARTERS[:max_count]

        # Apply casing to match user's input
        formatted = []
        is_all_upper = prefix.isupper() or (is_caps and len(prefix) > 0)
        is_title = prefix[0].isupper() and not is_all_upper

        for w in results:
            if is_all_upper:
                formatted.append(w.upper())
            elif is_title:
                formatted.append(w.capitalize())
            else:
                formatted.append(w.lower())

        return formatted

    @staticmethod
    def apply_suggestion(text: str, suggestion: str) -> str:
        """
        Replaces the trailing word prefix in `text` with `suggestion` + ' '.
        This is ONLY executed when the user explicitly clicks a suggestion chip.
        """
        if not text:
            return suggestion + " "

        match = re.search(r"([A-Za-z0-9'\-]+)$", text)
        if match:
            start_idx = match.start(1)
            return text[:start_idx] + suggestion + " "
        else:
            # If text ended with space or punctuation, just append
            if text.endswith(" "):
                return text + suggestion + " "
            return text + " " + suggestion + " "

