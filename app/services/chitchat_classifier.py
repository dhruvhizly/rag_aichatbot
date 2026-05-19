"""Semantic chitchat classifier.

Used as a fallback when the strict regex in `chat_service` doesn't match.
Embeds the user message with the existing MiniLM embedder and compares against
a small canonical corpus of chitchat utterances. If the max cosine similarity
is above the threshold, the message is treated as chitchat (skip RAG).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from app.services.embeddings import embed_query_cached, get_embeddings

logger = logging.getLogger(__name__)

# Canonical chitchat utterances. The classifier uses semantic similarity, so
# you don't need exhaustive variants — a few representative phrasings per
# category generalize well. Add or remove freely.
CHITCHAT_CORPUS: list[str] = [
    # ── Greetings ────────────────────────────────────────────────────────────
    "hi", "hello", "hey", "hey there", "hiya", "howdy",
    "good morning", "good afternoon", "good evening", "good night",
    "what's up", "what's going on", "how's it going", "how are you",
    "how are you doing", "how have you been", "how do you do",
    "nice to meet you", "pleased to meet you", "greetings",

    # ── Thanks / Appreciation ────────────────────────────────────────────────
    "thanks", "thank you", "thanks a lot", "thank you very much",
    "thanks a bunch", "thanks a ton", "many thanks", "thanks so much",
    "thank you so much", "thanks for the answer", "thanks for your help",
    "thanks for explaining", "thanks for the clarification",
    "thanks that was helpful", "that was really helpful, thanks",
    "i appreciate it", "appreciate the help", "much appreciated",
    "i really appreciate that", "i appreciate your help",
    "i'm grateful", "grateful for the help",

    # ── Acknowledgements / Confirmations ────────────────────────────────────
    "ok", "okay", "ok thanks", "okay got it", "got it", "got it thanks",
    "understood", "noted", "alright", "alright thanks",
    "that makes sense", "that helps", "that was helpful",
    "that clears it up", "that answers my question",
    "fair enough", "sounds good", "sounds right", "sounds about right",
    "great", "perfect", "cool", "awesome", "wonderful", "excellent",
    "makes sense", "i see", "i understand", "i get it", "i get that",
    "roger that", "copy that",

    # ── Praise / Compliments on the answer ──────────────────────────────────
    "great answer", "nice answer", "good explanation", "well explained",
    "clear answer", "well done", "nicely put", "good job", "great job",
    "that's a great answer", "that's a good explanation",
    "very clear", "very helpful", "super helpful", "really helpful",
    "excellent explanation", "perfect explanation", "brilliant",
    "you explained that well", "nice work", "impressive",

    # ── Requests to continue / follow-ups with no content ───────────────────
    "go on", "continue", "please continue", "keep going",
    "tell me more", "and then what", "what else",
    "anything else i should know",

    # ── Expressions of surprise or emotion ──────────────────────────────────
    "wow", "oh wow", "interesting", "oh interesting", "fascinating",
    "that's interesting", "that's fascinating", "no way", "really",
    "seriously", "oh really", "i didn't know that", "good to know",

    # ── Apologies / Corrections ──────────────────────────────────────────────
    "sorry", "my bad", "apologies", "sorry about that",
    "never mind that", "disregard that", "ignore my last message",

    # ── Negations / Mild dismissals ──────────────────────────────────────────
    "no thanks", "not now", "never mind", "forget it",
    "don't worry about it", "it's fine", "no worries", "no problem",
    "not a problem", "that's okay", "that's fine",

    # ── Farewells ────────────────────────────────────────────────────────────
    "bye", "goodbye", "bye bye", "see you", "see you later",
    "see you around", "have a good day", "have a great day",
    "have a good one", "take care", "talk to you later",
    "until next time", "catch you later", "so long",
    "thanks and goodbye", "thanks bye",

    # ── Meta / Identity ──────────────────────────────────────────────────────
    "who are you", "what are you", "what can you do",
    "what are your capabilities", "are you a bot", "are you an ai",
    "are you human", "are you a robot", "are you chatgpt",
    "what's your name", "what do you do",

    # ── Conversation enders ──────────────────────────────────────────────────
    "that's all", "that's all for now", "no more questions",
    "i'm done", "i'm good", "i'm all set", "we're done here",
    "nothing else", "nothing else for now", "i think that's it",
    "that's everything", "all good", "all done",

    # ── Affirmations / Negations ─────────────────────────────────────────────
    "yes", "yeah", "yep", "yup", "uh huh", "mhm", "sure", "of course",
    "absolutely", "definitely", "exactly", "correct", "right",
    "no", "nope", "nah", "not really",

    # ── Filler / thinking out loud ───────────────────────────────────────────
    "hmm", "hm", "let me think", "i'm not sure", "i don't know",
    "good question", "interesting question",

    # ── Repeated/retry nudges (no real content) ──────────────────────────────
    "can you repeat that", "say that again", "what did you say",
    "could you rephrase that", "i didn't understand",
]

# Cosine threshold. Embeddings are L2-normalized, so cosine == dot product.
# Tune downward to catch more, upward to be stricter.
DEFAULT_THRESHOLD = 0.62

# Length cap: messages longer than this are almost certainly not chitchat.
MAX_LEN = 80

@lru_cache(maxsize=1)
def _corpus_matrix() -> np.ndarray:
    emb = get_embeddings()
    vectors = emb.embed_documents(CHITCHAT_CORPUS)
    return np.asarray(vectors, dtype=np.float32)


def is_semantic_chitchat(text: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    t = text.strip()
    if not t or len(t) > MAX_LEN:
        return False
    try:
        query_vec = np.asarray(embed_query_cached(t), dtype=np.float32)
        sims = _corpus_matrix() @ query_vec
        score = float(sims.max())
    except Exception:
        logger.exception("Semantic chitchat check failed; falling through")
        return False
    if score >= threshold:
        logger.info("semantic chitchat match (score=%.3f) on: %r", score, t)
        return True
    return False


def prewarm() -> None:
    _corpus_matrix()
