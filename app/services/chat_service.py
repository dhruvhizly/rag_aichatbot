import asyncio
import logging
import random
import re
import time
from collections import deque
from typing import AsyncGenerator

from app.config import get_settings
from app.services.chitchat_classifier import is_semantic_chitchat
from app.services.llm_service import SYSTEM_PROMPT, CLASSIFIER_TEMPLATE, LLMService
from app.services.rag_registry import get_rag_service

logger = logging.getLogger(__name__)

# Shown while the request is being processed (retrieval + LLM, until first token).
# Add/remove freely — one is picked at random per request, then cycled every
# STATUS_CYCLE_INTERVAL_SECS while still waiting.
GENERATING_STATUSES = [
    # retrieval / search
    "Searching for relevant information…",
    "Scanning for relevant sections…",
    "Pulling up the most relevant passages…",
    "Locating the right details…",
    "Sifting through the source material…",
    "Tracking down the details…",
    "Combing through the context…",
    "Cross-referencing relevant sections…",
    "Pinpointing the right information…",
    "Retrieving relevant context…",

    # thinking / reasoning
    "Thinking this through…",
    "Working through the details…",
    "Making sense of the information…",
    "Putting the pieces together…",
    "Connecting the dots…",
    "Reasoning through this…",
    "Figuring out the best answer…",
    "Processing what I found…",

    # writing / composing
    "Writing up the answer…",
    "Drafting a response…",
    "Putting this into words…",
    "Composing a clear response…",
    "Structuring the answer…",
    "Pulling the answer together…",
    "Finishing up the response…",
    "Tying it all together…",
    "Polishing the response…",
    "Almost ready…",

    # light / conversational
    "On it…",
    "Just a moment…",
    "One second…",
    "Bear with me…",
    "Right with you…",
    "Give me just a second…",
    "Let me check on that…",
    "Let me look into that…",
    "Hang tight…",
    "A bit longer, thanks for your patience..."
]

_PRONOUN_RE = re.compile(
    r"\b(it|its|that|those|this|these|they|them|their|he|she|him|her|same)\b",
    re.IGNORECASE,
)

# Continuation starters — the current message extends a prior topic.
_CONTINUATION_RE = re.compile(
    r"^\s*("
    # conjunctions / connectors
    r"and(\s+(also|then|so|yet|still|now))?|"
    r"also|but|or|nor|yet|so|for|"
    r"plus|additionally|furthermore|moreover|besides|likewise|"
    r"on\s+top\s+of\s+(that|this)|in\s+addition(\s+to\s+that)?|"
    r"not\s+only\s+that|what'?s\s+more|more\s+than\s+that|beyond\s+that|"
    r"at\s+the\s+same\s+time|along\s+with\s+that|coupled\s+with\s+that|"
    # contrast / pivot
    r"but(\s+(also|then|wait))?|however|although|though|even\s+so|"
    r"that\s+said|having\s+said\s+that|with\s+that\s+said|then\s+again|"
    r"on\s+the\s+other\s+hand|on\s+the\s+flip\s+side|at\s+the\s+same\s+time|"
    r"regardless|nevertheless|nonetheless|still|yet|even\s+then|"
    # hypotheticals / alternatives
    r"what\s+(about|if|else|else\s+about)|how\s+about|"
    r"what\s+(would|could|might|should|will|does|did)\s+happen\s+if|"
    r"suppose(\s+(that|so))?|assuming(\s+that)?|given\s+that|"
    r"in\s+that\s+case|if\s+so|if\s+not|either\s+way|"
    r"alternatively|as\s+an\s+alternative|instead(\s+of\s+that)?|"
    r"or\s+(else|rather|instead|alternatively)|rather(\s+than)?|"
    # requests for more
    r"more(\s+(details?|info|information|examples?|context|explanation|depth|specifics?))?|"
    r"another(\s+(one|example|way|option|approach|case|question))?|"
    r"any(\s+(other|more|additional|further))(\s+(examples?|options?|ways?|thoughts?|ideas?|info|details?))?|"
    r"(what|are\s+there)(\s+any)?\s+(other|more|additional|alternative)\s+\w+|"
    r"give\s+me\s+(more|another|an?\s+example|some\s+examples?)|"
    r"show\s+me\s+(more|another|an?\s+example)|"
    r"list\s+(more|the\s+rest|all(\s+of\s+(them|those))?)|"
    # explicit continuation prompts
    r"tell\s+me\s+more(\s+about\s+(that|this|it))?|"
    r"continue(\s+(please|from\s+(there|here|that|where\s+you\s+(left|stopped))))?|"
    r"go\s+on(\s+please)?|keep\s+going(\s+please)?|keep\s+it\s+going|"
    r"carry\s+on(\s+from\s+(there|that))?|proceed(\s+(please|from\s+there))?|"
    r"don'?t\s+stop(\s+there)?|don'?t\s+hold\s+back|keep\s+it\s+coming|"
    r"finish(\s+(it|that|the\s+(list|sentence|thought|answer|explanation)))?|"
    r"complete(\s+(it|that|the\s+(list|answer|thought|explanation)))?|"
    r"(please\s+)?elaborate(\s+(on\s+(that|this|it))?)?|"
    r"(can\s+you\s+)?expand(\s+on(\s+(that|this|it))?)?|"
    r"(can\s+you\s+)?go\s+(deeper|further|into\s+more\s+detail)(\s+on\s+(that|this|it))?|"
    r"(please\s+)?clarify(\s+(that|this|it|further|more))?|"
    r"(can\s+you\s+)?explain(\s+(more|further|that|this|it|in\s+more\s+detail))?|"
    # follow-up pivots
    r"(and\s+)?what\s+about\s+(the\s+)?(rest|others?|remaining|next\s+(one|part|step|point))|"
    r"(and\s+)?how\s+does\s+(that|this)\s+(work|relate|apply|connect|fit)|"
    r"(and\s+)?(now\s+)?what(\s+else)?|then\s+what(\s+happens?)?|"
    r"(and\s+)?(what|how|where|when|who|why)\s+(comes?\s+)?(next|after(\s+(that|this))?)|"
    r"(what|how)\s+(about|does)\s+(the\s+)?(next|last|first|second|other)\s+(part|step|point|one)|"
    # affirmative re-prompts
    r"(yes\s+)?and\s+(then|so|also|but)|"
    r"(ok|okay|right|sure|got\s+it|noted|understood|alright|yes|yeah|yep)[,.]?\s+"
    r"(and|but|so|also|now|then|what\s+about|how\s+about|tell\s+me|keep\s+going|continue|more)|"
    # implicit continuers (standalone)
    r"next(\s+(one|please|up|step|point|part|question))?|"
    r"(what'?s\s+)?after\s+that|following(\s+that)?|subsequently|"
    r"then(\s+(what|how|where|when|who|why))?|"
    r"so(\s+(then|what|how|now))?|now(\s+what)?|"
    r"furthermore|in\s+turn|by\s+extension|consequently|as\s+a\s+result|"
    r"(and\s+)?lastly|finally(\s+(then))?|to\s+(finish|conclude|wrap\s+up)|"
    r"in\s+(closing|conclusion|summary|short)|to\s+sum\s+(up|it\s+up)|"
    r"anything\s+else(\s+(you\s+can\s+add|on\s+that|to\s+add))?|"
    r"is\s+that\s+(all|it|everything)|that'?s?\s+it\??|what\s+else(\s+is\s+there)?"
    r")\b",
    re.IGNORECASE,
)

# One-word / fragment follow-ups that only make sense given prior context.
_SHORT_FOLLOWUP_RE = re.compile(
    r"^\s*("
    # requests for more content
    r"more(\s+(please|info|information|details?|examples?|context|depth|specifics?|data|options?|results?))?|"
    r"examples?(\s+(please|of\s+that|of\s+this))?|"
    r"details?(\s+(please|on\s+that|on\s+this))?|"
    r"specifics?(\s+(please))?|specifically|"
    r"in\s+detail|in\s+depth|in\s+full|in\s+full\s+detail|"
    r"fully|completely|thoroughly|exhaustively|"
    # elaboration requests
    r"elaborate(\s+on\s+(that|this|it))?|"
    r"expand(\s+on(\s+(that|this|it))?)?|"
    r"explain(\s+(that|this|it|more|further|please|again|differently|in\s+simpler\s+terms))?|"
    r"clarify(\s+(that|this|it|please|further))?|"
    r"rephrase(\s+(that|this|it|please))?|"
    r"simplify(\s+(that|this|it|please))?|"
    r"summarize(\s+(that|this|it|please))?|"
    r"break\s+(it|that|this)\s+down(\s+(for\s+me|please))?|"
    r"walk\s+me\s+through\s+(it|that|this)|"
    r"spell\s+it\s+out(\s+(for\s+me|please))?|"
    r"dumb\s+it\s+down(\s+(for\s+me|please))?|"
    r"(in\s+)?(plain|simple|simpler|layman'?s?)\s+(terms|english|words|language)|"
    r"eli5|explain\s+like\s+i'?m\s+5|"
    # causal / reasoning probes
    r"why(\s+(not|so|though|exactly|specifically|is\s+that|would\s+that\s+be|does\s+that\s+matter))?|"
    r"how\s+come(\s+(though|so))?|"
    r"what'?s\s+the\s+(reason|point|purpose|rationale|logic|idea|catch|trade\s*off)|"
    r"for\s+what\s+(reason|purpose)|"
    r"what\s+caused\s+(that|this)|what\s+led\s+to\s+(that|this)|"
    r"how\s+(so|though|exactly|does\s+that\s+work)|"
    r"in\s+what\s+way(s)?|to\s+what\s+extent|"
    # factual/value probes (expanded)
    r"the\s+(date|dates|time|times|number|numbers|amount|amounts|"
    r"figure|figures|stat|stats|statistic|statistics|"
    r"name|names|term|terms|word|words|phrase|"
    r"price|prices|cost|costs|rate|rates|fee|fees|"
    r"deadline|deadlines|due\s+date|timeframe|timeline|duration|"
    r"reason|reasons|cause|causes|source|sources|origin|"
    r"result|results|outcome|outcomes|impact|effect|effects|"
    r"difference|differences|similarity|similarities|"
    r"advantage|advantages|disadvantage|disadvantages|"
    r"pro|pros|con|cons|benefit|benefits|downside|downsides|"
    r"step|steps|stage|stages|phase|phases|"
    r"type|types|kind|kinds|category|categories|"
    r"option|options|choice|choices|alternative|alternatives|"
    r"example|examples|exception|exceptions|"
    r"limit|limits|limitation|limitations|constraint|constraints|"
    r"requirement|requirements|condition|conditions|"
    r"definition|definitions|meaning|meanings)s?|"
    # implicit short probes
    r"and\??|then\??|so\??|but\??|or\??|now\??|"
    r"really\??|seriously\??|actually\??|"
    r"like\s+what\??|such\s+as\??|for\s+example\??|for\s+instance\??|"
    r"(which|what|who)\s+(one|ones|type|kind|part|aspect|piece)\??|"
    r"compared\s+to\s+what\??|relative\s+to\s+what\??|"
    r"how\s+(many|much|long|often|far|big|small|fast|slow|likely|important|significant|common|rare|good|bad)\??|"
    r"what\s+(size|kind|type|sort|form|version|format|level|degree|scale)\??|"
    r"since\s+when\??|as\s+of\s+when\??|until\s+when\??|by\s+when\??|"
    r"where\s+(exactly|specifically|though)\??|"
    r"who\s+(exactly|specifically|else|though)\??|"
    r"when\s+(exactly|specifically|though|did\s+that\s+happen)\??|"
    # confirmation / challenge probes
    r"are\s+you\s+sure(\s+(about\s+that|of\s+that))?|"
    r"is\s+that\s+(right|correct|accurate|true|certain|confirmed|so)\??|"
    r"really(\s+(though|so))?|"
    r"you\s+sure(\s+(about\s+that))?|"
    r"double[- ]check(\s+that)?|verify(\s+(that|this|it|please))?|"
    r"can\s+you\s+(confirm|verify|check|double[- ]check)(\s+(that|this|it))?|"
    r"is\s+that\s+still\s+(true|valid|accurate|the\s+case)\??|"
    r"has\s+that\s+(changed|been\s+updated)\??|"
    r"source(\s+(for\s+that|please))?|"
    r"(any\s+)?(proof|evidence|citation|citations|reference|references|link|links)\??|"
    r"where\s+did\s+you\s+(get|find|read|hear)\s+that\??|"
    r"according\s+to\s+who(m)?\??|"
    # agreement / challenge
    r"(is\s+that|that'?s)\s+(all|it|everything|correct|right|the\s+whole\s+(story|picture))\??|"
    r"nothing\s+(else|more|further)\??|"
    r"that'?s?\s+it\??|that'?s?\s+all\??|"
    r"no\s+(other|more|additional)\s+(options?|ways?|reasons?|factors?|examples?)\??"
    r")\??\s*$",
    re.IGNORECASE,
)

_CHITCHAT_RE = re.compile(
    r"^\s*("
    # greetings
    r"hi+|hello+|hey+|yo+|hiya|howdy|sup|greetings|what('s|\s+is)\s+good|"
    r"good\s*(morning|afternoon|evening|night|day|one)|"
    r"morning|afternoon|evening|g'?morning|g'?day|"
    r"ello|'ello|heya|heyy+|hii+|ohh?i+|"
    # thanks / acknowledgements
    r"thanks?(\s+(a\s+(lot|ton|million|bunch)|so\s+much|much|again|for\s+(everything|that|this)))?|"
    r"thank\s+you(\s+(so\s+much|very\s+much|a\s+lot|for\s+(everything|that|this)))?|"
    r"ty|thx|thnx|tnx|cheers|appreciated|much\s+appreciated|"
    r"many\s+thanks|thanks\s+a\s+(million|ton|bunch)|bless\s+you|"
    # affirmations / fillers
    r"ok(ay)?|k|kk|kkk|okie(\s+dokie)?|roger(\s+that)?|copy(\s+that)?|"
    r"cool|nice|great|awesome|perfect|excellent|fantastic|wonderful|brilliant|"
    r"sure|fine|alright|right|sounds\s+good|makes\s+sense|fair\s+enough|"
    r"got\s+it|understood|noted|gotcha|i\s+see|i\s+get\s+(it|that)|"
    r"10\s*[–-]?\s*4|good\s+to\s+know|good\s+to\s+go|g2g|"
    r"yes|yep|yeah|yup|yh|yass+|yas+|yea|yah|affirmative|"
    r"no|nope|nah|nay|negative|"
    r"lol|lmao|lmfao|rofl|haha+|hehe+|hihi+|heh|ha|"
    r"wow|whoa|woah|oh|ah|aha|ohh|ahh|ooh|omg|oh\s+my|"
    # farewells
    r"bye+|goodbye|good\s*bye|see\s+ya|see\s+you(\s+(around|soon|later|tomorrow))?|"
    r"cya|later|talk\s+(to\s+you\s+)?later|ttyl|ttys|"
    r"peace(\s+out)?|take\s+(care|it\s+easy)|have\s+a\s+(good|great|nice)\s+(one|day|night|evening|weekend)|"
    r"good\s+night|gn|g'?night|nighty\s*night|night|"
    r"catch\s+you\s+later|until\s+next\s+time|fare?well|so\s+long|adios|ciao|"
    # apologies / politeness
    r"sorry|my\s+bad|my\s+mistake|my\s+fault|apolog(ize|ies)|pardon(\s+me)?|excuse\s+me|"
    r"no\s+(problem|worries|issue|stress)|np|nw|all\s+good|no\s+biggie|"
    r"you('|\s+a)re\s+welcome|you'?re\s+welcome|welcome|of\s+course|"
    r"please|kindly|if\s+you\s+(don'?t\s+mind|could|would)|"
    # meta / identity / capability
    r"who\s+are\s+you|what\s+(are\s+you|do\s+you\s+do|can\s+you\s+do)|"
    r"what\s+are\s+you\s+capable\s+of|what\s+are\s+your\s+(capabilities|features|skills)|"
    r"help|help\s+me|what\s+can\s+you\s+help\s+(with|me\s+with)|how\s+do\s+you\s+work|"
    r"are\s+you\s+(there|alive|working|online|an?\s+ai|a\s+bot|a\s+robot|human)|"
    r"you\s+there|is\s+(anyone|somebody|someone)\s+there|"
    r"can\s+you\s+help(\s+me)?|do\s+you\s+understand|do\s+you\s+know|"
    r"are\s+you\s+ready|ready\??|let'?s\s+(go|start|begin|do\s+this)|"
    # small talk
    r"how\s+are\s+you(\s+doing)?|how('s|\s+is)\s+it\s+going|how('s|\s+is)\s+everything|"
    r"how('s|\s+is)\s+your\s+day(\s+going)?|how\s+have\s+you\s+been|how\s+do\s+you\s+do|"
    r"what('s|\s+is)\s+up|whats\s+up|wassup|what'?s\s+new|what'?s\s+good|"
    r"i('m|\s+am)\s+(good|well|fine|great|okay|alright|doing\s+well)(\s+(thanks?|thank\s+you))?|"
    r"not\s+(bad|much|too\s+bad)|same\s+(here|old|as\s+always)|just\s+(chilling|hanging|browsing)|"
    # conversation enders / fillers
    r"that('s|\s+is)\s+all(\s+(for\s+now|thanks?|today))?|no\s+more\s+questions|"
    r"i('m|\s+am)\s+done|done|all\s+done|all\s+set|we'?re\s+done|"
    r"nothing(\s+(else|more|further))?|never\s*mind|nvm|forget\s+it|skip(\s+it)?|"
    r"ignore(\s+(that|it))?|scratch\s+that|disregard(\s+that)?|"
    r"carry\s+on|continue|proceed|go\s+ahead|go\s+on|"
    r"i\s+was\s+just\s+(testing|checking|trying)|just\s+(testing|checking|kidding|joking)|"
    r"test|testing|check|checking|hello\s+world|ping|"
    # filler sounds / discourse markers
    r"umm*|uhh*|hmm+|huh|mhm+|mmm+|uh\s*huh|uh\s*oh|oops|whoops|"
    r"welp|whelp|anyhoo|anyway(s)?|so+|well+|right\s+then|"
    r"ok\s+so|so\s+um|ok\s+um|"
    # emoji-style text / reactions
    r":\)|:\(|:D|:P|;\)|<3|\\o/|o/|/o\\"
    r")[\s!.?,~\-]*$",
    re.IGNORECASE,
)

# How often to rotate to a new status while a phase is still running.
STATUS_CYCLE_INTERVAL_SECS = 5.0

def _looks_dependent(text: str) -> bool:
    """Whether `text` likely depends on prior conversation context."""
    t = text.strip()
    if not t or len(t) > 120:
        return False
    if _PRONOUN_RE.search(t):
        return True
    if _CONTINUATION_RE.match(t):
        return True
    if _SHORT_FOLLOWUP_RE.match(t):
        return True
    return False


def _is_chitchat(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # Fast path: strict regex matches single-phrase exact chitchat (no embed cost).
    if len(t) <= 40 and _CHITCHAT_RE.match(t):
        return True
    # Fallback: semantic similarity catches compound phrasings the regex misses
    # ("thanks for the answer", "that was helpful", "great explanation", etc.).
    return is_semantic_chitchat(t)


_RECENT_QUERIES_MAX = 3  # How many prior user messages to keep for query rewriting.

class ChatService:
    def __init__(self) -> None:
        settings = get_settings()
        self._histories: dict[str, list[dict]] = {}
        self._recent_queries: dict[str, deque[str]] = {}
        self._llm = LLMService()
        self._history_max_turns = settings.history_max_turns

    def _get_history(self, session_id: str) -> list[dict]:
        if session_id not in self._histories:
            self._histories[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        return self._histories[session_id]

    def _get_recent(self, session_id: str) -> deque[str]:
        if session_id not in self._recent_queries:
            self._recent_queries[session_id] = deque(maxlen=_RECENT_QUERIES_MAX)
        return self._recent_queries[session_id]

    def clear_history(self, session_id: str) -> None:
        self._histories.pop(session_id, None)
        self._recent_queries.pop(session_id, None)

    async def stream_response(
        self,
        user_message: str,
        session_id: str,
    ) -> AsyncGenerator[tuple[str, str], None]:
        history = self._get_history(session_id)
        rag = get_rag_service()

        # Fast path: chitchat — skip RAG entirely, talk straight to the LLM.
        if _is_chitchat(user_message):
            messages = [*history, {"role": "user", "content": user_message}]
            full_reply: list[str] = []
            async for token in self._llm.stream_chat(messages):
                full_reply.append(token)
                yield ("token", token)

            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "".join(full_reply)})
            self._trim_history(history)
            return

        # Build the embedding query. For messages that look dependent on prior
        # context, concatenate the last few user messages so the embedding
        # captures the topic even when it was introduced several turns back
        # (handles chained dependents: "Tell me about X" → "explain more" → "what's its range").
        # The LLM still sees only the current message in the prompt.
        recent = self._get_recent(session_id)
        if recent and _looks_dependent(user_message):
            embed_query = " ".join([*recent, user_message])
            logger.info(
                "dependent query — rewriting for retrieval (using %d prior turns)",
                len(recent),
            )
        else:
            embed_query = user_message

        # Relevance Classifier: check if the query is relevant to the documents or motives
        classifier_prompt = f"{CLASSIFIER_TEMPLATE}\n\nUser query: {embed_query}"
        is_relevant_resp = await self._llm.chat([
            {"role": "user", "content": classifier_prompt}
        ])
        
        if is_relevant_resp.strip().lower().startswith("no"):
            generic_reply = "I am sorry, but I can only assist with questions related to ARI Simulation and its products. If you have any questions about ARI Simulation, feel free to ask!"
            yield ("token", generic_reply)
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": generic_reply})
            self._trim_history(history)
            return

        # Full RAG path with cycling status messages.
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        async def cycle(messages: list[str], stop: asyncio.Event) -> None:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=STATUS_CYCLE_INTERVAL_SECS
                    )
                    return
                except asyncio.TimeoutError:
                    await queue.put(("status", random.choice(messages)))

        async def worker() -> None:
            full_reply: list[str] = []
            t0 = time.perf_counter()
            try:
                # Single status phase: cycle from retrieval start until first token.
                await queue.put(("status", random.choice(GENERATING_STATUSES)))
                stop = asyncio.Event()
                cycler = asyncio.create_task(cycle(GENERATING_STATUSES, stop))
                try:
                    retrieval, doc_session = await asyncio.to_thread(
                        rag.retrieval_and_presence, session_id, embed_query
                    )
                    sources = rag.list_sources(session_id)
                    self._get_recent(session_id).append(user_message)
                    t_retrieval = time.perf_counter() - t0

                    user_for_llm = self._user_turn_with_retrieval(
                        user_message,
                        retrieval,
                        doc_session=doc_session,
                        sources=sources,
                    )
                    messages = [*history, {"role": "user", "content": user_for_llm}]

                    t1 = time.perf_counter()
                    first_token_at: float | None = None
                    async for token in self._llm.stream_chat(messages):
                        if first_token_at is None:
                            first_token_at = time.perf_counter() - t1
                            stop.set()
                        full_reply.append(token)
                        await queue.put(("token", token))
                finally:
                    stop.set()
                    await cycler

                t_total = time.perf_counter() - t1
                logger.info(
                    "chat timing: retrieval=%.2fs ttft=%.2fs llm_total=%.2fs",
                    t_retrieval,
                    first_token_at if first_token_at is not None else -1.0,
                    t_total,
                )

                history.append({"role": "user", "content": user_message})
                history.append(
                    {"role": "assistant", "content": "".join(full_reply)}
                )
                self._trim_history(history)
            except Exception as exc:  # surface to consumer via sentinel
                await queue.put(("__error__", repr(exc)))
            finally:
                await queue.put(SENTINEL)

        worker_task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                kind, payload = item
                if kind == "__error__":
                    raise RuntimeError(payload)
                yield (kind, payload)
        finally:
            await worker_task

    def _trim_history(self, history: list[dict]) -> None:
        max_messages = 1 + 2 * self._history_max_turns
        if len(history) <= max_messages:
            return
        keep_tail = max_messages - 1
        del history[1 : len(history) - keep_tail]

    @staticmethod
    def _user_turn_with_retrieval(
        user_message: str,
        retrieval_block: str,
        *,
        doc_session: bool,
        sources: list[str],
    ) -> str:
        product_names = _sources_to_product_names(sources)
        if product_names:
            inventory = "\n".join(f"- {name}" for name in product_names)
        else:
            inventory = "(none)"
        return (
            f"[Available products]\n{inventory}\n\n"
            f"[Retrieved excerpts]\n{retrieval_block}\n\n"
            f"[User]\n{user_message}"
        )


_FILENAME_NOISE_RE = re.compile(
    r"\b("
    r"operator'?s?|user'?s?|owner'?s?|"
    r"manuals?|guide|guides|instructions?|handbook|"
    r"datasheets?|spec|specs|specification|specifications|"
    r"document|documents|doc|docs|"
    r"brochure|catalog|catalogue|reference"
    r")\b",
    re.IGNORECASE,
)


def _filename_to_product_name(filename: str) -> str | None:
    """Convert a raw filename into a clean product/device name.

    Strips extension, removes manual/datasheet noise words, and normalizes
    separators. Returns None if nothing meaningful remains.
    """
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", filename).strip()
    cleaned = _FILENAME_NOISE_RE.sub(" ", stem)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" /-_")
    if not cleaned:
        return None
    if cleaned.isdigit():
        return None
    return cleaned


def _sources_to_product_names(sources: list[str]) -> list[str]:
    """Map a list of source filenames to deduped, ordered product names."""
    seen: set[str] = set()
    out: list[str] = []
    for src in sources:
        name = _filename_to_product_name(src)
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out
 