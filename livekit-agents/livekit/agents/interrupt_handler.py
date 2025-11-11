from .handler import InterruptHandler, Action
from .config import Config
from .middleware import LivekitInterruptMiddleware
from .types import ASRResult

__all__ = ["InterruptHandler", "Action", "Config", "LivekitInterruptMiddleware", "ASRResult"]

import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    ignored_words: List[str] = field(default_factory=lambda: ["uh", "umm", "hmm", "haan"])
    command_words: List[str] = field(default_factory=lambda: ["stop", "wait", "no", "hold on", "pause", "cancel"])
    min_confidence_for_interrupt: float = 0.6
    case_insensitive: bool = True

    @classmethod
    def from_env(cls):
        ignored = os.getenv("LIH_IGNORED_WORDS", None)
        commands = os.getenv("LIH_COMMAND_WORDS", None)
        min_conf = os.getenv("LIH_MIN_CONF", None)
        return cls(
            ignored_words=[w.strip() for w in ignored.split(",")] if ignored else cls().ignored_words,
            command_words=[w.strip() for w in commands.split(",")] if commands else cls().command_words,
            min_confidence_for_interrupt=float(min_conf) if min_conf else cls().min_confidence_for_interrupt,
        )

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any

@dataclass
class ASRResult:
    text: str
    confidence: Optional[float] = None
    tokens: Optional[Sequence[str]] = None
    metadata: Optional[Dict[str, Any]] = None

import unicodedata
import re
from typing import List

_token_re = re.compile(r"\w+['-]?\w*|\S+")

def normalize_text(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFKD", text)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_from_text(text: str) -> List[str]:
    text = normalize_text(text)
    matches = _token_re.findall(text)
    return [m for m in matches if any(c.isalnum() for c in m)]

import asyncio
import logging
from enum import Enum, auto
from typing import Optional, Iterable

from .types import ASRResult
from .config import Config
from .utils import normalize_text, tokens_from_text

logger = logging.getLogger("livekit.interrupt_handler")
logger.addHandler(logging.NullHandler())

class Action(Enum):
    IGNORE = auto()
    ACCEPT = auto()
    STOP = auto()

class InterruptHandler:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self._lock = asyncio.Lock()
        self._build_sets()

    def _build_sets(self):
        self._ignored = set(normalize_text(x) for x in (self.config.ignored_words or []) if x)
        self._commands = set(normalize_text(x) for x in (self.config.command_words or []) if x)

    async def update_ignored(self, new_list: Iterable[str]):
        async with self._lock:
            self.config.ignored_words = list(new_list)
            self._build_sets()
            logger.info("Updated ignored_words -> %s", self.config.ignored_words)

    async def decide(self, asr: ASRResult, agent_speaking: bool) -> Action:
        async with self._lock:
            text = normalize_text(asr.text or "")
            tokens = asr.tokens or tokens_from_text(text)
            tokens_n = [normalize_text(t) for t in tokens]

            if not text:
                logger.debug("Empty ASR -> %s", "IGNORE" if agent_speaking else "ACCEPT")
                return Action.IGNORE if agent_speaking else Action.ACCEPT

            # command presence -> STOP immediately
            for t in tokens_n:
                if t in self._commands:
                    logger.info("Command token detected '%s' -> STOP", t)
                    return Action.STOP

            # low-confidence while agent speaking -> ignore as murmur
            if agent_speaking and asr.confidence is not None and asr.confidence < self.config.min_confidence_for_interrupt:
                logger.debug("Low confidence %.2f while agent speaking -> IGNORE", asr.confidence)
                return Action.IGNORE

            # filler-only -> IGNORE when agent speaking, ACCEPT when quiet
            if tokens_n and all((t in self._ignored) for t in tokens_n):
                if agent_speaking:
                    logger.info("Filler-only while agent speaking ('%s') -> IGNORE", text)
                    return Action.IGNORE
                else:
                    logger.info("Filler-only while agent quiet ('%s') -> ACCEPT", text)
                    return Action.ACCEPT

            # mixed or other words -> ACCEPT
            logger.info("Valid user speech detected ('%s') -> ACCEPT", text)
            return Action.ACCEPT
        
import asyncio
import logging
from typing import Callable, Awaitable, Optional

from .handler import InterruptHandler, Action
from .types import ASRResult

logger = logging.getLogger("livekit.interrupt_handler.middleware")
logger.addHandler(logging.NullHandler())

class LivekitInterruptMiddleware:
    def __init__(
        self,
        handler: InterruptHandler,
        *,
        is_agent_speaking_cb: Callable[[], bool],
        pause_agent_audio: Callable[[], Awaitable[None]],
        resume_agent_audio: Callable[[], Awaitable[None]],
        stop_agent_cb: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.handler = handler
        self.is_agent_speaking_cb = is_agent_speaking_cb
        self.pause_agent_audio = pause_agent_audio
        self.resume_agent_audio = resume_agent_audio
        self.stop_agent_cb = stop_agent_cb

        self._lock = asyncio.Lock()
        self._suppressed = False

    async def on_transcript(self, asr_result: ASRResult):
        async with self._lock:
            agent_speaking = bool(self.is_agent_speaking_cb())
            decision = await self.handler.decide(asr_result, agent_speaking)

            if decision == Action.IGNORE:
                self._suppressed = True
                logger.debug("Ignored transcript while agent speaking: '%s'", asr_result.text)
                return

            # resume if we had been suppressing before
            if self._suppressed:
                logger.debug("Suppression cleared due to valid speech: '%s'", asr_result.text)
                self._suppressed = False

            if decision == Action.STOP:
                logger.info("STOP decision -> stopping agent immediately")
                if self.stop_agent_cb:
                    await self.stop_agent_cb()
                else:
                    try:
                        await self.pause_agent_audio()
                    except Exception:
                        logger.exception("pause_agent_audio failed during STOP")
                return

            if decision == Action.ACCEPT:
                logger.info("ACCEPT decision -> pausing agent audio and propagating user speech")
                try:
                    await self.pause_agent_audio()
                except Exception:
                    logger.exception("pause_agent_audio failed on ACCEPT")
                # downstream processing (transcript delivery to agent) should happen in original ASR pipeline
                return
