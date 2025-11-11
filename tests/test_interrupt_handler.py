import pytest
from livekit.agents.interrupt_handler.handler import InterruptHandler, Action
from livekit.agents.interrupt_handler.types import ASRResult
from livekit.agents.interrupt_handler.config import Config

@pytest.mark.asyncio
async def test_filler_ignored_while_agent_speaking():
    cfg = Config(ignored_words=["uh", "umm"], command_words=["stop"], min_confidence_for_interrupt=0.5)
    h = InterruptHandler(cfg)
    res = await h.decide(ASRResult(text="umm", confidence=0.9), agent_speaking=True)
    assert res == Action.IGNORE

@pytest.mark.asyncio
async def test_filler_accepted_when_quiet():
    h = InterruptHandler(Config(ignored_words=["uh", "umm"]))
    res = await h.decide(ASRResult(text="uh", confidence=0.9), agent_speaking=False)
    assert res == Action.ACCEPT

@pytest.mark.asyncio
async def test_command_stops_agent():
    h = InterruptHandler(Config(command_words=["stop"]))
    res = await h.decide(ASRResult(text="please stop", confidence=0.9), agent_speaking=True)
    assert res == Action.STOP

@pytest.mark.asyncio
async def test_low_confidence_ignored():
    h = InterruptHandler(Config(min_confidence_for_interrupt=0.8))
    res = await h.decide(ASRResult(text="hmm yeah", confidence=0.3), agent_speaking=True)
    assert res == Action.IGNORE
