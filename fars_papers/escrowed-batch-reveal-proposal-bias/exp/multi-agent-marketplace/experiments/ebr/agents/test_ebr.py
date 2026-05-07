"""Integration test for EBRCustomerAgent escrow buffer logic.

Mocks super().fetch_messages() to simulate sequential proposal arrivals
and verifies the buffering/batching/shuffling behavior without needing
a live PostgreSQL instance.
"""

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "magentic-marketplace" / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from magentic_marketplace.marketplace.actions.actions import (
    FetchMessagesResponse,
    ReceivedMessage,
)
from magentic_marketplace.marketplace.actions.messaging import (
    OrderItem,
    OrderProposal,
    TextMessage,
)


def _make_proposal_msg(proposal_id: str, price: float, index: int) -> ReceivedMessage:
    return ReceivedMessage(
        from_agent_id=f"biz_{proposal_id}",
        to_agent_id="customer_1",
        created_at=datetime.now(UTC),
        message=OrderProposal(
            id=proposal_id,
            items=[OrderItem(id="i1", item_name="Widget", quantity=1, unit_price=price)],
            total_price=price,
        ),
        index=index,
    )


def _make_text_msg(content: str, index: int) -> ReceivedMessage:
    return ReceivedMessage(
        from_agent_id="biz_text",
        to_agent_id="customer_1",
        created_at=datetime.now(UTC),
        message=TextMessage(content=content),
        index=index,
    )


async def test_ebr_buffering():
    from ebr.agents.ebr_agent import EBRCustomerAgent
    from ebr.agents.hardgate_agent import HardGateCustomerAgent

    profile = MagicMock()
    profile.id = "customer_1"
    profile.agent_id = "customer_1"

    agent = EBRCustomerAgent.__new__(EBRCustomerAgent)
    agent._ebr_buffer = []
    agent._ebr_released = False
    agent._ebr_reveal_order = None
    agent._ebr_arrival_order = []
    agent._K = 3
    agent._logger = MagicMock()

    fetch_responses = [
        FetchMessagesResponse(
            messages=[
                _make_text_msg("Hi there", 1),
                _make_proposal_msg("p1", 10.0, 2),
            ],
            has_more=False,
        ),
        FetchMessagesResponse(
            messages=[_make_proposal_msg("p2", 20.0, 3)],
            has_more=False,
        ),
        FetchMessagesResponse(
            messages=[
                _make_text_msg("Reminder", 4),
                _make_proposal_msg("p3", 15.0, 5),
            ],
            has_more=False,
        ),
    ]

    call_idx = 0

    async def mock_super_fetch():
        nonlocal call_idx
        resp = fetch_responses[call_idx]
        call_idx += 1
        return resp

    with patch.object(HardGateCustomerAgent, "fetch_messages", side_effect=mock_super_fetch):
        r1 = await agent.fetch_messages()
        assert len([m for m in r1.messages if isinstance(m.message, OrderProposal)]) == 0, \
            f"Call 1: expected 0 proposals, got {len([m for m in r1.messages if isinstance(m.message, OrderProposal)])}"
        assert len([m for m in r1.messages if isinstance(m.message, TextMessage)]) == 1, \
            "Call 1: expected 1 text message"
        print("  Call 1 OK: 0 proposals, 1 text message passed through")

        r2 = await agent.fetch_messages()
        assert len([m for m in r2.messages if isinstance(m.message, OrderProposal)]) == 0, \
            "Call 2: expected 0 proposals"
        assert len(r2.messages) == 0, "Call 2: expected 0 messages total"
        print("  Call 2 OK: 0 proposals, 0 messages (only proposal was buffered)")

        r3 = await agent.fetch_messages()
        proposals_in_r3 = [m for m in r3.messages if isinstance(m.message, OrderProposal)]
        texts_in_r3 = [m for m in r3.messages if isinstance(m.message, TextMessage)]
        assert len(proposals_in_r3) == 3, \
            f"Call 3: expected 3 proposals, got {len(proposals_in_r3)}"
        assert len(texts_in_r3) == 1, \
            f"Call 3: expected 1 text message, got {len(texts_in_r3)}"
        print(f"  Call 3 OK: 3 proposals released, 1 text message")

        proposal_ids = {m.message.id for m in proposals_in_r3}
        assert proposal_ids == {"p1", "p2", "p3"}, f"Wrong proposal IDs: {proposal_ids}"

        assert agent._ebr_released is True, "Expected _ebr_released=True"
        assert agent._ebr_reveal_order is not None, "Expected reveal order to be logged"

        reveal_map = agent._ebr_reveal_order["reveal_position_to_arrival_rank"]
        assert len(reveal_map) == 3, f"Expected 3 entries in reveal map, got {len(reveal_map)}"
        assert set(reveal_map.values()) == {1, 2, 3}, f"Reveal map values wrong: {reveal_map}"
        print(f"  Reveal order: {agent._ebr_reveal_order}")

    print("\nAll EBR integration tests PASSED.")


if __name__ == "__main__":
    asyncio.run(test_ebr_buffering())
