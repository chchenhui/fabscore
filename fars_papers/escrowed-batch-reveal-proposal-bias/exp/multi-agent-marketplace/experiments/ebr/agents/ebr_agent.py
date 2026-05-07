"""EBR (Escrowed Batch Reveal) customer agent: Condition C.

Subclasses HardGateCustomerAgent to add proposal buffering. Proposals are
hidden from the LLM until K arrive, then revealed simultaneously in shuffled
order. This isolates the sequential-visibility mechanism: the B->C comparison
holds payment gating constant while changing only whether proposals are seen
one-at-a-time or all-at-once.

Nudge mechanism: after NUDGE_AFTER_EMPTY consecutive empty fetches with a
partial buffer, injects a synthetic text message prompting the LLM to send
follow-up messages to businesses that haven't sent formal proposals yet.

Fallback: if MAX_EMPTY_FETCHES consecutive fetch_messages calls return no new
proposals and the buffer is non-empty but < K, release whatever is buffered
and lower the HardGate threshold to match.
"""

import random
from datetime import datetime, timezone

from magentic_marketplace.marketplace.actions.actions import (
    FetchMessagesResponse,
    ReceivedMessage,
)
from magentic_marketplace.marketplace.actions.messaging import (
    OrderProposal,
    TextMessage,
)

from .hardgate_agent import HardGateCustomerAgent

NUDGE_AFTER_EMPTY = 4
MAX_EMPTY_FETCHES = 20


class EBRCustomerAgent(HardGateCustomerAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ebr_buffer = []
        self._ebr_released = False
        self._ebr_reveal_order = None
        self._ebr_arrival_order = []
        self._ebr_empty_fetch_count = 0
        self._ebr_nudge_sent = False
        self._ebr_proposal_senders = set()

    def _make_nudge_message(self):
        n_buffered = len(self._ebr_buffer)
        return ReceivedMessage(
            from_agent_id="system",
            to_agent_id="customer_0001",
            created_at=datetime.now(timezone.utc),
            message=TextMessage(
                content=(
                    f"NOTICE: You have been waiting for proposals. "
                    f"So far {n_buffered} of {self._K} expected proposals have arrived. "
                    f"Some businesses may have responded with text messages but not yet "
                    f"sent formal order proposals. Please send follow-up messages to any "
                    f"businesses that haven't provided a formal order_proposal yet, "
                    f"requesting that they submit a formal quote with pricing."
                )
            ),
            index=-1,
        )

    def _release_batch(self, non_proposals, batch_size, is_fallback=False):
        batch = list(self._ebr_buffer[:batch_size])
        arrival_ids = [m.message.id for m in batch]
        random.shuffle(batch)
        shuffled_ids = [m.message.id for m in batch]

        self._ebr_reveal_order = {
            "reveal_position_to_arrival_rank": {},
            "arrival_order": arrival_ids,
            "shuffled_order": shuffled_ids,
        }
        for reveal_pos, msg in enumerate(batch, start=1):
            arrival_rank = arrival_ids.index(msg.message.id) + 1
            self._ebr_reveal_order["reveal_position_to_arrival_rank"][reveal_pos] = arrival_rank

        self._ebr_released = True
        self._ebr_buffer = self._ebr_buffer[batch_size:]

        if is_fallback:
            self._K = len(batch)

        self.logger.info(
            f"[EBR] Released batch of {len(batch)} proposals"
            f"{' (fallback)' if is_fallback else ''}. "
            f"Arrival order: {arrival_ids}, Shuffled order: {shuffled_ids}"
        )

        return FetchMessagesResponse(
            messages=non_proposals + batch, has_more=False
        )

    async def fetch_messages(self) -> FetchMessagesResponse:
        if self._ebr_released:
            return await super().fetch_messages()

        response = await super().fetch_messages()

        proposals = []
        non_proposals = []
        for msg in response.messages:
            if isinstance(msg.message, OrderProposal):
                proposals.append(msg)
                self._ebr_proposal_senders.add(msg.from_agent_id)
            else:
                non_proposals.append(msg)

        self._ebr_buffer.extend(proposals)
        for p in proposals:
            self._ebr_arrival_order.append(p.message.id)

        if len(proposals) == 0:
            self._ebr_empty_fetch_count += 1
        else:
            self._ebr_empty_fetch_count = 0

        if len(self._ebr_buffer) >= self._K:
            return self._release_batch(non_proposals, self._K)

        if (
            not self._ebr_nudge_sent
            and self._ebr_empty_fetch_count >= NUDGE_AFTER_EMPTY
            and len(self._ebr_buffer) > 0
        ):
            self._ebr_nudge_sent = True
            self.logger.info(
                f"[EBR] Nudge: {self._ebr_empty_fetch_count} empty fetches, "
                f"buffer has {len(self._ebr_buffer)}/{self._K} proposals. "
                f"Injecting follow-up prompt."
            )
            non_proposals.append(self._make_nudge_message())

        if (
            self._ebr_empty_fetch_count >= MAX_EMPTY_FETCHES
            and len(self._ebr_buffer) > 0
        ):
            self.logger.info(
                f"[EBR] Fallback: {self._ebr_empty_fetch_count} empty fetches, "
                f"buffer has {len(self._ebr_buffer)} proposals (< K={self._K})"
            )
            return self._release_batch(
                non_proposals, len(self._ebr_buffer), is_fallback=True
            )

        return FetchMessagesResponse(
            messages=non_proposals, has_more=False
        )
