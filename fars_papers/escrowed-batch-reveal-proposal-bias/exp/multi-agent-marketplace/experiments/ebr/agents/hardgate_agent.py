"""HardGate customer agent: code-level payment gating baseline (Condition B).

Subclasses SoftWaitCustomerAgent to block payment until the customer has
received at least K proposals. Uses the same SoftWait prompt augmentation
as Condition A to avoid prompt confounds. Proposals are still revealed
sequentially -- only payment timing is enforced.
"""

from magentic_marketplace.marketplace.agents.customer.models import (
    CustomerAction,
    CustomerSendMessageResults,
)

from .softwait_agent import SoftWaitCustomerAgent

K_DEFAULT = 3


class HardGateCustomerAgent(SoftWaitCustomerAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._K = K_DEFAULT

    async def _execute_customer_action(self, action: CustomerAction):
        if (
            action.action_type == "send_messages"
            and action.messages is not None
            and len(action.messages.pay_messages) > 0
        ):
            if self.proposal_storage.count_proposals() < self._K:
                result = CustomerSendMessageResults()
                result.pay_message_results = [
                    (False, "ACTION_UNAVAILABLE")
                    for _ in action.messages.pay_messages
                ]
                self._event_history.append((action, result))
                self.logger.info(
                    f"[HardGate] Payment blocked: {self.proposal_storage.count_proposals()}/{self._K} proposals received"
                )
                return False
        return await super()._execute_customer_action(action)
