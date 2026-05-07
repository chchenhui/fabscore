"""SoftWait customer agent: prompt-only debiasing baseline (Condition A).

Subclasses CustomerAgent to prepend a "wait for K proposals" instruction
to the system prompt. No code-level payment enforcement is applied.
"""

import traceback

from magentic_marketplace.marketplace.agents.customer.agent import CustomerAgent
from magentic_marketplace.marketplace.agents.customer.models import CustomerAction

SOFTWAIT_INSTRUCTION = """
IMPORTANT: Before making any payment, you MUST wait until you have received at least 3 order proposals from different businesses. When you have received 3 or more proposals, you MUST explicitly compare ALL proposals by examining each one's price and quality. Select the BEST VALUE option — do NOT automatically choose the first proposal you see in the list. Carefully evaluate every option before deciding.
If businesses have responded with text messages but have not yet sent formal order proposals, send them a follow-up message requesting a formal order_proposal with pricing.
""".strip()


class SoftWaitCustomerAgent(CustomerAgent):

    async def _generate_customer_action(self) -> CustomerAction | None:
        prompts = self._get_prompts_handler()
        system_prompt = prompts.format_system_prompt().strip()
        state_context, step_counter = prompts.format_state_context()
        state_context = state_context.strip()
        step_prompt = prompts.format_step_prompt(step_counter).strip()

        system_prompt = f"{SOFTWAIT_INSTRUCTION}\n\n{system_prompt}"

        full_prompt = f"{system_prompt}\n\n\n\n{state_context}\n\n{step_prompt}"

        try:
            action, _ = await self.generate_struct(
                prompt=full_prompt,
                response_format=CustomerAction,
            )
            self.logger.info(
                f"[Step {self.conversation_step}/{self._max_steps or 'inf'}] Action: {action.action_type}. Reason: {action.reason}"
            )
            return action
        except Exception:
            self.logger.exception(
                f"[Step {self.conversation_step}/{self._max_steps or 'inf'}] LLM decision failed"
            )
            self._event_history.append(f"LLM decision failed: {traceback.format_exc()}")
            return None
