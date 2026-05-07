"""Inference-Time Scaling (ITS) customer agent: best-of-N at payment time (Condition A').

Subclasses SoftWaitCustomerAgent. When the agent decides to pay, samples
the LLM N=5 times and selects the sample that pays the lowest-price proposal.
Non-payment actions use standard single-sample behavior.
"""

import asyncio
import traceback

from magentic_marketplace.marketplace.agents.customer.models import CustomerAction

from .softwait_agent import SoftWaitCustomerAgent

N_SAMPLES = 5


class ITSCustomerAgent(SoftWaitCustomerAgent):

    async def _generate_customer_action(self) -> CustomerAction | None:
        prompts = self._get_prompts_handler()
        system_prompt = prompts.format_system_prompt().strip()
        state_context, step_counter = prompts.format_state_context()
        state_context = state_context.strip()
        step_prompt = prompts.format_step_prompt(step_counter).strip()

        from .softwait_agent import SOFTWAIT_INSTRUCTION
        system_prompt = f"{SOFTWAIT_INSTRUCTION}\n\n{system_prompt}"
        full_prompt = f"{system_prompt}\n\n\n\n{state_context}\n\n{step_prompt}"

        try:
            action, _ = await self.generate_struct(
                prompt=full_prompt,
                response_format=CustomerAction,
            )
        except Exception:
            self.logger.exception(
                f"[Step {self.conversation_step}/{self._max_steps or 'inf'}] LLM decision failed"
            )
            self._event_history.append(f"LLM decision failed: {traceback.format_exc()}")
            return None

        has_payment = (
            action.action_type == "send_messages"
            and action.messages is not None
            and len(action.messages.pay_messages) > 0
        )

        if not has_payment:
            self.logger.info(
                f"[Step {self.conversation_step}/{self._max_steps or 'inf'}] Action: {action.action_type}. Reason: {action.reason}"
            )
            return action

        self.logger.info(
            f"[Step {self.conversation_step}/{self._max_steps or 'inf'}] Payment detected -- sampling N={N_SAMPLES} for best-of-N selection"
        )

        async def _sample_once():
            try:
                a, _ = await self.generate_struct(
                    prompt=full_prompt,
                    response_format=CustomerAction,
                )
                return a
            except Exception:
                return None

        samples = await asyncio.gather(*[_sample_once() for _ in range(N_SAMPLES)])

        best_action = action
        best_price = float("inf")

        for s in samples:
            if s is None:
                continue
            if s.action_type != "send_messages" or s.messages is None:
                continue
            if len(s.messages.pay_messages) == 0:
                continue

            proposal_id = s.messages.pay_messages[0].proposal_message_id
            stored = self.proposal_storage.get_proposal(proposal_id)
            if stored is None:
                continue

            price = stored.proposal.total_price
            if price < best_price:
                best_price = price
                best_action = s

        if best_price < float("inf"):
            pid = best_action.messages.pay_messages[0].proposal_message_id
            self.logger.info(
                f"[ITS] Selected lowest-price proposal {pid} at ${best_price:.2f} from {N_SAMPLES} samples"
            )
        else:
            self.logger.info(
                f"[ITS] No valid payment in {N_SAMPLES} samples, using initial action"
            )

        self.logger.info(
            f"[Step {self.conversation_step}/{self._max_steps or 'inf'}] Action: {best_action.action_type}. Reason: {best_action.reason}"
        )
        return best_action
