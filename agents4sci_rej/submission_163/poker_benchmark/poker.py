import random
from collections import namedtuple, deque

State = namedtuple('State', 'street hero_cards board pot stacks to_act legal_actions')

class MultiwayHoldemEnv:
    def __init__(self, n_players=4, starting_stack=100, blinds=(1,2)):
        assert n_players >= 3, "Use HUHoldemEnv for 2 players"
        self.n_players = n_players
        self.starting_stack = starting_stack
        self.small_blind, self.big_blind = blinds
        self.deck_template = [r+s for r in '23456789TJQKA' for s in 'shdc']
        self.reset()

    def reset(self):
        # fresh deck
        self.deck = self.deck_template[:]
        random.shuffle(self.deck)

        # deal hole cards
        self.hole_cards = [[self.deck.pop(), self.deck.pop()]
                           for _ in range(self.n_players)]

        # initialize stacks and pot
        self.stacks = [self.starting_stack] * self.n_players
        self.pot = 0

        # post blinds (seat 0 SB, seat 1 BB)
        self.stacks[0] -= self.small_blind
        self.stacks[1] -= self.big_blind
        self.pot += self.small_blind + self.big_blind

        # active players queue
        self.active = deque(range(self.n_players))
        self.to_act = self.active[0]
        self.street = 'preflop'
        self.board = []
        return self._get_state()

    def step(self, action):
        """
        action ∈ {'fold','call','raise'}
        """
        reward = [0]*self.n_players
        done = False

        if action == 'fold':
            self.active.remove(self.to_act)
            if len(self.active) == 1:
                winner = self.active[0]
                reward[winner] = self.pot
                done = True

        elif action == 'call':
            if self.street == 'river' and self._all_called():
                winner = self._showdown()
                reward[winner] = self.pot
                done = True
            else:
                self._advance_street()

        elif action == 'raise':
            bet_size = 10  # fixed for demo
            if self.stacks[self.to_act] >= bet_size:
                self.stacks[self.to_act] -= bet_size
                self.pot += bet_size
            # all others must respond, so order naturally continues

        if not done:
            self._next_player()
        return self._get_state(), reward, done

    # ---- helpers ----
    def _advance_street(self):
        if self.street == 'preflop':
            self.board.extend([self.deck.pop() for _ in range(3)])
            self.street = 'flop'
        elif self.street == 'flop':
            self.board.append(self.deck.pop()); self.street = 'turn'
        elif self.street == 'turn':
            self.board.append(self.deck.pop()); self.street = 'river'

    def _next_player(self):
        self.active.rotate(-1)
        self.to_act = self.active[0]

    def _all_called(self):
        # Simplified: everyone checked/called last bet
        return True

    def _showdown(self):
        # TODO: implement real hand evaluation across all remaining players
        return random.choice(list(self.active))

    def _get_state(self):
        legal = ['fold','call','raise']
        return State(self.street,
                     self.hole_cards[0],  # hero’s perspective
                     self.board, self.pot,
                     tuple(self.stacks),
                     self.to_act,
                     legal)