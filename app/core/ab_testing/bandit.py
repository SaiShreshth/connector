"""
Simple Bandit controller (epsilon-greedy) for routing A/B test traffic.

Usage patterns:
- From your routing code (e.g., ab_framework.py), instantiate BanditController and call:
    bc = BanditController(state_path="app/core/ab_testing/bandit_state.json")
    chosen = bc.select_variant(variants)   # variants is a list of dicts from your config
- After observing outcome, call:
    bc.register_result(chosen['name'], success=True/False)

This file also provides a small CLI for quick local inspection/simulation.

State format (JSON):
{
  "<experiment_id>": {
      "variants": {
          "<variant_name>": {"trials": int, "successes": int}
      },
      "epsilon": 0.1
  },
  ...
}
"""

from __future__ import annotations
import json
import os
import random
from typing import List, Dict, Optional, Any
import argparse

DEFAULT_STATE_PATH = os.path.join(os.path.dirname(__file__), "bandit_state.json")


class BanditController:
    def __init__(self, experiment_id: str = "default", state_path: str = DEFAULT_STATE_PATH, epsilon: float = 0.1):
        assert 0.0 <= epsilon <= 1.0, "epsilon must be between 0 and 1"
        self.experiment_id = experiment_id
        self.state_path = state_path
        self.epsilon = float(epsilon)
        self._state = self._load_state()

        # ensure experiment entry exists
        if experiment_id not in self._state:
            self._state[experiment_id] = {"variants": {}, "epsilon": self.epsilon}
            self._save_state()

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Corrupt file? back up and start fresh
            backup = self.state_path + ".bak"
            try:
                os.rename(self.state_path, backup)
            except Exception:
                pass
            return {}

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)
        os.replace(tmp, self.state_path)

    def _ensure_variant(self, variant_name: str):
        vmap = self._state[self.experiment_id].setdefault("variants", {})
        if variant_name not in vmap:
            vmap[variant_name] = {"trials": 0, "successes": 0}
            self._save_state()

    def select_variant(self, variants: List[Dict]) -> Dict:
        """
        Choose a variant from the given list (each variant is a dict with at least 'name' key).
        Uses epsilon-greedy: with probability epsilon choose random variant (explore),
        otherwise choose the variant with highest observed success rate (exploit).
        """
        # Ensure state has entries for all variants
        for v in variants:
            self._ensure_variant(v["name"])

        # Exploration
        if random.random() < float(self._state[self.experiment_id].get("epsilon", self.epsilon)):
            return random.choice(variants)

        # Exploitation: compute empirical success rate
        best = None
        best_rate = -1.0
        for v in variants:
            s = self._state[self.experiment_id]["variants"].get(v["name"], {"trials": 0, "successes": 0})
            trials = s.get("trials", 0)
            successes = s.get("successes", 0)
            rate = (successes / trials) if trials > 0 else 0.0
            if rate > best_rate:
                best_rate = rate
                best = v
        # if all rates are 0 (no trials), fallback to random choice
        return best if best is not None else random.choice(variants)

    def register_result(self, variant_name: str, success: bool):
        """Record outcome for a variant (success True/False)."""
        self._ensure_variant(variant_name)
        v = self._state[self.experiment_id]["variants"][variant_name]
        v["trials"] = int(v.get("trials", 0)) + 1
        if success:
            v["successes"] = int(v.get("successes", 0)) + 1
        # Persist
        self._save_state()

    def get_state(self) -> Dict[str, Any]:
        """Return the in-memory state for the experiment."""
        return self._state.get(self.experiment_id, {})

    def reset_experiment(self, confirm: bool = True):
        """Reset counts for the current experiment. Set confirm=False to force reset programmatically."""
        if confirm:
            raise RuntimeError("reset_experiment(confirm=True) requires explicit confirmation=False for programmatic reset.")
        self._state[self.experiment_id] = {"variants": {}, "epsilon": self.epsilon}
        self._save_state()

    # convenience helpers for CLI/debugging
    @staticmethod
    def pretty_print_state(state: Dict[str, Any]):
        print(json.dumps(state, indent=2))


# Small CLI to inspect and simulate
def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true", help="Show bandit state")
    p.add_argument("--experiment", type=str, default="default", help="Experiment id")
    p.add_argument("--epsilon", type=float, default=None, help="Set epsilon for this invocation (saves to state)")
    p.add_argument("--simulate-select", action="store_true", help="Select a variant from a provided comma list")
    p.add_argument("--variants", type=str, help="Comma-separated variant names for simulate-select")
    p.add_argument("--register", type=str, help="Register result in format variant:success (e.g., variant_a:1)")
    args = p.parse_args()

    bc = BanditController(experiment_id=args.experiment, state_path=DEFAULT_STATE_PATH, epsilon=(args.epsilon if args.epsilon is not None else 0.1))

    if args.epsilon is not None:
        bc._state[args.experiment]["epsilon"] = float(args.epsilon)
        bc._save_state()

    if args.show:
        print("Bandit state:")
        BanditController.pretty_print_state(bc.get_state())
        return

    if args.simulate_select:
        if not args.variants:
            print("Provide --variants a,b,c")
            return
        variants = [{"name": n.strip()} for n in args.variants.split(",")]
        chosen = bc.select_variant(variants)
        print("Chosen variant:", chosen["name"])
        return

    if args.register:
        # expected form variant:0 or variant:1
        try:
            name, val = args.register.split(":")
            succ = bool(int(val))
            bc.register_result(name, succ)
            print(f"Registered result for {name} success={succ}")
            BanditController.pretty_print_state(bc.get_state())
        except Exception as e:
            print("Invalid --register format. Use variant:0 or variant:1. Error:", e)


if __name__ == "__main__":
    _cli()
