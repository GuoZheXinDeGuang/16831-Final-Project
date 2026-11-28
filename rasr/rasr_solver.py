"""RASR hierarchical solver: reachability-aware subgoal selection (P2) + failure monitoring and
local subgoal repair with full-replanning fallback (P3), on top of the frozen upstream H-JEPA planner.

`RASRHierarchicalSolver.solve` is a line-for-line copy of upstream `HierarchicalSolver.solve` with a
single hook after the TOP level's solve (the level that emits the subgoal for level 1). With
`mode='baseline'` the hook is a no-op and the planner is bit-identical to upstream
(tests/test_p2p3_solver.py).

Modes
  baseline     upstream behaviour
  rerank       (B3)  every call: choose candidate by progress cost - lambda * log(R + delta)
  full_replan  (B4)  same trigger as ours; on trigger re-solve the top level (fresh GD restart),
                     pick min cost over old + new pool; no reachability
  rasr         (Ours) rerank every call + on trigger local repair from the candidate pool
                     (reachable first subgoal, downstream waypoints preserved), fallback = full replan
  repair_noR   (ablation) ours without reachability: rerank off, repair scores cost + downstream only,
                     fallback after `max_consecutive_repairs`
Only planner-visible quantities are used: latents of the current frame / goal frame, the model's
predicted plans, the reachability predictor, and the subgoal-miss test on latents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from hierarchical_solver import HierarchicalSolver
from stable_worldmodel.solver.solver import planning_call_index


@dataclass
class RASRConfig:
    mode: str = "baseline"
    lam: float = 0.5                # rerank: relative reachability weight (see _scores)
    delta: float = 0.05
    reach_H: int = 10               # controller condition used to query R (= baseline execution window)
    reach_n_steps: int = 90
    reach_num_samples: int = 100
    eps: float = 6.6966             # subgoal-reached threshold (P1, train-calibrated)
    miss_m: int = 3                 # trigger: m consecutive missed subgoals
    low_reach_tau: float = 0.0      # trigger: R(chosen) < tau (0 disables)
    cooldown: int = 1               # calls after a repair with no new trigger
    stall_w: int = 5                # trigger: real latent goal distance not improved by stall_frac in w calls
    stall_frac: float = 0.05
    repair_lam: float = 1.0         # weight of the differentiable reachability term in repair GD
    novelty_beta: float = 1.0       # weight of the visited-region penalty in repair GD
    novelty_sigma2: float = 25.0    # kernel width (squared level-2 latent distance)
    novelty_steps: int = 2          # plan waypoints penalised
    repair_hold: int = 3            # calls the repair objective stays active after a trigger
    repair_min_reach: float = 0.3   # repaired subgoal must have R >= this (and novelty <= max), else fallback
    repair_max_novelty: float = 1.0 # repaired subgoal must leave the visited region (1.0 disables)
    fallback: str = "full_replan"
    passthrough: bool = False       # log pool statistics but return the upstream choice unchanged
    dump_alt_k: int = 0             # log k non-selected candidate subgoals (R-stratified) per call   # full_replan | goal_direct (bypass the top level; level 1 -> real goal)
    # v2 (post-hoc, AntMaze): how R enters selection / repair scores
    reach_form: str = "log"         # log: -lam c_ref log(R+delta) | hinge: lam c_ref relu(log tau - log(R+delta))
                                    # expected: lam (1-R) relu(C_cur - C)  (lam=1: expected final cost R C + (1-R) C_cur)
    reach_tau: float = 0.5          # hinge threshold
    miss_memory: bool = False       # R_eff(g) = R(g) * prod_j (1 - exp(-||g - m_j||^2 / sigma2)), m_j = missed subgoals
    # v3 (post-hoc ablations / ideas for the repair itself)
    always_repair: bool = False     # apply the repair objective at every call (no stall trigger)
    adaptive_window: int = 0        # >0: novelty beta x (number of triggers in the last `adaptive_window` calls)


class RASRHierarchicalSolver(HierarchicalSolver):
    def __init__(self, level_solvers, level_models, cfg: RASRConfig, predictor=None):
        super().__init__(level_solvers=level_solvers, level_models=level_models)
        self.rcfg = cfg
        self.predictor = predictor
        self.call_log: list[dict] = []
        self._goal_z = None
        self._reset_episode()

    # ---------------------------------------------------------------- episode state
    def _reset_episode(self):
        self.prev_subgoal = None
        self.prev_plan = None
        self.prev_actions = None
        self.miss_run = 0
        self.cooldown_left = 0
        self.hold_left = 0
        self._goal_direct_left = 0
        self._trigger = None
        self._goal_direct_now = False
        self.visited = []
        self.goal_dists = []
        self.missed = []
        self.trigger_calls = []
        self.n_calls = 0

    def pop_episode_log(self):
        log, self.call_log = self.call_log, []
        return log

    # ---------------------------------------------------------------- upstream copy + hook
    def solve(self, info_dict: dict, steps_taken: int, eval_budget: int) -> dict:
        if steps_taken == 0:
            self._reset_episode()
        subgoal_latents = None
        subgoal_actions = None
        skipped_goal_level = None
        upper_anchored = True
        horizon_schedules = self._horizon_schedules(eval_budget=eval_budget)
        call_idx = planning_call_index(replanning_interval=self._replanning_interval(), steps_taken=steps_taken)
        obs_embeddings = self._encode_all_levels(info_dict=info_dict, key='pixels')
        goal_embeddings = self._encode_all_levels(info_dict=info_dict, key='goal')
        top_level = len(self.level_models)
        self._log = {"call_idx": int(call_idx), "env_step": int(steps_taken), "mode": self.rcfg.mode}
        self._goal_z = goal_embeddings['embed_2'][0, 0].float()
        self._pre_call_monitor(obs_embeddings)
        self._update_trigger()
        bypass_top = self.rcfg.mode != "baseline" and self._goal_direct_now

        for level in range(len(self.level_models), 0, -1):
            level_cfg = self._level_plan_config(level)
            planning_horizon = horizon_schedules[level][call_idx]

            if level > 1 and (planning_horizon == 1 or (bypass_top and level == top_level)):
                skipped_goal_level = (
                    (skipped_goal_level or level)
                    if level_cfg.horizon_one_goal_cost_space == 'upper' else None
                )
                subgoal_latents = None
                subgoal_actions = None
                continue

            level_info = dict(info_dict)
            level_info['embed_0'] = obs_embeddings[f'embed_{level}']
            level_info['action_cost_weight'] = level_cfg.action_cost_weight

            if subgoal_latents is None:
                goal_level = skipped_goal_level or level
                level_info['goal_embed_0'] = goal_embeddings[f'embed_{goal_level}']
                level_info['goal_embed_0_level'] = goal_level
                level_info['goal_embed_0_in_upper_space'] = goal_level > level
                level_info['dense_upper_goal_projection'] = goal_level > level
                this_level_anchored = True
            else:
                num_subgoals = self._level_plan_config(level + 1).num_subgoals
                first_pred_idx = 1 if subgoal_latents.shape[1] > 1 else 0
                first_action_idx = max(0, first_pred_idx - 1)
                includes_last = (first_pred_idx + num_subgoals >= subgoal_latents.shape[1])
                next_goal = subgoal_latents[:, first_pred_idx:first_pred_idx + num_subgoals].clone()
                next_action = subgoal_actions[:, first_action_idx:first_action_idx + num_subgoals].clone()
                if next_action.shape[1] != next_goal.shape[1]:
                    raise ValueError(
                        f"Level {level}: got {next_action.shape[1]} action "
                        f"subgoals for {next_goal.shape[1]} state subgoals."
                    )
                this_level_anchored = False
                if includes_last and upper_anchored:
                    next_goal[:, -1] = goal_embeddings[f'embed_{level + 1}'].squeeze(1)
                    this_level_anchored = True
                level_info['goal_embed_0'] = next_goal
                level_info['goal_action'] = next_action
                level_info['goal_embed_0_in_upper_space'] = True

            goal_emb = level_info.get('goal_embed_0')
            if (
                level_info.get('goal_embed_0_in_upper_space', False)
                and torch.is_tensor(goal_emb)
                and goal_emb.shape[-2] > 1
            ):
                _stride = self.level_solvers[level].model.upper_models[0].temporal_stride
                _expected = (planning_horizon + _stride) // _stride - 1
                assert goal_emb.shape[-2] == _expected

            result = self.level_solvers[level].solve(
                info_dict=level_info,
                planning_horizon=planning_horizon,
                steps_taken=steps_taken,
                eval_budget=eval_budget,
            )
            upper_anchored = this_level_anchored
            skipped_goal_level = None
            subgoal_latents = result['predictions']['predicted_embed_0']
            subgoal_actions = result['actions']

            # ------------------------------------------------ RASR hook (top level only)
            if level == top_level and level > 1 and self.rcfg.mode != "baseline":
                subgoal_latents, subgoal_actions = self._rasr_select(
                    level, level_info, planning_horizon, steps_taken, eval_budget,
                    obs_embeddings, subgoal_latents, subgoal_actions)

        # remember what level 1 was asked to reach (for the next call's miss test)
        self._post_call_bookkeeping(level_info)
        self.call_log.append(self._log)
        return {'actions': result['actions']}

    # ---------------------------------------------------------------- monitoring
    def _update_trigger(self):
        """Debounced failure detection from planner-visible history (before planning this call)."""
        rc = self.rcfg
        self._trigger = None
        self._goal_direct_now = False
        if self._goal_direct_left > 0:          # a goal-direct fallback is in progress
            self._goal_direct_now = True
            self._goal_direct_left -= 1
            self._consume_hold()
            self._log.update({"trigger": None, "repair": "goal_direct", "fallback": True, "hold_left": int(self.hold_left)})
            return
        self.n_calls += 1
        if rc.always_repair and rc.mode == "repair_noR":
            self.cooldown_left = 0
            if self.hold_left == 0:
                self._trigger = "always"
        elif rc.mode in ("rasr", "full_replan", "repair_noR") and self.cooldown_left == 0 and self.hold_left == 0:
            gd = self.goal_dists
            if self.miss_run >= rc.miss_m:
                self._trigger = "subgoal_miss"
            elif rc.stall_w > 0 and len(gd) > rc.stall_w and min(gd[-rc.stall_w:]) > (1 - rc.stall_frac) * min(gd[:-rc.stall_w]):
                self._trigger = "goal_stall"
        if self._trigger is not None:
            self.trigger_calls.append(self.n_calls)
            self.miss_run = 0
            self.goal_dists = self.goal_dists[-1:]
            self.hold_left = 1 if (rc.mode == "full_replan" or rc.always_repair) else rc.repair_hold
        self._log.update({"trigger": self._trigger, "hold_left": int(self.hold_left)})

    def _consume_hold(self):
        if self.hold_left > 0:
            self.hold_left -= 1
            if self.hold_left == 0:
                self.cooldown_left = self.rcfg.cooldown + 1

    def _pre_call_monitor(self, obs_embeddings):
        """Subgoal-miss test for the previous call's subgoal, using only latents."""
        z = obs_embeddings['embed_2'][0, 0].float()
        self.visited.append(z.detach())
        if self._goal_z is not None:
            self.goal_dists.append(float(((z - self._goal_z) ** 2).sum()))
        if self.prev_subgoal is not None:
            d = float(((z - self.prev_subgoal.to(z.device)) ** 2).sum())
            missed = d > self.rcfg.eps
            if missed:
                self.missed.append(self.prev_subgoal.detach().float())
            self.miss_run = self.miss_run + 1 if missed else 0
            self._log.update({"prev_subgoal_dist": d, "prev_subgoal_missed": bool(missed)})
        self._log["miss_run"] = int(self.miss_run)

    def _post_call_bookkeeping(self, level1_info):
        g = level1_info.get('goal_embed_0')
        if torch.is_tensor(g) and level1_info.get('goal_embed_0_in_upper_space', False):
            self.prev_subgoal = g.reshape(-1, g.shape[-1])[-1].detach().float().to(g.device)
        if self.cooldown_left > 0:
            self.cooldown_left -= 1

    # ---------------------------------------------------------------- candidate pool
    @torch.no_grad()
    def _candidate_pool(self, level, level_info, lsolver):
        """Roll out every optimised candidate of the top-level GD once; return predicted latents
        (S, T+1, D), actions (S, T, A) and progress costs (S,)."""
        acts = lsolver.init.detach()  # (1, S, T, A) after optimisation
        S = acts.shape[1]
        info = lsolver._expand_info_dict_for_samples(level_info, 0, 1, S)
        info['cost_last_n'] = max(1, int(lsolver._config.cost_last_n))
        info['intermediate_cost_weight'] = float(lsolver._config.intermediate_cost_weight)
        cost = lsolver.model.get_cost(info, acts)[0]  # (S,); get_cost rolls out into `info`
        preds = info['predicted_embed_0'][0]  # (S, T+1, D)
        return preds.float(), acts[0].float(), cost.float()

    @torch.no_grad()
    def _reach(self, z, g):
        """R(z, g | H, B) for a batch of candidate subgoals g (K, D)."""
        p = self.predictor
        K = g.shape[0]
        c = torch.tensor([np.log(self.rcfg.reach_H), np.log(self.rcfg.reach_n_steps),
                          np.log(self.rcfg.reach_num_samples)], device=g.device, dtype=torch.float32).expand(K, 3)
        return p(z.expand(K, -1).to(g.device), g, c) * self._miss_factor(g)

    def _miss_factor(self, g):
        """Failure memory (planner-visible): subgoals near ones the controller already missed this episode
        are less reachable than the static predictor says."""
        if not self.rcfg.miss_memory or not self.missed:
            return torch.ones(g.shape[0], device=g.device)
        M = torch.stack(self.missed).to(g.device)
        d = ((g.float()[:, None] - M[None]) ** 2).sum(-1)
        return torch.prod(1 - torch.exp(-d / self.rcfg.novelty_sigma2), dim=-1).clamp_min(0.0)

    def _reach_pen(self, R, C, c_ref, lam):
        """Additive reachability term of the selection / repair score (lower is better)."""
        rc = self.rcfg
        if rc.reach_form == "log":
            return -lam * c_ref * torch.log(R + rc.delta)
        if rc.reach_form == "hinge":
            return lam * c_ref * torch.relu(np.log(rc.reach_tau) - torch.log(R + rc.delta))
        if rc.reach_form == "expected":
            c_cur = self.goal_dists[-1] if self.goal_dists else float(C.max())
            return lam * (1 - R) * torch.relu(c_cur - C)
        raise ValueError(rc.reach_form)

    def _scores(self, cost, R, lam):
        """Reachability-aware selection score. lam is a *relative* trade-off: the penalty is
        scaled by the call's best progress cost, so lam=0.5 accepts a candidate whose cost is
        up to 0.5 * c_min * delta(-log R) worse."""
        c_ref = cost.min().clamp_min(1e-6)
        return cost + self._reach_pen(R, cost, c_ref, lam)

    def _reach_c(self, K, device):
        return torch.tensor([np.log(self.rcfg.reach_H), np.log(self.rcfg.reach_n_steps),
                             np.log(self.rcfg.reach_num_samples)], device=device, dtype=torch.float32).expand(K, 3)

    # ---------------------------------------------------------------- selection / repair
    def _rasr_select(self, level, level_info, planning_horizon, steps_taken, eval_budget,
                     obs_embeddings, subgoal_latents, subgoal_actions):
        rc = self.rcfg
        lsolver = self.level_solvers[level]
        P, A, C = self._candidate_pool(level, level_info, lsolver)
        base_idx = int(torch.argmin(C))
        z = obs_embeddings['embed_2'][0, 0].float()
        use_R = self.predictor is not None and rc.mode in ("rerank", "rasr")
        R = self._reach(z, P[:, 1]) if self.predictor is not None else None
        log = self._log
        log.update({"pool_size": int(P.shape[0]), "base_idx": base_idx, "base_cost": float(C[base_idx]),
                    "extra_toplevel_rollouts": 1, "extra_toplevel_solves": 0})
        if R is not None:
            log["R_base"] = float(R[base_idx])
            log["R_max_pool"] = float(R.max())

        if rc.dump_alt_k > 0 and R is not None:
            others = torch.tensor([i for i in torch.argsort(R).tolist() if i != base_idx], device=R.device)
            pick = others[torch.linspace(0, len(others) - 1, rc.dump_alt_k).round().long()]
            rank = torch.argsort(torch.argsort(C))
            log["alt_idx"] = pick.tolist()
            log["alt_R"] = R[pick].tolist()
            log["alt_cost_rank"] = rank[pick].tolist()
            log["alt_subgoals"] = P[pick, 1].cpu().tolist()
        if rc.passthrough:
            return subgoal_latents, subgoal_actions

        # P2: reachability-aware reranking of the unchanged candidate pool
        sel = base_idx
        if rc.mode in ("rerank", "rasr"):
            sel = int(torch.argmin(self._scores(C, R, rc.lam)))
        log["rerank_changed"] = sel != base_idx
        if R is not None:
            log["R_sel"] = float(R[sel])

        trigger = self._trigger
        log.update({"repair": None, "fallback": False})
        R2 = None
        if self.hold_left > 0 and not self._goal_direct_now:
            c_ref = float(C[base_idx])
            if rc.mode == "full_replan":
                P, A, C, sel, R2 = self._resolve(level, level_info, planning_horizon, steps_taken, eval_budget,
                                                 z, init=None, lam=0.0, beta=0.0, c_ref=c_ref)
                log.update({"repair": "full_replan", "fallback": True})
                self._consume_hold()
            else:
                lam_r = rc.repair_lam if rc.mode == "rasr" else 0.0
                P, A, C, sel, R2 = self._resolve(level, level_info, planning_horizon, steps_taken, eval_budget,
                                                 z, init=None, lam=lam_r, beta=self._beta_eff(), c_ref=c_ref)
                log.update({"repair": "subgoal_repair", "repair_R": None if R2 is None else float(R2[sel]),
                            "repair_novelty": float(self._novelty(P[sel][None])[0])})
                bad = (rc.mode == "rasr" and R2 is not None and float(R2[sel]) < rc.repair_min_reach) or \
                      float(self._novelty(P[sel][None])[0]) > rc.repair_max_novelty
                if bad:
                    if rc.fallback == "goal_direct":
                        # next calls of this hold bypass the top level (level 1 plans to the real goal)
                        self._goal_direct_left = self.hold_left - 1
                        log.update({"repair": "subgoal_repair_rejected->goal_direct", "fallback": True})
                    else:
                        P, A, C, sel, R2 = self._resolve(level, level_info, planning_horizon, steps_taken, eval_budget,
                                                         z, init=None, lam=0.0, beta=0.0, c_ref=c_ref)
                        log.update({"repair": "subgoal_repair+fallback_full_replan", "fallback": True})
            self._consume_hold()
            if R2 is not None:
                log["R_sel"] = float(R2[sel])

        log["final_cost"] = float(C[sel])
        self.prev_plan = P[sel].detach()
        self.prev_actions = A[sel].detach()
        return P[sel][None].cpu(), A[sel][None].cpu()

    def _beta_eff(self):
        rc = self.rcfg
        if rc.adaptive_window <= 0:
            return rc.novelty_beta
        n = sum(1 for c in self.trigger_calls if c > self.n_calls - rc.adaptive_window)
        self._log["beta_eff"] = rc.novelty_beta * max(1, n)
        return rc.novelty_beta * max(1, n)

    def _novelty(self, preds):
        """Visited-region penalty of plan waypoints 1..novelty_steps: max_j exp(-||w - z_j||^2 / sigma2)
        over level-2 latents of frames observed so far in this episode (planner-visible history)."""
        if not self.visited:
            return torch.zeros(preds.shape[0], device=preds.device)
        V = torch.stack(self.visited).to(preds.device)  # (Nv, D)
        w = preds[:, 1:1 + self.rcfg.novelty_steps].float()  # (S, k, D)
        d = ((w[:, :, None] - V[None, None]) ** 2).sum(-1)  # (S, k, Nv)
        return torch.exp(-d / self.rcfg.novelty_sigma2).max(-1).values.mean(-1)

    def _resolve(self, level, level_info, planning_horizon, steps_taken, eval_budget, z, *, init, lam, c_ref, beta=0.0):
        """Re-run the upstream top-level GradientSolver (optionally warm-started), with the cost
        augmented by -lam * c_ref * log(R(z, first predicted latent) + delta) (differentiable MLP)."""
        rc = self.rcfg
        lsolver = self.level_solvers[level]
        model = lsolver.model
        orig_get_cost = model.get_cost
        pred = self.predictor
        use_r = lam > 0 and pred is not None
        if use_r or beta > 0:
            def get_cost(info, acts):
                cost = orig_get_cost(info, acts)  # (1, S); rollout written into info
                preds = info["predicted_embed_0"][0]
                if use_r:
                    g = preds[:, 1].float()
                    S = g.shape[0]
                    r = pred(z.expand(S, -1), g, self._reach_c(S, g.device)) * self._miss_factor(g)
                    cost = cost + self._reach_pen(r, cost[0], c_ref, lam)[None]
                if beta > 0:
                    cost = cost + beta * c_ref * self._novelty(preds)[None]
                return cost
            model.get_cost = get_cost
        try:
            lsolver.solve(info_dict=level_info, init_action=None if init is None else init.cpu(),
                          planning_horizon=planning_horizon, steps_taken=steps_taken, eval_budget=eval_budget)
        finally:
            model.__dict__.pop("get_cost", None)
        self._log["extra_toplevel_solves"] = self._log.get("extra_toplevel_solves", 0) + 1
        P, A, C = self._candidate_pool(level, level_info, lsolver)  # un-augmented progress cost
        R = self._reach(z, P[:, 1]) if pred is not None else None
        score = C.clone()
        if lam > 0 and R is not None:
            score = score + self._reach_pen(R, C, c_ref, lam)
        if beta > 0:
            score = score + beta * c_ref * self._novelty(P)
        sel = int(torch.argmin(score))
        return P, A, C, sel, R

    def config_dict(self):
        return asdict(self.rcfg)


def load_predictor(path, device="cuda"):
    from rasr.reachability import DistLogRegReachability, ReachabilityMLP

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("type") == "dist_logreg":
        return DistLogRegReachability(ck["coef"], ck["intercept"]).to(device).eval()
    m = ReachabilityMLP(ck["d_latent"], hidden=ck["hidden"], dropout=ck["dropout"])
    m.load_state_dict(ck["state_dict"])
    for q in m.parameters():
        q.requires_grad_(False)
    return m.to(device).eval()
