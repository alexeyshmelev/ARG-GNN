"""Isolated exact execution selection; no patches to live modules or models.

The level-local arm composes the previously qualified categorical site encoder
and layout cache with functional level-local tree sweeps. History, decoder,
checkpoint recomputation, clade/episode coupling, and parameter keys are original.
"""
from types import SimpleNamespace
import time
import torch
from research.scalability_20261024.integration.executors import (
    IntegratedStagedExecution, build_executor as previous_executor)
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import (
    profiled, need)
from .level_sweep import LevelSweepPlan, sweep_tree_state


class LevelLocalStagedExecution(IntegratedStagedExecution):
    def __init__(self, *args, max_sweep_plan_bytes, sweep_plan_admit=None, **kwargs):
        need(type(max_sweep_plan_bytes) is int and max_sweep_plan_bytes > 0,
             "Explicit additional CPU sweep-plan reservation required")
        need(sweep_plan_admit is None or callable(sweep_plan_admit),
             "Optional caller admission callback must be callable")
        self.sweep_plans = {}
        self.sweep_plan_bytes = 4096
        self.sweep_plan_build_seconds = 0.
        self.sweep_plan_cap = max_sweep_plan_bytes
        super().__init__(*args, **kwargs)
        try:
            started = time.perf_counter()
            with self.profile.stage("level_sweep_plan_constructor"):
                for z in self.layout:
                    self.tick("level_sweep_plan_admit")
                    # Includes eventual tensors and conservative Python/control
                    # allowance. Peak sorting/source-copy scratch is separately
                    # reserved by the caller before allocation. No GPU tensors.
                    n = z[3]-z[2]
                    levels = len(z[5]) + len(z[8])
                    estimated = 96*n + 4096*max(1, levels*levels)
                    need(self.sweep_plan_bytes + estimated <= max_sweep_plan_bytes,
                         "Declared additional sweep-plan host cap exceeded")
                    if sweep_plan_admit is not None:
                        sweep_plan_admit(estimated + 128*n, "level_sweep_plan_build")
                    def tensor(x):
                        return torch.tensor(x, dtype=torch.long, device="cpu")
                    bank = SimpleNamespace(parent=tensor(z[4]),
                        upward_levels=tuple(tensor(x) for x in z[5]),
                        upward_children=tuple(tensor(x) for x in z[6]),
                        upward_parent_rank=tuple(tensor(x) for x in z[7]),
                        downward_levels=tuple(tensor(x) for x in z[8]))
                    plan = LevelSweepPlan.build(bank)
                    charge = plan.tensor_bytes() + 4096*max(1, levels*levels)
                    need(charge <= estimated, "Measured sweep-plan charge exceeds admission")
                    self.sweep_plans[z[0]] = plan
                    self.sweep_plan_bytes += charge
                    del bank
            self.sweep_plan_build_seconds = time.perf_counter()-started
        except BaseException:
            self.release_layout()
            raise

    def _sweep(self, z, bank, base, ug, dg, degree):
        # The numerical graph can retain CUDA route tensors only within the
        # original block checkpoint lifetime; there is no whole-parent GPU cache.
        plan = self.sweep_plans[z[0]].to(self.device)
        return sweep_tree_state(self.model, bank, base, ug, dg, degree, plan=plan)

    @profiled
    def _first(self, z):
        self.tick("bounded_first_sweep")
        bank = self._block(z)
        base, ug, dg, degree = self._base(bank, self._sites(z[0], z[1]))
        up, out = self._sweep(z, bank, base, ug, dg, degree)
        return up.to("cpu"), out.to("cpu")

    @profiled
    def _second(self, z, up, value, eligible):
        self.tick("bounded_second_sweep")
        a,b,lo,hi,_,_,_,_,_,ml,mr = z
        bank = self._block(z)
        base,ug,dg,degree = self._base(bank, self._sites(a,b))
        up = up.to(self.device)
        m = self.model
        channels = self.bundle.episodes.flags.node_flags[lo:hi].to(self.device,dtype=up.dtype)
        common = m.clade_coupling.U(up)
        flag = m.clade_coupling.O(channels[:,0:1]*common*m.clade_coupling.V(up)
            -channels[:,1:2]*common*m.clade_coupling.W(up))
        member_node = (self.bundle.episodes.member_node[ml:mr].long()-lo).to(self.device)
        member = up.index_select(0,member_node)
        ep = m.episode_coupling
        residual = ep.O(ep.U(member)*ep.V(member)+ep.W(member))
        residual = residual*eligible[ml:mr,None].to(self.device,dtype=torch.float32)
        episode = up.new_zeros(up.shape).index_add(0,member_node,residual)
        injection = flag+episode
        cut = self.bundle.cut
        first,last = cut.group_offsets[a],cut.group_offsets[b]
        representatives = (cut.representatives[first:last]-lo).to(self.device)
        delta = m.memory.O(value.to(self.device))
        injection = injection.index_add(0,representatives,delta)
        second_up,second_out = self._sweep(z,bank,base+injection,ug,dg,degree)
        return m.leaf_context_v1(bank,second_up,second_out).to("cpu")

    def sweep_plan_snapshot(self):
        return dict(additional_CPU_cap_bytes=self.sweep_plan_cap,
            charged_CPU_bytes=self.sweep_plan_bytes, plan_count=len(self.sweep_plans),
            build_seconds=self.sweep_plan_build_seconds,
            persistent_GPU_plan_cache=False, numerical_states_cached=False,
            plans_rebuilt_per_executor=True)

    def release_layout(self):
        self.sweep_plans = {}
        self.sweep_plan_bytes = 4096
        super().release_layout()


def build_executor(kind, model, bundle, metadata, *, execution="staged",
                   layout_cache=None, frozen_module=None, **kwargs):
    """Compatible explicit factory for original, integrated and level_local arms.

    level_local requires max_sweep_plan_bytes. This is extra CPU memory beyond
    the original observation and layout caches. The executor owns plans until
    release_layout after all backward/optimizer work. The first qualification
    intentionally measures per-executor plan construction, without an unbounded
    whole-cohort cache or persistent learned state.
    """
    if kind in ("original", "integrated"):
        kwargs.pop("max_sweep_plan_bytes", None)
        kwargs.pop("sweep_plan_admit", None)
        return previous_executor(kind, model, bundle, metadata,
            execution=execution, layout_cache=layout_cache,
            frozen_module=frozen_module, **kwargs)
    need(kind == "level_local" and execution == "staged",
         "Only the explicitly selected staged level-local arm is implemented")
    return LevelLocalStagedExecution(model,bundle,metadata,layout_cache=layout_cache,
        frozen_module=frozen_module,**kwargs)
