"""Explore-and-fix workflow: drain the fix queue, explore, repeat.

``--limit`` is a *cap*, not a guarantee: the loop stops early as soon as any
of these happen, in order:

* the attempt budget is exhausted (``--limit`` issues attempted);
* the creation budget is exhausted (``--limit`` issues created); with the
  default ``limit=0`` (unlimited) this check is skipped, so the run keeps
  draining until explore creates zero issues (or the ``max_cycles`` guard
  fires);
* explore creates zero issues (no progress);
* ``max_cycles`` cycles elapse without converging.

So a run may end having fixed far fewer than ``--limit`` issues, or having
explored only once.  The loop is a bounded search, not a guaranteed
drain-to-zero pass.

The loop, implemented literally:

```
attempted = set()
cycles = 0
loop:
  drain queue (each issue attempted at most once per run)
  if --limit reached: stop
  created = explore(remaining budget)
  if created == 0: stop
  cycles += 1
  if cycles >= max_cycles: warn and stop
```

The budget applies independently to the issues *attempted* and the issues
*created*: each drain gets the remaining attempt budget and each explore gets
the remaining creation budget.  Exploration stops once it has created its
budget; the loop then drains those freshly created issues (their attempts use
the independent attempt budget) before the next top-of-loop budget check stops
the run.  Failed issues stay open but are remembered in ``attempted``, so a
second drain can never retry them in the same run; only a successful fix (or a
fresh issue discovered by the next explore) makes the loop go around again.
``max_cycles`` is the no-progress guard against a model that keeps re-reporting
issues after every fix.

The workflow does not decide exit codes: it returns the counters, and the CLI
maps "any issue remains open" to exit code 1.
"""

from __future__ import annotations

from khorek.context import AppContext
from khorek.report import CombinedStats
from khorek.workflows.explore import run_explore
from khorek.workflows.fix import run_fix


def run_explore_and_fix(app: AppContext, limit: int = 0, *, pruned: int = 0) -> CombinedStats:
    """Run the combined workflow and return its unified counters.

    ``pruned`` is the drift-prune count from the run preamble (reported in the
    unified summary); the workflow itself never prunes.
    """
    attempted_ids: set[int] = set()
    attempted = 0
    created = 0
    cycles = 0
    max_cycles = app.config.defaults.max_cycles
    stats = CombinedStats(pruned=pruned)

    while True:
        # -- drain the fix queue (each issue at most once per run) ------
        if limit > 0 and attempted >= limit:
            break
        fix_limit = 0 if limit <= 0 else limit - attempted
        fix_stats = run_fix(
            app,
            limit=fix_limit,
            exclude_ids=attempted_ids,
            attempted_ids=attempted_ids,
            summarize=False,
        )
        attempted += fix_stats.attempted
        stats.fixed += fix_stats.fixed
        stats.failed += fix_stats.failed
        stats.skipped_stale += fix_stats.skipped_stale

        if limit > 0 and attempted >= limit:
            break

        # -- explore for new issues within the remaining budget --------
        remaining_created = 0 if limit <= 0 else limit - created
        if limit > 0 and remaining_created <= 0:
            break
        explore_stats = run_explore(app, limit=remaining_created)
        created += explore_stats.issues_created
        if explore_stats.issues_created == 0:
            break

        cycles += 1
        if cycles >= max_cycles:
            app.reporter.message(
                f"explore-and-fix: reached max_cycles={max_cycles} without "
                "converging; stopping (remaining issues stay open for the next run)"
            )
            break

    stats.created = created
    stats.cycles = cycles
    stats.remaining = app.database.count_issues()
    app.reporter.combined_summary(stats)
    return stats


__all__ = ["run_explore_and_fix"]
