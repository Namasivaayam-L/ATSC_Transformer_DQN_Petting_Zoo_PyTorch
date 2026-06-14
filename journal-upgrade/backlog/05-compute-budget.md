# Backlog 05 — Compute Budget Reality-Check

## Why (Week 0)
The plan targets <2 months on a single GPU. The experiment matrix (methods × scenarios × rewards × 5
seeds × episodes) must be estimated up front or the timeline silently breaks.

## Actions
- Estimate GPU-hours: count configurations × episodes × wall-clock per episode (measure one short run
  with `libsumo`). Multiply out.
- Confirm it fits the available GPU before the deadline. If not, secure backup: lab cluster, Colab Pro,
  or Kaggle.
- Use `libsumo` (not TraCI) and run seeds in parallel where the GPU allows. Cache the road graph.
- Define the cut-order (from `99-decision-gates.md` standing rule) if compute runs short.

## Definition of done
A GPU-hour estimate, a confirmed fit (or a secured backup), and a documented cut-order.
