# 00 — Session Reproduction Guide

## What this folder is
A complete record of the planning session that produced this `journal-upgrade/` tree. Its purpose is to
let a NEW chat session (or a future you) reconstruct exactly where we landed and WHY, without re-deriving
anything. If you are an agent resuming this work, read these files in order before acting.

## Read order
1. `01-session-context.md` — the paper, the repo, the tech stack, the goal. The "where we started".
2. `02-decisions-log.md` — every decision made and the exact value chosen. The "what we settled on".
3. `03-options-presented.md` — every option that was offered for each decision, including the ones NOT
   chosen and the trade-off previews. The "what else was on the table".
4. `04-diagnosis-and-suggestions.md` — the technical diagnosis (repo bugs, paper flaws) and the package/
   method recommendations. The "why the plan looks the way it does".
5. `05-open-tensions-and-risks.md` — unresolved tensions and the single biggest risk. The "watch out for".

## How to reproduce the session in a fresh chat
Paste this prompt to a new agent:

> Read `journal-upgrade/memory/` files 00→05 in order, then `journal-upgrade/plans/00-overview.md`. The
> repo under upgrade is `../ATSC_Transformer_DQN_Petting_Zoo_PyTorch`. All planning decisions are already
> made and recorded — do not re-ask them. Confirm you understand the locked decisions and the two
> decision gates, then continue from wherever the plans/ phases were last left off.

## One-line summary of the whole session
Diagnosed that the original FYP paper's transformer-DQN was never properly trained (a cluster of RL
bugs), then planned a simulation-only, journal-grade rebuild on CleanRL with a spatial neighbour-
coordination transformer as the real novelty, benchmarked on RESCO grid4x4 + cologne3, primary metric
average travel time, under a <2-month single-GPU budget with a Week-4 fallback to a temporal transformer.

## Provenance
- Original paper: `fyp_journal_ieee.pdf` (11 pages, IEEE format, dated 2024-12-16). Title: "Traffic Sense:
  Optimizing City Traffic with Transformer-infused DRL and Computer Vision."
- Repo: https://github.com/Namasivaayam-L/ATSC_Transformer_DQN_Petting_Zoo_PyTorch
- Planning session date: 2026-06-13.
