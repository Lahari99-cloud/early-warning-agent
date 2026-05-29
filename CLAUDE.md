# Attrition Early-Warning Agent

## What this is
Demo for a Data Scientist (People) application. Predicts 90-day regretted-attrition
risk per employee, explains the drivers, and an LLM agent drafts a manager 1:1 prep
note. Human reviews, never auto-sends. Synthetic data, production-shaped architecture.

## Stack
Python 3.11, xgboost, shap, pandas, faker (synthetic data), anthropic (agent),
streamlit (UI), pytest. Deploy target: Replit.

## Principles
- Synthetic data only. No real HR data anywhere.
- Every prediction must be explainable (SHAP drivers attached).
- Agent output is a DRAFT for human review. Never an action.
- Small, tested, committed increments. One concern per module.
- Determinism where possible: seed everything (np, random, faker).

## Structure
data/        synthetic generator + cached parquet
model/       train, evaluate, explain, persisted artifacts
agent/       prompt builder + LLM call + guardrails
app/         streamlit dashboard
tests/       pytest, one file per module

## Done = code + passing test + committed.