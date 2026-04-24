# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-24

### Added

- Vite + React 19 + TypeScript app for a **Neuro-Pong** demo.
- Headless Pong simulation with configurable physics; **greedy** right-paddle opponent.
- **State encoder** (fixed-size game feature vector) feeding a configurable deep MLP.
- Default topology **12 → 10 → 10 → 10 → 3** (three hidden layers, three discrete actions).
- **REINFORCE** policy-gradient updates with EMA **baseline** and manual backprop through ReLU layers.
- **Hebbian**-style local updates scaled by interval reward.
- **Neurogenesis** on the last hidden layer (probability + max width cap).
- Separate **game Hz** and **brain tick** clocks; reward accumulated per brain interval.
- UI: Pong canvas, **live weight heatmaps**, mean-|w| sparkline, and **configuration** panel (clocks, topology string, learning rates, opponent scale, stochastic policy, seed, resets).
- Network panel: parameter counts (weights, biases, total), topology string, and explanatory copy.
- **Docker** multi-stage image (Node build, nginx serve) and **docker-compose** (`8080:80`).
- **README** with overview, scripts, Docker, and repo layout.
