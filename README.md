# agi-experiment

A small browser demo: **Pong** where the left paddle is a **growing neural network** trained online with **REINFORCE** (policy gradient), optional **Hebbian** updates, and **neurogenesis** on the last hidden layer. The right paddle uses a simple **greedy** policy. This is a toy visualization, not general intelligence.

## Stack

- [Vite](https://vitejs.dev/) + [React](https://react.dev/) 19 + TypeScript  
- Headless Pong sim and policy code in plain TypeScript (no RL framework)

## Quick start

```bash
npm install
npm run dev
```

Then open the URL Vite prints (usually `http://localhost:5173`).

### Scripts

| Command        | Description                          |
|----------------|--------------------------------------|
| `npm run dev`  | Dev server with HMR                  |
| `npm run build`| Typecheck + production build to `dist/` |
| `npm run preview` | Serve the production build locally |

## Docker

Build a static production bundle and serve it with nginx.

```bash
docker build -t agi-experiment .
docker run --rm -p 8080:80 agi-experiment
```

Or with Compose (maps **host port 8080** → container **80**):

```bash
docker compose up --build
```

Open **http://localhost:8080**.

## How it works (short)

1. **State** — The game is encoded into a fixed-length vector (ball, paddles, scores, etc.).
2. **Policy** — A small MLP (default `12 → 10 → 10 → 10 → 3`) outputs logits for **up / down / stay**; softmax gives action probabilities.
3. **Clocks** — Physics runs at **game Hz**, optionally **faster than real time** via a **physics time scale** multiplier (brain tick stays in real milliseconds). The network picks a new action and learns on each **brain tick** (default 1s wall clock). Reward is summed over that interval.
4. **REINFORCE** — After each interval, weights get a manual backward pass from the softmax policy gradient scaled by **advantage** (return minus an EMA baseline).
5. **Hebbian** — An extra local update scaled by interval reward co-activates pre/post units.
6. **Neurogenesis** — With configurable probability, the **last** hidden layer can grow (new row + output column), up to a max width.
7. **Unplugging** — After each weight-changing step, any **connection** weight ≤ 0 is set to 0. Random init and new growth weights use strictly positive draws so the net does not collapse to all zeros.

The UI has three areas: **Pong** (canvas), **live weight heatmaps** + mean-|w| sparkline, and **configuration** (layer sizes, tick rates, physics speed, learning rates, opponent strength, seed, resets).

## Repository layout

```
src/
  sim/           # Pong engine, state encoder, greedy opponent
  brain/         # Deep MLP, softmax, REINFORCE + Hebbian + neurogenesis
  hooks/         # Game + learning loop (physics vs brain clocks)
  components/    # Canvas, weight viz, config panel
docker/
  nginx.conf     # SPA + gzip (used by production image)
```

## License

MIT — see [LICENSE](LICENSE).
