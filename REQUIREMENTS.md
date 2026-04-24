# Requirements

What you need to **develop**, **build**, or **run** this project.

## Local development and production build

| Requirement | Notes |
|-------------|--------|
| **Node.js** | **20.x or newer** recommended (Vite 6 and the toolchain expect a current Node). The Docker image uses **22** for builds. |
| **npm** | Comes with Node; used for `npm install` / `npm ci`. **10+** is typical on current Node installs. |
| **Git** | Optional for cloning; required if you use version control workflows. |

No global installs are required beyond Node/npm unless you prefer another package manager (then use its equivalent of `npm ci` / `npm run build`).

## Runtime (browser)

- A **recent evergreen browser** (Chrome, Firefox, Safari, Edge) with JavaScript enabled.
- The app is a static SPA after `npm run build`; no backend server is required for the UI itself.

## Docker (optional)

Only if you use the container workflow:

| Requirement | Notes |
|-------------|--------|
| **Docker Engine** | Recent version with BuildKit (default on current Docker Desktop / Linux). |
| **Docker Compose** | v2 plugin (`docker compose`) for the bundled `docker-compose.yml`. |

Compose maps **host port 8080** to the container’s **HTTP port 80** (nginx).

## Hardware

- Ordinary laptop specs are sufficient; the simulation and networks are tiny.
