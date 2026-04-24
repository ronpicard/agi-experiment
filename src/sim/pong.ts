import type { PaddleAction, PongParams, PongState } from "./types";

export const defaultPongParams: PongParams = {
  fieldWidth: 400,
  fieldHeight: 300,
  paddleWidth: 10,
  paddleHeight: 50,
  paddleMaxSpeed: 220,
  ballRadius: 6,
  baseBallSpeed: 200,
  ballSpeedGainOnHit: 1.04,
};

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function createInitialState(p: PongParams, rng: () => number): PongState {
  const angle = (rng() * 0.6 + 0.2) * Math.PI; // avoid too horizontal
  const dir = rng() < 0.5 ? -1 : 1;
  const sp = p.baseBallSpeed;
  return {
    ballX: p.fieldWidth * 0.5,
    ballY: p.fieldHeight * (0.3 + rng() * 0.4),
    ballVx: Math.cos(angle) * sp * dir,
    ballVy: Math.sin(angle) * sp * (rng() < 0.5 ? -1 : 1),
    leftY: p.fieldHeight * 0.5,
    rightY: p.fieldHeight * 0.5,
    scoreLeft: 0,
    scoreRight: 0,
  };
}

function actionToVy(action: PaddleAction, maxSpeed: number): number {
  if (action === 0) return -maxSpeed;
  if (action === 1) return maxSpeed;
  return 0;
}

/** Advance simulation by dt seconds. Returns reward shaping for the left (neural) agent. */
export function stepPong(
  s: PongState,
  p: PongParams,
  leftAction: PaddleAction,
  rightVy: number,
  dt: number,
  rng: () => number,
): { state: PongState; reward: number; pointLeft: boolean; pointRight: boolean } {
  const next: PongState = {
    ...s,
    leftY: clamp(
      s.leftY + actionToVy(leftAction, p.paddleMaxSpeed) * dt,
      p.paddleHeight * 0.5,
      p.fieldHeight - p.paddleHeight * 0.5,
    ),
    rightY: clamp(s.rightY + rightVy * dt, p.paddleHeight * 0.5, p.fieldHeight - p.paddleHeight * 0.5),
    ballX: s.ballX + s.ballVx * dt,
    ballY: s.ballY + s.ballVy * dt,
    ballVx: s.ballVx,
    ballVy: s.ballVy,
    scoreLeft: s.scoreLeft,
    scoreRight: s.scoreRight,
  };

  let reward = 0;
  let pointLeft = false;
  let pointRight = false;

  // Walls top/bottom
  if (next.ballY - p.ballRadius <= 0) {
    next.ballY = p.ballRadius;
    next.ballVy = Math.abs(next.ballVy);
  } else if (next.ballY + p.ballRadius >= p.fieldHeight) {
    next.ballY = p.fieldHeight - p.ballRadius;
    next.ballVy = -Math.abs(next.ballVy);
  }

  const halfP = p.paddleHeight * 0.5;
  const leftPadTop = next.leftY - halfP;
  const leftPadBot = next.leftY + halfP;
  const rightPadTop = next.rightY - halfP;
  const rightPadBot = next.rightY + halfP;

  // Left paddle collision
  if (
    next.ballX - p.ballRadius <= p.paddleWidth &&
    next.ballVx < 0 &&
    next.ballY >= leftPadTop &&
    next.ballY <= leftPadBot
  ) {
    next.ballX = p.paddleWidth + p.ballRadius;
    next.ballVx = Math.abs(next.ballVx) * p.ballSpeedGainOnHit;
    const hit = (next.ballY - next.leftY) / halfP;
    next.ballVy += hit * 80;
    reward += 0.02;
  }

  // Right paddle collision
  if (
    next.ballX + p.ballRadius >= p.fieldWidth - p.paddleWidth &&
    next.ballVx > 0 &&
    next.ballY >= rightPadTop &&
    next.ballY <= rightPadBot
  ) {
    next.ballX = p.fieldWidth - p.paddleWidth - p.ballRadius;
    next.ballVx = -Math.abs(next.ballVx) * p.ballSpeedGainOnHit;
    const hit = (next.ballY - next.rightY) / halfP;
    next.ballVy += hit * 80;
    reward -= 0.01;
  }

  // Score: ball past left
  if (next.ballX + p.ballRadius < 0) {
    next.scoreRight += 1;
    pointRight = true;
    reward -= 1;
    resetBallAfterPoint(next, p, 1, rng);
  } else if (next.ballX - p.ballRadius > p.fieldWidth) {
    next.scoreLeft += 1;
    pointLeft = true;
    reward += 1;
    resetBallAfterPoint(next, p, -1, rng);
  }

  // Light shaping: ball moving toward opponent helps left agent
  reward += (next.ballVx / p.baseBallSpeed) * 0.0005;

  return { state: next, reward, pointLeft, pointRight };
}

function resetBallAfterPoint(
  s: PongState,
  p: PongParams,
  toward: -1 | 1,
  rng: () => number,
): void {
  s.ballX = p.fieldWidth * 0.5;
  s.ballY = p.fieldHeight * 0.5;
  const angle = (rng() * 0.5 + 0.25) * Math.PI;
  const sp = p.baseBallSpeed;
  s.ballVx = Math.cos(angle) * sp * toward;
  s.ballVy = Math.sin(angle) * sp * (rng() < 0.5 ? -1 : 1);
}
