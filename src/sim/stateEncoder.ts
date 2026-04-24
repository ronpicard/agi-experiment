import type { PongParams, PongState } from "./types";

/** Fixed feature size for the policy network input */
export const STATE_DIM = 12;

/**
 * Normalized game features in roughly [-1, 1].
 * Order: ball x,y,vx,vy, leftY, rightY, score diff, ball rel left paddle,
 * ball above left, dist to left paddle (norm), same for right (simplified).
 */
export function encodeState(s: PongState, p: PongParams): Float32Array {
  const out = new Float32Array(STATE_DIM);
  const fx = (s.ballX / p.fieldWidth) * 2 - 1;
  const fy = (s.ballY / p.fieldHeight) * 2 - 1;
  const fvx = clamp((s.ballVx / (p.baseBallSpeed * 2)) * 2 - 1, -1, 1);
  const fvy = clamp((s.ballVy / (p.baseBallSpeed * 2)) * 2 - 1, -1, 1);
  const fl = (s.leftY / p.fieldHeight) * 2 - 1;
  const fr = (s.rightY / p.fieldHeight) * 2 - 1;
  const score = clamp((s.scoreLeft - s.scoreRight) / 10, -1, 1);
  const relBallLeft = fy - fl;
  const relBallRight = fy - fr;

  out[0] = fx;
  out[1] = fy;
  out[2] = fvx;
  out[3] = fvy;
  out[4] = fl;
  out[5] = fr;
  out[6] = score;
  out[7] = clamp(relBallLeft, -1, 1);
  out[8] = clamp(relBallRight, -1, 1);
  out[9] = s.ballVx > 0 ? 1 : -1;
  const faceLeft = p.paddleWidth + p.ballRadius;
  out[10] = clamp((s.ballX - faceLeft) / p.fieldWidth, -1, 1);
  out[11] = clamp((s.ballY - s.leftY) / p.fieldHeight, -1, 1);

  return out;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
