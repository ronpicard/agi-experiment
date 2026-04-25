import type { PongParams, PongState } from "./types";

/** Move right paddle toward ball y, clamped by max speed. */
export function greedyRightPaddleVy(
  s: PongState,
  p: PongParams,
  speedScale: number,
): number {
  const max = p.paddleMaxSpeed * speedScale;
  const dy = s.ballY - s.rightY;
  const dead = Math.max(1.5, p.fieldHeight * (4 / 150));
  if (dy > dead) return max;
  if (dy < -dead) return -max;
  return 0;
}
