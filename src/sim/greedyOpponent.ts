import type { PongParams, PongState } from "./types";

/** Move right paddle toward ball y, clamped by max speed. */
export function greedyRightPaddleVy(
  s: PongState,
  p: PongParams,
  speedScale: number,
): number {
  const max = p.paddleMaxSpeed * speedScale;
  const dy = s.ballY - s.rightY;
  if (dy > 4) return max;
  if (dy < -4) return -max;
  return 0;
}
