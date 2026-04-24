/** Discrete paddle control for one side */
export type PaddleAction = 0 | 1 | 2; // up, down, stay

export interface PongParams {
  fieldWidth: number;
  fieldHeight: number;
  paddleWidth: number;
  paddleHeight: number;
  paddleMaxSpeed: number;
  ballRadius: number;
  baseBallSpeed: number;
  ballSpeedGainOnHit: number;
}

export interface PongState {
  ballX: number;
  ballY: number;
  ballVx: number;
  ballVy: number;
  leftY: number;
  rightY: number;
  scoreLeft: number;
  scoreRight: number;
}
