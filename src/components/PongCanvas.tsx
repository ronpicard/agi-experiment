import { useEffect, useRef } from "react";
import type { PongParams, PongState } from "../sim/types";

interface Props {
  state: PongState;
  params: PongParams;
}

export function PongCanvas({ state, params: p }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.clientWidth || p.fieldWidth;
    const h = canvas.clientHeight;
    const scale =
      h > 0
        ? Math.max(0.25, Math.min(w / p.fieldWidth, h / p.fieldHeight))
        : Math.max(0.25, w / p.fieldWidth);
    canvas.width = p.fieldWidth * scale;
    canvas.height = p.fieldHeight * scale;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);

    ctx.fillStyle = "#0b0c0f";
    ctx.fillRect(0, 0, p.fieldWidth, p.fieldHeight);

    ctx.strokeStyle = "#2a2d34";
    ctx.setLineDash([6, 8]);
    ctx.beginPath();
    ctx.moveTo(p.fieldWidth * 0.5, 0);
    ctx.lineTo(p.fieldWidth * 0.5, p.fieldHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    const halfP = p.paddleHeight * 0.5;

    ctx.fillStyle = "#5c9ded";
    ctx.fillRect(0, state.leftY - halfP, p.paddleWidth, p.paddleHeight);

    ctx.fillStyle = "#e07a5f";
    ctx.fillRect(
      p.fieldWidth - p.paddleWidth,
      state.rightY - halfP,
      p.paddleWidth,
      p.paddleHeight,
    );

    ctx.fillStyle = "#f4f1de";
    ctx.beginPath();
    ctx.arc(state.ballX, state.ballY, p.ballRadius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#9aa0a6";
    const scoreFs = Math.max(10, Math.round(14 * (p.fieldHeight / 300)));
    ctx.font = `${scoreFs}px system-ui,sans-serif`;
    const scoreY = p.fieldHeight * 0.073;
    ctx.fillText(String(state.scoreLeft), p.fieldWidth * 0.25, scoreY);
    ctx.fillText(String(state.scoreRight), p.fieldWidth * 0.72, scoreY);
  }, [state, p]);

  return (
    <canvas
      ref={ref}
      style={{
        display: "block",
        width: "100%",
        maxWidth: "50%",
        margin: "0 auto",
        height: "auto",
        aspectRatio: `${p.fieldWidth} / ${p.fieldHeight}`,
        borderRadius: 8,
        border: "1px solid #3c4043",
      }}
    />
  );
}
