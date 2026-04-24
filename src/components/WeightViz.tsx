import { useMemo } from "react";
import { countNetworkParameters, type DeepPolicySnapshot } from "../brain/deepPolicyNet";

interface MatrixProps {
  data: Float32Array;
  rows: number;
  cols: number;
  label: string;
  revision: number;
}

function Sparkline({ values }: { values: number[] }) {
  const points = useMemo(() => {
    if (values.length === 0) return "";
    const hi = Math.max(...values, 1e-9);
    const n = values.length;
    return values
      .map((v, i) => {
        const x = n <= 1 ? 60 : (i / (n - 1)) * 118 + 1;
        const y = 40 - (v / hi) * 36 + 2;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }, [values]);

  return (
    <svg
      viewBox="0 0 120 44"
      preserveAspectRatio="none"
      style={{ width: "100%", height: 48, border: "1px solid #3c4043", borderRadius: 6, background: "#0b0c0f" }}
    >
      {points && (
        <polyline fill="none" stroke="#5c9ded" strokeWidth="0.9" points={points} />
      )}
    </svg>
  );
}

function heatColor(t: number): string {
  const x = Math.max(0, Math.min(1, (t + 1) * 0.5));
  const h = (1 - x) * 240;
  const l = 35 + x * 35;
  return `hsl(${h} 70% ${l}%)`;
}

function MatrixHeatmap({ data, rows, cols, label, revision }: MatrixProps) {
  const { min, max, cells } = useMemo(() => {
    let minV = Infinity;
    let maxV = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    if (!Number.isFinite(minV) || !Number.isFinite(maxV)) {
      minV = -1;
      maxV = 1;
    }
    const span = maxV - minV || 1;
    const cells = Array.from({ length: rows * cols }, (_, i) => {
      const v = data[i];
      const t = (v - minV) / span * 2 - 1;
      return heatColor(t);
    });
    return { min: minV, max: maxV, cells };
  }, [data, rows, cols, revision]);

  return (
    <div style={{ marginBottom: "0.65rem" }}>
      <div style={{ fontSize: "0.72rem", color: "#9aa0a6", marginBottom: 4 }}>{label}</div>
      <svg
        viewBox={`0 0 ${cols} ${rows}`}
        preserveAspectRatio="none"
        style={{
          width: "100%",
          maxHeight: 140,
          borderRadius: 6,
          border: "1px solid #3c4043",
          background: "#0b0c0f",
        }}
      >
        {cells.map((fill, i) => {
          const r = Math.floor(i / cols);
          const c = i % cols;
          return <rect key={i} x={c} y={r} width={1} height={1} fill={fill} />;
        })}
      </svg>
      <div style={{ fontSize: "0.65rem", color: "#6f7378", marginTop: 2 }}>
        min {min.toFixed(3)} · max {max.toFixed(3)} · {rows}×{cols}
      </div>
    </div>
  );
}

function formatTopology(snapshot: DeepPolicySnapshot): string {
  const h = snapshot.hiddenLayerWidths.join(" → ");
  const last = 3;
  return `${snapshot.inputDim} → ${h} → ${last}`;
}

export function WeightViz({
  policy,
  meanAbsHistory,
  revision,
}: {
  policy: DeepPolicySnapshot;
  meanAbsHistory: number[];
  revision: number;
}) {
  const counts = useMemo(() => countNetworkParameters(policy), [policy]);

  const blocks: { data: Float32Array; rows: number; cols: number; label: string }[] = [];
  let inD = policy.inputDim;
  for (let l = 0; l < policy.weights.length; l++) {
    const isOut = l === policy.weights.length - 1;
    const outD = isOut ? policy.weights[l].length / inD : policy.hiddenLayerWidths[l];
    const w = policy.weights[l];
    blocks.push({
      data: w,
      rows: outD,
      cols: inD,
      label: isOut ? `W_out (${outD}×${inD})` : `W_${l + 1} (${outD}×${inD})`,
    });
    inD = outD;
  }

  return (
    <div>
      <div
        style={{
          fontSize: "0.8125rem",
          lineHeight: 1.45,
          color: "#bdc1c6",
          marginBottom: "0.75rem",
          padding: "0.5rem 0.65rem",
          background: "#121418",
          borderRadius: 8,
          border: "1px solid #2a2d34",
        }}
      >
        <p style={{ margin: "0 0 0.5rem" }}>
          <strong style={{ color: "#e8eaed" }}>How this panel maps to the run</strong>
        </p>
        <p className="hint" style={{ margin: "0 0 0.45rem", color: "#9aa0a6" }}>
          Each <strong style={{ color: "#bdc1c6" }}>heatmap</strong> is one weight matrix: rows are
          outputs from that layer, columns are inputs. Color encodes each weight value (cooler =
          lower, warmer = higher) after normalizing min–max inside that matrix so you can see
          structure even when scales differ between layers.
        </p>
        <p className="hint" style={{ margin: "0 0 0.45rem", color: "#9aa0a6" }}>
          <strong style={{ color: "#bdc1c6" }}>Data flow:</strong> the game state is compressed to{" "}
          {policy.inputDim} numbers, then multiplied by{" "}
          <code style={{ fontSize: "0.75rem" }}>W₁, W₂, …</code> with ReLU between hidden blocks, then{" "}
          <code style={{ fontSize: "0.75rem" }}>W_out</code> produces three logits (up / down / stay).
          A softmax turns those into probabilities; on each brain tick the left paddle uses one
          action (stochastic or greedy argmax, per config).
        </p>
        <p className="hint" style={{ margin: "0 0 0.45rem", color: "#9aa0a6" }}>
          <strong style={{ color: "#bdc1c6" }}>Learning:</strong> when a brain tick ends, REINFORCE
          nudges weights using the sum of rewards collected during that interval (advantage vs a
          moving-average baseline). A small <strong style={{ color: "#bdc1c6" }}>Hebbian</strong> term
          then reinforces co-active pre/post pairs scaled by that same interval reward.{" "}
          <strong style={{ color: "#bdc1c6" }}>Neurogenesis</strong> may append a new unit to the{" "}
          <em>last</em> hidden layer (new row in the last hidden matrix, new column in{" "}
          <code style={{ fontSize: "0.75rem" }}>W_out</code>), up to the max width in settings—so the
          heatmaps can grow over time.
        </p>
        <p className="hint" style={{ margin: 0, color: "#9aa0a6" }}>
          The <strong style={{ color: "#bdc1c6" }}>sparkline</strong> tracks mean absolute weight
          magnitude after each learning tick so you can see overall drift or stabilization at a
          glance.
        </p>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "baseline",
          gap: "0.35rem 1rem",
          marginBottom: "0.65rem",
          fontSize: "0.8125rem",
          color: "#e8eaed",
        }}
      >
        <span>
          <strong>Weight scalars</strong>: {counts.weightScalars.toLocaleString()}
        </span>
        <span style={{ color: "#6f7378" }}>·</span>
        <span>
          <strong>Bias scalars</strong>: {counts.biasScalars.toLocaleString()}
        </span>
        <span style={{ color: "#6f7378" }}>·</span>
        <span>
          <strong>Total parameters</strong>: {counts.totalScalars.toLocaleString()}
        </span>
      </div>
      <p className="hint" style={{ marginTop: "-0.35rem", marginBottom: "0.65rem" }}>
        “Weights” here means every connection weight; biases are listed separately. Topology:{" "}
        <code style={{ fontSize: "0.75rem" }}>{formatTopology(policy)}</code> (last is three paddle
        actions).
      </p>

      {blocks.map((b, i) => (
        <MatrixHeatmap key={i} {...b} revision={revision} />
      ))}
      <div style={{ fontSize: "0.72rem", color: "#9aa0a6", marginTop: 6 }}>Mean |w| over time</div>
      <Sparkline values={meanAbsHistory} />
    </div>
  );
}
