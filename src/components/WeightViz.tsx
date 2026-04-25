import { useMemo } from "react";
import { countNetworkParameters, type DeepPolicySnapshot } from "../brain/deepPolicyNet";

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

function formatTopology(snapshot: DeepPolicySnapshot): string {
  const h = snapshot.hiddenLayerWidths.join(" → ");
  const last = 3;
  return `${snapshot.inputDim} → ${h} → ${last}`;
}

type LayerKind = "input" | "hidden" | "output";
type LayerSpec = {
  kind: LayerKind;
  label: string;
  size: number;
  // Each displayed node corresponds to a slice of original neurons.
  // For unsampled layers, these are singletons.
  slices: Array<{ start: number; end: number }>; // [start,end)
};

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function pickSlices(size: number, maxNodes: number): Array<{ start: number; end: number }> {
  if (size <= 0) return [];
  const n = Math.max(1, Math.min(maxNodes, size));
  if (n === size) return Array.from({ length: size }, (_, i) => ({ start: i, end: i + 1 }));
  // Evenly partition [0,size) into n slices.
  const slices: Array<{ start: number; end: number }> = [];
  for (let i = 0; i < n; i++) {
    const a = Math.floor((i * size) / n);
    const b = Math.floor(((i + 1) * size) / n);
    slices.push({ start: a, end: Math.max(a + 1, b) });
  }
  return slices;
}

function meanSignedAndAbs(
  W: Float32Array,
  outStart: number,
  outEnd: number,
  inStart: number,
  inEnd: number,
  inD: number,
): { mean: number; meanAbs: number } {
  let s = 0;
  let sa = 0;
  let n = 0;
  for (let oi = outStart; oi < outEnd; oi++) {
    const row = oi * inD;
    for (let ij = inStart; ij < inEnd; ij++) {
      const v = W[row + ij];
      s += v;
      sa += Math.abs(v);
      n++;
    }
  }
  if (!n) return { mean: 0, meanAbs: 0 };
  return { mean: s / n, meanAbs: sa / n };
}

function meanAbsSlice(b: Float32Array, start: number, end: number): number {
  let s = 0;
  let n = 0;
  for (let i = start; i < end; i++) {
    s += Math.abs(b[i]);
    n++;
  }
  return n ? s / n : 0;
}

function NetworkDiagram({
  policy,
  revision,
}: {
  policy: DeepPolicySnapshot;
  revision: number;
}) {
  const { layers, edges, biasAbs, maxAbs } = useMemo(() => {
    const hidden = policy.hiddenLayerWidths;
    const layerSizes = [policy.inputDim, ...hidden, 3];

    // Keep this responsive even with huge layers (up to 10k).
    const maxNodesByKind: Record<LayerKind, number> = {
      input: 18,
      hidden: 22,
      output: 3,
    };

    const layers: LayerSpec[] = layerSizes.map((size, idx) => {
      const kind: LayerKind =
        idx === 0 ? "input" : idx === layerSizes.length - 1 ? "output" : "hidden";
      const label =
        kind === "input"
          ? `Input (${size})`
          : kind === "output"
            ? `Output (${size})`
            : `Hidden ${idx} (${size})`;
      return {
        kind,
        label,
        size,
        slices: pickSlices(size, maxNodesByKind[kind]),
      };
    });

    // Bias magnitudes for displayed nodes (for non-input layers).
    const biasAbs: number[][] = [];
    for (let li = 0; li < layers.length; li++) {
      const L = layers[li];
      if (L.kind === "input") {
        biasAbs.push(new Array(L.slices.length).fill(0));
        continue;
      }
      // biases index: hidden layer l-1, output layer = hidden.length
      const b =
        L.kind === "output" ? policy.biases[hidden.length] : policy.biases[li - 1];
      biasAbs.push(L.slices.map((s) => meanAbsSlice(b, s.start, s.end)));
    }

    type Edge = {
      fromLayer: number;
      fromIdx: number;
      toLayer: number;
      toIdx: number;
      mean: number;
      abs: number;
    };
    const edges: Edge[] = [];

    // Weight matrices: policy.weights[0..hidden.length-1] are hidden weights; last is output weights.
    for (let li = 0; li < layers.length - 1; li++) {
      const from = layers[li];
      const to = layers[li + 1];
      const inD = from.size;
      const W = policy.weights[li]; // outD x inD
      for (let ti = 0; ti < to.slices.length; ti++) {
        const outS = to.slices[ti];
        for (let fi = 0; fi < from.slices.length; fi++) {
          const inS = from.slices[fi];
          const { mean, meanAbs } = meanSignedAndAbs(W, outS.start, outS.end, inS.start, inS.end, inD);
          edges.push({
            fromLayer: li,
            fromIdx: fi,
            toLayer: li + 1,
            toIdx: ti,
            mean,
            abs: meanAbs,
          });
        }
      }
    }

    let maxAbs = 1e-9;
    for (const e of edges) maxAbs = Math.max(maxAbs, e.abs);
    for (const row of biasAbs) for (const v of row) maxAbs = Math.max(maxAbs, v);

    return { layers, edges, biasAbs, maxAbs };
    // revision forces recompute after growth/prune
  }, [policy, revision]);

  const W = 860;
  const topPad = 30;
  const bottomPad = 22;
  const leftPad = 18;
  const rightPad = 18;
  const layerGap = 130;
  const H = topPad + bottomPad + (layers.length - 1) * layerGap + 20;

  const layerY = (li: number) => topPad + li * layerGap;

  const nodePos = (li: number, ni: number): { x: number; y: number } => {
    const L = layers[li];
    const n = Math.max(1, L.slices.length);
    const span = W - leftPad - rightPad;
    const x = leftPad + (n === 1 ? span / 2 : (ni / (n - 1)) * span);
    const y = layerY(li);
    return { x, y };
  };

  const edgeStroke = (mean: number) => (mean >= 0 ? "#79d28a" : "#f28b82");

  return (
    <div style={{ margin: "0.65rem 0 0.9rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div style={{ fontSize: "0.72rem", color: "#9aa0a6" }}>
          Traditional view (top→bottom). Opacity encodes |weight| and |bias|. Edge color encodes mean
          sign (green if mean ≥ 0, red if mean negative); weights ≤ 0 are unplugged after each tick
          so bundles are often all-positive.
        </div>
        <div style={{ fontSize: "0.72rem", color: "#6f7378" }}>
          Topology: <code style={{ fontSize: "0.72rem" }}>{formatTopology(policy)}</code>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{
          width: "100%",
          height: 420,
          borderRadius: 10,
          border: "1px solid #3c4043",
          background: "linear-gradient(180deg, #0b0c0f 0%, #0e1016 100%)",
        }}
        role="img"
        aria-label="Neural network diagram"
      >
        {/* Edges */}
        {edges.map((e, idx) => {
          const a = nodePos(e.fromLayer, e.fromIdx);
          const b = nodePos(e.toLayer, e.toIdx);
          const alpha = clamp01(0.04 + (e.abs / maxAbs) * 0.9);
          const w = 0.4 + (e.abs / maxAbs) * 1.6;
          return (
            <line
              key={idx}
              x1={a.x}
              y1={a.y + 7}
              x2={b.x}
              y2={b.y - 7}
              stroke={edgeStroke(e.mean)}
              strokeOpacity={alpha}
              strokeWidth={w}
            />
          );
        })}

        {/* Nodes */}
        {layers.map((L, li) => {
          const y = layerY(li);
          const labelX = leftPad;
          return (
            <g key={li}>
              <text
                x={labelX}
                y={y - 16}
                fill="#9aa0a6"
                fontSize="12"
                fontFamily="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto"
              >
                {L.label}
                {L.slices.length < L.size ? ` (showing ${L.slices.length})` : ""}
              </text>

              {L.slices.map((s, ni) => {
                const p = nodePos(li, ni);
                const bAbs = biasAbs[li][ni] ?? 0;
                const bAlpha = clamp01(0.08 + (bAbs / maxAbs) * 0.92);
                const r = L.kind === "output" ? 7.5 : 6.2;
                const fill = L.kind === "input" ? "#8ab4f8" : "#e8eaed";
                const stroke = L.kind === "input" ? "#5c9ded" : "#3c4043";
                return (
                  <g key={ni}>
                    {/* bias ring */}
                    {L.kind !== "input" && (
                      <circle
                        cx={p.x}
                        cy={p.y}
                        r={r + 2.6}
                        fill="none"
                        stroke="#fbbc04"
                        strokeWidth={1.4}
                        strokeOpacity={bAlpha}
                      />
                    )}
                    <circle cx={p.x} cy={p.y} r={r} fill={fill} fillOpacity={0.88} stroke={stroke} strokeWidth={1} />
                    {L.slices.length <= 12 && (
                      <text
                        x={p.x}
                        y={p.y + 3.6}
                        textAnchor="middle"
                        fill="#0b0c0f"
                        fontSize="9"
                        fontFamily="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto"
                      >
                        {s.start === s.end - 1 ? s.start : `${s.start}…`}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
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

  return (
    <div>
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
        “Weights” here means every connection weight; biases are listed separately.
      </p>

      <NetworkDiagram policy={policy} revision={revision} />

      <div style={{ fontSize: "0.72rem", color: "#9aa0a6", marginTop: 6 }}>Mean |w| over time</div>
      <Sparkline values={meanAbsHistory} />
    </div>
  );
}
