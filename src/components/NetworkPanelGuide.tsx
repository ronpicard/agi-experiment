/** Explanatory copy for the network / weights UI (lives in its own app section). */
export function NetworkPanelGuide({ inputDim }: { inputDim: number }) {
  return (
    <div
      style={{
        fontSize: "0.8125rem",
        lineHeight: 1.45,
        color: "#bdc1c6",
        padding: "0.5rem 0.65rem",
        background: "#121418",
        borderRadius: 8,
        border: "1px solid #2a2d34",
      }}
    >
      <p className="hint" style={{ margin: "0 0 0.45rem", color: "#9aa0a6" }}>
        Each <strong style={{ color: "#bdc1c6" }}>heatmap</strong> is one weight matrix: rows are
        outputs from that layer, columns are inputs. Color encodes each weight value (cooler =
        lower, warmer = higher) after normalizing min–max inside that matrix so you can see structure
        even when scales differ between layers.
      </p>
      <p className="hint" style={{ margin: "0 0 0.45rem", color: "#9aa0a6" }}>
        <strong style={{ color: "#bdc1c6" }}>Data flow:</strong> the game state is compressed to{" "}
        {inputDim} numbers, then multiplied by{" "}
        <code style={{ fontSize: "0.75rem" }}>W₁, W₂, …</code> with ReLU between hidden blocks, then{" "}
        <code style={{ fontSize: "0.75rem" }}>W_out</code> produces three logits (up / down / stay). A
        softmax turns those into probabilities; on each brain tick the left paddle uses one action
        (stochastic or greedy argmax, per config).
      </p>
      <p className="hint" style={{ margin: "0 0 0.45rem", color: "#9aa0a6" }}>
        <strong style={{ color: "#bdc1c6" }}>Learning:</strong> when a brain tick ends, REINFORCE
        nudges weights using the sum of rewards collected during that interval (advantage vs a
        moving-average baseline). A small <strong style={{ color: "#bdc1c6" }}>Hebbian</strong> term
        then reinforces co-active pre/post pairs scaled by that same interval reward.
      </p>
      <p className="hint" style={{ margin: "0 0 0.45rem", color: "#9aa0a6" }}>
        <strong style={{ color: "#bdc1c6" }}>Neurogenesis</strong> may append a new unit to the{" "}
        <em>last</em> hidden layer (new row in the last hidden matrix, new column in{" "}
        <code style={{ fontSize: "0.75rem" }}>W_out</code>), up to the max width in settings—so the
        heatmaps can grow over time.
      </p>
      <p className="hint" style={{ margin: "0 0 0.45rem", color: "#9aa0a6" }}>
        <strong style={{ color: "#bdc1c6" }}>Unplugging:</strong> after each weight-changing step, any{" "}
        <em>connection</em> weight ≤ 0 is set to 0 (that synapse no longer contributes). Random init and
        new growth weights use strictly positive draws so the network does not start or grow as
        all-disconnected.
      </p>
      <p className="hint" style={{ margin: "0 0 0.45rem", color: "#9aa0a6" }}>
        <strong style={{ color: "#bdc1c6" }}>Synapse weakening + pruning</strong> applies a small per-tick
        decay to weights/biases, then zeros out tiny values. Optionally, the weakest unit in the{" "}
        <em>last</em> hidden layer can be removed when its total in/out connectivity is below a
        threshold—so the heatmaps can also shrink over time.
      </p>
      <p className="hint" style={{ margin: 0, color: "#9aa0a6" }}>
        The <strong style={{ color: "#bdc1c6" }}>sparkline</strong> tracks mean absolute weight magnitude
        after each learning tick so you can see overall drift or stabilization at a glance.
      </p>
    </div>
  );
}
