import { useCallback, useState } from "react";
import { PongCanvas } from "./components/PongCanvas";
import { ConfigPanel } from "./components/ConfigPanel";
import { NetworkPanelGuide } from "./components/NetworkPanelGuide";
import { WeightViz } from "./components/WeightViz";
import {
  defaultExperimentConfig,
  serializeHiddenWidths,
  useNeuroPongLoop,
} from "./hooks/useNeuroPongLoop";

export default function App() {
  const [config, setConfig] = useState<typeof defaultExperimentConfig>(defaultExperimentConfig);
  const [hiddenField, setHiddenField] = useState(serializeHiddenWidths(defaultExperimentConfig.hiddenLayerWidths));
  const [parseError, setParseError] = useState<string | null>(null);

  const { snapshot, setRunning, resetGame, resetWeights, resetAll, parseHiddenWidthsField } =
    useNeuroPongLoop(config);

  const onApplyHidden = useCallback(() => {
    const parsed = parseHiddenWidthsField(hiddenField);
    if (!parsed) {
      setParseError("Use positive integers, e.g. 10, 10, 10");
      return;
    }
    setParseError(null);
    setConfig((c) => ({ ...c, hiddenLayerWidths: parsed }));
  }, [hiddenField, parseHiddenWidthsField]);

  return (
    <div className="app">
      <h1>AGI-Pong (DRL + Hebbian Learning + Neurogenesis + Synapse Weakening)</h1>
      <p className="hint">
        Physics runs at the game Hz and can run faster than real time via the physics time scale;
        the brain still uses wall-clock brain ticks and updates weights from reward accumulated
        over that interval. Large brain ticks look intentionally chunky.
      </p>

      <div className="toolbar">
        <button type="button" onClick={() => setRunning(!snapshot.running)}>
          {snapshot.running ? "Pause" : "Run"}
        </button>
        <button type="button" onClick={resetGame}>
          Reset game
        </button>
        <button type="button" onClick={resetWeights}>
          Randomize weights
        </button>
        <button type="button" onClick={resetAll}>
          Full reset
        </button>
      </div>

      <section className="panel" style={{ marginTop: "1rem" }}>
        <h2>Network weights (live)</h2>
        <WeightViz
          policy={snapshot.policy}
          meanAbsHistory={snapshot.meanAbsHistory}
          revision={snapshot.tickCount}
        />
      </section>

      <section className="panel" style={{ marginTop: "1rem" }}>
        <h2>Pong</h2>
        <PongCanvas state={snapshot.pong} params={snapshot.params} />
        <p className="hint">
          Left (blue) = policy · Right (red) = greedy · Last interval reward:{" "}
          <strong>{snapshot.lastReward.toFixed(4)}</strong> · Brain ticks: {snapshot.tickCount}
        </p>
      </section>

      <section className="panel" style={{ marginTop: "1rem" }} aria-labelledby="network-guide-heading">
        <h2 id="network-guide-heading">How this panel maps to the run</h2>
        <NetworkPanelGuide inputDim={snapshot.policy.inputDim} />
      </section>

      <section className="panel" style={{ marginTop: "1rem" }}>
        <h2>Configuration</h2>
        <ConfigPanel
          config={config}
          onChange={setConfig}
          hiddenField={hiddenField}
          onHiddenFieldChange={setHiddenField}
          onApplyHidden={onApplyHidden}
          parseError={parseError}
        />
      </section>
    </div>
  );
}
