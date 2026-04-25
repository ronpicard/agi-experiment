import type { ExperimentConfig } from "../hooks/useNeuroPongLoop";
import { serializeHiddenWidths } from "../hooks/useNeuroPongLoop";

interface Props {
  config: ExperimentConfig;
  onChange: (c: ExperimentConfig) => void;
  hiddenField: string;
  onHiddenFieldChange: (s: string) => void;
  onApplyHidden: () => void;
  parseError: string | null;
}

export function ConfigPanel({
  config,
  onChange,
  hiddenField,
  onHiddenFieldChange,
  onApplyHidden,
  parseError,
}: Props) {
  const n = (k: keyof ExperimentConfig, v: number) => onChange({ ...config, [k]: v });

  return (
    <div className="config-grid">
      <fieldset>
        <legend>Clocks</legend>
        <label htmlFor="gameHz">Game physics (Hz)</label>
        <input
          id="gameHz"
          type="number"
          min={1}
          max={240}
          value={config.gameHz}
          onChange={(e) => n("gameHz", Number(e.target.value))}
        />
        <label htmlFor="physScale" style={{ marginTop: 8 }}>
          Physics time scale (× real time)
        </label>
        <input
          id="physScale"
          type="number"
          min={0.05}
          max={128}
          step={0.05}
          value={config.physicsTimeScale}
          onChange={(e) => n("physicsTimeScale", Number(e.target.value))}
        />
        <p className="hint" style={{ marginTop: 4 }}>
          Values above 1 advance the Pong sim faster than your clock (more physics steps per
          frame). Brain tick below is still in real milliseconds.
        </p>
        <label htmlFor="brainMs" style={{ marginTop: 8 }}>
          Brain tick (ms)
        </label>
        <input
          id="brainMs"
          type="number"
          min={50}
          max={60000}
          step={50}
          value={config.brainTickMs}
          onChange={(e) => n("brainTickMs", Number(e.target.value))}
        />
      </fieldset>

      <fieldset>
        <legend>Network topology</legend>
        <label htmlFor="hidden">Hidden layer widths (comma-separated)</label>
        <input
          id="hidden"
          type="text"
          value={hiddenField}
          onChange={(e) => onHiddenFieldChange(e.target.value)}
          placeholder={serializeHiddenWidths([10, 10, 10])}
        />
        {parseError && (
          <div className="hint" style={{ color: "#f4a261" }}>
            {parseError}
          </div>
        )}
        <button type="button" style={{ marginTop: 8 }} onClick={onApplyHidden}>
          Apply layer sizes
        </button>
        <label htmlFor="maxH" style={{ marginTop: 8 }}>
          Max width per hidden layer (neurogenesis cap)
        </label>
        <input
          id="maxH"
          type="number"
          min={4}
          max={10000}
          value={config.maxHiddenPerLayer}
          onChange={(e) => n("maxHiddenPerLayer", Number(e.target.value))}
        />
        <label htmlFor="minH" style={{ marginTop: 8 }}>
          Min width per hidden layer (prune floor)
        </label>
        <input
          id="minH"
          type="number"
          min={1}
          max={256}
          value={config.minHiddenPerLayer}
          onChange={(e) => n("minHiddenPerLayer", Number(e.target.value))}
        />
      </fieldset>

      <fieldset>
        <legend>Grow / prune (activity)</legend>
        <label htmlFor="actEmaB">Activation EMA β</label>
        <input
          id="actEmaB"
          type="number"
          min={0.5}
          max={0.999}
          step={0.001}
          value={config.actEmaBeta}
          onChange={(e) => n("actEmaBeta", Number(e.target.value))}
        />
        <label htmlFor="actPrune" style={{ marginTop: 8 }}>
          Prune neuron if its activation EMA is below
        </label>
        <input
          id="actPrune"
          type="number"
          min={0}
          max={0.2}
          step={0.001}
          value={config.actPruneThreshold}
          onChange={(e) => n("actPruneThreshold", Number(e.target.value))}
        />
        <label htmlFor="hotK" style={{ marginTop: 8 }}>
          Hot presynaptic slots for new neurons
        </label>
        <input
          id="hotK"
          type="number"
          min={1}
          max={64}
          value={config.neurogenesisHotK}
          onChange={(e) => n("neurogenesisHotK", Number(e.target.value))}
        />
        <label htmlFor="inBon" style={{ marginTop: 8 }}>
          Incoming weight bonus on hot inputs
        </label>
        <input
          id="inBon"
          type="number"
          min={1}
          max={10}
          step={0.1}
          value={config.neurogenesisInBonus}
          onChange={(e) => n("neurogenesisInBonus", Number(e.target.value))}
        />
        <label htmlFor="outBon" style={{ marginTop: 8 }}>
          Outgoing weight bonus toward hot next layer
        </label>
        <input
          id="outBon"
          type="number"
          min={1}
          max={10}
          step={0.1}
          value={config.neurogenesisOutBonus}
          onChange={(e) => n("neurogenesisOutBonus", Number(e.target.value))}
        />
      </fieldset>

      <fieldset>
        <legend>Learning</legend>
        <label htmlFor="rlLr">REINFORCE learning rate</label>
        <input
          id="rlLr"
          type="number"
          min={0}
          max={1}
          step={0.001}
          value={config.rlLr}
          onChange={(e) => n("rlLr", Number(e.target.value))}
        />
        <label htmlFor="hebb" style={{ marginTop: 8 }}>
          Hebbian η (0 disables)
        </label>
        <input
          id="hebb"
          type="number"
          min={0}
          max={0.1}
          step={0.0005}
          value={config.hebbianEta}
          onChange={(e) => n("hebbianEta", Number(e.target.value))}
        />
        <label htmlFor="decay" style={{ marginTop: 8 }}>
          Synapse decay / tick (0 disables)
        </label>
        <input
          id="decay"
          type="number"
          min={0}
          max={0.2}
          step={0.0001}
          value={config.synapseDecay}
          onChange={(e) => n("synapseDecay", Number(e.target.value))}
        />
        <label htmlFor="pruneW" style={{ marginTop: 8 }}>
          Prune tiny weights (|w| &lt; threshold → 0)
        </label>
        <input
          id="pruneW"
          type="number"
          min={0}
          max={1}
          step={0.00001}
          value={config.pruneWeightAbs}
          onChange={(e) => n("pruneWeightAbs", Number(e.target.value))}
        />
        <label htmlFor="pruneN" style={{ marginTop: 8 }}>
          Prune weak last-hidden neurons (0 disables)
        </label>
        <input
          id="pruneN"
          type="number"
          min={0}
          max={10}
          step={0.001}
          value={config.pruneNeuronAbs}
          onChange={(e) => n("pruneNeuronAbs", Number(e.target.value))}
        />
        <label htmlFor="base" style={{ marginTop: 8 }}>
          Baseline EMA β
        </label>
        <input
          id="base"
          type="number"
          min={0}
          max={0.999}
          step={0.001}
          value={config.baselineBeta}
          onChange={(e) => n("baselineBeta", Number(e.target.value))}
        />
        <label htmlFor="ng" style={{ marginTop: 8 }}>
          Neurogenesis probability / tick
        </label>
        <input
          id="ng"
          type="number"
          min={0}
          max={1}
          step={0.01}
          value={config.neurogenesisProb}
          onChange={(e) => n("neurogenesisProb", Number(e.target.value))}
        />
      </fieldset>

      <fieldset>
        <legend>Opponent & policy</legend>
        <label htmlFor="greedy">Greedy opponent speed scale</label>
        <input
          id="greedy"
          type="number"
          min={0.1}
          max={3}
          step={0.05}
          value={config.greedySpeedScale}
          onChange={(e) => n("greedySpeedScale", Number(e.target.value))}
        />
        <label style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={config.stochasticPolicy}
            onChange={(e) => onChange({ ...config, stochasticPolicy: e.target.checked })}
          />
          Stochastic policy (explore)
        </label>
      </fieldset>

      <fieldset>
        <legend>Randomness</legend>
        <label htmlFor="seed">RNG seed</label>
        <input
          id="seed"
          type="number"
          value={config.seed}
          onChange={(e) => n("seed", Number(e.target.value) | 0)}
        />
      </fieldset>
    </div>
  );
}
