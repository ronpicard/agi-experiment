import { useCallback, useEffect, useRef, useState } from "react";
import { DeepPolicyNet } from "../brain/deepPolicyNet";
import { mulberry32 } from "../brain/rng";
import { greedyRightPaddleVy } from "../sim/greedyOpponent";
import { createInitialState, defaultPongParams, stepPong } from "../sim/pong";
import type { PongParams, PongState } from "../sim/types";
import { encodeState, STATE_DIM } from "../sim/stateEncoder";
import type { PaddleAction } from "../sim/types";

export interface ExperimentConfig {
  gameHz: number;
  /** >1 runs Pong physics faster than wall clock; brain tick stays wall-clock ms. */
  physicsTimeScale: number;
  brainTickMs: number;
  hiddenLayerWidths: number[];
  /** Max width for any hidden layer (neurogenesis cap per layer). */
  maxHiddenPerLayer: number;
  /** Minimum hidden units per layer (prune will not go below). */
  minHiddenPerLayer: number;
  /** EMA β for tracking input/hidden activations (hotspots for growth, low EMA for prune). */
  actEmaBeta: number;
  /** Neurons with activation EMA below this may be pruned (rebalance). */
  actPruneThreshold: number;
  /** How many presynaptic units get stronger weights when a new neuron is born. */
  neurogenesisHotK: number;
  neurogenesisInBonus: number;
  neurogenesisOutBonus: number;
  neurogenesisProb: number;
  rlLr: number;
  hebbianEta: number;
  /** Multiplicative per-tick weight decay (0 disables). */
  synapseDecay: number;
  /** After decay/update, weights with |w| below this are zeroed. */
  pruneWeightAbs: number;
  /** If the last-hidden unit's (incoming+outgoing) magnitude is below this, it may be removed. */
  pruneNeuronAbs: number;
  baselineBeta: number;
  stochasticPolicy: boolean;
  greedySpeedScale: number;
  seed: number;
}

export const defaultExperimentConfig: ExperimentConfig = {
  gameHz: 45,
  physicsTimeScale: 1,
  brainTickMs: 1000,
  hiddenLayerWidths: [10, 10, 10],
  maxHiddenPerLayer: 40,
  minHiddenPerLayer: 4,
  actEmaBeta: 0.94,
  actPruneThreshold: 0.012,
  neurogenesisHotK: 6,
  neurogenesisInBonus: 2.2,
  neurogenesisOutBonus: 2,
  neurogenesisProb: 0.08,
  rlLr: 0.06,
  hebbianEta: 0.002,
  synapseDecay: 0.0005,
  pruneWeightAbs: 0.00002,
  pruneNeuronAbs: 0.02,
  baselineBeta: 0.98,
  stochasticPolicy: true,
  greedySpeedScale: 1,
  seed: 42,
};

export interface LoopSnapshot {
  pong: PongState;
  params: PongParams;
  policy: ReturnType<DeepPolicyNet["snapshot"]>;
  lastReward: number;
  running: boolean;
  tickCount: number;
  meanAbsHistory: number[];
}

function parseWidths(s: string): number[] | null {
  const parts = s
    .split(/[, ]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => Number(x));
  if (parts.some((n) => !Number.isFinite(n) || n < 1 || n > 512)) return null;
  if (parts.length === 0) return null;
  return parts.map((n) => Math.floor(n));
}

export function serializeHiddenWidths(w: number[]): string {
  return w.join(", ");
}

export function parseHiddenWidthsField(text: string): number[] | null {
  return parseWidths(text);
}

export function useNeuroPongLoop(config: ExperimentConfig) {
  const rng0 = mulberry32(config.seed);
  const initialSim = createInitialState(defaultPongParams, rng0);
  const initialPolicy = new DeepPolicyNet(
    STATE_DIM,
    config.hiddenLayerWidths,
    mulberry32(config.seed + 1),
    config.minHiddenPerLayer,
    config.actEmaBeta,
  );

  const [snapshot, setSnapshot] = useState<LoopSnapshot>(() => ({
    pong: initialSim,
    params: { ...defaultPongParams },
    policy: initialPolicy.snapshot(),
    lastReward: 0,
    running: false,
    tickCount: 0,
    meanAbsHistory: [],
  }));

  const configRef = useRef(config);
  configRef.current = config;

  const simRef = useRef<PongState>({ ...initialSim });
  const paramsRef = useRef<PongParams>({ ...defaultPongParams });
  const rngSimRef = useRef(mulberry32(config.seed));
  const rngPolicyRef = useRef(mulberry32(config.seed + 1));
  const policyRef = useRef(initialPolicy);
  const leftActionRef = useRef<PaddleAction>(2);
  const physicsAccRef = useRef(0);
  const brainAccRef = useRef(0);
  const lastTsRef = useRef<number | null>(null);
  const rafRef = useRef<number>(0);
  const tickCountRef = useRef(0);
  const meanHistRef = useRef<number[]>([]);
  const runningRef = useRef(false);

  const prevXRef = useRef<Float32Array | null>(null);
  const prevActionRef = useRef<PaddleAction>(2);
  const rewardAccumRef = useRef(0);

  const rebuildPolicy = useCallback((widths: number[], seed: number) => {
    const c = configRef.current;
    rngPolicyRef.current = mulberry32(seed + 1);
    policyRef.current = new DeepPolicyNet(
      STATE_DIM,
      widths,
      rngPolicyRef.current,
      c.minHiddenPerLayer,
      c.actEmaBeta,
    );
  }, []);

  const resetGame = useCallback(() => {
    const c = configRef.current;
    rngSimRef.current = mulberry32(c.seed);
    simRef.current = createInitialState(paramsRef.current, rngSimRef.current);
    leftActionRef.current = 2;
    prevXRef.current = null;
    rewardAccumRef.current = 0;
    tickCountRef.current = 0;
    setSnapshot((prev) => ({
      ...prev,
      pong: { ...simRef.current },
      lastReward: 0,
      tickCount: 0,
    }));
  }, []);

  const resetWeights = useCallback(() => {
    const c = configRef.current;
    rebuildPolicy(c.hiddenLayerWidths, c.seed);
    policyRef.current.randomizeWeights();
    prevXRef.current = null;
    rewardAccumRef.current = 0;
    setSnapshot((prev) => ({
      ...prev,
      policy: policyRef.current.snapshot(),
    }));
  }, [rebuildPolicy]);

  const resetAll = useCallback(() => {
    const c = configRef.current;
    rngSimRef.current = mulberry32(c.seed);
    simRef.current = createInitialState(paramsRef.current, rngSimRef.current);
    rebuildPolicy(c.hiddenLayerWidths, c.seed);
    policyRef.current.randomizeWeights();
    leftActionRef.current = 2;
    prevXRef.current = null;
    rewardAccumRef.current = 0;
    tickCountRef.current = 0;
    meanHistRef.current = [];
    setSnapshot({
      pong: { ...simRef.current },
      params: { ...paramsRef.current },
      policy: policyRef.current.snapshot(),
      lastReward: 0,
      running: runningRef.current,
      tickCount: 0,
      meanAbsHistory: [],
    });
  }, [rebuildPolicy]);

  useEffect(() => {
    const c = configRef.current;
    const widths = c.hiddenLayerWidths;
    const prev = policyRef.current.hiddenLayerWidths.join(",");
    const next = widths.join(",");
    if (prev !== next) {
      rebuildPolicy(widths, c.seed);
      policyRef.current.randomizeWeights();
      prevXRef.current = null;
      setSnapshot((s) => ({ ...s, policy: policyRef.current.snapshot() }));
    }
  }, [config.hiddenLayerWidths, config.seed, rebuildPolicy]);

  const onBrainFire = useCallback(() => {
    const c = configRef.current;
    const sim = simRef.current;
    const p = paramsRef.current;
    const pol = policyRef.current;
    const px = prevXRef.current;
    const totalR = rewardAccumRef.current;
    const prevA = prevActionRef.current;

    if (px) {
      pol.forward(px);
      const adv = totalR - pol.baseline;
      pol.baseline = c.baselineBeta * pol.baseline + (1 - c.baselineBeta) * totalR;
      pol.reinforceUpdate(px, prevA, adv, c.rlLr);
      pol.hebbianUpdate(px, totalR, c.hebbianEta);
      pol.minHiddenPerLayer = c.minHiddenPerLayer;
      pol.actEmaBeta = c.actEmaBeta;
      pol.forward(px, true);
      pol.tryPruneLowImportanceNeuron(c.actPruneThreshold);
      pol.tryNeurogenesisTargeted(
        c.neurogenesisProb,
        c.maxHiddenPerLayer,
        c.neurogenesisHotK,
        c.neurogenesisInBonus,
        c.neurogenesisOutBonus,
        px,
      );
      pol.synapseWeakenAndPrune(c.synapseDecay, c.pruneWeightAbs, c.pruneNeuronAbs);
    }

    const x = encodeState(sim, p);
    pol.forward(x);
    const action = pol.pickAction(c.stochasticPolicy);
    prevXRef.current = new Float32Array(x);
    prevActionRef.current = action;
    leftActionRef.current = action;
    rewardAccumRef.current = 0;

    tickCountRef.current += 1;
    const m = pol.meanAbsWeight();
    const hist = meanHistRef.current;
    hist.push(m);
    if (hist.length > 120) hist.shift();

    setSnapshot({
      pong: { ...sim },
      params: { ...p },
      policy: pol.snapshot(),
      lastReward: totalR,
      running: runningRef.current,
      tickCount: tickCountRef.current,
      meanAbsHistory: [...hist],
    });
  }, []);

  useEffect(() => {
    if (!snapshot.running) return;

    const tick = (ts: number) => {
      if (!runningRef.current) return;
      const last = lastTsRef.current ?? ts;
      lastTsRef.current = ts;
      const dt = Math.min(0.1, (ts - last) / 1000);
      const c = configRef.current;
      const rawScale = c.physicsTimeScale;
      const physScale = Number.isFinite(rawScale)
        ? Math.max(0.05, Math.min(128, rawScale))
        : 1;

      physicsAccRef.current += dt * physScale;
      const stepDt = 1 / Math.max(1, c.gameHz);
      const rng = rngSimRef.current;
      const sim = simRef.current;
      const p = paramsRef.current;
      while (physicsAccRef.current >= stepDt) {
        physicsAccRef.current -= stepDt;
        const rightVy = greedyRightPaddleVy(sim, p, c.greedySpeedScale);
        const { state, reward } = stepPong(sim, p, leftActionRef.current, rightVy, stepDt, rng);
        Object.assign(sim, state);
        rewardAccumRef.current += reward;
      }

      brainAccRef.current += dt;
      const brainSec = Math.max(0.05, c.brainTickMs / 1000);
      while (brainAccRef.current >= brainSec) {
        brainAccRef.current -= brainSec;
        onBrainFire();
      }

      setSnapshot((prev) => ({
        ...prev,
        pong: { ...sim },
        policy: policyRef.current.snapshot(),
      }));

      rafRef.current = requestAnimationFrame(tick);
    };

    lastTsRef.current = null;
    physicsAccRef.current = 0;
    brainAccRef.current = 0;
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [snapshot.running, onBrainFire]);

  const setRunning = useCallback((running: boolean) => {
    runningRef.current = running;
    if (!running) {
      lastTsRef.current = null;
      physicsAccRef.current = 0;
      brainAccRef.current = 0;
    } else {
      prevXRef.current = null;
      rewardAccumRef.current = 0;
      onBrainFire();
    }
    setSnapshot((s) => ({ ...s, running }));
  }, [onBrainFire]);

  return {
    snapshot,
    setRunning,
    resetGame,
    resetWeights,
    resetAll,
    parseHiddenWidthsField,
  };
}
