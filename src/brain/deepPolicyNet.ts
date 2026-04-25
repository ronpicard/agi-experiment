import { argmax, sampleCategorical, softmaxInPlace } from "./math";
import type { PaddleAction } from "../sim/types";

const NUM_ACTIONS = 3;

export interface DeepPolicySnapshot {
  inputDim: number;
  hiddenLayerWidths: number[];
  weights: Float32Array[];
  biases: Float32Array[];
  /** Cumulative hidden units successfully added via neurogenesis (not reduced by pruning). */
  neuronsAddedLifetime: number;
}

/** Counts for the live network panel (connections vs weight slots, neurons, totals). */
export function countNetworkDisplayStats(snapshot: DeepPolicySnapshot): {
  connectionsNonzero: number;
  neurons: number;
  neuronsAdded: number;
  weightSlots: number;
  biasScalars: number;
  totalScalars: number;
} {
  let weightSlots = 0;
  let connectionsNonzero = 0;
  for (const w of snapshot.weights) {
    weightSlots += w.length;
    for (let i = 0; i < w.length; i++) {
      if (w[i] !== 0) connectionsNonzero++;
    }
  }
  let biasScalars = 0;
  for (const b of snapshot.biases) {
    biasScalars += b.length;
  }
  const neurons =
    snapshot.hiddenLayerWidths.reduce((s, d) => s + d, 0) + NUM_ACTIONS;
  return {
    connectionsNonzero,
    neurons,
    neuronsAdded: snapshot.neuronsAddedLifetime,
    weightSlots,
    biasScalars,
    totalScalars: weightSlots + biasScalars,
  };
}

function topKIndices(scores: Float32Array, k: number): number[] {
  const n = scores.length;
  const kk = Math.min(Math.max(1, k), n);
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, kk);
}

/** Row-major weight matrix: outRows x inCols, index row*inCols + col */
export class DeepPolicyNet {
  readonly inputDim: number;
  hiddenLayerWidths: number[];
  weights: Float32Array[];
  biases: Float32Array[];
  private zs: Float32Array[];
  private hs: Float32Array[];
  private readonly logits: Float32Array;
  private readonly probs: Float32Array;
  private readonly rng: () => number;
  /** Successful neurogenesis events (hidden units added). */
  neuronsAddedLifetime = 0;
  /** Minimum width for any hidden layer (prune will not go below). */
  minHiddenPerLayer: number;
  /** EMA factor for activation importance (0..1), higher = slower change. */
  actEmaBeta: number;
  private inputActEma: Float32Array;
  private actEma: Float32Array[];
  baseline = 0;

  constructor(
    inputDim: number,
    hiddenLayerWidths: number[],
    rng: () => number,
    minHiddenPerLayer = 4,
    actEmaBeta = 0.94,
  ) {
    this.inputDim = inputDim;
    this.hiddenLayerWidths = [...hiddenLayerWidths];
    this.rng = rng;
    this.minHiddenPerLayer = Math.max(1, minHiddenPerLayer);
    this.actEmaBeta = actEmaBeta;
    this.inputActEma = new Float32Array(inputDim);
    this.actEma = [];
    this.weights = [];
    this.biases = [];
    this.zs = [];
    this.hs = [];
    this.logits = new Float32Array(NUM_ACTIONS);
    this.probs = new Float32Array(NUM_ACTIONS);

    let inD = inputDim;
    for (let l = 0; l < hiddenLayerWidths.length; l++) {
      const outD = hiddenLayerWidths[l];
      this.weights.push(new Float32Array(outD * inD));
      this.biases.push(new Float32Array(outD));
      this.zs.push(new Float32Array(outD));
      this.hs.push(new Float32Array(outD));
      this.actEma.push(new Float32Array(outD));
      inD = outD;
    }
    const lastIn = inD;
    this.weights.push(new Float32Array(NUM_ACTIONS * lastIn));
    this.biases.push(new Float32Array(NUM_ACTIONS));

    this.randomizeWeights();
  }

  private resetActivationEmas(): void {
    this.inputActEma.fill(0);
    for (const a of this.actEma) {
      a.fill(0);
    }
  }

  /**
   * Remove any synapse whose weight is zero or negative ("unplugged").
   * Only connection weights are clipped; biases are unchanged.
   */
  unplugNonPositiveWeights(): void {
    for (const w of this.weights) {
      for (let i = 0; i < w.length; i++) {
        if (w[i] <= 0) w[i] = 0;
      }
    }
  }

  randomizeWeights(): void {
    const rng = this.rng;
    let inD = this.inputDim;
    for (let l = 0; l < this.weights.length; l++) {
      const isOut = l === this.weights.length - 1;
      const outD = isOut ? NUM_ACTIONS : this.hiddenLayerWidths[l];
      const scale = 1 / Math.sqrt(inD);
      const w = this.weights[l];
      for (let i = 0; i < w.length; i++) {
        w[i] = (0.02 + rng() * 0.98) * scale;
      }
      const b = this.biases[l];
      for (let i = 0; i < b.length; i++) {
        b[i] = (rng() * 2 - 1) * 0.05;
      }
      inD = outD;
    }
    this.baseline = 0;
    this.neuronsAddedLifetime = 0;
    this.resetActivationEmas();
    this.unplugNonPositiveWeights();
  }

  private recordActivationEmas(x: Float32Array): void {
    const b = this.actEmaBeta;
    for (let j = 0; j < this.inputDim; j++) {
      const v = Math.abs(x[j]);
      this.inputActEma[j] = b * this.inputActEma[j] + (1 - b) * v;
    }
    for (let l = 0; l < this.hiddenLayerWidths.length; l++) {
      const h = this.hs[l];
      const ema = this.actEma[l];
      for (let i = 0; i < h.length; i++) {
        ema[i] = b * ema[i] + (1 - b) * h[i];
      }
    }
  }

  forward(
    x: Float32Array,
    recordActivationEma = true,
  ): {
    logits: Float32Array;
    probs: Float32Array;
    zs: Float32Array[];
    hs: Float32Array[];
  } {
    let a = x;
    for (let l = 0; l < this.hiddenLayerWidths.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const z = this.zs[l];
      const h = this.hs[l];
      const outD = this.hiddenLayerWidths[l];
      const inD = a.length;
      for (let i = 0; i < outD; i++) {
        let sum = b[i];
        const row = i * inD;
        for (let j = 0; j < inD; j++) {
          sum += W[row + j] * a[j];
        }
        z[i] = sum;
        h[i] = sum > 0 ? sum : 0;
      }
      a = h;
    }
    const L = this.hiddenLayerWidths.length;
    const W = this.weights[L];
    const b = this.biases[L];
    const lastIn = this.hiddenLayerWidths[L - 1] ?? this.inputDim;
    for (let k = 0; k < NUM_ACTIONS; k++) {
      let sum = b[k];
      for (let i = 0; i < lastIn; i++) {
        sum += W[k * lastIn + i] * a[i];
      }
      this.logits[k] = sum;
    }
    softmaxInPlace(this.logits, this.probs);
    if (recordActivationEma) {
      this.recordActivationEmas(x);
    }
    return { logits: this.logits, probs: this.probs, zs: this.zs, hs: this.hs };
  }

  pickAction(stochastic: boolean): PaddleAction {
    if (stochastic) {
      return sampleCategorical(this.probs, this.rng) as PaddleAction;
    }
    return argmax(this.probs) as PaddleAction;
  }

  reinforceUpdate(x: Float32Array, action: PaddleAction, advantage: number, lr: number): void {
    const nHidden = this.hiddenLayerWidths.length;
    const dLogits = new Float32Array(NUM_ACTIONS);
    for (let k = 0; k < NUM_ACTIONS; k++) {
      const indicator = k === action ? 1 : 0;
      dLogits[k] = advantage * (indicator - this.probs[k]);
    }

    const dHidden = this.hs.map((h) => new Float32Array(h.length));

    let inD = this.hiddenLayerWidths[nHidden - 1];
    const dLast = dHidden[nHidden - 1];
    const WL = this.weights[nHidden];
    for (let i = 0; i < inD; i++) {
      let s = 0;
      for (let k = 0; k < NUM_ACTIONS; k++) {
        s += dLogits[k] * WL[k * inD + i];
      }
      dLast[i] = s;
    }

    for (let k = 0; k < NUM_ACTIONS; k++) {
      const g = dLogits[k];
      for (let i = 0; i < inD; i++) {
        WL[k * inD + i] += lr * g * this.hs[nHidden - 1][i];
      }
      this.biases[nHidden][k] += lr * g;
    }

    for (let l = nHidden - 1; l >= 0; l--) {
      const z = this.zs[l];
      const dh = dHidden[l];
      const dz = new Float32Array(dh.length);
      for (let i = 0; i < dh.length; i++) {
        dz[i] = dh[i] * (z[i] > 0 ? 1 : 0);
      }

      const W = this.weights[l];
      const outD = this.hiddenLayerWidths[l];
      const prevIn = l === 0 ? this.inputDim : this.hiddenLayerWidths[l - 1];
      const prevH = l === 0 ? x : this.hs[l - 1];

      if (l > 0) {
        const dPrev = dHidden[l - 1];
        dPrev.fill(0);
        for (let i = 0; i < outD; i++) {
          const g = dz[i];
          const row = i * prevIn;
          for (let j = 0; j < prevIn; j++) {
            dPrev[j] += g * W[row + j];
          }
        }
      }

      for (let i = 0; i < outD; i++) {
        const g = dz[i];
        const row = i * prevIn;
        for (let j = 0; j < prevIn; j++) {
          W[row + j] += lr * g * prevH[j];
        }
        this.biases[l][i] += lr * g;
      }
    }
    this.unplugNonPositiveWeights();
  }

  hebbianUpdate(x: Float32Array, reward: number, eta: number): void {
    if (eta === 0 || reward === 0) return;
    const r = reward;
    let a = x;
    for (let l = 0; l < this.hiddenLayerWidths.length; l++) {
      const W = this.weights[l];
      const h = this.hs[l];
      const outD = this.hiddenLayerWidths[l];
      const inD = a.length;
      for (let i = 0; i < outD; i++) {
        for (let j = 0; j < inD; j++) {
          W[i * inD + j] += eta * r * a[j] * h[i];
        }
      }
      a = h;
    }
    const L = this.hiddenLayerWidths.length;
    const W = this.weights[L];
    const lastIn = this.hiddenLayerWidths[L - 1];
    const hLast = this.hs[L - 1];
    for (let k = 0; k < NUM_ACTIONS; k++) {
      for (let i = 0; i < lastIn; i++) {
        W[k * lastIn + i] += eta * r * hLast[i] * this.probs[k];
      }
    }
    this.unplugNonPositiveWeights();
  }

  synapseWeakenAndPrune(decay: number, pruneWeightAbs: number, pruneNeuronAbs: number): void {
    const d = Number.isFinite(decay) ? Math.max(0, Math.min(0.2, decay)) : 0;
    const wFloor = Number.isFinite(pruneWeightAbs) ? Math.max(0, pruneWeightAbs) : 0;
    const nFloor = Number.isFinite(pruneNeuronAbs) ? Math.max(0, pruneNeuronAbs) : 0;

    if (d > 0) {
      const s = 1 - d;
      for (const w of this.weights) {
        for (let i = 0; i < w.length; i++) w[i] *= s;
      }
      for (const b of this.biases) {
        for (let i = 0; i < b.length; i++) b[i] *= s;
      }
    }

    if (wFloor > 0) {
      for (const w of this.weights) {
        for (let i = 0; i < w.length; i++) {
          const v = w[i];
          if (v !== 0 && Math.abs(v) < wFloor) w[i] = 0;
        }
      }
      for (const b of this.biases) {
        for (let i = 0; i < b.length; i++) {
          const v = b[i];
          if (v !== 0 && Math.abs(v) < wFloor) b[i] = 0;
        }
      }
    }

    if (nFloor > 0) {
      this.pruneWeakLastHidden(nFloor);
    }

    this.unplugNonPositiveWeights();
  }

  /**
   * Remove one last-hidden unit with smallest total |weight| if below threshold (optional extra shrink).
   */
  private pruneWeakLastHidden(neuronAbsThreshold: number): boolean {
    const nH = this.hiddenLayerWidths.length;
    if (nH === 0) return false;
    const lastIdx = nH - 1;
    const cur = this.hiddenLayerWidths[lastIdx];
    if (cur <= this.minHiddenPerLayer) return false;

    const prevIn = lastIdx === 0 ? this.inputDim : this.hiddenLayerWidths[lastIdx - 1];
    const Wmid = this.weights[lastIdx];
    const bmid = this.biases[lastIdx];
    const Wout = this.weights[nH];

    let bestI = -1;
    let bestScore = Infinity;

    for (let i = 0; i < cur; i++) {
      let incoming = Math.abs(bmid[i]);
      const row = i * prevIn;
      for (let j = 0; j < prevIn; j++) incoming += Math.abs(Wmid[row + j]);

      let outgoing = 0;
      for (let k = 0; k < NUM_ACTIONS; k++) outgoing += Math.abs(Wout[k * cur + i]);

      const score = incoming + outgoing;
      if (score < bestScore) {
        bestScore = score;
        bestI = i;
      }
    }

    if (bestI < 0 || bestScore >= neuronAbsThreshold) return false;
    this.removeNeuronAtLayer(lastIdx, bestI);
    return true;
  }

  /**
   * Drop the least important hidden unit whose EMA activation is below threshold (rebalances over time).
   */
  tryPruneLowImportanceNeuron(actPruneThreshold: number): boolean {
    const thr = Number.isFinite(actPruneThreshold) ? Math.max(1e-6, actPruneThreshold) : 0.012;
    const nH = this.hiddenLayerWidths.length;
    let bestL = -1;
    let bestI = -1;
    let bestEma = Infinity;
    for (let l = 0; l < nH; l++) {
      const w = this.hiddenLayerWidths[l];
      if (w <= this.minHiddenPerLayer) continue;
      const ema = this.actEma[l];
      for (let i = 0; i < w; i++) {
        const e = ema[i];
        if (e < thr && e < bestEma) {
          bestEma = e;
          bestL = l;
          bestI = i;
        }
      }
    }
    if (bestL < 0 || bestI < 0) return false;
    this.removeNeuronAtLayer(bestL, bestI);
    return true;
  }

  private removeNeuronAtLayer(L: number, rm: number): void {
    const nH = this.hiddenLayerWidths.length;
    if (L < 0 || L >= nH) return;
    const cur = this.hiddenLayerWidths[L];
    if (cur <= this.minHiddenPerLayer) return;
    if (rm < 0 || rm >= cur) return;

    const prevIn = L === 0 ? this.inputDim : this.hiddenLayerWidths[L - 1];
    const W = this.weights[L];
    const newRows = cur - 1;
    const wNew = new Float32Array(newRows * prevIn);
    let dst = 0;
    for (let i = 0; i < cur; i++) {
      if (i === rm) continue;
      const row = i * prevIn;
      for (let j = 0; j < prevIn; j++) wNew[dst++] = W[row + j];
    }
    this.weights[L] = wNew;

    const outNext = L + 1 >= nH ? NUM_ACTIONS : this.hiddenLayerWidths[L + 1];
    const Wnext = this.weights[L + 1];
    const wNextNew = new Float32Array(outNext * newRows);
    for (let r = 0; r < outNext; r++) {
      let cWrite = 0;
      for (let c = 0; c < cur; c++) {
        if (c === rm) continue;
        wNextNew[r * newRows + cWrite++] = Wnext[r * cur + c];
      }
    }
    this.weights[L + 1] = wNextNew;

    const bOld = this.biases[L];
    const bNew = new Float32Array(newRows);
    const zNew = new Float32Array(newRows);
    const hNew = new Float32Array(newRows);
    const emaNew = new Float32Array(newRows);
    let t = 0;
    for (let i = 0; i < cur; i++) {
      if (i === rm) continue;
      bNew[t] = bOld[i];
      zNew[t] = this.zs[L][i];
      hNew[t] = this.hs[L][i];
      emaNew[t] = this.actEma[L][i];
      t++;
    }
    this.biases[L] = bNew;
    this.zs[L] = zNew;
    this.hs[L] = hNew;
    this.actEma[L] = emaNew;

    this.hiddenLayerWidths[L] = newRows;
    this.unplugNonPositiveWeights();
  }

  /**
   * Grow a random eligible hidden layer; new unit connects strongly to historically hot inputs/outputs.
   */
  tryNeurogenesisTargeted(
    prob: number,
    maxHiddenPerLayer: number,
    hotK: number,
    inBonus: number,
    outBonus: number,
    x: Float32Array,
  ): boolean {
    const nH = this.hiddenLayerWidths.length;
    if (nH === 0) return false;
    if (this.rng() >= prob) return false;

    const cap = Math.max(1, Math.min(10000, Math.floor(maxHiddenPerLayer)));
    const eligible: number[] = [];
    for (let l = 0; l < nH; l++) {
      if (this.hiddenLayerWidths[l] < cap) eligible.push(l);
    }
    if (eligible.length === 0) return false;
    const L = eligible[Math.floor(this.rng() * eligible.length)];

    const prevIn = L === 0 ? this.inputDim : this.hiddenLayerWidths[L - 1];
    const pre = L === 0 ? x : this.hs[L - 1];
    const scores = new Float32Array(prevIn);
    for (let j = 0; j < prevIn; j++) {
      const pv = Math.abs(pre[j]);
      const em = L === 0 ? this.inputActEma[j] : this.actEma[L - 1][j];
      scores[j] = pv * (1 + em);
    }
    const hot = new Set(topKIndices(scores, hotK));

    const cur = this.hiddenLayerWidths[L];
    const newCur = cur + 1;
    const base = (1 / Math.sqrt(prevIn)) * 0.22;
    const ib = Number.isFinite(inBonus) ? Math.max(1, inBonus) : 2;
    const ob = Number.isFinite(outBonus) ? Math.max(1, outBonus) : 2;

    const W = this.weights[L];
    const wRow = new Float32Array(newCur * prevIn);
    wRow.set(W);
    const rowOff = cur * prevIn;
    for (let j = 0; j < prevIn; j++) {
      const mul = hot.has(j) ? ib : 1;
      wRow[rowOff + j] = (0.02 + this.rng() * 0.98) * base * mul;
    }
    this.weights[L] = wRow;

    const bOld = this.biases[L];
    const bNew = new Float32Array(newCur);
    bNew.set(bOld);
    bNew[cur] = (this.rng() * 2 - 1) * 0.05;

    const zNew = new Float32Array(newCur);
    zNew.set(this.zs[L]);
    const hNew = new Float32Array(newCur);
    hNew.set(this.hs[L]);
    const emaNew = new Float32Array(newCur);
    emaNew.set(this.actEma[L]);
    emaNew[cur] = Math.max(emaNew[cur], 0.05);

    this.biases[L] = bNew;
    this.zs[L] = zNew;
    this.hs[L] = hNew;
    this.actEma[L] = emaNew;

    const outNext = L + 1 >= nH ? NUM_ACTIONS : this.hiddenLayerWidths[L + 1];
    const Wnext = this.weights[L + 1];
    const wNextNew = new Float32Array(outNext * newCur);
    if (L + 1 >= nH) {
      let maxP = 1e-8;
      for (let k = 0; k < NUM_ACTIONS; k++) maxP = Math.max(maxP, this.probs[k]);
      for (let k = 0; k < NUM_ACTIONS; k++) {
        for (let c = 0; c < cur; c++) {
          wNextNew[k * newCur + c] = Wnext[k * cur + c];
        }
        const col = (0.02 + this.rng() * 0.98) * 0.1 * (1 + ob * (this.probs[k] / maxP));
        wNextNew[k * newCur + cur] = col;
      }
    } else {
      let maxE = 1e-8;
      const emaDown = this.actEma[L + 1];
      for (let r = 0; r < outNext; r++) maxE = Math.max(maxE, emaDown[r]);
      for (let r = 0; r < outNext; r++) {
        for (let c = 0; c < cur; c++) {
          wNextNew[r * newCur + c] = Wnext[r * cur + c];
        }
        const col = (0.02 + this.rng() * 0.98) * 0.12 * (1 + ob * (emaDown[r] / maxE));
        wNextNew[r * newCur + cur] = col;
      }
    }
    this.weights[L + 1] = wNextNew;

    this.hiddenLayerWidths[L] = newCur;
    this.unplugNonPositiveWeights();
    this.neuronsAddedLifetime++;
    return true;
  }

  snapshot(): DeepPolicySnapshot {
    return {
      inputDim: this.inputDim,
      hiddenLayerWidths: [...this.hiddenLayerWidths],
      weights: this.weights.map((w) => w),
      biases: this.biases.map((b) => b),
      neuronsAddedLifetime: this.neuronsAddedLifetime,
    };
  }

  meanAbsWeight(): number {
    let s = 0;
    let n = 0;
    for (const w of this.weights) {
      for (let i = 0; i < w.length; i++) {
        s += Math.abs(w[i]);
        n++;
      }
    }
    return n ? s / n : 0;
  }
}
