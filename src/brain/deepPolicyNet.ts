import { argmax, sampleCategorical, softmaxInPlace } from "./math";
import type { PaddleAction } from "../sim/types";

const NUM_ACTIONS = 3;

export interface DeepPolicySnapshot {
  inputDim: number;
  hiddenLayerWidths: number[];
  weights: Float32Array[];
  biases: Float32Array[];
}

/** Scalar counts for the policy (every entry in each weight matrix and bias vector). */
export function countNetworkParameters(snapshot: DeepPolicySnapshot): {
  weightScalars: number;
  biasScalars: number;
  totalScalars: number;
} {
  let weightScalars = 0;
  for (const w of snapshot.weights) {
    weightScalars += w.length;
  }
  let biasScalars = 0;
  for (const b of snapshot.biases) {
    biasScalars += b.length;
  }
  return {
    weightScalars,
    biasScalars,
    totalScalars: weightScalars + biasScalars,
  };
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
  baseline = 0;

  constructor(inputDim: number, hiddenLayerWidths: number[], rng: () => number) {
    this.inputDim = inputDim;
    this.hiddenLayerWidths = [...hiddenLayerWidths];
    this.rng = rng;
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
      inD = outD;
    }
    const lastIn = inD;
    this.weights.push(new Float32Array(NUM_ACTIONS * lastIn));
    this.biases.push(new Float32Array(NUM_ACTIONS));

    this.randomizeWeights();
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
        w[i] = (rng() * 2 - 1) * scale;
      }
      const b = this.biases[l];
      for (let i = 0; i < b.length; i++) {
        b[i] = (rng() * 2 - 1) * 0.05;
      }
      inD = outD;
    }
    this.baseline = 0;
  }

  forward(x: Float32Array): {
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
  }

  tryNeurogenesis(prob: number, maxLastHidden: number): boolean {
    const nH = this.hiddenLayerWidths.length;
    if (nH === 0) return false;
    const lastIdx = nH - 1;
    const cur = this.hiddenLayerWidths[lastIdx];
    if (cur >= maxLastHidden) return false;
    if (this.rng() >= prob) return false;

    const prevIn = lastIdx === 0 ? this.inputDim : this.hiddenLayerWidths[lastIdx - 1];
    const newW = cur + 1;

    const Wmid = this.weights[lastIdx];
    const wNew = new Float32Array(newW * prevIn);
    wNew.set(Wmid);
    const scale = 1 / Math.sqrt(prevIn);
    for (let j = 0; j < prevIn; j++) {
      wNew[cur * prevIn + j] = (this.rng() * 2 - 1) * scale * 0.25;
    }
    this.weights[lastIdx] = wNew;

    const bmid = this.biases[lastIdx];
    const bNew = new Float32Array(newW);
    bNew.set(bmid);
    bNew[cur] = (this.rng() * 2 - 1) * 0.05;
    this.biases[lastIdx] = bNew;

    const zNew = new Float32Array(newW);
    zNew.set(this.zs[lastIdx]);
    this.zs[lastIdx] = zNew;
    const hNew = new Float32Array(newW);
    hNew.set(this.hs[lastIdx]);
    this.hs[lastIdx] = hNew;

    const Wout = this.weights[nH];
    const wOutNew = new Float32Array(NUM_ACTIONS * newW);
    for (let k = 0; k < NUM_ACTIONS; k++) {
      for (let i = 0; i < cur; i++) {
        wOutNew[k * newW + i] = Wout[k * cur + i];
      }
      wOutNew[k * newW + cur] = (this.rng() * 2 - 1) * 0.05;
    }
    this.weights[nH] = wOutNew;

    this.hiddenLayerWidths[lastIdx] = newW;
    return true;
  }

  snapshot(): DeepPolicySnapshot {
    return {
      inputDim: this.inputDim,
      hiddenLayerWidths: [...this.hiddenLayerWidths],
      weights: this.weights.map((w) => w),
      biases: this.biases.map((b) => b),
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
