/**
 * Rainbow-lite DQN: Double DQN + n-step returns + Dueling heads.
 *
 * Dueling decomposition:
 *   Q(s, a) = V(s) + A(s, a) - mean_{a' in currentActions} A(s, a')
 * Two single-hidden-layer MLPs back V and A respectively. The mean term is
 * computed at training time over the candidate action set captured when the
 * transition was recorded (the agent stores currentActions in each
 * Transition, mirroring how the vanilla agent stores nextActions).
 *
 * n-step returns:
 *   R^{(n)}_t = sum_{k=0..n-1} gamma^k r_{t+k} + gamma^n * Q(s_{t+n}, argmax_a Q(s_{t+n}, a))
 * with a 3-step ring buffer in front of the main replay. The bootstrap uses
 * gamma^{n_effective} where n_effective shrinks if an intermediate
 * transition was marked done (we cut the chain there) or if the buffer is
 * drained at episode end.
 *
 * Double DQN:
 *   target uses the online net to pick the argmax action at s_{t+n} but the
 *   target net to score it - decouples action selection from value
 *   estimation and reduces optimistic bias.
 *
 * Everything else (epsilon-greedy decay, target-net sync, replay capacity)
 * matches DQNAgent so the comparison is apples-to-apples.
 */
package org.workflowsim.scheduling;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Random;

public class RainbowDQNAgent {

    public static final int STATE_DIM = 4;
    public static final int ACTION_DIM = 3;
    public static final int A_INPUT_DIM = STATE_DIM + ACTION_DIM;

    private final int hiddenDim = 16;
    private final double learningRate = 0.01;
    private final double gamma = 0.9;
    private final int batchSize = 32;
    private final int replayCapacity = 2000;
    private final int targetSync = 50;
    private final int nStep = 3;

    private double epsilon = 1.0;
    private final double epsilonMin = 0.05;
    private final double epsilonDecay = 0.995;

    private final Random rng;
    private final Deque<NStepEntry> nStepBuffer;
    private final Deque<Transition> replay;

    // Value net: state -> V scalar.
    private double[][] vW1; private double[] vB1;
    private double[][] vW2; private double[] vB2;
    private double[][] vW1Tar; private double[] vB1Tar;
    private double[][] vW2Tar; private double[] vB2Tar;

    // Advantage net: (state, action) -> A scalar.
    private double[][] aW1; private double[] aB1;
    private double[][] aW2; private double[] aB2;
    private double[][] aW1Tar; private double[] aB1Tar;
    private double[][] aW2Tar; private double[] aB2Tar;

    private int trainStep;

    public RainbowDQNAgent(long seed) {
        this.rng = new Random(seed);
        this.nStepBuffer = new ArrayDeque<>();
        this.replay = new ArrayDeque<>();

        vW1 = randMatrix(STATE_DIM, hiddenDim);
        vB1 = new double[hiddenDim];
        vW2 = randMatrix(hiddenDim, 1);
        vB2 = new double[1];

        aW1 = randMatrix(A_INPUT_DIM, hiddenDim);
        aB1 = new double[hiddenDim];
        aW2 = randMatrix(hiddenDim, 1);
        aB2 = new double[1];

        syncTargets();
    }

    private double[][] randMatrix(int rows, int cols) {
        double[][] m = new double[rows][cols];
        double scale = Math.sqrt(2.0 / rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[i][j] = rng.nextGaussian() * scale;
            }
        }
        return m;
    }

    private double[][] copy(double[][] src) {
        double[][] dst = new double[src.length][];
        for (int i = 0; i < src.length; i++) {
            dst[i] = src[i].clone();
        }
        return dst;
    }

    private void syncTargets() {
        vW1Tar = copy(vW1); vB1Tar = vB1.clone();
        vW2Tar = copy(vW2); vB2Tar = vB2.clone();
        aW1Tar = copy(aW1); aB1Tar = aB1.clone();
        aW2Tar = copy(aW2); aB2Tar = aB2.clone();
    }

    /** Single hidden-layer forward; writes pre-output hidden activations into hiddenOut. */
    private double forwardOne(double[] x, int inDim,
                              double[][] w1, double[] b1, double[][] w2, double[] b2,
                              double[] hiddenOut) {
        for (int j = 0; j < hiddenDim; j++) {
            double s = b1[j];
            for (int i = 0; i < inDim; i++) {
                s += x[i] * w1[i][j];
            }
            hiddenOut[j] = s > 0 ? s : 0;
        }
        double out = b2[0];
        for (int j = 0; j < hiddenDim; j++) {
            out += hiddenOut[j] * w2[j][0];
        }
        return out;
    }

    private double vValue(double[] state, boolean useTarget) {
        double[] h = new double[hiddenDim];
        if (useTarget) {
            return forwardOne(state, STATE_DIM, vW1Tar, vB1Tar, vW2Tar, vB2Tar, h);
        }
        return forwardOne(state, STATE_DIM, vW1, vB1, vW2, vB2, h);
    }

    private double aValue(double[] state, double[] action, boolean useTarget) {
        double[] x = concat(state, action);
        double[] h = new double[hiddenDim];
        if (useTarget) {
            return forwardOne(x, A_INPUT_DIM, aW1Tar, aB1Tar, aW2Tar, aB2Tar, h);
        }
        return forwardOne(x, A_INPUT_DIM, aW1, aB1, aW2, aB2, h);
    }

    /**
     * argmax_a Q(s, a). Since V(s) and mean_a' A(s, a') are constant across a,
     * the argmax over Q reduces to argmax over A. Epsilon-greedy as usual.
     */
    public int selectAction(double[] state, List<double[]> candidateActions) {
        if (candidateActions.isEmpty()) {
            return -1;
        }
        if (rng.nextDouble() < epsilon) {
            return rng.nextInt(candidateActions.size());
        }
        int best = 0;
        double bestA = aValue(state, candidateActions.get(0), false);
        for (int i = 1; i < candidateActions.size(); i++) {
            double a = aValue(state, candidateActions.get(i), false);
            if (a > bestA) {
                bestA = a;
                best = i;
            }
        }
        return best;
    }

    /** Push a single-step transition; emit n-step transitions as the buffer fills. */
    public void remember(double[] state, double[] action, List<double[]> currentActions,
                         double reward, double[] nextState, List<double[]> nextActions,
                         boolean done) {
        nStepBuffer.addLast(new NStepEntry(state, action, currentActions, reward,
                nextState, nextActions, done));
        if (nStepBuffer.size() >= nStep) {
            emitOldest();
        }
        if (done) {
            while (!nStepBuffer.isEmpty()) {
                emitOldest();
            }
        }
    }

    /** Drain remaining short n-step transitions at episode/workflow boundary. */
    public void endEpisode() {
        while (!nStepBuffer.isEmpty()) {
            emitOldest();
        }
    }

    private void emitOldest() {
        if (nStepBuffer.isEmpty()) {
            return;
        }
        List<NStepEntry> entries = new ArrayList<>(nStepBuffer);
        int k = entries.size();
        double rSum = 0.0;
        double g = 1.0;
        int effectiveK = k;
        for (int i = 0; i < k; i++) {
            rSum += g * entries.get(i).reward;
            g *= gamma;
            if (entries.get(i).done) {
                effectiveK = i + 1;
                break;
            }
        }
        NStepEntry first = entries.get(0);
        NStepEntry last = entries.get(effectiveK - 1);
        Transition t = new Transition(
                first.state, first.action, first.currentActions,
                rSum, last.nextState, last.nextActions, last.done, effectiveK);
        if (replay.size() >= replayCapacity) {
            replay.removeFirst();
        }
        replay.addLast(t);
        nStepBuffer.pollFirst();
    }

    public void trainStep() {
        if (replay.size() < batchSize) {
            return;
        }
        List<Transition> all = new ArrayList<>(replay);

        double[][] gVW1 = new double[STATE_DIM][hiddenDim];
        double[] gVB1 = new double[hiddenDim];
        double[][] gVW2 = new double[hiddenDim][1];
        double[] gVB2 = new double[1];

        double[][] gAW1 = new double[A_INPUT_DIM][hiddenDim];
        double[] gAB1 = new double[hiddenDim];
        double[][] gAW2 = new double[hiddenDim][1];
        double[] gAB2 = new double[1];

        for (int b = 0; b < batchSize; b++) {
            Transition t = all.get(rng.nextInt(all.size()));

            // ----- Target (Double DQN + n-step) ---------------------------------
            double target = t.rewardSum;
            if (!t.done && t.nextActions != null && !t.nextActions.isEmpty()) {
                int bestIdx = 0;
                double bestA = aValue(t.nextState, t.nextActions.get(0), false);
                for (int i = 1; i < t.nextActions.size(); i++) {
                    double av = aValue(t.nextState, t.nextActions.get(i), false);
                    if (av > bestA) { bestA = av; bestIdx = i; }
                }
                double vTar = vValue(t.nextState, true);
                double aTar = aValue(t.nextState, t.nextActions.get(bestIdx), true);
                double meanATar = 0.0;
                for (double[] na : t.nextActions) {
                    meanATar += aValue(t.nextState, na, true);
                }
                meanATar /= t.nextActions.size();
                double qNext = vTar + aTar - meanATar;
                target += Math.pow(gamma, t.nEffective) * qNext;
            }

            // ----- Online forward for Q(s, a_taken) -----------------------------
            double[] vH = new double[hiddenDim];
            double vOut = forwardOne(t.state, STATE_DIM, vW1, vB1, vW2, vB2, vH);

            double[] xTaken = concat(t.state, t.action);
            double[] aHTaken = new double[hiddenDim];
            double aTaken = forwardOne(xTaken, A_INPUT_DIM, aW1, aB1, aW2, aB2, aHTaken);

            int nA = t.currentActions.size();
            double[][] aXs = new double[nA][];
            double[][] aHs = new double[nA][hiddenDim];
            double meanA = 0.0;
            for (int j = 0; j < nA; j++) {
                aXs[j] = concat(t.state, t.currentActions.get(j));
                aHs[j] = new double[hiddenDim];
                meanA += forwardOne(aXs[j], A_INPUT_DIM, aW1, aB1, aW2, aB2, aHs[j]);
            }
            meanA /= nA;

            double qPred = vOut + aTaken - meanA;
            double err = qPred - target;

            // ----- Backprop -----------------------------------------------------
            // V branch: dQ/dV = 1.
            backpropMLP(gVW1, gVB1, gVW2, gVB2, vW2, t.state, STATE_DIM, vH, err);

            // A branch: contribution from the +A(s, a_taken) term (weight 1).
            backpropMLP(gAW1, gAB1, gAW2, gAB2, aW2, xTaken, A_INPUT_DIM, aHTaken, err);

            // A branch: contribution from the -mean A term (weight -1/n per candidate).
            // If a_taken matches one of currentActions[j], this loop double-counts
            // it with weight -1/n; combined with the +1 above the net weight is
            // (1 - 1/n) -- exactly the dueling decomposition.
            double meanWeight = -err / nA;
            for (int j = 0; j < nA; j++) {
                backpropMLP(gAW1, gAB1, gAW2, gAB2, aW2, aXs[j], A_INPUT_DIM, aHs[j], meanWeight);
            }
        }

        double scale = learningRate / batchSize;
        applyGrad(vW1, gVW1, scale);
        for (int j = 0; j < hiddenDim; j++) vB1[j] -= scale * gVB1[j];
        applyGrad(vW2, gVW2, scale);
        vB2[0] -= scale * gVB2[0];

        applyGrad(aW1, gAW1, scale);
        for (int j = 0; j < hiddenDim; j++) aB1[j] -= scale * gAB1[j];
        applyGrad(aW2, gAW2, scale);
        aB2[0] -= scale * gAB2[0];

        trainStep++;
        if (trainStep % targetSync == 0) {
            syncTargets();
        }
    }

    private void applyGrad(double[][] w, double[][] g, double scale) {
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[i].length; j++) {
                w[i][j] -= scale * g[i][j];
            }
        }
    }

    /** Accumulate gradients of (err * forwardOne) into gW1, gB1, gW2, gB2. */
    private void backpropMLP(double[][] gW1, double[] gB1, double[][] gW2, double[] gB2,
                             double[][] w2,
                             double[] x, int inDim, double[] hidden, double err) {
        gB2[0] += err;
        for (int j = 0; j < hiddenDim; j++) {
            gW2[j][0] += err * hidden[j];
        }
        for (int j = 0; j < hiddenDim; j++) {
            if (hidden[j] <= 0) continue;
            double dh = err * w2[j][0];
            gB1[j] += dh;
            for (int i = 0; i < inDim; i++) {
                gW1[i][j] += dh * x[i];
            }
        }
    }

    public void decayEpsilon() {
        epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
    }

    public double getEpsilon() {
        return epsilon;
    }

    private static double[] concat(double[] a, double[] b) {
        double[] c = new double[a.length + b.length];
        System.arraycopy(a, 0, c, 0, a.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    /** Single-step transition kept in the n-step ring buffer. */
    private static class NStepEntry {
        final double[] state;
        final double[] action;
        final List<double[]> currentActions;
        final double reward;
        final double[] nextState;
        final List<double[]> nextActions;
        final boolean done;

        NStepEntry(double[] s, double[] a, List<double[]> ca, double r,
                   double[] ns, List<double[]> na, boolean d) {
            this.state = s;
            this.action = a;
            this.currentActions = ca;
            this.reward = r;
            this.nextState = ns;
            this.nextActions = na;
            this.done = d;
        }
    }

    /** n-step transition kept in the main replay buffer. */
    private static class Transition {
        final double[] state;
        final double[] action;
        final List<double[]> currentActions;
        final double rewardSum;
        final double[] nextState;
        final List<double[]> nextActions;
        final boolean done;
        final int nEffective;

        Transition(double[] s, double[] a, List<double[]> ca, double rs,
                   double[] ns, List<double[]> na, boolean d, int n) {
            this.state = s;
            this.action = a;
            this.currentActions = ca;
            this.rewardSum = rs;
            this.nextState = ns;
            this.nextActions = na;
            this.done = d;
            this.nEffective = n;
        }
    }
}
