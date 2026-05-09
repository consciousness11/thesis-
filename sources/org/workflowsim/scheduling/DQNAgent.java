/**
 * Pure-Java Deep Q-Network agent used by DQNSchedulingAlgorithm.
 *
 * Q(s, a) is parameterised with (state, action) features concatenated as the
 * network input and a single scalar Q value as the output.  This lets the same
 * network handle a varying number of candidate VMs each step (we score each
 * (cloudlet, vm) pair independently and pick the argmax).
 *
 * Implements the paper's training loop: epsilon-greedy with decay, an
 * experience-replay buffer of capacity m_sigma, minibatch updates of size
 * b_sigma, and a target network synced every {@code targetSync} steps.
 */
package org.workflowsim.scheduling;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Random;

public class DQNAgent {

    public static final int STATE_DIM = 4;
    public static final int ACTION_DIM = 3;
    public static final int INPUT_DIM = STATE_DIM + ACTION_DIM;

    private final int hiddenDim;
    private final double learningRate;
    private final double gamma;
    private final int batchSize;
    private final int replayCapacity;
    private final int targetSync;

    private double epsilon;
    private final double epsilonMin;
    private final double epsilonDecay;

    private final Random rng;
    private final Deque<Transition> replay;

    private double[][] w1, w1Tar;
    private double[] b1, b1Tar;
    private double[][] w2, w2Tar;
    private double[] b2, b2Tar;

    private int trainStep;

    public DQNAgent(long seed) {
        this.hiddenDim = 16;
        this.learningRate = 0.01;
        this.gamma = 0.9;
        this.batchSize = 32;
        this.replayCapacity = 2000;
        this.targetSync = 50;

        this.epsilon = 1.0;
        this.epsilonMin = 0.05;
        this.epsilonDecay = 0.995;

        this.rng = new Random(seed);
        this.replay = new ArrayDeque<>();

        this.w1 = randMatrix(INPUT_DIM, hiddenDim);
        this.b1 = new double[hiddenDim];
        this.w2 = randMatrix(hiddenDim, 1);
        this.b2 = new double[1];

        this.w1Tar = copy(w1);
        this.b1Tar = b1.clone();
        this.w2Tar = copy(w2);
        this.b2Tar = b2.clone();
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

    private double[] forward(double[] x, double[][] w1, double[] b1, double[][] w2, double[] b2,
                              double[] hiddenOut) {
        for (int j = 0; j < hiddenDim; j++) {
            double s = b1[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                s += x[i] * w1[i][j];
            }
            hiddenOut[j] = s > 0 ? s : 0;
        }
        double q = b2[0];
        for (int j = 0; j < hiddenDim; j++) {
            q += hiddenOut[j] * w2[j][0];
        }
        return new double[]{q};
    }

    public double qValue(double[] state, double[] action) {
        double[] x = concat(state, action);
        double[] h = new double[hiddenDim];
        return forward(x, w1, b1, w2, b2, h)[0];
    }

    private double qValueTarget(double[] input) {
        double[] h = new double[hiddenDim];
        return forward(input, w1Tar, b1Tar, w2Tar, b2Tar, h)[0];
    }

    /**
     * Pick the index of the best action given the current state and a list of
     * candidate action feature vectors.  Uses epsilon-greedy exploration.
     */
    public int selectAction(double[] state, List<double[]> candidateActions) {
        if (candidateActions.isEmpty()) {
            return -1;
        }
        if (rng.nextDouble() < epsilon) {
            return rng.nextInt(candidateActions.size());
        }
        int best = 0;
        double bestQ = qValue(state, candidateActions.get(0));
        for (int i = 1; i < candidateActions.size(); i++) {
            double q = qValue(state, candidateActions.get(i));
            if (q > bestQ) {
                bestQ = q;
                best = i;
            }
        }
        return best;
    }

    public void remember(double[] state, double[] action, double reward,
                         double[] nextState, List<double[]> nextActions, boolean done) {
        if (replay.size() >= replayCapacity) {
            replay.removeFirst();
        }
        replay.addLast(new Transition(state, action, reward, nextState, nextActions, done));
    }

    public void trainStep() {
        if (replay.size() < batchSize) {
            return;
        }
        List<Transition> batch = sampleBatch();

        double[][] gW1 = new double[INPUT_DIM][hiddenDim];
        double[] gB1 = new double[hiddenDim];
        double[][] gW2 = new double[hiddenDim][1];
        double[] gB2 = new double[1];

        for (Transition t : batch) {
            double target = t.reward;
            if (!t.done && t.nextActions != null && !t.nextActions.isEmpty()) {
                double maxNext = Double.NEGATIVE_INFINITY;
                for (double[] na : t.nextActions) {
                    double q = qValueTarget(concat(t.nextState, na));
                    if (q > maxNext) {
                        maxNext = q;
                    }
                }
                target += gamma * maxNext;
            }

            double[] x = concat(t.state, t.action);
            double[] h = new double[hiddenDim];
            double q = forward(x, w1, b1, w2, b2, h)[0];
            double err = q - target;

            gB2[0] += err;
            for (int j = 0; j < hiddenDim; j++) {
                gW2[j][0] += err * h[j];
            }
            for (int j = 0; j < hiddenDim; j++) {
                if (h[j] > 0) {
                    double dh = err * w2[j][0];
                    gB1[j] += dh;
                    for (int i = 0; i < INPUT_DIM; i++) {
                        gW1[i][j] += dh * x[i];
                    }
                }
            }
        }

        double scale = learningRate / batch.size();
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                w1[i][j] -= scale * gW1[i][j];
            }
        }
        for (int j = 0; j < hiddenDim; j++) {
            b1[j] -= scale * gB1[j];
        }
        for (int j = 0; j < hiddenDim; j++) {
            w2[j][0] -= scale * gW2[j][0];
        }
        b2[0] -= scale * gB2[0];

        trainStep++;
        if (trainStep % targetSync == 0) {
            w1Tar = copy(w1);
            b1Tar = b1.clone();
            w2Tar = copy(w2);
            b2Tar = b2.clone();
        }
    }

    public void decayEpsilon() {
        epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
    }

    public double getEpsilon() {
        return epsilon;
    }

    private List<Transition> sampleBatch() {
        List<Transition> all = new ArrayList<>(replay);
        List<Transition> batch = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            batch.add(all.get(rng.nextInt(all.size())));
        }
        return batch;
    }

    private static double[] concat(double[] a, double[] b) {
        double[] c = new double[a.length + b.length];
        System.arraycopy(a, 0, c, 0, a.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }

    private static class Transition {
        final double[] state;
        final double[] action;
        final double reward;
        final double[] nextState;
        final List<double[]> nextActions;
        final boolean done;

        Transition(double[] s, double[] a, double r, double[] ns, List<double[]> na, boolean d) {
            this.state = s;
            this.action = a;
            this.reward = r;
            this.nextState = ns;
            this.nextActions = na;
            this.done = d;
        }
    }
}
