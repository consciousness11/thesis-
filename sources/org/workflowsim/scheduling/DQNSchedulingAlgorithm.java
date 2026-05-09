/**
 * MOPWSDRL: Multi-Objective Prioritized Workflow Scheduling using Deep
 * Reinforcement Learning, after Mangalampalli et al., IEEE Access 2024.
 *
 * The scheduler keeps a singleton {@link DQNAgent} so the Q-network is shared
 * across scheduling rounds and across workflows in a single JVM run (this
 * matches the paper's "100 iterations" training loop).
 *
 * Each call to {@link #run()} corresponds to one decision episode: ready
 * cloudlets are matched to idle VMs one at a time, with the agent selecting
 * the VM via epsilon-greedy on Q(s, a).  The state is a 4-D vector built from
 * task priority, VM priority, current makespan, and current energy
 * (eqs. 6, 7, 11, 16 of the paper); the action features describe a candidate
 * VM (its electricity priority, normalised speed, current load).
 *
 * Reward = -(makespan_increment + alpha * energy_increment).  The paper
 * states R = min(MSP, Energy) (eq. 21) which we read as "minimise both";
 * standard RL convention is to negate so the agent maximises.
 */
package org.workflowsim.scheduling;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Vm;
import org.workflowsim.CondorVM;
import org.workflowsim.WorkflowSimTags;

public class DQNSchedulingAlgorithm extends BaseSchedulingAlgorithm {

    private static final double P_ACTIVE = 200.0;
    private static final double P_IDLE = 100.0;
    private static final double ENERGY_WEIGHT = 0.5;

    private static DQNAgent agent;
    private static final Map<Integer, Double> vmElectricityCost = new HashMap<>();
    private static final Map<Integer, Double> vmPendingFinish = new HashMap<>();
    private static double maxElectricityCost = 1.0;
    private static double maxVmMips = 1.0;
    private static double globalMakespan = 0.0;
    private static double globalEnergy = 0.0;
    private static long totalDecisions = 0;

    public DQNSchedulingAlgorithm() {
        super();
    }

    public static synchronized DQNAgent getAgent() {
        if (agent == null) {
            agent = new DQNAgent(42L);
        }
        return agent;
    }

    /** Reset the singleton agent and per-VM bookkeeping. */
    public static synchronized void reset() {
        agent = null;
        vmElectricityCost.clear();
        vmPendingFinish.clear();
        maxElectricityCost = 1.0;
        maxVmMips = 1.0;
        globalMakespan = 0.0;
        globalEnergy = 0.0;
        totalDecisions = 0;
    }

    public static double getGlobalMakespan() {
        return globalMakespan;
    }

    public static double getGlobalEnergy() {
        return globalEnergy;
    }

    private void registerVms(List<? extends Vm> vms) {
        for (Vm v : vms) {
            int id = v.getId();
            if (!vmElectricityCost.containsKey(id)) {
                double cost = 1.0 + (id * 0.07) % 1.5;
                vmElectricityCost.put(id, cost);
                vmPendingFinish.put(id, 0.0);
                if (cost > maxElectricityCost) {
                    maxElectricityCost = cost;
                }
            }
            double mips = v.getMips() * v.getNumberOfPes();
            if (mips > maxVmMips) {
                maxVmMips = mips;
            }
        }
    }

    private double[] buildState(List<Cloudlet> ready, List<CondorVM> idle, double cloudletLen) {
        double avgTaskLen = 0;
        for (Cloudlet c : ready) {
            avgTaskLen += c.getCloudletLength();
        }
        avgTaskLen /= Math.max(1, ready.size());

        double avgVmPrio = 0;
        for (CondorVM v : idle) {
            avgVmPrio += maxElectricityCost / vmElectricityCost.get(v.getId());
        }
        avgVmPrio /= Math.max(1, idle.size());

        return new double[]{
            cloudletLen / Math.max(1.0, avgTaskLen),
            avgVmPrio,
            globalMakespan / 1000.0,
            globalEnergy / 10000.0
        };
    }

    private double[] buildActionFeatures(CondorVM vm, double cloudletLen) {
        double mips = vm.getMips() * vm.getNumberOfPes();
        double cost = vmElectricityCost.get(vm.getId());
        double pending = vmPendingFinish.get(vm.getId());
        return new double[]{
            maxElectricityCost / cost,
            mips / maxVmMips,
            1.0 / (1.0 + pending / 100.0)
        };
    }

    private List<double[]> buildActionList(List<CondorVM> idle, double cloudletLen) {
        List<double[]> out = new ArrayList<>(idle.size());
        for (CondorVM v : idle) {
            out.add(buildActionFeatures(v, cloudletLen));
        }
        return out;
    }

    private List<CondorVM> idleVms() {
        List<CondorVM> out = new ArrayList<>();
        for (Object o : getVmList()) {
            CondorVM v = (CondorVM) o;
            if (v.getState() == WorkflowSimTags.VM_STATUS_IDLE) {
                out.add(v);
            }
        }
        return out;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void run() {
        registerVms(getVmList());
        DQNAgent net = getAgent();

        List<Cloudlet> ready = new ArrayList<>(getCloudletList());

        for (Cloudlet cl : ready) {
            List<CondorVM> idle = idleVms();
            if (idle.isEmpty()) {
                break;
            }

            double cloudletLen = cl.getCloudletLength();
            double[] state = buildState(ready, idle, cloudletLen);
            List<double[]> actions = buildActionList(idle, cloudletLen);
            int chosen = net.selectAction(state, actions);
            CondorVM vm = idle.get(chosen);

            double mips = vm.getMips() * vm.getNumberOfPes();
            double execTime = cloudletLen / Math.max(1.0, mips);
            double pending = vmPendingFinish.get(vm.getId());
            double finishOnVm = pending + execTime;
            double energyInc = execTime * P_ACTIVE
                + Math.max(0, globalMakespan - finishOnVm) * P_IDLE / 1000.0;

            vmPendingFinish.put(vm.getId(), finishOnVm);
            if (finishOnVm > globalMakespan) {
                globalMakespan = finishOnVm;
            }
            globalEnergy += energyInc;
            totalDecisions++;

            double makespanNorm = execTime / 100.0;
            double energyNorm = energyInc / 10000.0;
            double reward = -makespanNorm - ENERGY_WEIGHT * energyNorm;
            reward += vmElectricityCost.get(vm.getId()) < maxElectricityCost ? 0.05 : -0.05;

            List<CondorVM> idleAfter = new ArrayList<>(idle);
            idleAfter.remove(chosen);
            double[] nextState = buildState(ready, idleAfter,
                cloudletLen);
            List<double[]> nextActions = buildActionList(idleAfter, cloudletLen);
            boolean done = idleAfter.isEmpty();

            net.remember(state, actions.get(chosen), reward, nextState, nextActions, done);
            net.trainStep();

            vm.setState(WorkflowSimTags.VM_STATUS_BUSY);
            cl.setVmId(vm.getId());
            getScheduledList().add(cl);
        }

        net.decayEpsilon();
    }
}
