/**
 * PPO + GAT workflow scheduler that delegates VM-selection decisions to a
 * Python sidecar (PyTorch + Stable-Baselines3) reached over a TCP socket via
 * {@link RLBridge}.
 *
 * Episode flow:
 *   1. Trainer calls {@link #beginEpisode(String)} before
 *      {@code CloudSim.startSimulation()} to reset bookkeeping.
 *   2. On the first {@link #run()} call inside an episode, this scheduler
 *      lazily captures the workflow DAG from the cloudlet list and emits
 *      {@code EPISODE_START} to Python.
 *   3. For each ready cloudlet (FIFO over the current ready queue) the
 *      scheduler emits {@code OBSERVE}, receives an {@code ACTION} naming a
 *      VM, applies the assignment, and emits {@code STEP_RESULT}.
 *   4. Trainer calls {@link #endEpisode(double, double, int)} after
 *      {@code CloudSim.stopSimulation()} to emit {@code EPISODE_END}.
 *
 * Python only chooses a VM each step (single {@code Discrete(MAX_VMS)} action
 * masked by the set of idle VM ids).  Java decides task ordering (FIFO over
 * the ready queue) so the action space stays flat.
 *
 * Resilience: if the bridge isn't connected or any I/O fails, this scheduler
 * silently falls back to MIN-MIN on the current call so the simulation never
 * deadlocks while the Python side is being debugged.
 */
package org.workflowsim.scheduling;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Vm;
import org.workflowsim.CondorVM;
import org.workflowsim.Job;
import org.workflowsim.Task;
import org.workflowsim.WorkflowSimTags;

public class PPOSchedulingAlgorithm extends BaseSchedulingAlgorithm {

    private static final double P_ACTIVE = 200.0;
    private static final double P_IDLE = 100.0;
    private static final double TASK_LEN_SCALE = 100.0;
    private static final double ENERGY_SCALE = 10000.0;

    private static String currentWorkflow = "";
    private static boolean episodeStarted = false;
    private static boolean dagSent = false;

    private static final Map<Integer, Double> vmElectricityCost = new HashMap<>();
    private static final Map<Integer, Double> vmPendingFinish = new HashMap<>();
    private static double maxElectricityCost = 1.0;
    private static double globalMakespan = 0.0;
    private static double globalEnergy = 0.0;

    public PPOSchedulingAlgorithm() {
        super();
    }

    /** Called by the trainer before each CloudSim.startSimulation(). */
    public static synchronized void beginEpisode(String workflowName) {
        currentWorkflow = workflowName;
        episodeStarted = true;
        dagSent = false;
        vmPendingFinish.clear();
        globalMakespan = 0.0;
        globalEnergy = 0.0;
    }

    /** Called by the trainer after each CloudSim.stopSimulation(). */
    public static synchronized void endEpisode(double makespan, double energy, int jobsCompleted) {
        RLBridge bridge = RLBridge.get();
        if (!bridge.isHealthy()) {
            return;
        }
        try {
            double terminalReward = -makespan / 100.0 - 0.5 * energy / 10000.0;
            Map<String, Object> frame = new LinkedHashMap<>();
            frame.put("type", "EPISODE_END");
            frame.put("makespan", makespan);
            frame.put("energy", energy);
            frame.put("jobs_completed", jobsCompleted);
            frame.put("terminal_reward", terminalReward);
            bridge.send(frame);
        } catch (IOException e) {
            Log.printLine("[PPO] EPISODE_END send failed: " + e.getMessage());
        }
        episodeStarted = false;
    }

    public static synchronized void resetAll() {
        currentWorkflow = "";
        episodeStarted = false;
        dagSent = false;
        vmElectricityCost.clear();
        vmPendingFinish.clear();
        maxElectricityCost = 1.0;
        globalMakespan = 0.0;
        globalEnergy = 0.0;
    }

    private void registerVms(List<? extends Vm> vms) {
        for (Vm v : vms) {
            int id = v.getId();
            if (!vmElectricityCost.containsKey(id)) {
                double cost = 1.0 + (id * 0.07) % 1.5;
                vmElectricityCost.put(id, cost);
                if (cost > maxElectricityCost) {
                    maxElectricityCost = cost;
                }
            }
            vmPendingFinish.putIfAbsent(id, 0.0);
        }
    }

    private Map<String, Object> buildDagPayload() {
        List<Map<String, Object>> nodes = new ArrayList<>();
        List<int[]> edges = new ArrayList<>();
        for (Object o : getCloudletList()) {
            if (!(o instanceof Job)) {
                continue;
            }
            Job job = (Job) o;
            Map<String, Object> node = new LinkedHashMap<>();
            node.put("id", job.getCloudletId());
            node.put("length", job.getCloudletLength());
            node.put("depth", job.getDepth());
            nodes.add(node);
            for (Object p : job.getParentList()) {
                Task parent = (Task) p;
                edges.add(new int[]{parent.getCloudletId(), job.getCloudletId()});
            }
        }
        Map<String, Object> dag = new LinkedHashMap<>();
        dag.put("nodes", nodes);
        List<Object> edgeList = new ArrayList<>();
        for (int[] e : edges) {
            edgeList.add(e);
        }
        dag.put("edges", edgeList);
        return dag;
    }

    private List<Map<String, Object>> buildVmPayload(List<? extends Vm> vms) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Vm v : vms) {
            Map<String, Object> vmInfo = new LinkedHashMap<>();
            vmInfo.put("id", v.getId());
            vmInfo.put("mips", v.getMips());
            vmInfo.put("pes", v.getNumberOfPes());
            vmInfo.put("elec_cost", vmElectricityCost.get(v.getId()));
            out.add(vmInfo);
        }
        return out;
    }

    private boolean ensureEpisodeStarted(RLBridge bridge) {
        if (!episodeStarted || dagSent) {
            return true;
        }
        try {
            Map<String, Object> frame = new LinkedHashMap<>();
            frame.put("type", "EPISODE_START");
            frame.put("workflow", currentWorkflow);
            frame.put("num_tasks", getCloudletList().size());
            frame.put("dag", buildDagPayload());
            frame.put("vms", buildVmPayload(getVmList()));
            bridge.send(frame);
            dagSent = true;
            return true;
        } catch (IOException e) {
            Log.printLine("[PPO] EPISODE_START send failed: " + e.getMessage());
            return false;
        }
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
        RLBridge bridge = RLBridge.get();
        boolean useBridge = bridge.connect() && ensureEpisodeStarted(bridge);

        List<Cloudlet> ready = new ArrayList<>(getCloudletList());

        for (Cloudlet cl : ready) {
            List<CondorVM> idle = idleVms();
            if (idle.isEmpty()) {
                break;
            }
            CondorVM chosen = null;
            if (useBridge) {
                chosen = chooseViaBridge(bridge, cl, idle);
                if (chosen == null) {
                    useBridge = false;
                }
            }
            if (chosen == null) {
                chosen = minMinFallback(cl, idle);
            }
            applyAssignment(cl, chosen, bridge, useBridge);
        }
    }

    private CondorVM chooseViaBridge(RLBridge bridge, Cloudlet cl, List<CondorVM> idle) {
        try {
            Map<String, Object> obs = new LinkedHashMap<>();
            obs.put("type", "OBSERVE");
            obs.put("task_id", cl.getCloudletId());

            Map<String, Object> taskFeatures = new LinkedHashMap<>();
            taskFeatures.put("length", cl.getCloudletLength());
            int depth = 0;
            int parentCount = 0;
            int childCount = 0;
            if (cl instanceof Job) {
                Job job = (Job) cl;
                depth = job.getDepth();
                parentCount = job.getParentList().size();
                childCount = job.getChildList().size();
            }
            taskFeatures.put("depth", depth);
            taskFeatures.put("parent_count", parentCount);
            taskFeatures.put("child_count", childCount);
            obs.put("task_features", taskFeatures);

            int[] idleIds = new int[idle.size()];
            List<Object> vmFeatures = new ArrayList<>();
            for (int i = 0; i < idle.size(); i++) {
                CondorVM v = idle.get(i);
                idleIds[i] = v.getId();
                double[] f = new double[]{
                    v.getMips() * v.getNumberOfPes(),
                    vmElectricityCost.get(v.getId()),
                    vmPendingFinish.get(v.getId())
                };
                vmFeatures.add(f);
            }
            obs.put("idle_vm_ids", idleIds);
            obs.put("vm_features", vmFeatures);
            obs.put("makespan_so_far", globalMakespan);
            obs.put("energy_so_far", globalEnergy);

            bridge.send(obs);
            Map<String, Object> action = bridge.recv();
            int vmId = ((Number) action.get("vm_id")).intValue();
            for (CondorVM v : idle) {
                if (v.getId() == vmId) {
                    return v;
                }
            }
            Log.printLine("[PPO] Python returned vm_id " + vmId
                + " not in idle set; falling back to MIN-MIN");
            return null;
        } catch (IOException e) {
            Log.printLine("[PPO] OBSERVE/ACTION exchange failed: " + e.getMessage());
            return null;
        }
    }

    private CondorVM minMinFallback(Cloudlet cl, List<CondorVM> idle) {
        CondorVM best = idle.get(0);
        double bestMips = best.getMips() * best.getNumberOfPes();
        for (CondorVM v : idle) {
            double mips = v.getMips() * v.getNumberOfPes();
            if (mips > bestMips) {
                bestMips = mips;
                best = v;
            }
        }
        return best;
    }

    private void applyAssignment(Cloudlet cl, CondorVM vm, RLBridge bridge, boolean useBridge) {
        double mips = vm.getMips() * vm.getNumberOfPes();
        double execTime = cl.getCloudletLength() / Math.max(1.0, mips);
        double pending = vmPendingFinish.get(vm.getId());
        double finishOnVm = pending + execTime;
        double energyInc = execTime * P_ACTIVE
            + Math.max(0, globalMakespan - finishOnVm) * P_IDLE / 1000.0;

        vmPendingFinish.put(vm.getId(), finishOnVm);
        double makespanInc = Math.max(0, finishOnVm - globalMakespan);
        if (finishOnVm > globalMakespan) {
            globalMakespan = finishOnVm;
        }
        globalEnergy += energyInc;

        if (useBridge) {
            double reward = -(makespanInc / TASK_LEN_SCALE)
                - 0.5 * (energyInc / ENERGY_SCALE);
            try {
                Map<String, Object> sr = new LinkedHashMap<>();
                sr.put("type", "STEP_RESULT");
                sr.put("reward", reward);
                sr.put("done", false);
                bridge.send(sr);
            } catch (IOException e) {
                Log.printLine("[PPO] STEP_RESULT send failed: " + e.getMessage());
            }
        }

        vm.setState(WorkflowSimTags.VM_STATUS_BUSY);
        cl.setVmId(vm.getId());
        getScheduledList().add(cl);
    }
}
