/**
 * Drives PPO + GAT training over the four scientific workflows from the paper
 * (Montage, CyberShake, Epigenomics, LIGO) at the 100-task setting.
 *
 * For each (workflow, episode) pair this example:
 *   1. Resets the PPO bookkeeping and re-initialises CloudSim.
 *   2. Calls {@code beginEpisode(name)} so the next OBSERVE/ACTION exchange
 *      starts a fresh trajectory in Python.
 *   3. Runs the CloudSim simulation; the {@link PPOSchedulingAlgorithm}
 *      contacts the Python sidecar over the TCP bridge for every decision.
 *   4. Calls {@code endEpisode(makespan, energy, jobs)} so the Python side
 *      can finalise the trajectory and update its PPO policy.
 *
 * Requires the Python sidecar to be listening before this main() runs.
 * Defaults to {@code 127.0.0.1:7777}; override with
 * {@code -Dppo.host=...} / {@code -Dppo.port=...}.  Episode count per workflow
 * is configurable via {@code -Dppo.episodes=10}.
 */
package org.workflowsim.examples.scheduling;

import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.HarddriveStorage;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.VmAllocationPolicySimple;
import org.cloudbus.cloudsim.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.workflowsim.CondorVM;
import org.workflowsim.Job;
import org.workflowsim.WorkflowDatacenter;
import org.workflowsim.WorkflowEngine;
import org.workflowsim.WorkflowPlanner;
import org.workflowsim.scheduling.PPOSchedulingAlgorithm;
import org.workflowsim.scheduling.RLBridge;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

public class PPOTrainerExample {

    private static final double P_ACTIVE_W = 200.0;
    private static final double P_IDLE_W = 100.0;
    private static final int VM_NUM = 20;

    public static void main(String[] args) {
        int episodesPerWorkflow = Integer.parseInt(System.getProperty("ppo.episodes", "10"));

        String[][] workflows = {
            {"Montage",     "config/dax/Montage_100.xml"},
            {"CyberShake",  "config/dax/CyberShake_100.xml"},
            {"Epigenomics", "config/dax/Epigenomics_100.xml"},
            {"LIGO",        "config/dax/Inspiral_100.xml"}
        };

        RLBridge bridge = RLBridge.get();
        if (!bridge.connect()) {
            Log.printLine("[trainer] Cannot reach Python sidecar; aborting.");
            return;
        }

        PPOSchedulingAlgorithm.resetAll();

        DecimalFormat df = new DecimalFormat("###0.00");
        for (int ep = 1; ep <= episodesPerWorkflow; ep++) {
            for (String[] wf : workflows) {
                double[] r = runEpisode(wf[0], wf[1]);
                if (r == null) {
                    continue;
                }
                Log.enable();
                Log.printLine(String.format("[ep %02d/%02d %-12s] makespan=%s energy=%s jobs=%d",
                        ep, episodesPerWorkflow, wf[0],
                        df.format(r[0]), df.format(r[1]), (int) r[2]));
                Log.disable();
            }
        }
        Log.enable();
        bridge.close();
        Log.printLine("[trainer] done.");
    }

    private static double[] runEpisode(String name, String daxPath) {
        File daxFile = new File(daxPath);
        if (!daxFile.exists()) {
            Log.enable();
            Log.printLine("[trainer] skipping " + name + " -- missing " + daxPath);
            Log.disable();
            return null;
        }
        try {
            Parameters.SchedulingAlgorithm sch = Parameters.SchedulingAlgorithm.PPO;
            Parameters.PlanningAlgorithm pln = Parameters.PlanningAlgorithm.INVALID;
            ReplicaCatalog.FileSystem fs = ReplicaCatalog.FileSystem.SHARED;

            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
            ClusteringParameters cp = new ClusteringParameters(0, 0,
                    ClusteringParameters.ClusteringMethod.NONE, null);

            Parameters.init(VM_NUM, daxPath, null, null, op, cp, sch, pln, null, 0);
            ReplicaCatalog.init(fs);

            CloudSim.init(1, Calendar.getInstance(), false);

            PPOSchedulingAlgorithm.beginEpisode(name);

            WorkflowDatacenter dc = createDatacenter("DC_" + name);
            WorkflowPlanner planner = new WorkflowPlanner("planner_" + name, 1);
            WorkflowEngine engine = planner.getWorkflowEngine();
            List<CondorVM> vms = createVMs(engine.getSchedulerId(0), Parameters.getVmNum());
            engine.submitVmList(vms, 0);
            engine.bindSchedulerDatacenter(dc.getId(), 0);

            Log.disable();
            CloudSim.startSimulation();
            List<Job> finished = engine.getJobsReceivedList();
            CloudSim.stopSimulation();

            double makespan = 0;
            double activeSeconds = 0;
            int jobs = 0;
            for (Job j : finished) {
                if (j.getCloudletStatus() == Cloudlet.SUCCESS) {
                    if (j.getFinishTime() > makespan) {
                        makespan = j.getFinishTime();
                    }
                    activeSeconds += j.getActualCPUTime();
                    jobs++;
                }
            }
            double idleSeconds = Math.max(0, VM_NUM * makespan - activeSeconds);
            double energy = activeSeconds * P_ACTIVE_W + idleSeconds * P_IDLE_W;

            PPOSchedulingAlgorithm.endEpisode(makespan, energy, jobs);
            return new double[]{makespan, energy, jobs};
        } catch (Exception e) {
            Log.enable();
            Log.printLine("[trainer] " + name + " episode failed: " + e.getMessage());
            return null;
        }
    }

    private static List<CondorVM> createVMs(int userId, int vms) {
        LinkedList<CondorVM> list = new LinkedList<>();
        long size = 10000;
        int ram = 512;
        int mips = 1000;
        long bw = 1000;
        int pesNumber = 1;
        String vmm = "Xen";
        for (int i = 0; i < vms; i++) {
            list.add(new CondorVM(i, userId, mips, pesNumber, ram, bw, size, vmm,
                    new CloudletSchedulerSpaceShared()));
        }
        return list;
    }

    private static WorkflowDatacenter createDatacenter(String name) {
        List<Host> hostList = new ArrayList<>();
        for (int i = 1; i <= 20; i++) {
            List<Pe> peList = new ArrayList<>();
            int mips = 2000;
            peList.add(new Pe(0, new PeProvisionerSimple(mips)));
            peList.add(new Pe(1, new PeProvisionerSimple(mips)));
            int ram = 2048;
            long storage = 1000000;
            int bw = 10000;
            hostList.add(new Host(0,
                    new RamProvisionerSimple(ram),
                    new BwProvisionerSimple(bw),
                    storage, peList,
                    new VmSchedulerTimeShared(peList)));
        }
        DatacenterCharacteristics ch = new DatacenterCharacteristics(
                "x86", "Linux", "Xen", hostList, 10.0, 3.0, 0.05, 0.1, 0.1);
        LinkedList<Storage> storageList = new LinkedList<>();
        try {
            HarddriveStorage s = new HarddriveStorage(name, 1e12);
            s.setMaxTransferRate(15);
            storageList.add(s);
            return new WorkflowDatacenter(name, ch,
                    new VmAllocationPolicySimple(hostList), storageList, 0);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
