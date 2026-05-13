/**
 * Rainbow-lite DQN scheduler over the same four scientific workflows used by
 * the DQN / PPO / HEFT examples (Montage, CyberShake, Epigenomics, LIGO at
 * the 1000-task setting). Same 20-VM heterogeneous datacenter and the same
 * energy formula so makespan / energy are directly comparable.
 *
 * Between workflows we call {@link RainbowSchedulingAlgorithm#endEpisode()}
 * so the agent's n-step ring buffer is flushed and the next workflow does
 * not inherit truncated partial trajectories.
 */
package org.workflowsim.examples.scheduling;

import java.io.File;
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
import org.workflowsim.scheduling.RainbowSchedulingAlgorithm;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

public class RainbowSchedulingAlgorithmExample {

    private static final double P_ACTIVE_W = 200.0;
    private static final double P_IDLE_W = 100.0;
    private static final int VM_NUM = 20;

    public static void main(String[] args) {
        RainbowSchedulingAlgorithm.reset();

        String[][] workflows = {
            {"Montage",     "config/dax/Montage_1000.xml"},
            {"CyberShake",  "config/dax/CyberShake_1000.xml"},
            {"Epigenomics", "config/dax/Epigenomics_997.xml"},
            {"LIGO",        "config/dax/Inspiral_1000.xml"}
        };

        List<double[]> summary = new ArrayList<>();
        for (String[] wf : workflows) {
            double[] result = runOne(wf[0], wf[1]);
            if (result != null) {
                summary.add(result);
                Log.enable();
                Log.printLine(String.format("[%s] makespan=%.2f  energy=%.2f W*s  jobs=%d",
                        wf[0], result[0], result[1], (int) result[2]));
            }
            RainbowSchedulingAlgorithm.endEpisode();
            Log.disable();
        }
        Log.enable();
        Log.printLine();
        Log.printLine("===== Rainbow-lite DQN summary (1000-task workflows) =====");
        Log.printLine(String.format("%-14s %12s %12s %8s", "Workflow", "Makespan", "Energy(W*s)", "Jobs"));
        for (int i = 0; i < workflows.length && i < summary.size(); i++) {
            double[] r = summary.get(i);
            Log.printLine(String.format("%-14s %12.2f %12.2f %8d",
                    workflows[i][0], r[0], r[1], (int) r[2]));
        }
    }

    private static double[] runOne(String name, String daxPath) {
        File daxFile = new File(daxPath);
        if (!daxFile.exists()) {
            Log.printLine("Skipping " + name + " -- missing " + daxPath);
            return null;
        }
        try {
            Parameters.SchedulingAlgorithm sch = Parameters.SchedulingAlgorithm.RAINBOW;
            Parameters.PlanningAlgorithm pln = Parameters.PlanningAlgorithm.INVALID;
            ReplicaCatalog.FileSystem fs = ReplicaCatalog.FileSystem.SHARED;

            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
            ClusteringParameters cp = new ClusteringParameters(0, 0,
                    ClusteringParameters.ClusteringMethod.NONE, null);

            Parameters.init(VM_NUM, daxPath, null, null, op, cp, sch, pln, null, 0);
            ReplicaCatalog.init(fs);

            CloudSim.init(1, Calendar.getInstance(), false);

            WorkflowDatacenter dc = createDatacenter("Datacenter_" + name);
            WorkflowPlanner planner = new WorkflowPlanner("planner_" + name, 1);
            WorkflowEngine engine = planner.getWorkflowEngine();
            List<CondorVM> vms = createVMs(engine.getSchedulerId(0), Parameters.getVmNum());
            engine.submitVmList(vms, 0);
            engine.bindSchedulerDatacenter(dc.getId(), 0);

            CloudSim.startSimulation();
            List<Job> finished = engine.getJobsReceivedList();
            CloudSim.stopSimulation();

            double makespan = 0.0;
            double activeSeconds = 0.0;
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

            return new double[]{makespan, energy, jobs};
        } catch (Exception e) {
            Log.printLine("Run for " + name + " failed: " + e.getMessage());
            return null;
        }
    }

    private static List<CondorVM> createVMs(int userId, int vms) {
        LinkedList<CondorVM> list = new LinkedList<>();
        long size = 10000;
        int ram = 512;
        long bw = 1000;
        int pesNumber = 1;
        String vmm = "Xen";
        // Heterogeneous mips: 500 / 1000 / 1500 / 2000 cycling across VMs.
        for (int i = 0; i < vms; i++) {
            int mips = 500 + (i % 4) * 500;
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
