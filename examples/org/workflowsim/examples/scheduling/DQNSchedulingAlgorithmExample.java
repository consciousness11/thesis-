/**
 * Runs the MOPWSDRL DQN scheduler from the paper
 *   Mangalampalli et al., "Multi Objective Prioritized Workflow Scheduling
 *   Using Deep Reinforcement Based Learning in Cloud Computing", IEEE Access
 *   2024.
 *
 * Iterates over the four scientific workflows used in the paper -- Montage,
 * CyberShake, Epigenomics, and LIGO (Inspiral) -- at the 100-task setting and
 * reports makespan and an estimated energy consumption per workflow.
 *
 * The Q-network learned from earlier workflows is intentionally carried over
 * to later runs (transfer learning), matching the paper's "100 iterations"
 * training scheme that runs across multiple workloads.
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
import org.workflowsim.scheduling.DQNSchedulingAlgorithm;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;

public class DQNSchedulingAlgorithmExample {

    private static final double P_ACTIVE_W = 200.0;
    private static final double P_IDLE_W = 100.0;

    public static void main(String[] args) {
        DQNSchedulingAlgorithm.reset();

        String[][] workflows = {
            {"Montage",     "config/dax/Montage_100.xml"},
            {"CyberShake",  "config/dax/CyberShake_100.xml"},
            {"Epigenomics", "config/dax/Epigenomics_100.xml"},
            {"LIGO",        "config/dax/Inspiral_100.xml"}
        };

        List<double[]> summary = new ArrayList<>();
        for (String[] wf : workflows) {
            double[] result = runOne(wf[0], wf[1]);
            if (result != null) {
                summary.add(new double[]{result[0], result[1], result[2]});
                Log.printLine(String.format("[%s] makespan=%.2f  energy=%.2f W*s  jobs=%d",
                        wf[0], result[0], result[1], (int) result[2]));
            }
            // suppress CloudSim chatter for subsequent runs
            Log.disable();
        }
        Log.enable();

        Log.printLine();
        Log.printLine("===== MOPWSDRL summary (100 tasks per workflow) =====");
        Log.printLine(String.format("%-14s %12s %12s %8s", "Workflow", "Makespan", "Energy(W*s)", "Jobs"));
        for (int i = 0; i < workflows.length && i < summary.size(); i++) {
            double[] r = summary.get(i);
            Log.printLine(String.format("%-14s %12.2f %12.2f %8d",
                    workflows[i][0], r[0], r[1], (int) r[2]));
        }
    }

    /** Returns {makespan, energy, jobsCompleted} or null on failure. */
    private static double[] runOne(String name, String daxPath) {
        File daxFile = new File(daxPath);
        if (!daxFile.exists()) {
            Log.printLine("Skipping " + name + " -- missing " + daxPath);
            return null;
        }
        try {
            int vmNum = 20;

            Parameters.SchedulingAlgorithm sch = Parameters.SchedulingAlgorithm.DQN;
            Parameters.PlanningAlgorithm pln = Parameters.PlanningAlgorithm.INVALID;
            ReplicaCatalog.FileSystem fs = ReplicaCatalog.FileSystem.SHARED;

            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
            ClusteringParameters cp = new ClusteringParameters(0, 0,
                    ClusteringParameters.ClusteringMethod.NONE, null);

            Parameters.init(vmNum, daxPath, null, null, op, cp, sch, pln, null, 0);
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
            double idleSeconds = Math.max(0, vmNum * makespan - activeSeconds);
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
