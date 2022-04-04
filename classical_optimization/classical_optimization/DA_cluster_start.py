from tqdm import tqdm

"""
python dispatch_jobs.py
"""
from classical_optimization.terra.utils import write_graph
import networkx as nx
import numpy as np
from subprocess import call
import sys
num_processors = 16
SLURM = f"""#!/bin/bash
# This is a sample slurm job script for the JILA cluster
# Edit below as required, but delete any lines you do not need
# email unix@jila.colorado.edu with questions, or come see us
#
# Keep the # in front of the SBATCH lines or they won't work!  

# A name for this job
#SBATCH -J QAOA_graphs

# The partition (queue) to run on: Most jobs should specify both
# jila and nistq.  However, long-running jobs are eligible to run
# on nistq only if the user is a member of a NIST quantum theory group
# (i.e., Rey, Holland, or Bohn)
#SBATCH -p jila

# QOS determines the job priority and permitted maximum duration.
# Jobs up to 7 days long should use 'standard', otherwise 'long'
# 'priority' is available only by special request by a PI
#SBATCH -q standard

# Number of processor cores required, e.g., 4
#SBATCH -n {num_processors}

# Number of nodes to run on.  The JILA Cluster is not suitable for
# spanning nodes (Summit, XSEDE, etc are good places to run these jobs)
# so leave this set to 1
#SBATCH -N 1

# How much RAM the job will require.  Make sure this is large enough, 
# or the job will likely fail.  If you're not sure, estimate high.
#
# The cluster is designed for roughly 7.5GiB/CPU, making that a fine guess.
#SBATCH --mem=8G

# Maximum expected wall time this job will require
# Format is DD-HH:MM:SS, this one will end in 15 seconds
#SBATCH -t 00-02:0:00

# Scratch disk is storage local to a compute node and required for high
# i/o jobs using temporary files (typically quantum chemistry jobs).  
# Some nodes have limited scratch space; failure to request an appropriate
# allocation may result in job or node failure and unhappy sysadmins.
#
# Many jobs will not require this space at all.
#SBATCH --tmp=1GB

# Request email about this job: options include BEGIN, END, FAIL, ALL or NONE
# Additionally, TIME_LIMIT_50, 80 or 90 will send email when the job reaches
# 50%, 80% or 90% of the job walltime limit
# Multiple options can be separated with a comma
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

# Lines after this will be executed as if typed on a command line.
# This script is executed from the directory you submitted your job from;
# unlike the old cluster, there is no need for "cd $PBS_O_WORKDIR"

# You should make sure the appropriate environment module is loaded
# for the software you want to use: this is the "module load" command.
# Replace matlab with the package you'll use.

# module load matlab

# The following example runs a MATLAB program stored in example.m
# Replace this with commands to run your job. 
"""

shots_per_points = [1, 10, 100, 1000]
maxiters = [1, 10, 100, 1000]
initial_temps = [1, 10, 100, 1000]
restart_temp_ratios = [1E-10, 1E-9, 1E-8, 1E-7]

for shots_per_point in shots_per_points:
    for maxiter in  maxiters:
        for initial_temp in initial_temps:
            for restart_temp_ratio in restart_temp_ratios:
                cmd = f"python ~/repos/qaoa/classical_optimization/classical_optimization/DA_cluster_start.py" \
                      f" {shots_per_point} {maxiter} {initial_temp} {restart_temp_ratio}"
                print("Dispatching...")
                with open('scratch.txt', 'w') as filehandle:
                    filehandle.write(SLURM + "\n" + cmd)
                call('sbatch scratch.txt', shell=True)
