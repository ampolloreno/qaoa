The purpose of this document is to specify near to intermediate term goals for the classical optimization portion of this project, with an approximate timeline.

The most near-term goal has been accomplished - the demonstration a small (2) qubit example of non-weighted MAXCUT QAOA using qiskit, with the ColdQuanta Noise model, in a Jupyter notebook. Additionally, use the built-in scipy optimizers to see that some subset of them can find the optimal cut. This functionality is being moved into the python module classical_optimization, with unit tests. So the first goal is:

[] Move functions from jupyter notebook into python module.  

Following this, there are (at least) five lines of work to pursue. 

[] Investigate other models not included in scipy. Most existing literature has likely explored these optimizers, so if we want _novel_ optimization, this is necessary. Two routes to investigate are Google's recent Model Gradient Ascent, with code provided in Cirq, and Evolution Strategies, to see what we can get from using techniques in machine learning.

[] Identify the relevant metrics to optimize. In particular, existing works suggest that calls to the QPU are expensive. This is particularly true in the cloud model (cite Harrigan et al. and Karalekas et al.), however this is _not necessarily_ relevant if the computation is onsite, or if the jobs are submitted in a concatenated fashion, where per-circuit overhead can be minimized. Additionally, we might consider local search around returned bitstrings like Peter suggested.

[] Fixing resources (like evaluations on the QPU), construct the Pareto front of p versus qubit number for current simulator. Then with Phase I deliverables in mind, we can (concretely) identify for what instances the Argonne simulator will be useful/necessary.

[] Related to the previous goal, make sure to write the code so that we can swap the qiskit simulator for the Argonne Simulator.

[] QAOA statistics. Notice that we aren't actually interested in the expectation of the cost function. We are interested in the sampling distribution of the MAX cut value. There might be something interesting to say about the statistics we optimize with respect to, and their sampling distribution.

[] Build sensible serialization framework for storing results of running QAOA circuits.

[] What does introducing correlations between gamma and beta do?

[] What depth do we need to take? (i.e. unweighted maxcut cannot be solved with p=1)

[] RQAOA

[] We need to figure out how to pick intial \gamma, \beta.

