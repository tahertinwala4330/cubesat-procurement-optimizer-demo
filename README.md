# cubesat-procurement-optimizer-demo
This project implements a procurement planning model for CubeSat components considering supplier lead times, minimum order quantities (MOQ), and unit costs.
Problem Scope: The planner computes a cost-minimizing sourcing plan that satisfies constellation assembly demand while ensuring delivery feasibility.

Data Modeling:Input data is structured into three CSV datasets:
BOM.csv → component demand per satellite
Suppliers.csv → supplier cost, lead time, MOQ
Program.csv → constellation size & assembly date

Optimization Approach: The procurement problem is formulated as a Mixed-Integer Linear Program (MILP):
Continuous variables → order quantities
Binary variables → supplier selection
MOQ enforced via linking constraints

Solver: Implemented in Python using PuLP with the CBC MILP solver.
Handling Uncertainty (Future Extensions)

Potential extensions include:
Lead-time buffers
Supplier failure probability
Dual sourcing strategies
Safety stock modeling
