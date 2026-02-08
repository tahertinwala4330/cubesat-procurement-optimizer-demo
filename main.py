# ==============================
# CubeSat Procurement Optimizer
# ==============================

import os
import pandas as pd
from pulp import *

# ------------------------------
# PATH SETUP
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------
# LOAD DATA
# ------------------------------
def load_data():

    bom = pd.read_csv(os.path.join(DATA_DIR, "BOM.csv"))
    suppliers = pd.read_csv(os.path.join(DATA_DIR, "Suppliers.csv"))
    program = pd.read_csv(os.path.join(DATA_DIR, "Program.csv"))

    # Strip whitespace
    bom.columns = bom.columns.str.strip()
    suppliers.columns = suppliers.columns.str.strip()
    program.columns = program.columns.str.strip()

    return bom, suppliers, program

# ------------------------------
# PREPROCESS DATA
# ------------------------------
def preprocess_data(bom, suppliers, program):

    num_sats = program.loc[0, "Num_Satellites"]
    assembly_start = program.loc[0, "Assembly_Start_Day"]

    # Rename BOM columns
    bom = bom.rename(columns={
        "Component": "component",
        "Subsystem": "subsystem",
        "Qty/Sat": "qty_per_sat"
    })

    bom["total_demand"] = bom["qty_per_sat"] * num_sats
    demand_dict = dict(zip(bom.component, bom.total_demand))

    # Rename supplier columns
    suppliers = suppliers.rename(columns={
        "Components": "component",
        "Suppliers": "supplier",
        "Unit_Cost": "unit_cost",
        "Lead_Time_Days": "lead_time_days",
        "MOQ": "moq"
    })

    # Lead time feasibility
    suppliers = suppliers[
        suppliers["lead_time_days"] <= assembly_start
    ]

    if suppliers.empty:
        raise ValueError("No feasible suppliers available.")

    return demand_dict, suppliers

# ------------------------------
# BUILD MODEL
# ------------------------------
def build_model(demand_dict, suppliers):

    model = LpProblem(
        "CubeSat_Procurement_Optimization",
        LpMinimize
    )

    x, y = {}, {}

    for _, row in suppliers.iterrows():

        key = (row.component, row.supplier)

        x[key] = LpVariable(
            f"x_{row.component}_{row.supplier}",
            lowBound=0
        )

        y[key] = LpVariable(
            f"y_{row.component}_{row.supplier}",
            cat="Binary"
        )

    # Objective
    model += lpSum(
        row.unit_cost * x[(row.component, row.supplier)]
        for _, row in suppliers.iterrows()
    )

    # Demand constraints
    for comp in demand_dict:

        model += lpSum(
            x[(row.component, row.supplier)]
            for _, row in suppliers.iterrows()
            if row.component == comp
        ) >= demand_dict[comp]

    # MOQ constraints
    BigM = 10000

    for _, row in suppliers.iterrows():

        key = (row.component, row.supplier)

        model += x[key] >= row.moq * y[key]
        model += x[key] <= BigM * y[key]

    return model, x

# ------------------------------
# SOLVE & EXPORT
# ------------------------------
def solve_and_export(model, x):

    solver = PULP_CBC_CMD(msg=0)
    model.solve(solver)

    print("Solver Status:", LpStatus[model.status])
    print("Total Cost:", value(model.objective))

    results = []

    for (comp, supp), var in x.items():

        if var.varValue > 0:

            qty = var.varValue
            results.append([comp, supp, qty])

            print(f"{comp} → {supp} : {qty}")

    df = pd.DataFrame(
        results,
        columns=["Component", "Supplier", "OrderQty"]
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "procurement_plan.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nSaved to:", output_path)

# ------------------------------
# MAIN EXECUTION
# ------------------------------
def main():

    bom, suppliers, program = load_data()
    demand_dict, suppliers = preprocess_data(
        bom, suppliers, program
    )
    model, x = build_model(demand_dict, suppliers)
    solve_and_export(model, x)

if __name__ == "__main__":
    main()
