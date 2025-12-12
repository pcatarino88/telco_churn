import pandas as pd

def assign_customer_segment(df):
    """
    Assign customer segments based on Age, Dependents, and Tenure in Months.
    Segments:
    - New Customers: Tenure in Months <= 6
    - +65y: Age 65-80 (regardless of Dependents)
    - 0-64_Dep: Age 0-64 with Dependents
    - 0-64_NoDep: Age 0-64 without Dependents
    """

    # 1) Create AgeGroup with the new bins
    bins   = [0, 65, 81]
    labels = ["0-64", "65-80"]

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels,
        include_lowest=True,   
        right=False            # [0,35), [35,50), ...
    )

    # 2) Make Dependents a cleaner label
    df["DepFlag"] = df["Dependents"].map({"Yes": "Dep", "No": "NoDep"})

    # 3) Create the segment column: 5 age groups × 2 dep flags = 10 segments
    df["segment"] = df["AgeGroup"].astype(str) + "_" + df["DepFlag"]

    # 4) Override with "New Cust" segment when Tenure in Months <= 6
    mask_new = df["Tenure in Months"] <= 6
    df.loc[mask_new, "segment"] = "New Customer"

    # 5) Merge 65_80_Dep and 65_80_NoDep into '+65y'
    mask_senior = df["segment"].isin(["65-80_Dep", "65-80_NoDep"])
    df.loc[mask_senior, "segment"] = "+65y"

    # 6) Drop temporary columns
    df.drop(columns=["AgeGroup", "DepFlag"], inplace=True)

    return df