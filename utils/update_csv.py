import pandas as pd

def update_csv(info, ep, output_path):
    """Updates the CSV files with the given information."""
    mode = "w" if not info["step"] else "a"
    for stat in ["state", "rewards"]:
        if stat == "state":
            for key in info[stat].keys():
                info[stat][key] = info[stat][key].tolist()
        info[stat]["ep"] = ep
        df = pd.DataFrame([info[stat]])
        df.to_csv(output_path + f"{stat}.csv", index=False, mode=mode, header=not bool(info["step"]))
        if stat == "rewards":
            df["rewards"] = df[["1", "2", "5", "6"]].mean(axis=1)
            df.drop(columns=["1", "2", "5", "6"], inplace=True)
            df.to_csv(
                output_path + f"comb_rewards.csv", index=False, mode=mode, header=not bool(info["step"])
            )
        del info[stat]["ep"]

