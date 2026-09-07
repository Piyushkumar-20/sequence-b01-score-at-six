import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "scorecards"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINTS_FILE = PROCESSED_DIR / "checkpoints.csv"


def load_checkpoints():
    """Load the generated checkpoints dataset."""
    return pd.read_csv(CHECKPOINTS_FILE)


def build_checkpoints():
    """Build over-6 checkpoints from all scorecard JSON files."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = list(RAW_DIR.glob("*.json"))

    print(f"Found {len(files)} JSON scorecards.")

    rows = []

    for number, path in enumerate(files, start=1):

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        score = data["doc"][0]["data"]["score"]

        match_id = score["matchId"]

        innings_list = score["innings"]
        worm = score["wormAndManhattan"]

        # Find over 6 directly.
        over_6 = next(
            item
            for item in worm
            if int(item["overNumber"]) == 6
        )

        for innings in innings_list:

            innings_number = innings["inningsNumber"]

            innings_key = f"{innings_number}Innings"

            worm_value = over_6[innings_key]

            # Skip innings without data.
            if not worm_value:
                continue

            values = worm_value.split(",")

            runs_so_far = int(values[2])
            wickets_down = int(values[3])

            current_run_rate = runs_so_far / 6

            final_score = innings["runs"]

            batting_team = innings["teamName"]

            rows.append(
                {
                    "match_id": match_id,
                    "innings": innings_number,
                    "over_mark": 6,
                    "runs_so_far": runs_so_far,
                    "wickets_down": wickets_down,
                    "current_run_rate": current_run_rate,
                    "balls_since_boundary": 0,
                    "final_score": final_score,
                    "batting_team": batting_team,
                }
            )

        if number % 50 == 0:
            print(f"Processed {number}/{len(files)} files.")

    columns = [
        "match_id",
        "innings",
        "over_mark",
        "runs_so_far",
        "wickets_down",
        "current_run_rate",
        "balls_since_boundary",
        "final_score",
        "batting_team",
    ]

    df = pd.DataFrame(rows, columns=columns)

    df = df.drop_duplicates(
        subset=["match_id", "innings", "over_mark"]
    )

    df.to_csv(
        CHECKPOINTS_FILE,
        index=False
    )

    print()
    print("Checkpoint generation complete.")
    print(f"Rows: {len(df)}")
    print(f"Saved to: {CHECKPOINTS_FILE}")

    print()
    print("Dataset preview:")
    print(df.head(10).to_string(index=False))

    print()
    print("Shape:")
    print(df.shape)

    print()
    print("Missing values:")
    print(df.isna().sum())

    return df


if __name__ == "__main__":
    build_checkpoints()