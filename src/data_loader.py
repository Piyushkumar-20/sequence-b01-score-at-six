import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_SCORECARDS_DIR = PROJECT_ROOT / "data" / "raw" / "scorecards"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINTS_PATH = PROCESSED_DIR / "checkpoints.csv"


def load_scorecard(path: Path) -> dict:
    """Load one JSON scorecard."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_score_data(scorecard: dict) -> dict:
    """Get score data from the scorecard."""
    return scorecard["doc"][0]["data"]["score"]


def parse_worm_entry(value: str) -> tuple[int, int]:
    """
    Parse a wormAndManhattan value.

    Example:
        '17,0,58,0'

    Format:
        over_runs,
        wickets_in_over,
        cumulative_runs,
        cumulative_wickets
    """

    parts = str(value).split(",")

    if len(parts) != 4:
        raise ValueError(
            f"Unexpected worm entry: {value}"
        )

    runs_so_far = int(parts[2])
    wickets_down = int(parts[3])

    return runs_so_far, wickets_down


def calculate_balls_since_boundary(
    ball_summaries: list,
    innings_number: int,
    checkpoint_over: int = 6,
) -> int:
    """
    Calculate balls since the most recent 4 or 6
    up to the end of the checkpoint over.

    The scorecard stores entries such as:
        '0,1,1,1,1,4'
    """

    innings_key = f"{innings_number}Innings"

    balls = []

    for over in ball_summaries:

        if not isinstance(over, dict):
            continue

        over_number = over.get("overNumber")

        if over_number is None:
            continue

        try:
            over_number = int(over_number)
        except (TypeError, ValueError):
            continue

        if over_number > checkpoint_over:
            continue

        ball_string = over.get(
            innings_key,
            ""
        )

        if not ball_string:
            continue

        for ball in str(ball_string).split(","):

            ball = ball.strip()

            if ball:
                balls.append(ball)

    if not balls:
        return 0

    count = 0

    for ball in reversed(balls):

        token = ball.lower().strip()

        if token in {"4", "6"}:
            return count

        count += 1

    return count


def extract_checkpoints(scorecard: dict) -> list[dict]:
    """Extract over-6 checkpoints from one scorecard."""

    score = get_score_data(scorecard)

    match_id = score.get("matchId")

    innings_list = score.get(
        "innings",
        []
    )

    worm_data = score.get(
        "wormAndManhattan",
        []
    )

    ball_summaries = score.get(
        "ballByBallSummaries",
        []
    )

    rows = []

    # Find over 6.
    over_6 = None

    for worm_over in worm_data:

        try:
            over_number = int(
                worm_over.get("overNumber")
            )
        except (TypeError, ValueError):
            continue

        if over_number == 6:
            over_6 = worm_over
            break

    if over_6 is None:
        return rows

    # Process innings.
    for innings in innings_list:

        innings_number = innings.get(
            "inningsNumber"
        )

        if innings_number not in {
            1,
            2,
            3,
            4,
        }:
            continue

        innings_key = {
            1: "firstInnings",
            2: "secondInnings",
            3: "thirdInnings",
            4: "fourthInnings",
        }[innings_number]

        worm_value = over_6.get(
            innings_key,
            ""
        )

        if not worm_value:
            continue

        try:
            runs_so_far, wickets_down = (
                parse_worm_entry(worm_value)
            )
        except ValueError:
            continue

        current_run_rate = (
            runs_so_far / 6
        )

        balls_since_boundary = (
            calculate_balls_since_boundary(
                ball_summaries,
                innings_number,
                6,
            )
        )

        final_score = innings.get(
            "runs"
        )

        if final_score is None:
            continue

        batting_team = innings.get(
            "teamName"
        )

        if not batting_team:
            continue

        rows.append(
            {
                "match_id": match_id,
                "innings": innings_number,
                "over_mark": 6,
                "runs_so_far": runs_so_far,
                "wickets_down": wickets_down,
                "current_run_rate": current_run_rate,
                "balls_since_boundary": (
                    balls_since_boundary
                ),
                "final_score": final_score,
                "batting_team": batting_team,
            }
        )

    return rows


def build_checkpoints() -> pd.DataFrame:
    """Build checkpoints.csv from all scorecards."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_files = sorted(
        RAW_SCORECARDS_DIR.glob("*.json")
    )

    print(
        f"Found {len(json_files)} JSON scorecards."
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {RAW_SCORECARDS_DIR}"
        )

    all_rows = []

    for index, json_path in enumerate(
        json_files,
        start=1,
    ):

        try:
            scorecard = load_scorecard(
                json_path
            )

            rows = extract_checkpoints(
                scorecard
            )

            all_rows.extend(rows)

        except Exception as error:
            print(
                f"Error processing "
                f"{json_path.name}: {error}"
            )

        if index % 50 == 0:
            print(
                f"Processed "
                f"{index}/{len(json_files)} files."
            )

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

    checkpoints = pd.DataFrame(
        all_rows,
        columns=columns,
    )

    if checkpoints.empty:
        raise RuntimeError(
            "No checkpoint rows were generated. "
            "Check the scorecard structure."
        )

    checkpoints = checkpoints.drop_duplicates(
        subset=[
            "match_id",
            "innings",
            "over_mark",
        ]
    )

    checkpoints = checkpoints.sort_values(
        [
            "match_id",
            "innings",
        ]
    ).reset_index(drop=True)

    checkpoints.to_csv(
        CHECKPOINTS_PATH,
        index=False,
    )

    print()
    print(
        "Checkpoint generation complete."
    )
    print(
        f"Rows: {len(checkpoints)}"
    )
    print(
        f"Saved to: {CHECKPOINTS_PATH}"
    )

    print()
    print("Dataset preview:")
    print(
        checkpoints.head(10).to_string(
            index=False
        )
    )

    print()
    print("Shape:")
    print(checkpoints.shape)

    print()
    print("Missing values:")
    print(checkpoints.isna().sum())

    print()
    print("Innings distribution:")
    print(
        checkpoints["innings"]
        .value_counts()
        .sort_index()
    )

    return checkpoints


def load_checkpoints() -> pd.DataFrame:
    """Load checkpoints.csv, creating it if necessary."""

    if not CHECKPOINTS_PATH.exists():
        return build_checkpoints()

    return pd.read_csv(
        CHECKPOINTS_PATH
    )


if __name__ == "__main__":
    build_checkpoints()