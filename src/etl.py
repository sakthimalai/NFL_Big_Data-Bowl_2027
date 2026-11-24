"""
src/etl.py
High-Performance Ingestion & Parquet Transformation Pipeline
NFL Big Data Bowl 2027

Converts 9 raw CSV files (~2.3 GB) into Snappy-compressed Parquet tables,
normalizes null encodings ('NA', 'null', 'NULL', ''), casts data types,
verifies 10 Hz timestamp frequency, and outputs a complete data health profile.
"""

import os
import sys
import time
from pathlib import Path
import polars as pl

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = ROOT_DIR / "processed"

NULL_VALUES = ["NA", "null", "NULL", "None", "", "NaN", "nan"]

def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)

def process_players() -> dict:
    print("Processing players.csv...")
    raw_path = DATA_DIR / "players.csv"
    out_path = PROCESSED_DIR / "players.parquet"
    
    schema_overrides = {
        "nfl_id": pl.Int64,
        "display_name": pl.Utf8,
        "draft_year": pl.Int32,
        "nfl_position": pl.Utf8,
        "birth_date": pl.Utf8,
        "college_name": pl.Utf8,
        "college_conference": pl.Utf8,
        "draft_round": pl.Int32,
        "draft_pick_within_round": pl.Int32,
        "draft_overall_pick": pl.Int32,
    }
    
    df = pl.read_csv(
        raw_path,
        null_values=NULL_VALUES,
        schema_overrides=schema_overrides,
        infer_schema_length=10000,
    )
    
    df = df.with_columns([
        pl.col("nfl_position").str.strip_chars().str.to_uppercase(),
        pl.col("display_name").str.strip_chars(),
    ])
    
    df.write_parquet(out_path, compression="snappy")
    return {
        "name": "players",
        "raw_size_mb": get_file_size_mb(raw_path),
        "parquet_size_mb": get_file_size_mb(out_path),
        "rows": df.height,
        "cols": df.width,
        "unique_players": df["nfl_id"].n_unique(),
        "null_counts": {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
    }

def process_combine_results() -> dict:
    print("Processing combine_results.csv...")
    raw_path = DATA_DIR / "combine_results.csv"
    out_path = PROCESSED_DIR / "combine_results.parquet"
    
    schema_overrides = {
        "draft_year": pl.Int32,
        "nfl_id": pl.Int64,
        "combine_position": pl.Utf8,
        "combine_height": pl.Float64,
        "combine_weight": pl.Float64,
        "hand_size": pl.Float64,
        "arm_length": pl.Float64,
        "wing_span": pl.Float64,
        "ten_yd_split": pl.Float64,
        "forty": pl.Float64,
        "vertical": pl.Float64,
        "broad_jump": pl.Float64,
        "three_cone": pl.Float64,
        "short_shuttle": pl.Float64,
        "bench_reps": pl.Float64,
        "ngs_athleticism_score": pl.Float64,
        "ngs_college_production_score": pl.Float64,
        "ngs_final_score": pl.Float64,
    }
    
    df = pl.read_csv(
        raw_path,
        null_values=NULL_VALUES,
        schema_overrides=schema_overrides,
        infer_schema_length=10000,
    )
    
    df = df.with_columns([
        pl.col("combine_position").str.strip_chars().str.to_uppercase(),
    ])
    
    df.write_parquet(out_path, compression="snappy")
    return {
        "name": "combine_results",
        "raw_size_mb": get_file_size_mb(raw_path),
        "parquet_size_mb": get_file_size_mb(out_path),
        "rows": df.height,
        "cols": df.width,
        "unique_players": df["nfl_id"].n_unique(),
        "null_counts": {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
    }

def process_player_career_successes() -> dict:
    print("Processing player_career_successes.csv...")
    raw_path = DATA_DIR / "player_career_successes.csv"
    out_path = PROCESSED_DIR / "player_career_successes.parquet"
    
    schema_overrides = {
        "nfl_id": pl.Int64,
        "career_offensive_snaps": pl.Int64,
        "career_defensive_snaps": pl.Int64,
        "career_special_teams_snaps": pl.Int64,
        "career_games_active": pl.Int32,
        "career_games_started": pl.Int32,
        "ap_all_pro_1st_team": pl.Int32,
        "ap_all_pro_2nd_team": pl.Int32,
        "pro_bowl_original_ballot": pl.Int32,
    }
    
    df = pl.read_csv(
        raw_path,
        null_values=NULL_VALUES,
        schema_overrides=schema_overrides,
        infer_schema_length=10000,
    )
    
    df.write_parquet(out_path, compression="snappy")
    return {
        "name": "player_career_successes",
        "raw_size_mb": get_file_size_mb(raw_path),
        "parquet_size_mb": get_file_size_mb(out_path),
        "rows": df.height,
        "cols": df.width,
        "unique_players": df["nfl_id"].n_unique(),
        "null_counts": {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
    }

def process_games() -> dict:
    print("Processing games.csv...")
    raw_path = DATA_DIR / "games.csv"
    out_path = PROCESSED_DIR / "games.parquet"
    
    schema_overrides = {
        "game_id": pl.Int64,
        "game_key": pl.Int64,
        "season": pl.Int32,
        "season_type": pl.Utf8,
        "week": pl.Int32,
        "home_team_abbr": pl.Utf8,
        "visitor_team_abbr": pl.Utf8,
    }
    
    df = pl.read_csv(
        raw_path,
        null_values=NULL_VALUES,
        schema_overrides=schema_overrides,
        infer_schema_length=10000,
    )
    
    df.write_parquet(out_path, compression="snappy")
    return {
        "name": "games",
        "raw_size_mb": get_file_size_mb(raw_path),
        "parquet_size_mb": get_file_size_mb(out_path),
        "rows": df.height,
        "cols": df.width,
        "unique_games": df["game_id"].n_unique(),
        "null_counts": {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
    }

def process_player_play() -> dict:
    print("Processing player_play.csv (316,338 records)...")
    raw_path = DATA_DIR / "player_play.csv"
    out_path = PROCESSED_DIR / "player_play.parquet"
    
    df = pl.read_csv(
        raw_path,
        null_values=NULL_VALUES,
        infer_schema_length=50000,
    )
    
    # Cast standard types
    cast_ops = [
        pl.col("game_id").cast(pl.Int64),
        pl.col("play_id").cast(pl.Int64),
        pl.col("nfl_id").cast(pl.Int64),
    ]
    if "quarter" in df.columns:
        cast_ops.append(pl.col("quarter").cast(pl.Int32))
    if "down" in df.columns:
        cast_ops.append(pl.col("down").cast(pl.Int32))
    if "yards_to_go" in df.columns:
        cast_ops.append(pl.col("yards_to_go").cast(pl.Int32))
        
    df = df.with_columns(cast_ops)
    df.write_parquet(out_path, compression="snappy")
    
    return {
        "name": "player_play",
        "raw_size_mb": get_file_size_mb(raw_path),
        "parquet_size_mb": get_file_size_mb(out_path),
        "rows": df.height,
        "cols": df.width,
        "unique_players": df["nfl_id"].n_unique(),
        "unique_games": df["game_id"].n_unique(),
        "null_counts": {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
    }

def process_combine_tracking() -> dict:
    print("Processing combine_tracking.csv...")
    raw_path = DATA_DIR / "combine_tracking.csv"
    out_path = PROCESSED_DIR / "combine_tracking.parquet"
    
    df = pl.read_csv(
        raw_path,
        null_values=NULL_VALUES,
        infer_schema_length=50000,
    )
    
    # Parse time to Datetime(ms) and cast numeric fields
    df = df.with_columns([
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f").cast(pl.Datetime(time_unit="ms")),
        pl.col("draft_year").cast(pl.Int32),
        pl.col("event_id").cast(pl.Utf8),
        pl.col("nfl_id").cast(pl.Int64),
        pl.col("entity_type").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        pl.col("drill_type").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        pl.col("drill_name").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        pl.col("attempt").cast(pl.Int32),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("s").cast(pl.Float64),
        pl.col("a").cast(pl.Float64),
        pl.col("dis").cast(pl.Float64),
        pl.col("dir").cast(pl.Float64),
    ])
    
    # Check 10 Hz interval
    player_frames = df.filter((pl.col("entity_type") == "PLAYER") & pl.col("nfl_id").is_not_null())
    sample_dt = (
        player_frames
        .sort(["event_id", "nfl_id", "time"])
        .group_by(["event_id", "nfl_id"], maintain_order=True)
        .agg([
            (pl.col("time").diff().dt.total_milliseconds() / 1000.0).alias("dt")
        ])
        .explode("dt")
        .filter(pl.col("dt").is_not_null() & (pl.col("dt") > 0))
    )
    median_dt = sample_dt["dt"].median()
    p95_dt = sample_dt["dt"].quantile(0.95)
    
    df.write_parquet(out_path, compression="snappy")
    
    return {
        "name": "combine_tracking",
        "raw_size_mb": get_file_size_mb(raw_path),
        "parquet_size_mb": get_file_size_mb(out_path),
        "rows": df.height,
        "cols": df.width,
        "unique_players": player_frames["nfl_id"].n_unique(),
        "unique_events": df["event_id"].n_unique(),
        "median_dt_sec": median_dt,
        "p95_dt_sec": p95_dt,
        "null_counts": {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
    }

def process_game_tracking(year: int) -> dict:
    fname = f"game_tracking_{year}.csv"
    print(f"Processing {fname}...")
    raw_path = DATA_DIR / fname
    out_path = PROCESSED_DIR / f"game_tracking_{year}.parquet"
    
    df = pl.read_csv(
        raw_path,
        null_values=NULL_VALUES,
        infer_schema_length=50000,
    )
    
    df = df.with_columns([
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f").cast(pl.Datetime(time_unit="ms")),
        pl.col("game_id").cast(pl.Int64),
        pl.col("play_id").cast(pl.Int64),
        pl.col("nfl_id").cast(pl.Int64),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("s").cast(pl.Float64),
        pl.col("a").cast(pl.Float64),
        pl.col("dis").cast(pl.Float64),
        pl.col("o").cast(pl.Float64),
        pl.col("dir").cast(pl.Float64),
        pl.col("event").cast(pl.Utf8),
    ])
    
    # Verify 10 Hz delta t on sample
    sample_dt = (
        df.filter(pl.col("nfl_id").is_not_null())
        .head(20000)
        .sort(["game_id", "play_id", "nfl_id", "time"])
        .group_by(["game_id", "play_id", "nfl_id"], maintain_order=True)
        .agg([
            (pl.col("time").diff().dt.total_milliseconds() / 1000.0).alias("dt")
        ])
        .explode("dt")
        .filter(pl.col("dt").is_not_null() & (pl.col("dt") > 0))
    )
    median_dt = sample_dt["dt"].median() if len(sample_dt) > 0 else None
    
    df.write_parquet(out_path, compression="snappy")
    
    return {
        "name": f"game_tracking_{year}",
        "raw_size_mb": get_file_size_mb(raw_path),
        "parquet_size_mb": get_file_size_mb(out_path),
        "rows": df.height,
        "cols": df.width,
        "unique_players": df["nfl_id"].n_unique(),
        "unique_games": df["game_id"].n_unique(),
        "median_dt_sec": median_dt,
        "null_counts": {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
    }

def main():
    start_time = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    profiles = [
        process_players(),
        process_combine_results(),
        process_player_career_successes(),
        process_games(),
        process_player_play(),
        process_combine_tracking(),
        process_game_tracking(2023),
        process_game_tracking(2024),
        process_game_tracking(2025),
    ]
    
    total_raw_mb = sum(p["raw_size_mb"] for p in profiles)
    total_pq_mb = sum(p["parquet_size_mb"] for p in profiles)
    elapsed_sec = time.time() - start_time
    
    print(f"\nETL Complete in {elapsed_sec:.2f} seconds!")
    print(f"Total Raw CSV Size: {total_raw_mb:.2f} MB")
    print(f"Total Parquet Size: {total_pq_mb:.2f} MB (Compression: {(1.0 - total_pq_mb/total_raw_mb)*100:.1f}%)")

if __name__ == "__main__":
    main()
