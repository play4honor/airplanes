import polars as pl


def prepare_flight_chain_data(
    flight_data: list[str],
    supplemental_data: str,
    output_path: str,
):
    # Read data and correct dates.
    initial_data = pl.concat(
        [pl.read_csv(fd_path) for fd_path in flight_data]
    ).with_columns(
        pl.col("FL_DATE")
        .str.strptime(pl.Date, "%Y-%m-%d %H:%M:%S")
        .dt.strftime("%Y%m%d")
        .alias("gross_flight_date"),
        pl.col("CRS_DEP_TIME")
        .str.strptime(pl.Time, "%Y-%m-%d %H:%M:%S")
        .dt.strftime("%-H%M"),
        pl.col("DEP_TIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        pl.col("FL_DATE").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        pl.col("ARR_TIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
    )

    supplemental_data = (
        pl.read_parquet(supplemental_data)
        .filter(pl.col("flight_number") != "")
        .with_columns(pl.col("flight_number").cast(pl.Float64))
    )

    flight_data = (
        initial_data.join(
            supplemental_data,
            left_on=[
                "gross_flight_date",
                "OP_CARRIER",
                "OP_CARRIER_FL_NUM",
                "CRS_DEP_TIME",
            ],
            right_on=["date", "carrier", "flight_number", "departure_time"],
        )
        .with_columns(
            pl.concat_list(
                # Do we want an airport to have the same embedding as a departure and arrival?
                pl.concat_str(pl.lit("AIRPORT="), pl.col("ORIGIN")),
                pl.concat_str(pl.lit("AIRPORT="), pl.col("DEST")),
                pl.concat_str(
                    pl.lit("DEP_TIME="),
                    ((pl.col("DEP_TIME") - pl.col("FL_DATE")).dt.total_minutes() / 10)
                    .cast(pl.Int32)
                    .alias("departure_time"),
                ),
                pl.concat_str(
                    pl.lit("DEP_DELAY="),
                    pl.col("DEP_DELAY").sign()
                    * pl.col("DEP_DELAY").abs().sqrt().ceil(),
                ),
                pl.concat_str(
                    pl.lit("ARR_TIME="),
                    ((pl.col("ARR_TIME") - pl.col("FL_DATE")).dt.total_minutes() / 10)
                    .cast(pl.Int32)
                    .alias("arrival_time"),
                ),
                pl.concat_str(
                    pl.lit("ARR_DELAY="),
                    pl.col("ARR_DELAY").sign()
                    * pl.col("ARR_DELAY").abs().sqrt().ceil(),
                ),
            ).alias("flight_info")
        )
        .sort(["FL_DATE", "DEP_TIME"], descending=False)
        .group_by("FL_DATE", "tail_number")
        .agg(pl.col("flight_info").flatten().alias("flight_info"))
        .with_columns(
            pl.concat_list(
                pl.lit("<SOS>"), pl.col("flight_info"), pl.lit("<EOS>")
            ).alias("flight_info")
        )
    )

    print(f"Number of flight chains: {flight_data.height}")
    print(flight_data.row(0, named=True))
    flight_data.write_parquet(output_path)


if __name__ == "__main__":

    prepare_flight_chain_data(
        flight_data=[
            "./data/Flight_Tab/flight_with_weather_2021.csv",
            "./data/Flight_Tab/flight_with_weather_2022.csv",
            "./data/Flight_Tab/flight_with_weather_2023.csv",
            "./data/Flight_Tab/flight_with_weather_2024.csv",
        ],
        supplemental_data="./data/supplemental.parquet",
        output_path="./data/prepared_data.parquet",
    )
