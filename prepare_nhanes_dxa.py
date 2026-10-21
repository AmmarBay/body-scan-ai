from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "nhanes_dxa"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_PATH = DATA_DIR / "nhanes_dxa_bodyfat.csv"
CDC_DATA_URL = "https://wwwn.cdc.gov/nchs/data/nhanes/public"

CYCLES = {
    "1999-2000": ("1999", "", ""),
    "2001-2002": ("2001", "_B", "_B"),
    "2003-2004": ("2003", "_C", "_C"),
    "2005-2006": ("2005", "_D", "_D"),
}


def download_xpt(cycle_year, filename):
    directory = RAW_DIR / cycle_year / "datafiles"
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    if output_path.exists() and output_path.stat().st_size:
        return output_path

    url = f"{CDC_DATA_URL}/{cycle_year}/datafiles/{filename}"
    request = Request(url, headers={"User-Agent": "BodyFitAi dataset preparation"})
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def load_cycle(cycle_name, cycle_year, demo_suffix, body_suffix):
    demo_path = download_xpt(cycle_year, f"DEMO{demo_suffix}.xpt")
    body_path = download_xpt(cycle_year, f"BMX{body_suffix}.xpt")
    dxa_path = download_xpt(cycle_year, f"DXX{body_suffix}.xpt")

    demographics = pd.read_sas(
        demo_path,
        format="xport",
        encoding="latin-1",
    )[["SEQN", "RIAGENDR", "RIDAGEYR", "WTMEC2YR"]]
    body_measures = pd.read_sas(
        body_path,
        format="xport",
        encoding="latin-1",
    )[["SEQN", "BMXWT", "BMXHT", "BMXWAIST"]]
    dxa_all = pd.read_sas(dxa_path, format="xport", encoding="latin-1")
    imputation_flags = [
        column
        for column in ("DXITOTST", "DXITOTBN", "DXITOT")
        if column in dxa_all.columns
    ]
    if not imputation_flags:
        raise ValueError(f"Penanda imputasi DXA tidak ditemukan untuk siklus {cycle_name}.")
    dxa = dxa_all[
        ["SEQN", "_MULT_", "DXDTOPF", *imputation_flags]
    ].copy()
    dxa["imputation_flags_valid"] = dxa[imputation_flags].isin([0.0, 1.0]).all(axis=1)
    dxa = dxa.drop(columns=imputation_flags)

    merged = dxa.merge(
        demographics,
        on="SEQN",
        how="inner",
        validate="many_to_one",
    ).merge(
        body_measures,
        on="SEQN",
        how="inner",
        validate="many_to_one",
    )
    merged["cycle"] = cycle_name
    merged["participant_id"] = (
        cycle_name + "-" + merged["SEQN"].astype("Int64").astype("string")
    )
    merged = merged.rename(
        columns={
            "RIAGENDR": "sex_code",
            "RIDAGEYR": "age_years",
            "WTMEC2YR": "sample_weight",
            "BMXWT": "weight_kg",
            "BMXHT": "height_cm",
            "BMXWAIST": "abdomen_cm",
            "_MULT_": "imputation",
            "DXDTOPF": "bodyfat_dxa_pct",
        }
    )
    merged["sex_encoded"] = merged["sex_code"].map({1.0: 1, 2.0: 0})
    merged["height_m"] = merged["height_cm"] / 100
    merged["bmi"] = merged["weight_kg"] / merged["height_m"].pow(2)
    merged["sample_weight"] = merged["sample_weight"] / len(CYCLES)

    valid = (
        merged["sex_encoded"].notna()
        & merged["age_years"].between(18, 69)
        & merged["weight_kg"].between(30, 200)
        & merged["height_m"].between(1.0, 2.5, inclusive="both")
        & merged["abdomen_cm"].between(40, 150)
        & merged["bodyfat_dxa_pct"].between(0, 75, inclusive="neither")
        & merged["sample_weight"].gt(0)
        & merged["imputation_flags_valid"]
        & merged["imputation"].isin([1.0, 2.0, 3.0, 4.0, 5.0])
    )
    merged = merged.loc[valid]
    return merged[
        [
            "participant_id",
            "cycle",
            "imputation",
            "sex_encoded",
            "age_years",
            "weight_kg",
            "height_m",
            "abdomen_cm",
            "bmi",
            "bodyfat_dxa_pct",
            "sample_weight",
        ]
    ]


def main():
    frames = []
    for cycle_name, (cycle_year, demo_suffix, body_suffix) in CYCLES.items():
        print(f"Mengambil NHANES {cycle_name} dari CDC...")
        frame = load_cycle(cycle_name, cycle_year, demo_suffix, body_suffix)
        print(
            f"  {frame['participant_id'].nunique()} peserta, "
            f"{len(frame)} baris versi imputasi yang lolos filter."
        )
        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)
    imputation_counts = dataset.groupby("participant_id")["imputation"].nunique()
    complete_participants = imputation_counts[imputation_counts.eq(5)].index
    dataset = dataset[dataset["participant_id"].isin(complete_participants)].copy()
    dataset = dataset.sort_values(
        ["participant_id", "imputation"], kind="stable"
    ).reset_index(drop=True)
    if dataset.empty:
        raise ValueError("Tidak ada peserta dengan lima versi imputasi yang lengkap.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False, float_format="%.6g")
    print(
        f"\nDataset tersimpan: {OUTPUT_PATH.relative_to(BASE_DIR)}\n"
        f"Peserta unik: {dataset['participant_id'].nunique()}\n"
        f"Baris: {len(dataset)} (5 versi per peserta)\n"
        f"Versi imputasi: {sorted(dataset['imputation'].unique().astype(int).tolist())}"
    )


if __name__ == "__main__":
    main()
