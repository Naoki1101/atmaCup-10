import sys
import pandas as pd
import glob

sys.path.append("../src")
import const


def merge_color(df: pd.DataFrame) -> pd.DataFrame:
    color_df = pd.read_csv(const.INPUT_DATA_DIR / "color.csv")
    df = pd.merge(df, color_df, on=const.ID_COLS[0], how="left")
    return df


def merge_material(df: pd.DataFrame) -> pd.DataFrame:
    material_df = pd.read_csv(const.INPUT_DATA_DIR / "material.csv")
    material_df.rename(columns={"name": "material_name"}, inplace=True)
    df = pd.merge(df, material_df, on=const.ID_COLS[0], how="left")
    return df


def merge_historical_person(df: pd.DataFrame) -> pd.DataFrame:
    historical_person_df = pd.read_csv(const.INPUT_DATA_DIR / "historical_person.csv")
    historical_person_df.rename(
        columns={"name": "historical_person_name"}, inplace=True
    )
    df = pd.merge(df, historical_person_df, on=const.ID_COLS[0], how="left")
    return df


def merge_object_collection(df: pd.DataFrame) -> pd.DataFrame:
    object_collection_df = pd.read_csv(const.INPUT_DATA_DIR / "object_collection.csv")
    object_collection_df.rename(
        columns={"name": "object_collection_name"}, inplace=True
    )
    df = pd.merge(df, object_collection_df, on=const.ID_COLS[0], how="left")
    return df


def merge_production_place(df: pd.DataFrame) -> pd.DataFrame:
    production_place_df = pd.read_csv(const.INPUT_DATA_DIR / "production_place.csv")
    production_place_df.rename(columns={"name": "production_place_name"}, inplace=True)
    df = pd.merge(df, production_place_df, on=const.ID_COLS[0], how="left")
    return df


def merge_technique(df: pd.DataFrame) -> pd.DataFrame:
    technique_df = pd.read_csv(const.INPUT_DATA_DIR / "technique.csv")
    technique_df.rename(columns={"name": "technique_name"}, inplace=True)
    df = pd.merge(df, technique_df, on=const.ID_COLS[0], how="left")
    return df


def merge_palette(df: pd.DataFrame) -> pd.DataFrame:
    palette_df = pd.read_csv(const.INPUT_DATA_DIR / "palette.csv")
    palette_df.rename(columns={"name": "palette_name"}, inplace=True)
    df = pd.merge(df, palette_df, on=const.ID_COLS[0], how="left")
    return df


def merge_maker(df: pd.DataFrame) -> pd.DataFrame:
    maker_df = pd.read_csv(const.INPUT_DATA_DIR / "maker.csv")
    maker_df.rename(columns={"name": "principal_maker"}, inplace=True)
    df = pd.merge(df, maker_df, on="principal_maker", how="left")
    return df


def merge_principal_maker(df: pd.DataFrame) -> pd.DataFrame:
    principal_maker_df = pd.read_csv(const.INPUT_DATA_DIR / "principal_maker.csv")
    principal_maker_occupation_df = pd.read_csv(
        "../data/input/principal_maker_occupation.csv"
    )

    principal_maker_occupation_df = pd.crosstab(
        index=principal_maker_occupation_df["id"],
        columns=principal_maker_occupation_df["name"],
    ).reset_index()
    principal_maker_occupation_df.columns = ["id"] + [
        f"principal_maker_occupation_{col.replace(' ', '_')}"
        for col in principal_maker_occupation_df.columns[1:]
    ]

    principal_maker_df = pd.merge(
        principal_maker_df, principal_maker_occupation_df, on="id", how="left"
    )
    principal_maker_df = principal_maker_df.drop(columns=["id"])

    df = pd.merge(df, principal_maker_df, on=const.ID_COLS[0], how="left")
    return df


def main():
    train_df = pd.read_csv(const.INPUT_DATA_DIR / "train.csv")
    test_df = pd.read_csv(const.INPUT_DATA_DIR / "test.csv")

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)

    whole_df = merge_color(whole_df)
    whole_df = merge_material(whole_df)
    whole_df = merge_historical_person(whole_df)
    whole_df = merge_object_collection(whole_df)
    whole_df = merge_production_place(whole_df)
    whole_df = merge_technique(whole_df)
    whole_df = merge_palette(whole_df)
    whole_df = merge_maker(whole_df)
    whole_df = merge_principal_maker(whole_df)

    train_df = whole_df.iloc[: len(train_df)].reset_index(drop=True)
    test_df = whole_df.iloc[len(train_df) :].reset_index(drop=True)

    train_df.to_feather(const.INPUT_DATA_DIR / "train.feather")
    test_df.to_feather(const.INPUT_DATA_DIR / "test.feather")


if __name__ == "__main__":
    main()