import pandas as pd

def combine_files(output_file="used_car_financial_assets_800k.csv"):
    files = [
        "used_car_financial_assets_800k_part1.csv",
        "used_car_financial_assets_800k_part2.csv",
        "used_car_financial_assets_800k_part3.csv"
    ]

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(output_file, index=False)

    print(f"Combined file saved as: {output_file}")
    print(f"Total rows: {len(df):,}")

if __name__ == "__main__":
    combine_files()
