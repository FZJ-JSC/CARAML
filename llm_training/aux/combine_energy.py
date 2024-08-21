import argparse
import pandas as pd

def parse_args():

    default_interval = 100 #ms
    parser = argparse.ArgumentParser(description="Combine energy dataframes")
    parser.add_argument("--output",
        type=str,
        required=True,
        help=f"Filename to store the combined dataframes in")
    parser.add_argument(
        "files",
        nargs=argparse.REMAINDER,
        help="Files to combine",
    )

    return parser.parse_args()


def main():


    args = parse_args()

    dfs = [pd.read_csv(csv_file).T.reset_index(drop=True) for csv_file in args.files]


    for df in dfs:
        df.columns = df.iloc[0]
    dfs = [df.iloc[1:] for df in dfs]


    alldf = pd.concat(dfs).reset_index(drop=True)

    firstcolumns = ["nodename", "rank"]
    columns = firstcolumns + [c for c in alldf.columns if c not in firstcolumns]
    alldf = alldf[columns]

    print(f"Writing combined energy DataFrame to {args.output}")

    alldf.to_csv(args.output)


if "__main__" == __name__:
    main()
