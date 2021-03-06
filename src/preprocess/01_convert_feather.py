import sys
import pandas as pd
import glob

sys.path.append('../src')
import const

extension = 'csv'


def main():
    path_list = glob.glob(f'{str(const.INPUT_DATA_DIR)}/*.{extension}')

    for path in path_list:
        (pd.read_csv(path, encoding="utf-8"))\
            .to_feather(path.replace(extension, 'feather'))


if __name__ == '__main__':
    main()