import pathlib

from lib.datasets.anonymize import anonymize_directory

if __name__ == '__main__':
    anonymize_directory(str(pathlib.Path("data/fondo-normal")),
                        str(pathlib.Path("data/retinopathy-v2.1/fondo-normal-anon")),
                        rename=True)
