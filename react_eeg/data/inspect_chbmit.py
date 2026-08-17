import os
import pyedflib


DATA_ROOT = r"D:\Qiang\数据集\chb-mit-scalp-eeg-database-1.0.0"

EDF_FILE = os.path.join(
    DATA_ROOT,
    "chb01",
    "chb01_03.edf"
)


def inspect_edf(path):

    print("=" * 70)
    print("CHB-MIT EDF inspection")
    print("=" * 70)

    print("File:")
    print(path)

    if not os.path.exists(path):
        print("\nERROR: EDF file does not exist!")
        return

    f = pyedflib.EdfReader(path)

    try:

        print("\nNumber of channels:")
        print(f.signals_in_file)

        print("\nDuration:")
        print(f.file_duration, "seconds")

        print("\nSampling frequencies:")
        print(f.getSampleFrequencies())

        print("\nChannel labels:")

        labels = f.getSignalLabels()

        for i, label in enumerate(labels):
            print("{:02d}: {}".format(i, label))

        print("\nFirst channel information:")

        print("Label:", labels[0])
        print("Sample frequency:", f.getSampleFrequency(0))

        samples = f.readSignal(0, start=0, n=256)

        print("First 256 samples shape:", samples.shape)
        print("First 10 samples:")
        print(samples[:10])

    finally:
        f.close()

    print("\n" + "=" * 70)
    print("Inspection finished")
    print("=" * 70)


if __name__ == "__main__":
    inspect_edf(EDF_FILE)