import os
import re
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CHBMITDataset(Dataset):
    """
    CHB-MIT scalp EEG dataset.

    Output:
        signal: Tensor [C, L]
                C = 22 EEG channels
                L = window_size * sampling_rate

        label:
            0 = non-seizure
            1 = seizure
    """

    def __init__(
        self,
        root_dir,
        patient="chb01",
        window_seconds=5.0,
        sampling_rate=256,
        stride_seconds=5.0,
        normalize=True,
        files=None,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.patient = patient
        self.patient_dir = os.path.join(root_dir, patient)

        self.window_seconds = window_seconds
        self.sampling_rate = sampling_rate
        self.stride_seconds = stride_seconds
        self.normalize = normalize

        self.window_samples = int(window_seconds * sampling_rate)
        self.stride_samples = int(stride_seconds * sampling_rate)

        # CHB-MIT has 23 channels in this patient.
        # The last channel T8-P8 is duplicated.
        # For the first REACT-EEG version we keep the first 22 channels.
        self.num_channels = 22

        if not os.path.isdir(self.patient_dir):
            raise FileNotFoundError(
                f"Patient directory not found:\n{self.patient_dir}"
            )

        # If files are not explicitly provided, discover EDF files.
        if files is None:
            files = self._find_edf_files()
        else:
            files = [
                f if os.path.isabs(f)
                else os.path.join(self.patient_dir, f)
                for f in files
            ]

        self.files = sorted(files)

        if len(self.files) == 0:
            raise RuntimeError(
                f"No EDF files found in {self.patient_dir}"
            )

        # Each element in self.samples is one EEG window.
        #
        # {
        #     "file": EDF path,
        #     "start_sample": int,
        #     "end_sample": int,
        #     "label": 0/1
        # }
        self.samples = []

        self._build_index()

        print("=" * 70)
        print("CHB-MIT Dataset")
        print("=" * 70)
        print(f"Patient:              {self.patient}")
        print(f"Number of EDF files:  {len(self.files)}")
        print(f"Sampling rate:        {self.sampling_rate} Hz")
        print(f"Window:                {self.window_seconds} s")
        print(f"Window samples:        {self.window_samples}")
        print(f"Stride:                {self.stride_seconds} s")
        print(f"Channels:              {self.num_channels}")
        print(f"Total windows:         {len(self.samples)}")

        seizure_count = sum(
            sample["label"] == 1 for sample in self.samples
        )

        non_seizure_count = len(self.samples) - seizure_count

        print(f"Seizure windows:       {seizure_count}")
        print(f"Non-seizure windows:   {non_seizure_count}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Find EDF files
    # ------------------------------------------------------------------

    def _find_edf_files(self):
        files = []

        for name in os.listdir(self.patient_dir):
            if name.lower().endswith(".edf"):
                files.append(os.path.join(self.patient_dir, name))

        return sorted(files)

    # ------------------------------------------------------------------
    # Parse seizure annotation from summary file
    # ------------------------------------------------------------------

    def _read_summary(self):
        summary_path = os.path.join(
            self.patient_dir,
            f"{self.patient}-summary.txt"
        )

        if not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"Summary file not found:\n{summary_path}"
            )

        with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        return text

    def _parse_seizure_annotations(self):
        """
        Parse:

            File Name: chb01_03.edf
            ...
            Number of Seizures in File: 1
            Seizure Start Time: 2996 seconds
            Seizure End Time: 3036 seconds

        Return:

            {
                "chb01_03.edf": [(2996.0, 3036.0)],
                ...
            }
        """

        text = self._read_summary()

        annotations = {}

        # Split the summary by File Name.
        blocks = re.split(
            r"(?=File Name:\s*)",
            text
        )

        for block in blocks:

            file_match = re.search(
                r"File Name:\s*(\S+\.edf)",
                block,
                re.IGNORECASE
            )

            if file_match is None:
                continue

            filename = file_match.group(1)

            seizure_count_match = re.search(
                r"Number of Seizures in File:\s*(\d+)",
                block,
                re.IGNORECASE
            )

            if seizure_count_match is None:
                continue

            seizure_count = int(
                seizure_count_match.group(1)
            )

            seizures = []

            if seizure_count > 0:

                starts = re.findall(
                    r"Seizure Start Time:\s*([0-9.]+)\s*seconds",
                    block,
                    re.IGNORECASE
                )

                ends = re.findall(
                    r"Seizure End Time:\s*([0-9.]+)\s*seconds",
                    block,
                    re.IGNORECASE
                )

                if len(starts) != seizure_count:
                    raise RuntimeError(
                        f"Seizure start count mismatch for {filename}: "
                        f"expected {seizure_count}, found {len(starts)}"
                    )

                if len(ends) != seizure_count:
                    raise RuntimeError(
                        f"Seizure end count mismatch for {filename}: "
                        f"expected {seizure_count}, found {len(ends)}"
                    )

                for start, end in zip(starts, ends):
                    seizures.append(
                        (float(start), float(end))
                    )

            annotations[filename] = seizures

        return annotations

    # ------------------------------------------------------------------
    # Determine whether a window overlaps seizure
    # ------------------------------------------------------------------

    @staticmethod
    def _window_is_seizure(
        window_start,
        window_end,
        seizure_intervals
    ):
        """
        A window is labeled seizure if it overlaps any seizure interval.

        Window:
            [window_start, window_end)

        Seizure:
            [seizure_start, seizure_end)

        We use overlap rather than requiring the entire 5-second window
        to be inside the seizure interval.
        """

        for seizure_start, seizure_end in seizure_intervals:

            overlap = (
                window_start < seizure_end
                and window_end > seizure_start
            )

            if overlap:
                return True

        return False

    # ------------------------------------------------------------------
    # Build sample index
    # ------------------------------------------------------------------

    def _build_index(self):

        annotations = self._parse_seizure_annotations()

        # Import here so importing this file does not require pyedflib
        # until the dataset is actually constructed.
        import pyedflib

        for edf_path in self.files:

            filename = os.path.basename(edf_path)

            seizure_intervals = annotations.get(
                filename,
                []
            )

            try:
                reader = pyedflib.EdfReader(edf_path)

                duration = float(reader.file_duration)

                channels = reader.signals_in_file

                sample_frequencies = np.asarray(
                    reader.getSampleFrequencies(),
                    dtype=np.float64
                )

                reader.close()

            except Exception as e:
                raise RuntimeError(
                    f"Failed to inspect EDF:\n{edf_path}\n"
                    f"Error: {e}"
                )

            if channels < self.num_channels:
                raise RuntimeError(
                    f"{filename} has only {channels} channels, "
                    f"but {self.num_channels} are required."
                )

            # Verify sampling rate.
            if not np.allclose(
                sample_frequencies[:self.num_channels],
                self.sampling_rate
            ):
                raise RuntimeError(
                    f"Unexpected sampling rate in {filename}:\n"
                    f"{sample_frequencies}"
                )

            total_samples = int(
                duration * self.sampling_rate
            )

            start = 0

            while start + self.window_samples <= total_samples:

                end = start + self.window_samples

                window_start_sec = (
                    start / self.sampling_rate
                )

                window_end_sec = (
                    end / self.sampling_rate
                )

                label = int(
                    self._window_is_seizure(
                        window_start_sec,
                        window_end_sec,
                        seizure_intervals
                    )
                )

                self.samples.append(
                    {
                        "file": edf_path,
                        "filename": filename,
                        "start_sample": start,
                        "end_sample": end,
                        "start_sec": window_start_sec,
                        "end_sec": window_end_sec,
                        "label": label,
                    }
                )

                start += self.stride_samples

    # ------------------------------------------------------------------
    # Read one EEG window
    # ------------------------------------------------------------------

    def _load_window(self, sample_info):

        import pyedflib

        edf_path = sample_info["file"]

        start_sample = sample_info["start_sample"]
        end_sample = sample_info["end_sample"]

        reader = pyedflib.EdfReader(edf_path)

        try:

            signals = []

            for channel_idx in range(self.num_channels):

                signal = reader.readSignal(
                    channel_idx,
                    start=start_sample,
                    n=end_sample - start_sample
                )

                signals.append(signal)

        finally:
            reader.close()

        signal = np.asarray(
            signals,
            dtype=np.float32
        )

        # Expected:
        #
        # [22, 1280]
        #
        if signal.shape != (
            self.num_channels,
            self.window_samples
        ):
            raise RuntimeError(
                f"Unexpected EEG window shape: "
                f"{signal.shape}, expected "
                f"({self.num_channels}, {self.window_samples})"
            )

        # Per-channel normalization.
        #
        # This is intentionally simple for the first runnable version.
        # Later we can replace this with training-set statistics.
        if self.normalize:

            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)

            signal = (
                signal - mean
            ) / (
                std + 1e-6
            )

        return signal

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample_info = self.samples[index]

        signal = self._load_window(
            sample_info
        )

        label = sample_info["label"]

        signal = torch.from_numpy(
            signal
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return signal, label

    # ------------------------------------------------------------------
    # Useful information
    # ------------------------------------------------------------------

    def get_sample_info(self, index):
        return self.samples[index]