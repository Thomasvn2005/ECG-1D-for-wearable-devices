# ========================================================================================
#                TARGET Train/val/test on AF specialized dataset called IRIDIA-AF
# IRIDIA-AF 1-LEAD / 60-S AF DETECTION
# ResNet50-1D + Squeeze-and-Excitation (SE)
#   60-second Lead-I ECG -> AF / non-AF
# IMPORTANT:
#   First run with SMOKE_TEST = True  ---> done
#   After everything works, change SMOKE_TEST = False ---> not done
# ========================================================================================

from __future__ import annotations
import json
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,)
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================================================================
#                         1.FILES PATHS
# ================================================================
METADATA_CSV = Path(r"D:\Datasets\IRIDIA-AF\iridia-af-metadata-v1.0.1.csv")
RECORDS_ROOT = Path(r"D:\Datasets\IRIDIA-AF\iridia-af-records-v1.0.1\iridia-af-records-v1.0.1")
OUTPUT_DIR = Path(r"D:\Datasets\IRIDIA-AF\results_resnet50")

# ================================================================
#                      2. ECG CONFIGURATION
# ================================================================
# IRIDIA-AF sampling frequency
FS = 200
# Official IRIDIA-AF GitHub DL example uses:
# f[key][..., 0] with column 0 = ECG Lead I
LEAD_INDEX = 0
WINDOW_SEC = 60
WINDOW_SAMPLES = FS * WINDOW_SEC

# ------------------------------------------------
#                   STRIDE
# ------------------------------------------------
# 60s stride = no overlap (faster).
# Use 30s stride later for 50% overlap.
TRAIN_STRIDE_SEC = 60
EVAL_STRIDE_SEC = 60

# ------------------------------------------------
#              DEVICE CALIBRATION
# ------------------------------------------------
# Because IRIDIA paper states that the first 30 seconds of the recording are calibration.
# ==> Remove only the first 30 seconds of a RECORD, not 30 seconds from every 24-h HDF5 file.
CALIBRATION_SEC = 30

# ================================================================
#                     3. AF LABEL STRATEGY
# ================================================================
# Labeling:
# Clean mode
#   0% AF          → non-AF (0)
#   0% < AF < 80%  → exclude
#   AF ≥ 80%       → AF (1)
# Paper mode
#   59 sec normal + 1 sec AF → AF = 1
#   30 sec normal + 30 sec AF → AF = 1
#   1 sec normal + 59 sec AF → AF = 1
#   60 sec normal → non-AF = 0
LABEL_MODE = "clean"
AF_POSITIVE_THRESHOLD = 0.80

# ================================================================
#                       4. PATIENT-LEVEL SPLIT
# ================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# ================================================================
#                     5. TRAINING CONFIGURATION
# ================================================================
EPOCHS = 30
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8

NUM_WORKERS = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
DROPOUT = 0.20

# ================================================================
#                         6. MODEL OPTIONS
# ================================================================
# ResNet50-1D + Squeeze-and-Excitation
USE_SE = True

# Mixed precision:
USE_AMP = True

# Optional training augmentation.
# use for test noise robustness.
NOISE_STD = 0.00

# ================================================================
#                 7. OPERATING THRESHOLD
# ================================================================
# Threshold is selected ONLY on validation set.

# "youden": maximize sensitivity + specificity - 1
# "f1": maximize F1

THRESHOLD_METHOD = "youden"
# ===============================================================
#                  8. FIRST RUN: SMOKE TEST
# ================================================================
# VERY IMPORTANT:
# FIRST RUN:
# SMOKE_TEST = True
# It trains only a small subset for 2 epochs.
# If everything works:
# SMOKE_TEST = False
# and run the full experiment.
SMOKE_TEST = False
SMOKE_TRAIN_WINDOWS = 12000
SMOKE_EVAL_WINDOWS = 3000
# Rebuild CSV window index?
# False: reuse existing index
# True: recreate it from raw IRIDIA annotations
REBUILD_INDEX = False
# Resume training after interruption
AUTO_RESUME = True

# ================================================================
#                    9. REPRODUCIBILITY
# ================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

# ================================================================
#                 10. FIND IRIDIA RECORD DIRECTORY
# ================================================================
def resolve_records_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"RECORDS_ROOT does not exist:\n{root}"
        )
    # Expected:
    # root/
    #     record_000/
    #     record_001/
    direct_records = [
        p
        for p in root.glob("record_*")
        if p.is_dir()
    ]
    if direct_records:

        return root
    # Sometimes ZIP extraction creates:
    # root/
    #    another_folder/
    #       record_000/
    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        records = [
            p
            for p in child.glob("record_*")
            if p.is_dir()
        ]
        if records:
            candidates.append(child)

    if len(candidates) == 1:
        print(
            "\nAuto-detected records root:"
        )
        print(candidates[0])
        return candidates[0]
    raise FileNotFoundError(
        "Could not find record_000, record_001, ...\n"
        "Check RECORDS_ROOT."
    )
# ================================================================
#                      11. HDF5 INFORMATION
# ================================================================
def h5_info(path: Path):
    with h5py.File(
        path,
        "r"
    ) as f:
        keys = list(
            f.keys()
        )
        if len(keys) == 0:
            raise RuntimeError(
                f"No HDF5 dataset found:\n{path}"
            )
        key = keys[0]
        shape = f[key].shape

        if len(shape) != 2:
            raise RuntimeError(
                f"Expected ECG shape [samples, leads].\n"
                f"Got {shape}\n"
                f"File: {path}"
            )
        n_samples = int(
            shape[0]
        )
        n_channels = int(
            shape[1]
        )
    return (
        key,
        n_samples,
        n_channels
    )


# ================================================================
#                  12. AF INTERVAL HELPERS
# ================================================================

def merge_intervals(intervals):

    intervals = sorted(

        (
            int(start),
            int(end)
        )

        for start, end in intervals

        if int(end) > int(start)
    )


    if len(intervals) == 0:

        return []


    merged = [

        [
            intervals[0][0],
            intervals[0][1]
        ]

    ]


    for start, end in intervals[1:]:

        if start <= merged[-1][1]:

            merged[-1][1] = max(
                merged[-1][1],
                end
            )

        else:

            merged.append(
                [
                    start,
                    end
                ]
            )


    return [

        (
            int(start),
            int(end)
        )

        for start, end in merged
    ]


# ================================================================
# 13. CONVERT AF EPISODES TO PER-FILE INTERVALS
# ================================================================

def annotations_to_file_intervals(
    annotation_df: pd.DataFrame,
    file_lengths
):

    required_columns = {

        "start_file_index",
        "end_file_index",
        "start_qrs_index",
        "end_qrs_index"

    }


    missing = (

        required_columns
        -
        set(
            annotation_df.columns
        )

    )


    if missing:

        raise ValueError(

            "Annotation CSV is missing columns:\n"
            f"{sorted(missing)}"
        )


    intervals_per_file = defaultdict(
        list
    )


    for row in annotation_df.itertuples(
        index=False
    ):

        start_file = int(
            row.start_file_index
        )

        end_file = int(
            row.end_file_index
        )

        start_index = int(
            row.start_qrs_index
        )

        end_index = int(
            row.end_qrs_index
        )


        if start_file < 0:

            raise ValueError(
                "Negative start file index."
            )


        if end_file >= len(file_lengths):

            raise ValueError(

                f"AF episode refers to file {end_file}, "
                f"but record has only "
                f"{len(file_lengths)} files."
            )


        if end_file < start_file:

            raise ValueError(
                "AF end file occurs before start file."
            )


        # ----------------------------------------
        # AF starts and ends within same HDF5 file
        # ----------------------------------------

        if start_file == end_file:

            start_index = max(
                0,
                start_index
            )

            end_index = min(
                file_lengths[start_file],
                end_index
            )


            intervals_per_file[
                start_file
            ].append(

                (
                    start_index,
                    end_index
                )

            )


        # ----------------------------------------
        # AF crosses multiple HDF5 files
        # ----------------------------------------

        else:

            # Start file:
            #
            # AF start -> end of file

            intervals_per_file[
                start_file
            ].append(

                (
                    max(
                        0,
                        start_index
                    ),

                    file_lengths[
                        start_file
                    ]
                )

            )


            # Intermediate files:
            #
            # entire file = AF

            for file_index in range(

                start_file + 1,
                end_file

            ):

                intervals_per_file[
                    file_index
                ].append(

                    (
                        0,
                        file_lengths[
                            file_index
                        ]
                    )

                )


            # Final file:
            #
            # start of file -> AF termination

            intervals_per_file[
                end_file
            ].append(

                (
                    0,

                    min(
                        file_lengths[
                            end_file
                        ],
                        end_index
                    )
                )

            )


    result = {}


    for file_index, intervals in (
        intervals_per_file.items()
    ):

        result[
            file_index
        ] = merge_intervals(
            intervals
        )


    return result


# ================================================================
# 14. COUNT AF SAMPLES INSIDE WINDOW
# ================================================================

def overlap_len(
    start: int,
    end: int,
    intervals
):

    total = 0


    for interval_start, interval_end in intervals:

        if interval_end <= start:

            continue


        if interval_start >= end:

            break


        overlap_start = max(
            start,
            interval_start
        )

        overlap_end = min(
            end,
            interval_end
        )


        total += max(
            0,
            overlap_end - overlap_start
        )


    return total


# ================================================================
# 15. WINDOW LABEL
# ================================================================

def label_from_burden(
    burden: float
):

    # ----------------------------------------
    # Reproduce current GitHub-like strategy
    # ----------------------------------------

    if LABEL_MODE == "paper":

        if burden > 0.0:

            return 1

        return 0


    # ----------------------------------------
    # Recommended clean-label strategy
    # ----------------------------------------

    # Perfect non-AF window

    if burden == 0.0:

        return 0


    # Predominantly AF

    if burden >= AF_POSITIVE_THRESHOLD:

        return 1


    # Transition:
    #
    # contains both AF and non-AF
    #
    # We exclude it from training initially.

    return None


# ================================================================
# 16. PATIENT-LEVEL SPLIT
# ================================================================

def make_patient_split(
    metadata: pd.DataFrame
):

    patients = np.array(

        sorted(

            metadata[
                "patient_id"
            ]
            .astype(str)
            .unique()

        )

    )


    # ----------------------------------------
    # train+validation vs test
    # ----------------------------------------

    train_val_patients, test_patients = (
        train_test_split(

            patients,

            test_size=TEST_RATIO,

            random_state=SEED,

            shuffle=True

        )
    )


    validation_fraction = (

        VAL_RATIO
        /
        (
            TRAIN_RATIO
            +
            VAL_RATIO
        )

    )


    # ----------------------------------------
    # train vs validation
    # ----------------------------------------

    train_patients, val_patients = (
        train_test_split(

            train_val_patients,

            test_size=validation_fraction,

            random_state=SEED,

            shuffle=True

        )
    )


    mapping = {}


    for patient in train_patients:

        mapping[
            patient
        ] = "train"


    for patient in val_patients:

        mapping[
            patient
        ] = "val"


    for patient in test_patients:

        mapping[
            patient
        ] = "test"


    return mapping


# ================================================================
# 17. PRINT DATASET SUMMARY
# ================================================================

def print_split_summary(
    df: pd.DataFrame
):

    print(
        "\n================================="
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "================================="
    )


    for split in [
        "train",
        "val",
        "test"
    ]:

        split_df = df[
            df["split"] == split
        ]


        non_af = int(

            (
                split_df["label"] == 0
            ).sum()

        )


        af = int(

            (
                split_df["label"] == 1
            ).sum()

        )


        patients = (

            split_df[
                "patient_id"
            ].nunique()

        )


        total = len(
            split_df
        )


        af_percent = (

            100
            *
            af
            /
            max(
                total,
                1
            )

        )


        print(

            f"{split:>5} | "
            f"patients={patients:3d} | "
            f"windows={total:8,d} | "
            f"non-AF={non_af:8,d} | "
            f"AF={af:8,d} | "
            f"AF={af_percent:.2f}%"

        )


# ================================================================
# 18. BUILD 60-S WINDOW INDEX
# ================================================================

def build_window_index(
    metadata_csv: Path,
    records_root: Path,
    index_path: Path
):

    print(
        "\n================================="
    )

    print(
        "BUILDING IRIDIA-AF WINDOW INDEX"
    )

    print(
        "================================="
    )


    metadata = pd.read_csv(

        metadata_csv,

        dtype={

            "patient_id": str,
            "record_id": str

        }

    )


    required_metadata = {

        "patient_id",
        "record_id"

    }


    if not required_metadata.issubset(
        metadata.columns
    ):

        raise ValueError(

            "Metadata must contain:\n"
            "patient_id\n"
            "record_id"

        )


    patient_split = make_patient_split(
        metadata
    )


    patient_split_df = pd.DataFrame(

        sorted(
            patient_split.items()
        ),

        columns=[
            "patient_id",
            "split"
        ]

    )


    patient_split_df.to_csv(

        OUTPUT_DIR
        /
        "patient_splits.csv",

        index=False

    )


    train_stride = int(

        TRAIN_STRIDE_SEC
        *
        FS

    )


    eval_stride = int(

        EVAL_STRIDE_SEC
        *
        FS

    )


    calibration_samples = int(

        CALIBRATION_SEC
        *
        FS

    )


    rows = []

    dropped_transition = 0


    records = (

        metadata[
            [
                "patient_id",
                "record_id"
            ]
        ]
        .drop_duplicates()

    )


    iterator = tqdm(

        records.itertuples(
            index=False
        ),

        total=len(
            records
        ),

        desc="Indexing IRIDIA records"

    )


    for record in iterator:

        patient_id = str(
            record.patient_id
        )

        record_id = str(
            record.record_id
        )


        split = patient_split[
            patient_id
        ]


        if split == "train":

            stride = train_stride

        else:

            stride = eval_stride


        record_folder = (

            records_root
            /
            record_id

        )


        if not record_folder.exists():

            raise FileNotFoundError(

                f"Record directory not found:\n"
                f"{record_folder}"

            )


        # ----------------------------------------
        # ECG HDF5 files
        # ----------------------------------------

        ecg_files = sorted(

            record_folder.glob(
                "*_ecg_*.h5"
            )

        )


        if len(ecg_files) == 0:

            raise FileNotFoundError(

                f"No *_ecg_*.h5 found in:\n"
                f"{record_folder}"

            )


        # ----------------------------------------
        # AF annotations
        # ----------------------------------------

        annotation_files = sorted(

            record_folder.glob(
                "*ecg_labels.csv"
            )

        )


        if len(annotation_files) != 1:

            raise RuntimeError(

                f"Expected exactly one ecg_labels.csv.\n"
                f"Record: {record_folder}\n"
                f"Found: {len(annotation_files)}"

            )


        file_information = []

        file_lengths = []


        for h5_path in ecg_files:

            key, n_samples, n_channels = (
                h5_info(
                    h5_path
                )
            )


            if n_channels <= LEAD_INDEX:

                raise RuntimeError(

                    f"Lead index {LEAD_INDEX} "
                    f"not present in:\n"
                    f"{h5_path}"

                )


            file_information.append(

                (
                    key,
                    n_samples,
                    n_channels
                )

            )


            file_lengths.append(
                n_samples
            )


        annotation_df = pd.read_csv(
            annotation_files[0]
        )


        intervals_by_file = (
            annotations_to_file_intervals(

                annotation_df,

                file_lengths

            )
        )


        # ----------------------------------------
        # Create windows
        # ----------------------------------------

        for file_index, (
            h5_path,
            file_info
        ) in enumerate(

            zip(
                ecg_files,
                file_information
            )

        ):

            key = file_info[0]

            n_samples = file_info[1]


            # Only the first HDF5 file of a RECORD
            # contains the record calibration region
            # that we intentionally remove.

            if file_index == 0:

                first_start = (
                    calibration_samples
                )

            else:

                first_start = 0


            if (

                n_samples
                -
                first_start

                <
                WINDOW_SAMPLES

            ):

                continue


            af_intervals = (
                intervals_by_file.get(
                    file_index,
                    []
                )
            )


            last_start = (

                n_samples
                -
                WINDOW_SAMPLES

            )


            for start_index in range(

                first_start,

                last_start + 1,

                stride

            ):

                end_index = (

                    start_index
                    +
                    WINDOW_SAMPLES

                )


                af_samples = overlap_len(

                    start_index,

                    end_index,

                    af_intervals

                )


                af_burden = (

                    af_samples
                    /
                    WINDOW_SAMPLES

                )


                label = label_from_burden(
                    af_burden
                )


                if label is None:

                    dropped_transition += 1

                    continue


                rows.append(

                    {

                        "patient_id":
                            patient_id,

                        "record_id":
                            record_id,

                        "split":
                            split,

                        "file":
                            str(
                                h5_path.resolve()
                            ),

                        "h5_key":
                            key,

                        "file_index":
                            file_index,

                        "start_index":
                            start_index,

                        "end_index":
                            end_index,

                        "label":
                            int(
                                label
                            ),

                        "af_burden":
                            float(
                                af_burden
                            )

                    }

                )


    window_df = pd.DataFrame(
        rows
    )


    if len(window_df) == 0:

        raise RuntimeError(

            "No windows generated.\n"
            "Check your dataset paths."

        )


    window_df.to_csv(

        index_path,

        index=False

    )


    print(
        "\nWindow index saved:"
    )

    print(
        index_path
    )


    print(
        f"\nWindow duration: "
        f"{WINDOW_SEC} seconds"
    )


    print(
        f"Samples/window: "
        f"{WINDOW_SAMPLES}"
    )


    print(
        f"Dropped transition windows: "
        f"{dropped_transition:,}"
    )


    print_split_summary(
        window_df
    )


    return window_df


# ================================================================
# 19. PYTORCH DATASET
# ================================================================

class IRIDIAWindowDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        train=False,
        noise_std=0.0,
        cache_size=4
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )


        self.train = train

        self.noise_std = float(
            noise_std
        )

        self.cache_size = int(
            cache_size
        )


        # HDF5 file cache
        #
        # Prevents reopening same HDF5 file
        # for every single window.

        self._cache = OrderedDict()


    def __len__(
        self
    ):

        return len(
            self.df
        )


    def __getstate__(
        self
    ):

        # Windows DataLoader uses multiprocessing spawn.
        #
        # Do not pickle open HDF5 handles.

        state = self.__dict__.copy()

        state[
            "_cache"
        ] = OrderedDict()

        return state


    def get_h5_file(
        self,
        path
    ):

        path = str(
            path
        )


        if path in self._cache:

            file_handle = (
                self._cache.pop(
                    path
                )
            )

            self._cache[
                path
            ] = file_handle

            return file_handle


        file_handle = h5py.File(
            path,
            "r"
        )


        self._cache[
            path
        ] = file_handle


        while (

            len(
                self._cache
            )
            >
            self.cache_size

        ):

            _, old_file = (
                self._cache.popitem(
                    last=False
                )
            )

            try:

                old_file.close()

            except Exception:

                pass


        return file_handle


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]


        file_handle = self.get_h5_file(
            row["file"]
        )


        h5_key = str(
            row["h5_key"]
        )


        start_index = int(
            row["start_index"]
        )

        end_index = int(
            row["end_index"]
        )


        # ----------------------------------------
        # Read Lead I only
        # ----------------------------------------

        ecg = np.asarray(

            file_handle[
                h5_key
            ][

                start_index:end_index,

                LEAD_INDEX

            ],

            dtype=np.float32

        )


        if ecg.shape[0] != WINDOW_SAMPLES:

            raise RuntimeError(

                f"Expected {WINDOW_SAMPLES} samples, "
                f"got {ecg.shape[0]}.\n"
                f"File: {row['file']}"

            )


        # ----------------------------------------
        # Clean NaN / Inf
        # ----------------------------------------

        ecg = np.nan_to_num(

            ecg,

            nan=0.0,

            posinf=0.0,

            neginf=0.0

        )


        # ----------------------------------------
        # Per-window Z-score normalization
        # ----------------------------------------

        mean = float(
            ecg.mean()
        )

        std = float(
            ecg.std()
        )


        if std < 1e-6:

            std = 1.0


        ecg = (

            ecg
            -
            mean

        ) / std


        # Prevent extreme artifacts
        # dominating numerical values.

        ecg = np.clip(

            ecg,

            -8.0,

            8.0

        )


        ecg_tensor = torch.from_numpy(
            ecg
        )


        # [12000]
        #
        # ->
        #
        # [1, 12000]

        ecg_tensor = (
            ecg_tensor.unsqueeze(
                0
            )
        )


        # ----------------------------------------
        # Optional augmentation
        # ----------------------------------------

        if (

            self.train
            and
            self.noise_std > 0

        ):

            noise = (

                torch.randn_like(
                    ecg_tensor
                )

                *
                self.noise_std

            )

            ecg_tensor = (

                ecg_tensor
                +
                noise

            )


        label_tensor = torch.tensor(

            float(
                row["label"]
            ),

            dtype=torch.float32

        )


        return (
            ecg_tensor,
            label_tensor
        )


# ================================================================
# 20. SQUEEZE-AND-EXCITATION
# ================================================================

class SEBlock1D(
    nn.Module
):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()


        hidden = max(

            channels
            //
            reduction,

            8

        )


        self.pool = (
            nn.AdaptiveAvgPool1d(
                1
            )
        )


        self.fc = nn.Sequential(

            nn.Conv1d(

                channels,

                hidden,

                kernel_size=1

            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv1d(

                hidden,

                channels,

                kernel_size=1

            ),

            nn.Sigmoid()

        )


    def forward(
        self,
        x
    ):

        attention = self.pool(
            x
        )

        attention = self.fc(
            attention
        )


        return (

            x
            *
            attention

        )


# ================================================================
# 21. RESNET50 BOTTLENECK 1D
# ================================================================

class Bottleneck1D(
    nn.Module
):

    expansion = 4


    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        use_se=True
    ):

        super().__init__()


        out_channels = (

            channels
            *
            self.expansion

        )


        # ----------------------------------------
        # 1x1 convolution
        # ----------------------------------------

        self.conv1 = nn.Conv1d(

            in_channels,

            channels,

            kernel_size=1,

            bias=False

        )


        self.bn1 = (
            nn.BatchNorm1d(
                channels
            )
        )


        # ----------------------------------------
        # Temporal 3x1 convolution
        # ----------------------------------------

        self.conv2 = nn.Conv1d(

            channels,

            channels,

            kernel_size=3,

            stride=stride,

            padding=1,

            bias=False

        )


        self.bn2 = (
            nn.BatchNorm1d(
                channels
            )
        )


        # ----------------------------------------
        # 1x1 expansion
        # ----------------------------------------

        self.conv3 = nn.Conv1d(

            channels,

            out_channels,

            kernel_size=1,

            bias=False

        )


        self.bn3 = (
            nn.BatchNorm1d(
                out_channels
            )
        )


        # ----------------------------------------
        # SE attention
        # ----------------------------------------

        if use_se:

            self.se = SEBlock1D(
                out_channels
            )

        else:

            self.se = nn.Identity()


        self.relu = nn.ReLU(
            inplace=True
        )


        # ----------------------------------------
        # Residual shortcut
        # ----------------------------------------

        if (

            stride != 1

            or

            in_channels
            !=
            out_channels

        ):

            self.shortcut = (
                nn.Sequential(

                    nn.Conv1d(

                        in_channels,

                        out_channels,

                        kernel_size=1,

                        stride=stride,

                        bias=False

                    ),

                    nn.BatchNorm1d(
                        out_channels
                    )

                )
            )


        else:

            self.shortcut = (
                nn.Identity()
            )


    def forward(
        self,
        x
    ):

        identity = self.shortcut(
            x
        )


        out = self.conv1(
            x
        )

        out = self.bn1(
            out
        )

        out = self.relu(
            out
        )


        out = self.conv2(
            out
        )

        out = self.bn2(
            out
        )

        out = self.relu(
            out
        )


        out = self.conv3(
            out
        )

        out = self.bn3(
            out
        )


        out = self.se(
            out
        )


        out = (

            out
            +
            identity

        )


        out = self.relu(
            out
        )


        return out


# ================================================================
# 22. RESNET50-1D
# ================================================================

class ResNet50_1D(
    nn.Module
):

    def __init__(
        self,
        use_se=True,
        dropout=0.20
    ):

        super().__init__()


        self.in_channels = 64


        # ----------------------------------------
        # ECG temporal stem
        # ----------------------------------------

        self.stem = nn.Sequential(

            nn.Conv1d(

                in_channels=1,

                out_channels=64,

                kernel_size=15,

                stride=2,

                padding=7,

                bias=False

            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool1d(

                kernel_size=3,

                stride=2,

                padding=1

            )

        )


        # ----------------------------------------
        # Standard ResNet50 depth:
        #
        # 3, 4, 6, 3 bottleneck blocks
        # ----------------------------------------

        self.layer1 = self.make_layer(

            channels=64,

            blocks=3,

            stride=1,

            use_se=use_se

        )


        self.layer2 = self.make_layer(

            channels=128,

            blocks=4,

            stride=2,

            use_se=use_se

        )


        self.layer3 = self.make_layer(

            channels=256,

            blocks=6,

            stride=2,

            use_se=use_se

        )


        self.layer4 = self.make_layer(

            channels=512,

            blocks=3,

            stride=2,

            use_se=use_se

        )


        # Adaptive pooling means:
        #
        # 60 seconds works
        # 40 seconds works
        # 30 seconds works
        #
        # without fixed Flatten dimension.

        self.global_pool = (
            nn.AdaptiveAvgPool1d(
                1
            )
        )


        self.dropout = (
            nn.Dropout(
                dropout
            )
        )


        # Binary output:
        #
        # one logit
        #
        # Do NOT apply sigmoid here.
        #
        # BCEWithLogitsLoss does that
        # numerically safely.

        self.fc = nn.Linear(

            512
            *
            Bottleneck1D.expansion,

            1

        )


        self.initialize_weights()


    def make_layer(
        self,
        channels,
        blocks,
        stride,
        use_se
    ):

        layers = []


        layers.append(

            Bottleneck1D(

                self.in_channels,

                channels,

                stride=stride,

                use_se=use_se

            )

        )


        self.in_channels = (

            channels

            *
            Bottleneck1D.expansion

        )


        for _ in range(
            1,
            blocks
        ):

            layers.append(

                Bottleneck1D(

                    self.in_channels,

                    channels,

                    stride=1,

                    use_se=use_se

                )

            )


        return nn.Sequential(
            *layers
        )


    def initialize_weights(
        self
    ):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv1d
            ):

                nn.init.kaiming_normal_(

                    module.weight,

                    mode="fan_out",

                    nonlinearity="relu"

                )


            elif isinstance(
                module,
                nn.BatchNorm1d
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )


            elif isinstance(
                module,
                nn.Linear
            ):

                nn.init.normal_(

                    module.weight,

                    mean=0.0,

                    std=0.01

                )

                nn.init.zeros_(
                    module.bias
                )


    def forward(
        self,
        x
    ):

        x = self.stem(
            x
        )


        x = self.layer1(
            x
        )

        x = self.layer2(
            x
        )

        x = self.layer3(
            x
        )

        x = self.layer4(
            x
        )


        x = self.global_pool(
            x
        )


        x = x.flatten(
            1
        )


        x = self.dropout(
            x
        )


        logits = self.fc(
            x
        )


        # [B,1] -> [B]

        logits = logits.squeeze(
            1
        )


        return logits


# ================================================================
# 23. SELECT THRESHOLD USING VALIDATION SET ONLY
# ================================================================

def choose_threshold(
    y_true,
    probabilities,
    method="youden"
):

    y_true = np.asarray(
        y_true
    ).astype(
        int
    )


    probabilities = np.asarray(
        probabilities
    ).astype(
        float
    )


    if method == "youden":

        false_positive_rate, \
        true_positive_rate, \
        thresholds = roc_curve(

            y_true,

            probabilities

        )


        scores = (

            true_positive_rate
            -
            false_positive_rate

        )


        finite = np.isfinite(
            thresholds
        )


        valid_indices = np.where(
            finite
        )[0]


        if len(valid_indices) == 0:

            return 0.5


        best_index = valid_indices[

            np.argmax(
                scores[
                    finite
                ]
            )

        ]


        threshold = thresholds[
            best_index
        ]


        return float(

            np.clip(

                threshold,

                0.0,

                1.0

            )

        )


    elif method == "f1":

        thresholds = np.linspace(

            0.05,

            0.95,

            181

        )


        f1_values = []


        for threshold in thresholds:

            prediction = (

                probabilities
                >=
                threshold

            )


            score = f1_score(

                y_true,

                prediction,

                zero_division=0

            )


            f1_values.append(
                score
            )


        best_index = int(

            np.argmax(
                f1_values
            )

        )


        return float(
            thresholds[
                best_index
            ]
        )


    else:

        raise ValueError(

            "THRESHOLD_METHOD must be "
            "'youden' or 'f1'."

        )


# ================================================================
# 24. METRICS
# ================================================================

def compute_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    y_true = np.asarray(
        y_true
    ).astype(
        int
    )


    probabilities = np.asarray(
        probabilities
    ).astype(
        float
    )


    predictions = (

        probabilities
        >=
        threshold

    ).astype(
        int
    )


    tn, fp, fn, tp = (

        confusion_matrix(

            y_true,

            predictions,

            labels=[
                0,
                1
            ]

        ).ravel()

    )


    if (
        tp + fn
    ) > 0:

        sensitivity = (

            tp
            /
            (
                tp
                +
                fn
            )

        )

    else:

        sensitivity = np.nan


    if (
        tn + fp
    ) > 0:

        specificity = (

            tn
            /
            (
                tn
                +
                fp
            )

        )

    else:

        specificity = np.nan


    result = {

        "threshold":
            float(
                threshold
            ),

        "n":
            int(
                len(
                    y_true
                )
            ),

        "accuracy":
            float(

                accuracy_score(

                    y_true,

                    predictions

                )

            ),

        "sensitivity_recall":
            float(
                sensitivity
            ),

        "specificity":
            float(
                specificity
            ),

        "precision_ppv":
            float(

                precision_score(

                    y_true,

                    predictions,

                    zero_division=0

                )

            ),

        "f1":
            float(

                f1_score(

                    y_true,

                    predictions,

                    zero_division=0

                )

            ),

        "auroc":
            float(

                roc_auc_score(

                    y_true,

                    probabilities

                )

            ),

        "auprc":
            float(

                average_precision_score(

                    y_true,

                    probabilities

                )

            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp)

    }


    return result


# ================================================================
# 25. SMOKE-TEST SUBSET
# ================================================================

def stratified_subset(
    dataframe,
    n_samples
):

    if len(
        dataframe
    ) <= n_samples:

        return (

            dataframe
            .reset_index(
                drop=True
            )

        )


    fraction = (

        n_samples
        /
        len(
            dataframe
        )

    )


    groups = []


    for label, group in (

        dataframe.groupby(
            "label"
        )

    ):

        number = max(

            1,

            int(

                round(

                    len(group)
                    *
                    fraction

                )

            )

        )


        number = min(

            number,

            len(group)

        )


        sampled = group.sample(

            n=number,

            random_state=SEED

        )


        groups.append(
            sampled
        )


    subset = pd.concat(
        groups
    )


    subset = subset.sample(

        frac=1.0,

        random_state=SEED

    )


    subset = subset.head(
        n_samples
    )


    return (

        subset
        .reset_index(
            drop=True
        )

    )


# ================================================================
# 26. DATA LOADER
# ================================================================

def create_loader(
    dataframe,
    train,
    device
):

    dataset = IRIDIAWindowDataset(

        dataframe,

        train=train,

        noise_std=(
            NOISE_STD
            if train
            else 0.0
        ),

        cache_size=4

    )


    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )


    options = {

        "batch_size":
            BATCH_SIZE,

        "shuffle":
            train,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            (
                device.type
                ==
                "cuda"
            ),

        "drop_last":
            False,

        "worker_init_fn":
            worker_init_fn,

        "generator":
            generator

    }


    if NUM_WORKERS > 0:

        options[
            "persistent_workers"
        ] = True

        options[
            "prefetch_factor"
        ] = 2


    loader = DataLoader(

        dataset,

        **options

    )


    return loader


# ================================================================
# 27. MIXED PRECISION
# ================================================================

def autocast_context(
    enabled
):

    return torch.amp.autocast(

        device_type="cuda",

        dtype=torch.float16,

        enabled=enabled

    )


def make_scaler(
    enabled
):

    try:

        scaler = (
            torch.amp.GradScaler(

                "cuda",

                enabled=enabled

            )
        )


    except TypeError:

        scaler = (
            torch.cuda.amp.GradScaler(

                enabled=enabled

            )
        )


    return scaler


# ================================================================
# 28. EVALUATION
# ================================================================

@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
    loss_function,
    use_amp
):

    model.eval()


    labels = []

    probabilities = []


    total_loss = 0.0

    total_samples = 0


    progress = tqdm(

        data_loader,

        desc="Evaluate",

        leave=False

    )


    for ecg, label in progress:

        ecg = ecg.to(

            device,

            non_blocking=True

        )


        label = label.to(

            device,

            non_blocking=True

        )


        with autocast_context(
            use_amp
        ):

            logits = model(
                ecg
            )


            loss = loss_function(

                logits,

                label

            )


        batch_size = ecg.size(
            0
        )


        total_loss += (

            float(
                loss.item()
            )

            *
            batch_size

        )


        total_samples += (
            batch_size
        )


        probability = torch.sigmoid(
            logits
        )


        labels.append(

            label
            .detach()
            .cpu()
            .numpy()

        )


        probabilities.append(

            probability
            .float()
            .detach()
            .cpu()
            .numpy()

        )


    labels = np.concatenate(
        labels
    )


    probabilities = np.concatenate(
        probabilities
    )


    mean_loss = (

        total_loss
        /
        max(
            total_samples,
            1
        )

    )


    return (

        mean_loss,

        labels,

        probabilities

    )


# ================================================================
# 29. JSON SAVE
# ================================================================

def save_json(
    data,
    path
):

    with open(

        path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            data,

            file,

            indent=2,

            ensure_ascii=False

        )


# ================================================================
# 30. MAIN
# ================================================================

def main():

    seed_everything(
        SEED
    )


    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    if not np.isclose(

        TRAIN_RATIO
        +
        VAL_RATIO
        +
        TEST_RATIO,

        1.0

    ):

        raise ValueError(

            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO "
            "must equal 1.0"

        )


    if LABEL_MODE not in {

        "clean",
        "paper"

    }:

        raise ValueError(

            "LABEL_MODE must be "
            "'clean' or 'paper'."

        )


    metadata_path = (

        METADATA_CSV
        .expanduser()
        .resolve()

    )


    if not metadata_path.exists():

        raise FileNotFoundError(

            f"Metadata CSV not found:\n"
            f"{metadata_path}"

        )


    records_root = resolve_records_root(

        RECORDS_ROOT

    )


    # ============================================================
    # WINDOW INDEX
    # ============================================================

    index_filename = (

        f"window_index_"
        f"{WINDOW_SEC}s_"
        f"{LABEL_MODE}_"
        f"{AF_POSITIVE_THRESHOLD:.2f}.csv"

    )


    index_path = (

        OUTPUT_DIR
        /
        index_filename

    )


    if (

        REBUILD_INDEX

        or

        not index_path.exists()

    ):

        dataframe = build_window_index(

            metadata_path,

            records_root,

            index_path

        )


    else:

        print(

            "\nLoading existing window index:"
        )

        print(
            index_path
        )


        dataframe = pd.read_csv(

            index_path,

            dtype={

                "patient_id": str,

                "record_id": str

            }

        )


        print_split_summary(
            dataframe
        )


    # ============================================================
    # SPLITS
    # ============================================================

    train_df = dataframe[

        dataframe[
            "split"
        ]
        ==
        "train"

    ].copy()


    validation_df = dataframe[

        dataframe[
            "split"
        ]
        ==
        "val"

    ].copy()


    test_df = dataframe[

        dataframe[
            "split"
        ]
        ==
        "test"

    ].copy()


    training_epochs = EPOCHS


    # ============================================================
    # SMOKE TEST
    # ============================================================

    if SMOKE_TEST:

        print(

            "\n================================="
        )

        print(

            "SMOKE TEST ENABLED"
        )

        print(

            "Only a subset will be used."
        )

        print(

            "Training for only 2 epochs."
        )

        print(

            "================================="
        )


        train_df = stratified_subset(

            train_df,

            SMOKE_TRAIN_WINDOWS

        )


        validation_df = stratified_subset(

            validation_df,

            SMOKE_EVAL_WINDOWS

        )


        test_df = stratified_subset(

            test_df,

            SMOKE_EVAL_WINDOWS

        )


        training_epochs = 2


    # ============================================================
    # VERIFY CLASS DISTRIBUTION
    # ============================================================

    print(

        "\n================================="
    )

    print(

        "ACTIVE TRAINING DATA"
    )

    print(

        "================================="
    )


    for name, split_df in [

        (
            "train",
            train_df
        ),

        (
            "val",
            validation_df
        ),

        (
            "test",
            test_df
        )

    ]:

        non_af = int(

            (
                split_df[
                    "label"
                ]
                ==
                0
            ).sum()

        )


        af = int(

            (
                split_df[
                    "label"
                ]
                ==
                1
            ).sum()

        )


        patients = split_df[

            "patient_id"

        ].nunique()


        print(

            f"{name:>5}: "
            f"windows={len(split_df):,}, "
            f"patients={patients}, "
            f"non-AF={non_af:,}, "
            f"AF={af:,}"

        )


        if non_af == 0 or af == 0:

            raise RuntimeError(

                f"{name} split does not contain "
                f"both AF and non-AF."

            )


    # ============================================================
    # GPU
    # ============================================================

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )


    use_amp = (

        USE_AMP

        and

        device.type == "cuda"

    )


    print(

        "\n================================="
    )

    print(

        "DEVICE"
    )

    print(

        "================================="
    )


    print(

        "PyTorch version:",

        torch.__version__

    )


    print(

        "Device:",

        device

    )


    if device.type == "cuda":

        gpu_name = (
            torch.cuda.get_device_name(
                0
            )
        )


        gpu_memory = (

            torch.cuda
            .get_device_properties(
                0
            )
            .total_memory

            /
            (
                1024 ** 3
            )

        )


        print(

            "GPU:",

            gpu_name

        )


        print(

            f"VRAM: "
            f"{gpu_memory:.2f} GB"

        )


        print(

            "PyTorch CUDA:",

            torch.version.cuda

        )


        print(

            "AMP FP16:",

            use_amp

        )


        # Speed optimization

        torch.backends.cudnn.benchmark = True


        torch.backends.cuda.matmul.allow_tf32 = True


        torch.backends.cudnn.allow_tf32 = True


        try:

            torch.set_float32_matmul_precision(
                "high"
            )

        except Exception:

            pass


    else:

        print(

            "\nWARNING:"
        )

        print(

            "CUDA is NOT available."
        )

        print(

            "Do not full-train ResNet50 on CPU."
        )

        print(

            "Fix PyTorch CUDA first."
        )


    # ============================================================
    # DATA LOADERS
    # ============================================================

    train_loader = create_loader(

        train_df,

        train=True,

        device=device

    )


    validation_loader = create_loader(

        validation_df,

        train=False,

        device=device

    )


    test_loader = create_loader(

        test_df,

        train=False,

        device=device

    )


    # ============================================================
    # FIRST BATCH CHECK
    # ============================================================

    first_ecg, first_label = next(

        iter(
            train_loader
        )

    )


    print(

        "\n================================="
    )

    print(

        "FIRST BATCH CHECK"
    )

    print(

        "================================="
    )


    print(

        "ECG tensor shape:",

        tuple(
            first_ecg.shape
        )

    )


    print(

        "Label tensor shape:",

        tuple(
            first_label.shape
        )

    )


    print(

        "ECG mean:",

        float(
            first_ecg.mean()
        )

    )


    print(

        "ECG std:",

        float(
            first_ecg.std()
        )

    )


    # Expected:
    #
    # batch=4
    # lead=1
    # time=12000

    if (

        first_ecg.ndim != 3

        or

        first_ecg.shape[
            1
        ] != 1

        or

        first_ecg.shape[
            2
        ] != WINDOW_SAMPLES

    ):

        raise RuntimeError(

            f"Expected tensor "
            f"[B, 1, {WINDOW_SAMPLES}], "
            f"got "
            f"{tuple(first_ecg.shape)}"

        )


    # ============================================================
    # MODEL
    # ============================================================

    model = ResNet50_1D(

        use_se=USE_SE,

        dropout=DROPOUT

    )


    model = model.to(
        device
    )


    number_of_parameters = sum(

        parameter.numel()

        for parameter in (
            model.parameters()
        )

    )


    print(

        "\n================================="
    )

    print(

        "MODEL"
    )

    print(

        "================================="
    )


    print(

        "Architecture:",

        (
            "ResNet50-1D + SE"
            if USE_SE
            else
            "ResNet50-1D"
        )

    )


    print(

        "Parameters:",

        f"{number_of_parameters:,}"

    )


    print(

        "Physical batch size:",

        BATCH_SIZE

    )


    print(

        "Gradient accumulation:",

        GRAD_ACCUM_STEPS

    )


    print(

        "Effective batch size:",

        (
            BATCH_SIZE
            *
            GRAD_ACCUM_STEPS
        )

    )


    # ============================================================
    # CLASS IMBALANCE
    # ============================================================

    number_positive = int(

        (
            train_df[
                "label"
            ]
            ==
            1
        ).sum()

    )


    number_negative = int(

        (
            train_df[
                "label"
            ]
            ==
            0
        ).sum()

    )


    positive_weight_value = (

        number_negative
        /
        max(
            number_positive,
            1
        )

    )


    positive_weight = torch.tensor(

        positive_weight_value,

        dtype=torch.float32,

        device=device

    )


    print(

        "BCE positive weight:",

        f"{positive_weight_value:.4f}"

    )


    # ============================================================
    # LOSS
    # ============================================================

    loss_function = (
        nn.BCEWithLogitsLoss(

            pos_weight=
                positive_weight

        )
    )


    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=2,

        min_lr=1e-6

    )


    scaler = make_scaler(
        use_amp
    )


    # ============================================================
    # CHECKPOINT NAMES
    # ============================================================

    if SMOKE_TEST:

        best_model_path = (

            OUTPUT_DIR
            /
            "best_model_smoke.pt"

        )


        last_checkpoint_path = (

            OUTPUT_DIR
            /
            "last_checkpoint_smoke.pt"

        )


        history_path = (

            OUTPUT_DIR
            /
            "history_smoke.csv"

        )


    else:

        best_model_path = (

            OUTPUT_DIR
            /
            "best_model.pt"

        )


        last_checkpoint_path = (

            OUTPUT_DIR
            /
            "last_checkpoint.pt"

        )


        history_path = (

            OUTPUT_DIR
            /
            "history.csv"

        )


    # ============================================================
    # AUTO RESUME
    # ============================================================

    start_epoch = 1

    best_validation_loss = (
        float(
            "inf"
        )
    )

    bad_epochs = 0

    history = []


    if (

        AUTO_RESUME

        and

        last_checkpoint_path.exists()

    ):

        print(

            "\nLoading previous checkpoint:"
        )

        print(
            last_checkpoint_path
        )


        checkpoint = torch.load(

            last_checkpoint_path,

            map_location=device,

            weights_only=False

        )


        model.load_state_dict(

            checkpoint[
                "model"
            ]

        )


        optimizer.load_state_dict(

            checkpoint[
                "optimizer"
            ]

        )


        scaler.load_state_dict(

            checkpoint[
                "scaler"
            ]

        )


        start_epoch = (

            int(
                checkpoint[
                    "epoch"
                ]
            )
            +
            1

        )


        best_validation_loss = float(

            checkpoint[
                "best_val_loss"
            ]

        )


        bad_epochs = int(

            checkpoint[
                "bad_epochs"
            ]

        )


        if history_path.exists():

            history = (

                pd.read_csv(
                    history_path
                )
                .to_dict(
                    "records"
                )

            )


        print(

            f"Continuing from epoch "
            f"{start_epoch}"

        )


    # ============================================================
    # TRAIN
    # ============================================================

    print(

        "\n================================="
    )

    print(

        "START TRAINING"
    )

    print(

        "================================="
    )


    try:

        for epoch in range(

            start_epoch,

            training_epochs + 1

        ):

            epoch_start_time = time.time()


            model.train()


            optimizer.zero_grad(
                set_to_none=True
            )


            accumulated_loss = 0.0

            processed_samples = 0


            progress_bar = tqdm(

                train_loader,

                desc=(

                    f"Epoch "
                    f"{epoch}/"
                    f"{training_epochs}"

                )

            )


            for batch_index, (
                ecg,
                label
            ) in enumerate(

                progress_bar,

                start=1

            ):

                ecg = ecg.to(

                    device,

                    non_blocking=True

                )


                label = label.to(

                    device,

                    non_blocking=True

                )


                # --------------------------------
                # Forward
                # --------------------------------

                with autocast_context(
                    use_amp
                ):

                    logits = model(
                        ecg
                    )


                    raw_loss = loss_function(

                        logits,

                        label

                    )


                    loss = (

                        raw_loss
                        /
                        GRAD_ACCUM_STEPS

                    )


                # --------------------------------
                # Backpropagation
                # --------------------------------

                scaler.scale(
                    loss
                ).backward()


                # --------------------------------
                # Optimizer update
                # --------------------------------

                should_update = (

                    batch_index
                    %
                    GRAD_ACCUM_STEPS
                    ==
                    0

                )


                final_batch = (

                    batch_index
                    ==
                    len(
                        train_loader
                    )

                )


                if (

                    should_update

                    or

                    final_batch

                ):

                    scaler.unscale_(
                        optimizer
                    )


                    clip_grad_norm_(

                        model.parameters(),

                        max_norm=5.0

                    )


                    scaler.step(
                        optimizer
                    )


                    scaler.update()


                    optimizer.zero_grad(
                        set_to_none=True
                    )


                # --------------------------------
                # Statistics
                # --------------------------------

                current_batch_size = (
                    ecg.size(
                        0
                    )
                )


                accumulated_loss += (

                    float(
                        raw_loss.item()
                    )

                    *
                    current_batch_size

                )


                processed_samples += (
                    current_batch_size
                )


                average_loss = (

                    accumulated_loss
                    /
                    max(
                        processed_samples,
                        1
                    )

                )


                postfix = {

                    "loss":
                        f"{average_loss:.4f}"

                }


                if device.type == "cuda":

                    used_vram = (

                        torch.cuda
                        .memory_allocated()

                        /
                        (
                            1024 ** 3
                        )

                    )


                    postfix[
                        "VRAM"
                    ] = (
                        f"{used_vram:.2f}G"
                    )


                progress_bar.set_postfix(
                    **postfix
                )


            train_loss = (

                accumulated_loss
                /
                max(
                    processed_samples,
                    1
                )

            )


            # ====================================================
            # VALIDATION
            # ====================================================

            validation_loss, \
            validation_labels, \
            validation_probabilities = (
                evaluate_model(

                    model,

                    validation_loader,

                    device,

                    loss_function,

                    use_amp

                )
            )


            validation_metrics_05 = (
                compute_metrics(

                    validation_labels,

                    validation_probabilities,

                    threshold=0.5

                )
            )


            scheduler.step(
                validation_loss
            )


            current_learning_rate = (
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            )


            epoch_seconds = (

                time.time()
                -
                epoch_start_time

            )


            epoch_result = {

                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    validation_loss,

                "val_auroc":
                    validation_metrics_05[
                        "auroc"
                    ],

                "val_auprc":
                    validation_metrics_05[
                        "auprc"
                    ],

                "val_f1_at_0.5":
                    validation_metrics_05[
                        "f1"
                    ],

                "learning_rate":
                    current_learning_rate,

                "seconds":
                    epoch_seconds

            }


            history.append(
                epoch_result
            )


            pd.DataFrame(
                history
            ).to_csv(

                history_path,

                index=False

            )


            print(

                f"\nEpoch {epoch}: "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={validation_loss:.5f} | "
                f"AUROC={validation_metrics_05['auroc']:.4f} | "
                f"AUPRC={validation_metrics_05['auprc']:.4f} | "
                f"F1@0.5={validation_metrics_05['f1']:.4f} | "
                f"LR={current_learning_rate:.2e}"

            )


            # ====================================================
            # SAVE BEST MODEL
            # ====================================================

            improved = (

                validation_loss

                <

                best_validation_loss
                -
                1e-6

            )


            if improved:

                best_validation_loss = (
                    validation_loss
                )


                bad_epochs = 0


                torch.save(

                    {

                        "model":
                            model.state_dict(),

                        "epoch":
                            epoch,

                        "val_loss":
                            validation_loss,

                        "window_sec":
                            WINDOW_SEC,

                        "sample_rate":
                            FS,

                        "lead_index":
                            LEAD_INDEX,

                        "use_se":
                            USE_SE

                    },

                    best_model_path

                )


                print(

                    "Saved BEST model:"
                )

                print(
                    best_model_path
                )


            else:

                bad_epochs += 1


            # ====================================================
            # SAVE RESUMABLE CHECKPOINT
            # ====================================================

            torch.save(

                {

                    "model":
                        model.state_dict(),

                    "optimizer":
                        optimizer.state_dict(),

                    "scaler":
                        scaler.state_dict(),

                    "epoch":
                        epoch,

                    "best_val_loss":
                        best_validation_loss,

                    "bad_epochs":
                        bad_epochs

                },

                last_checkpoint_path

            )


            # ====================================================
            # EARLY STOPPING
            # ====================================================

            if (

                bad_epochs
                >=
                EARLY_STOPPING_PATIENCE

            ):

                print(

                    "\nEarly stopping."
                )

                break


    # ============================================================
    # CUDA OOM HANDLING
    # ============================================================

    except torch.cuda.OutOfMemoryError:

        print(

            "\n================================="
        )

        print(

            "CUDA OUT OF MEMORY"
        )

        print(

            "================================="
        )


        print(

            "\nFor RTX 5050 8 GB:"
        )


        print(

            "Change:"
        )


        print(

            "BATCH_SIZE = 2"
        )


        print(

            "GRAD_ACCUM_STEPS = 16"
        )


        print(

            "\nEffective batch stays 32."
        )


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        raise


    # ============================================================
    # LOAD BEST MODEL
    # ============================================================

    if not best_model_path.exists():

        raise RuntimeError(

            "No best model checkpoint found."

        )


    checkpoint = torch.load(

        best_model_path,

        map_location=device,

        weights_only=False

    )


    model.load_state_dict(

        checkpoint[
            "model"
        ]

    )


    # ============================================================
    # VALIDATION THRESHOLD
    # ============================================================

    validation_loss, \
    validation_labels, \
    validation_probabilities = (
        evaluate_model(

            model,

            validation_loader,

            device,

            loss_function,

            use_amp

        )
    )


    selected_threshold = choose_threshold(

        validation_labels,

        validation_probabilities,

        method=THRESHOLD_METHOD

    )


    validation_metrics = compute_metrics(

        validation_labels,

        validation_probabilities,

        threshold=selected_threshold

    )


    validation_metrics[
        "loss"
    ] = float(
        validation_loss
    )


    # ============================================================
    # FINAL TEST
    # ============================================================

    test_loss, \
    test_labels, \
    test_probabilities = (
        evaluate_model(

            model,

            test_loader,

            device,

            loss_function,

            use_amp

        )
    )


    test_metrics = compute_metrics(

        test_labels,

        test_probabilities,

        threshold=selected_threshold

    )


    test_metrics[
        "loss"
    ] = float(
        test_loss
    )


    test_metrics[
        "threshold_chosen_on"
    ] = "validation"


    test_metrics[
        "threshold_method"
    ] = THRESHOLD_METHOD


    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print(

        "\n================================="
    )

    print(

        "VALIDATION RESULTS"
    )

    print(

        "================================="
    )


    print(

        json.dumps(

            validation_metrics,

            indent=2

        )

    )


    print(
        "\n================================="
    )
    print(
        "INTERNAL TEST RESULTS"
    )
    print(
        "UNSEEN IRIDIA-AF PATIENTS"
    )
    print(

        "================================="
    )
    print(

        json.dumps(

            test_metrics,

            indent=2

        )

    )


    # ============================================================
    # SAVE RESULTS
    # ============================================================

    if SMOKE_TEST:

        experiment_tag = "smoke"

    else:

        experiment_tag = "full"


    save_json(

        validation_metrics,

        OUTPUT_DIR
        /
        f"val_metrics_{experiment_tag}.json"

    )


    save_json(

        test_metrics,

        OUTPUT_DIR
        /
        f"test_metrics_{experiment_tag}.json"

    )


    validation_predictions = pd.DataFrame(

        {

            "y_true":
                validation_labels,

            "p_af":
                validation_probabilities

        }

    )


    validation_predictions.to_csv(

        OUTPUT_DIR
        /
        f"val_predictions_{experiment_tag}.csv",

        index=False

    )


    test_predictions = pd.DataFrame(

        {

            "y_true":
                test_labels,

            "p_af":
                test_probabilities

        }

    )


    test_predictions.to_csv(

        OUTPUT_DIR
        /
        f"test_predictions_{experiment_tag}.csv",

        index=False

    )
    configuration = {

        "sample_rate":
            FS,

        "lead_index":
            LEAD_INDEX,

        "window_sec":
            WINDOW_SEC,

        "window_samples":
            WINDOW_SAMPLES,

        "train_stride_sec":
            TRAIN_STRIDE_SEC,

        "eval_stride_sec":
            EVAL_STRIDE_SEC,

        "calibration_sec":
            CALIBRATION_SEC,

        "label_mode":
            LABEL_MODE,

        "af_positive_threshold":
            AF_POSITIVE_THRESHOLD,

        "train_ratio":
            TRAIN_RATIO,

        "val_ratio":
            VAL_RATIO,

        "test_ratio":
            TEST_RATIO,

        "model":
            (
                "ResNet50-1D + SE"
                if USE_SE
                else
                "ResNet50-1D"
            ),

        "parameters":
            number_of_parameters,

        "batch_size":
            BATCH_SIZE,

        "grad_accum_steps":
            GRAD_ACCUM_STEPS,

        "effective_batch_size":
            (
                BATCH_SIZE
                *
                GRAD_ACCUM_STEPS
            ),
        "mixed_precision":
            use_amp,

        "validation_threshold":
            selected_threshold,

        "smoke_test":
            SMOKE_TEST

    }


    save_json(

        configuration,

        OUTPUT_DIR
        /
        f"config_{experiment_tag}.json"

    )


    print(

        "\n================================="
    )

    print(

        "FINISHED"
    )

    print(

        "================================="
    )


    print(

        "Results directory:"
    )


    print(

        OUTPUT_DIR.resolve()

    )


# ================================================================
# WINDOWS ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()