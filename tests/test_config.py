"""Regression tests for the composable CLI argument groups in compression_horizon.config.

These tests are CPU-only and fast. They guard the Stage 0 cleanup:

* the new dataclasses instantiate with sane defaults,
* their defaults match the legacy ``MyTrainingArguments`` (no silent drift),
* multiple groups compose via ``HfArgumentParser`` without field collisions,
* the three previously-duplicated fields on ``MyTrainingArguments`` now appear
  exactly once each.
"""

from __future__ import annotations

import dataclasses

import pytest
from transformers import HfArgumentParser

from compression_horizon.config import (
    AlignmentArgs,
    CompressionArgs,
    DataArgs,
    EvalArgs,
    LowDimArgs,
    ModelArgs,
    ProgressiveArgs,
)
from compression_horizon.train.arguments import MyTrainingArguments

# ---------------------------------------------------------------------------
# Stage 0 hygiene: duplicate-field guard on MyTrainingArguments.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name, expected_default",
    [
        ("max_optimization_steps_per_sample", 1_000),
        ("max_optimization_steps_per_token", 1_000),
        ("random_seed", 42),
        ("fix_position_ids", False),
    ],
)
def test_my_training_arguments_no_duplicates(field_name: str, expected_default):
    """Each previously-duplicated field is now declared exactly once with the same default."""
    field_names = [f.name for f in dataclasses.fields(MyTrainingArguments)]
    assert field_names.count(field_name) == 1, (
        f"{field_name} is declared multiple times in MyTrainingArguments — " f"this re-introduces the Stage 0 bug."
    )
    f = MyTrainingArguments.__dataclass_fields__[field_name]
    assert f.default == expected_default


# ---------------------------------------------------------------------------
# Each new dataclass instantiates and round-trips through HfArgumentParser.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataclass_cls",
    [
        ModelArgs,
        DataArgs,
        CompressionArgs,
        AlignmentArgs,
        LowDimArgs,
        ProgressiveArgs,
    ],
)
def test_dataclass_instantiates_with_defaults(dataclass_cls):
    """All groups except EvalArgs (which has a required field) instantiate with no args."""
    instance = dataclass_cls()
    assert isinstance(instance, dataclass_cls)


def test_eval_args_requires_dataset_path():
    """EvalArgs.embeddings_dataset_path has no default and must be provided."""
    parser = HfArgumentParser(EvalArgs)
    with pytest.raises(SystemExit):
        # argparse exits on missing required arg
        parser.parse_args_into_dataclasses([])
    (eval_args,) = parser.parse_args_into_dataclasses(["--embeddings_dataset_path", "/tmp/dummy"])
    assert eval_args.embeddings_dataset_path == "/tmp/dummy"


# ---------------------------------------------------------------------------
# Cross-dataclass composition via HfArgumentParser.
# ---------------------------------------------------------------------------


def test_no_field_name_collisions_across_groups():
    """Composing every group via HfArgumentParser must not raise on duplicate field names."""
    # The constructor is what raises on duplicate dest names.
    parser = HfArgumentParser(
        (
            ModelArgs,
            DataArgs,
            CompressionArgs,
            AlignmentArgs,
            LowDimArgs,
            ProgressiveArgs,
            EvalArgs,
        )
    )
    # Sanity-check: parser actually contains the union of fields.
    dests = {a.dest for a in parser._actions}  # noqa: SLF001 (private API, but stable)
    expected_fields = set()
    for cls in (
        ModelArgs,
        DataArgs,
        CompressionArgs,
        AlignmentArgs,
        LowDimArgs,
        ProgressiveArgs,
        EvalArgs,
    ):
        expected_fields.update(f.name for f in dataclasses.fields(cls))
    missing = expected_fields - dests
    assert not missing, f"HfArgumentParser dropped fields: {missing}"


def test_evaluation_script_style_composition_parses_cli():
    """Mimic the evaluation-script entry point: parse a typical CLI invocation."""
    parser = HfArgumentParser((ModelArgs, DataArgs, CompressionArgs, EvalArgs))
    model_args, data_args, comp_args, eval_args = parser.parse_args_into_dataclasses(
        [
            "--model_checkpoint",
            "EleutherAI/pythia-160m",
            "--dtype",
            "bf16",
            "--dataset_name",
            "mrsndmn/pg19",
            "--max_sequence_length",
            "64",
            "--limit_dataset_items",
            "4",
            "--no_bos_token",
            "--number_of_mem_tokens",
            "1",
            "--embedding_init_method",
            "random0.02",
            "--embeddings_dataset_path",
            "/tmp/dataset",
            "--limit_samples",
            "8",
        ]
    )
    assert model_args.model_checkpoint == "EleutherAI/pythia-160m"
    assert data_args.max_sequence_length == 64
    assert data_args.limit_dataset_items == 4
    assert data_args.no_bos_token is True
    assert comp_args.embedding_init_method == "random0.02"
    assert eval_args.embeddings_dataset_path == "/tmp/dataset"
    assert eval_args.limit_samples == 8


# ---------------------------------------------------------------------------
# Default-value parity with the legacy MyTrainingArguments.
#
# This matters because some scripts will eventually be migrated from
# MyTrainingArguments to the composed dataclasses; we want the migration to
# preserve behaviour. If the legacy default ever changes deliberately, this
# test pins the divergence and forces us to make the change in both places.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group_cls, field_pairs",
    [
        (
            ModelArgs,
            [("model_checkpoint", "model_checkpoint"), ("dtype", "dtype")],
        ),
        (
            DataArgs,
            [
                ("dataset_name", "dataset_name"),
                ("max_sequence_length", "max_sequence_length"),
                ("limit_dataset_items", "limit_dataset_items"),
                ("offset_dataset_items", "offset_dataset_items"),
                ("no_bos_token", "no_bos_token"),
            ],
        ),
        (
            CompressionArgs,
            [
                ("number_of_mem_tokens", "number_of_mem_tokens"),
                ("embedding_init_method", "embedding_init_method"),
                ("embedding_init_path", "embedding_init_path"),
                ("load_from_disk_embedding_init_method", "load_from_disk_embedding_init_method"),
                ("pretrained_pca_num_components", "pretrained_pca_num_components"),
                ("pretrained_pca_path", "pretrained_pca_path"),
                ("fix_position_ids", "fix_position_ids"),
            ],
        ),
        (
            AlignmentArgs,
            [
                ("loss_type", "loss_type"),
                ("hybrid_alpha", "hybrid_alpha"),
                ("num_alignment_layers", "num_alignment_layers"),
                ("inverted_alignment", "inverted_alignment"),
            ],
        ),
        (
            LowDimArgs,
            [
                ("low_dim_projection", "low_dim_projection"),
                ("low_dim_projection_global", "low_dim_projection_global"),
                ("low_dim_size", "low_dim_size"),
                ("low_dim_projection_checkpoint", "low_dim_projection_checkpoint"),
                ("low_dim_projection_train", "low_dim_projection_train"),
            ],
        ),
        (
            ProgressiveArgs,
            [
                ("progressive_train", "progressive_train"),
                ("progressive_min_seq_len", "progressive_min_seq_len"),
                ("progressive_step", "progressive_step"),
                ("progressive_convergence_threshold", "progressive_convergence_threshold"),
                ("progressive_max_stages", "progressive_max_stages"),
                (
                    "progressive_reset_lr_scheduler_on_non_convergence",
                    "progressive_reset_lr_scheduler_on_non_convergence",
                ),
                ("max_optimization_steps_per_token", "max_optimization_steps_per_token"),
                ("save_progressive_artifacts", "save_progressive_artifacts"),
            ],
        ),
    ],
)
def test_defaults_match_legacy_my_training_arguments(group_cls, field_pairs):
    """Defaults of every new dataclass field must equal the corresponding legacy field."""
    legacy_fields = MyTrainingArguments.__dataclass_fields__
    new_fields = group_cls.__dataclass_fields__

    for new_name, legacy_name in field_pairs:
        assert legacy_name in legacy_fields, f"Sanity check failed: {legacy_name!r} is not on MyTrainingArguments"
        legacy_default = legacy_fields[legacy_name].default
        new_default = new_fields[new_name].default
        assert new_default == legacy_default, (
            f"{group_cls.__name__}.{new_name} default {new_default!r} differs from "
            f"MyTrainingArguments.{legacy_name} default {legacy_default!r}. "
            f"If this divergence is intentional, update both sides explicitly."
        )
