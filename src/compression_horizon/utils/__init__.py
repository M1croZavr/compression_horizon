from __future__ import annotations

from typing import Optional


def to_mean_std_cell(
    val_mean: Optional[float], val_std: Optional[float], is_int: bool = False, use_latex: bool = True, float_precision=4
) -> str:
    if val_mean is None:
        return ""
    if is_int:
        mean_str = f"{int(round(val_mean))}"
        std_str = f"{int(round(val_std))}" if val_std is not None else "0"
    else:
        if float_precision == 0:
            mean_round = round(val_mean)
            std_round = round(val_std)
            mean_str = f"{mean_round}"
            std_str = f"{std_round}" if val_std is not None else "0"
        else:
            mean_round = round(val_mean, float_precision)
            std_round = round(val_std, float_precision)
            mean_str = f"{mean_round}".rstrip("0").rstrip(".")
            std_str = f"{std_round}".rstrip("0").rstrip(".") if val_std is not None else "0"

    if use_latex:
        return f"{mean_str} {{" + "\\small $\\pm$ " + f"{std_str}}}"

    if float(std_str) == 0:
        return mean_str

    return f"{mean_str} ± {std_str}"
