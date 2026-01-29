import argparse
import re

from mls.manager.job.utils import training_job_api_from_profile

_DURATION_RE = re.compile(r"(\d+)\s*([smhd])", flags=re.IGNORECASE)


def _parse_duration_seconds(duration: str | None) -> int | None:
    if not duration:
        return None

    duration = duration.strip()
    if not duration:
        return None

    if duration.isdigit():
        return int(duration)

    total = 0
    for value, unit in _DURATION_RE.findall(duration):
        v = int(value)
        u = unit.lower()
        if u == "s":
            total += v
        elif u == "m":
            total += 60 * v
        elif u == "h":
            total += 3600 * v
        elif u == "d":
            total += 86400 * v
    return total or None


def _format_hhmmss(total_seconds: int) -> str:
    seconds = int(total_seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--description_filter", type=str, required=True)

    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--offset", default=0, type=int)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print matched jobs and parsed durations.",
    )
    args = parser.parse_args()

    client, extra_options = training_job_api_from_profile("default")

    all_jobs = client.get_list_jobs(
        region=extra_options["region"],
        status=None,
        allocation_name="alloc-officecds-multimodal-2-sr004",
        limit=args.limit,
        offset=args.offset,
    )

    filters = [f.strip() for f in args.description_filter.split(",") if f.strip()]
    if not filters:
        raise ValueError("--description_filter is empty after parsing")

    matched_jobs: list[tuple[dict, int | None]] = []
    total_seconds = 0
    known_durations = 0
    unknown_durations = 0

    for job in all_jobs.get("jobs", []):
        desc = job.get("job_desc", "")
        if not any(f in desc for f in filters):
            continue

        duration_seconds = _parse_duration_seconds(job.get("duration"))
        matched_jobs.append((job, duration_seconds))
        if duration_seconds is None:
            unknown_durations += 1
            continue

        total_seconds += duration_seconds
        known_durations += 1

    print(f"Matched jobs: {len(matched_jobs)}")
    print(f"Known durations: {known_durations}")
    print(f"Unknown durations: {unknown_durations}")
    print(f"Total duration: {total_seconds}s ({_format_hhmmss(total_seconds)})")

    if args.verbose and matched_jobs:
        print("\nMatched job durations:")
        for job, duration_seconds in matched_jobs:
            job_name = job.get("job_name", "")
            status = job.get("status", "")
            duration_str = job.get("duration", "")
            if duration_seconds is None:
                print(f"- {job_name} [{status}]: {duration_str} (unparsed)")
            else:
                print(f"- {job_name} [{status}]: {duration_str} ({duration_seconds}s)")
