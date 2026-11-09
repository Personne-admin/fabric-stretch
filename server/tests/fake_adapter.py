import argparse
import json
import os
import sys

ADAPTER_VERSION = "0.0-fake"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--role", default="diffuse")
    parser.add_argument("--grid-cols", type=int, default=0)
    parser.add_argument("--grid-rows", type=int, default=0)
    parser.add_argument("--reference-error-limit", type=float, default=2.0)
    parser.add_argument("--output")
    parser.add_argument("--csv-output")
    parser.add_argument("--diagnostics-dir")
    parser.add_argument("--dry-run", action="store_true")
    args, rest = parser.parse_known_args()

    if os.path.basename(args.run_dir.rstrip("/")) == os.environ.get(
        "FAKE_ADAPTER_FAIL"
    ):
        print("error: simulated failure", file=sys.stderr)

        return 1
    with open(os.path.join(args.run_dir, "frames.json")) as fh:
        frames = json.load(fh)["frames"]
    key = f"image_{args.role}"
    ids = [f["frame_id"] for f in frames if key in f]
    if args.dry_run:
        print(f"fake dry run: {len(ids)} frames, role {args.role}")

        return 0

    points = [f"p{i}" for i in range(args.grid_cols * args.grid_rows)]
    bad = os.environ.get("FAKE_ADAPTER_BAD_REFERENCE") == "1"
    doc = {
        "format": "fabsim-tracks",
        "version": 1,
        "provenance": {
            "adapter_git_commit": os.environ.get("FABSIM_ADAPTER_COMMIT", ""),
            "checkpoint_sha256": os.environ.get("COWTRACKER_CHECKPOINT_SHA256", ""),
            "reference_error_limit_px": args.reference_error_limit,
            "source": os.environ.get("FABSIM_RUN_KEY", ""),
            "reference_errors_px": {p: 5.0 if bad else 0.0 for p in points},
            "runtime": {
                "gpu": "Fake GPU",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        },
        "frames": [
            {
                "frame_id": i,
                "points": [
                    {"id": p, "x": 1.0, "y": 2.0, "confidence": 0.9} for p in points
                ],
            }
            for i in ids
        ],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(doc, fh)

    with open(os.path.join(os.path.dirname(args.output), "argv.txt"), "w") as fh:
        fh.write("\n".join(sys.argv[1:]) + "\n")

    print(f"wrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
