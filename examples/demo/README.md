# Demo project

A downscaled `warp` capture with eight states per branch, also used by the tests.
The curves are coarse, the tiles small, and the force is close to load-cell noise.
Use full measurements for quantitative work.

## Regenerate

From a real capture:

```sh
python scripts/make_demo_dataset.py $DATA/fabric_002 --led-config $DATA/led_config.json
```

To rebuild just the derived files (copy the demo first if experimenting):

```sh
fabsim ps examples/demo --force
fabsim package examples/demo --force --tiling-window-size 8 16 --tiling-shift 3 --tiling-interp-dist 1
```

The default tiling window is too large for this demo. Force-quality warnings are expected.
