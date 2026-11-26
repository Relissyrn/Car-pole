# MuJoCo Box-Pushing

This project adds a configurable box-pushing task using one shared
Gymnasium/MuJoCo environment. Curriculum stages, action latency, action noise,
and parameter mismatch all operate on that same task.

Run the environment smoke checks with:

```powershell
python check_box_push.py
```

See [BOX_PUSH_SPEC.md](BOX_PUSH_SPEC.md) for the observation, reward, staged
training instructions, multi-seed protocol, fixed evaluation starts, failure
reporting, and heat-map outputs.
