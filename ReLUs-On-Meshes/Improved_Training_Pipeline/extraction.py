import torch, numpy as np, sys
ckpt = torch.load(sys.argv[1], map_location="cpu")
F = ckpt["F"].detach().cpu().numpy()
beta = 1.0
tc = ckpt.get("temp_ctrl")
if isinstance(tc, dict):
    beta = float(tc.get("beta") or tc.get("beta_c") or tc.get("beta_contour") or 1.0)
else:
    for k in ("beta", "beta_c", "beta_contour"):
        if hasattr(tc, k):
            beta = float(getattr(tc, k)); break
np.savez_compressed("field_values_245000.npz",
                    field_values=F, step=ckpt.get("step", -1), beta=beta)
print("Saved field_values_245000.npz:", F.shape, "beta=", beta)