import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np, pandas as pd, torch
from run_validation import prepare, Net
def test_normal_q_is_masked_not_zero_target():
 d=pd.DataFrame(dict(kT=[.1],JA=[.2],mu=[0.],delta_opt=[0.],q_opt=[0.],free_energy_gap_to_normal=[0.],topology_bulk_gap=[np.nan],topology_k_at_bulk_gap=[np.nan],topology_p0=[np.nan],topology_ppi=[np.nan],topology_pfaffian_margin=[np.nan],trusted_exact=[1],topology_trusted=[0],q_status=[0],q_unresolved=[0]))
 _,y,m=prepare(d); assert y[0,1]==0 and not m[0,1]
def test_heteroscedastic_shapes_and_finite_loss():
 net=Net(); x=torch.zeros((4,3)); mu,lv=net(x); assert mu.shape==(4,8) and lv.shape==(4,8); assert torch.isfinite(mu).all() and torch.isfinite(lv).all()
