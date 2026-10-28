import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np,pandas as pd
from run_stage_vi0 import level_data
def row(**kw):
 d=dict(kT=.1,JA=.2,mu=0.,delta_opt=0.,free_energy_gap_to_normal=0.,trusted_exact=1,training_eligible_exact=1,delta_unresolved=0,q_opt=0.,q_status=0,q_unresolved=0,topology_bulk_gap=np.nan,topology_k_at_bulk_gap=np.nan,topology_applicable=0,topology_pending=0,topology_spectral_status_code=-1,topology_p0=np.nan,topology_ppi=np.nan,topology_pfaffian_margin=np.nan,topology_trusted=0);d.update(kw);return pd.DataFrame([d])
def test_normal_is_level1_valid_but_branch_and_topology_not_applicable():
 d=row();assert level_data(d,"level1")[2].all();assert not level_data(d,"level2")[2].any();y=level_data(d,"level4")[1];assert not level_data(d,"level4")[2].any() and np.isfinite(y).all()
def test_trusted_gapped_sc_is_valid_at_all_levels():
 d=row(delta_opt=.2,free_energy_gap_to_normal=-.01,q_opt=.3,q_status=1,topology_bulk_gap=.1,topology_k_at_bulk_gap=1.,topology_applicable=1,topology_spectral_status_code=0,topology_p0=.2,topology_ppi=-.3,topology_pfaffian_margin=.2,topology_trusted=1);assert level_data(d,"level2")[2].all();assert level_data(d,"level3")[2].all();assert level_data(d,"level4")[2].all()
