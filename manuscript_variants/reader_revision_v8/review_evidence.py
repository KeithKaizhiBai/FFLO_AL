"""Read-only saved-label reconciliation; no oracle imports or physics calls."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"output/pdf/manuscript_reader_revision_v8"
ARCH=ROOT/"downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive"

def main():
    table=OUT/"tables";table.mkdir(exist_ok=True)
    raw=ROOT/"manuscript_langgraph/recovered_artifacts/stageIV/dataset_iter025.npz"
    cloud=ROOT/"figures/fig4_strict_data_v2/canonical/fig4_exact_3d_samples.csv"
    with np.load(raw,allow_pickle=False) as z:
        d=pd.read_csv(cloud)
        assert np.allclose(d[["kT","JA","mu"]],z["x"],rtol=0,atol=1e-12)
        assert np.array_equal(d.phase_code,z["y_phase"])
        base=z["topology_trusted"].astype(bool)&(z["topology_spectral_status_code"]==0)&(z["y_phase"]!=0)
        extra=base&(z["y_phase"]==1)
        rows=d.loc[extra,["kT","JA","mu","phase_code"]].copy()
        rows.insert(0,"endpoint_row_index",np.flatnonzero(extra))
        rows["stored_topology_label_code"]=z["topology_label_code"][extra]
        rows.to_csv(table/"R12_endpoint_uniform_sc_records.csv",index=False)
        counts=[]
        for phase in [1,2]:
            counts.append({"phase_code":phase,"stored_nu0":int(np.sum(base&(z["y_phase"]==phase)&(z["topology_label_code"]==0))),"stored_nu1":int(np.sum(base&(z["y_phase"]==phase)&(z["topology_label_code"]==1)))})
        pd.DataFrame(counts).to_csv(table/"R12_endpoint_reconciliation.csv",index=False)
        assert counts==[{"phase_code":1,"stored_nu0":120,"stored_nu1":21},{"phase_code":2,"stored_nu0":3145,"stored_nu1":1445}]
    rows=[]
    for center in [-.45,.15,.85,1.45]:
        b=d[np.abs(d.mu-center)<=.025+1e-12]
        c=int((b.topology_name=="cFFLO").sum());t=int((b.topology_name=="tFFLO").sum())
        rows.append({"mu_center":center,"half_width":.025,"all_retained":len(b),"FFLO":int((b.phase_code==2).sum()),"trusted_gapped_FFLO":c+t,"cFFLO":c,"tFFLO":t,"outside_displayed_topology_subset":len(b)-c-t})
    pd.DataFrame(rows).to_csv(table/"R01_proposed_bin_topology_counts.csv",index=False)
    paths=[raw,cloud,ROOT/"docs/paper_review/20260912_v7/codex_revision_tasks.md"]
    for stage in ["stage_i","stage_ii","stage_iii","stage_iv"]:
        paths.extend((ARCH/"data_archive"/stage).glob("*/run_config.json"))
        paths.extend((ARCH/"data_archive"/stage).glob("*/stop_state.json"))
    for stage in ["stage_iii","stage_iv"]:
        for name in ["exact_oracle.py","acquisition.py","active_refine.py","models.py","stageiv_3d.py"]:
            p=ARCH/"code_archive"/stage/"ml_phase"/name
            if p.exists():paths.append(p)
    paths.extend([ARCH/"code_archive/canonical/eta_phase_diagram_cuda.py",ARCH/"code_archive/canonical/tfflo_1d_cuda.py",ARCH/"code_archive/canonical/scripts/build_topology_convergence_audit.py"])
    records={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/"method_source_hashes.json").write_text(json.dumps(records,indent=2))
    (OUT/"evidence_checks.json").write_text(json.dumps({"endpoint_coordinate_order_matches":True,"endpoint_phase_labels_match":True,"R12_extra_rows":141,"R12_identity_checked":"endpoint only; not full history","R01_bin_counts":rows,"new_physics":False},indent=2))

if __name__=="__main__":main()
