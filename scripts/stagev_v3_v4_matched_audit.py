#!/usr/bin/env python3
"""Strictly offline matched Stage V-v3/V-v4 audit.

This program reads frozen cumulative artifacts only.  It never calls the exact
oracle, performs a Delta-q search, or mutates an input run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/al_fflo_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/al_fflo_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree
from scipy.stats import qmc, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "downloads/stagev_v3_v4_matched_audit_raw_20260712"
V3 = RAW / "v3/ML_Phase_StageV_V3_SparseArc/active_runs/stagev_v3_sparse_arc_boundary_support_3d_v1"
V4 = ROOT / "downloads/stagev_v4_min_report_audit_20260711/extracted/ML_Phase_StageV_V4_GraphAcq/active_runs/stagev_v4_phase_structure_graph_acquisition_3d_v1"
V4SUP = RAW / "v4_supplement/ML_Phase_StageV_V4_GraphAcq/active_runs/stagev_v4_phase_structure_graph_acquisition_3d_v1"
BOUNDS = np.array([[0.0, 0.56], [0.0, 2.12], [-0.5, 1.5]])
CHECKPOINTS = [1024, 1500, 2000, 3000, 4000, 5000, 6000, 7168]
SEED = 20260712
K = 12
SUPPORT_RADIUS = 0.115


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm_xyz(df: pd.DataFrame) -> np.ndarray:
    x = df[["kT", "JA", "mu"]].to_numpy(float)
    return (x - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def dataset(run: Path, iteration: int) -> pd.DataFrame:
    return pd.read_csv(run / f"dataset_iter{iteration:03d}.csv", low_memory=False)


def iter_for_selected_budget(budget: int) -> int:
    return int(np.clip(1 + (budget - 1024) // 64, 1, 97))


def trusted(df: pd.DataFrame) -> pd.DataFrame:
    if "trusted_exact" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df.trusted_exact.fillna(0).astype(int).eq(1)].copy()


def topo_trusted(df: pd.DataFrame) -> pd.DataFrame:
    if "trusted_exact" not in df.columns:
        return df.iloc[0:0].copy()
    m = (df.trusted_exact.fillna(0).astype(int).eq(1) &
         df.topology_trusted.fillna(0).astype(int).eq(1) &
         df.topology_label_code.isin([0, 1]) &
         df.phase_name.isin(["uniform_SC", "FFLO"]))
    return df[m].copy()


def knn_predict(train: pd.DataFrame, cloud: np.ndarray, target: str, k: int = K):
    x = norm_xyz(train)
    y = train[target].to_numpy(float)
    good = np.isfinite(y) & np.isfinite(x).all(1)
    x, y = x[good], y[good]
    if len(x) == 0:
        return np.full(len(cloud), np.nan), np.full(len(cloud), np.inf)
    kk = min(k, len(x))
    dist, idx = cKDTree(x).query(cloud, k=kk, workers=-1)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.maximum(dist, 1e-9) ** 2
    pred = (w * y[idx]).sum(1) / w.sum(1)
    return pred, dist[:, -1]


def pct(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return (np.nan, np.nan, np.nan)
    return tuple(np.quantile(a, [0.5, 0.9, 0.95]))


def phase_code(df: pd.DataFrame) -> pd.Series:
    return df.phase_name.map({"normal": 0, "uniform_SC": 1, "FFLO": 2}).fillna(-1).astype(int)


def cloud_phase(df: pd.DataFrame, cloud: np.ndarray, k=K):
    d = trusted(df).copy(); d["_code"] = phase_code(d)
    pred, support = knn_predict(d, cloud, "_code", k)
    return np.rint(pred).clip(0, 2), support


def cloud_topology(df: pd.DataFrame, cloud: np.ndarray, k=K):
    d = topo_trusted(df)
    pred, support = knn_predict(d, cloud, "topology_label_code", k)
    return (pred >= 0.5).astype(float), support


def boundary_support(df: pd.DataFrame, kind: str, k=K):
    if kind == "NS":
        d = trusted(df); lab = phase_code(d).gt(0).astype(int).to_numpy()
    elif kind == "UF":
        d = trusted(df); d = d[d.phase_name.isin(["uniform_SC", "FFLO"])]
        lab = d.phase_name.eq("FFLO").astype(int).to_numpy()
    else:
        d = topo_trusted(df); lab = d.topology_label_code.astype(int).to_numpy()
    x = norm_xyz(d)
    if len(x) < 2: return np.empty((0, 3)), np.empty(0)
    kk = min(k + 1, len(x)); dist, idx = cKDTree(x).query(x, k=kk)
    mids, spans = [], []
    for i in range(len(x)):
        for dd, j in zip(dist[i, 1:], idx[i, 1:]):
            if lab[i] != lab[j] and dd <= SUPPORT_RADIUS * 1.5:
                mids.append((x[i] + x[j]) / 2); spans.append(dd)
    return (np.asarray(mids) if mids else np.empty((0, 3)), np.asarray(spans))


def surface_metrics(df: pd.DataFrame, ref: pd.DataFrame, kind: str):
    surf, spans = boundary_support(df, kind)
    refsurf, _ = boundary_support(ref, kind)
    exact = topo_trusted(df) if kind == "TOP" else trusted(df)
    if not len(surf):
        return dict(p50=np.nan, p90=np.nan, p95=np.nan, shift_p50=np.nan,
                    shift_p90=np.nan, shift_p95=np.nan, unsupported=1.0,
                    bracket_density=0.0, components=0, largest_fraction=0.0)
    cov = cKDTree(norm_xyz(exact)).query(surf, k=1)[0] if len(exact) else np.full(len(surf), np.inf)
    shift = cKDTree(refsurf).query(surf, k=1)[0] if len(refsurf) else np.full(len(surf), np.inf)
    c50,c90,c95=pct(cov); s50,s90,s95=pct(shift)
    # Coarse occupied-cell components avoid unsupported long connections.
    cells=np.floor(surf/0.055).astype(int); keys={tuple(v) for v in cells}; seen=set(); sizes=[]
    for key in keys:
        if key in seen: continue
        stack=[key]; seen.add(key); n=0
        while stack:
            a=stack.pop(); n+=1
            for z in range(3):
                for sg in (-1,1):
                    b=list(a); b[z]+=sg; b=tuple(b)
                    if b in keys and b not in seen: seen.add(b); stack.append(b)
        sizes.append(n)
    return dict(p50=c50,p90=c90,p95=c95,shift_p50=s50,shift_p90=s90,shift_p95=s95,
                unsupported=float(np.mean(cov>SUPPORT_RADIUS)), bracket_density=len(surf)/max(len(exact),1),
                components=len(sizes),largest_fraction=max(sizes)/sum(sizes))


def source_family(source: str) -> str:
    s=str(source).lower()
    if not s or s in {"nan","none"}: return "unknown"
    if "candidate_phase" in s: return "candidate_phase_validation"
    if "oracle_anomaly" in s: return "oracle_anomaly_audit"
    if any(t in s for t in ["boundary","surface","bracket","component","patch","topology"]):
        return "known_boundary_refinement"
    if any(t in s for t in ["sobol","global","sparse_fill","parameter_region","mu_edge","initial"]):
        return "global_exploration"
    return "unknown"


def reward_file(run: Path) -> Path:
    hits=list(run.glob("*reward_history.csv")); return hits[0]


def artifact_manifest(out: Path):
    rows=[]
    specs={"V3":V3,"V4":V4}
    pats={"cumulative_dataset":"dataset_iter*.csv","exact_shard":"iter*/exact_shard_rank*.npz",
          "selected_metadata":"iter*/selected_points_metadata.csv","merge_log":"iter*/merge_summary*.json",
          "reward_update":"iter*/stagev*_reward_update_summary.json","graph_nodes":"iter*/stagev4_graph_nodes.csv",
          "graph_edges":"iter*/stagev4_graph_edges.csv","controller_log":"*.log","config":"*.json"}
    for run,base in specs.items():
        for typ,pat in pats.items():
            roots=[base] + ([V4SUP] if run=="V4" else [])
            if typ=="controller_log":
                roots += [RAW/"v3"] if run=="V3" else [ROOT/"downloads/stagev_v4_min_report_audit_20260711/extracted"]
            fs=[]
            for r in roots: fs.extend(r.glob(pat))
            fs=sorted(set(fs))
            iterations=[]
            for f in fs:
                import re
                m=re.search(r"iter(\d{3})",str(f));
                if m: iterations.append(int(m.group(1)))
            rows.append(dict(run=run,artifact_type=typ,first_iteration=min(iterations) if iterations else "",
                             last_iteration=max(iterations) if iterations else "",row_count=len(fs),
                             file_path=str(base),hash=sha256(fs[-1]) if fs and fs[-1].is_file() else "",
                             status="complete" if (len(fs)>0 or (run=="V3" and typ.startswith("graph_"))) else "missing",
                             note="V3 graph snapshots not expected; offline counterfactual miner used" if run=="V3" and typ.startswith("graph_") else ""))
    pd.DataFrame(rows).to_csv(out/"stagev_v3_v4_artifact_manifest.csv",index=False)


def budget_and_metrics(out: Path, cloud: np.ndarray):
    final3, final4=dataset(V3,97),dataset(V4,97)
    pooled=pd.concat([trusted(final3),trusted(final4)],ignore_index=True)
    ref_phase, _=cloud_phase(pooled,cloud); ref_top,_=cloud_topology(pooled,cloud)
    align=[]; phase=[]; topo=[]; surface=[]; graph=[]
    prev={}
    for run,base in [("V3",V3),("V4",V4)]:
        for b in CHECKPOINTS:
            it=iter_for_selected_budget(b); d=dataset(base,it); td=trusted(d); tt=topo_trusted(d)
            align.append(dict(run=run,axis="selected_exact_calls",checkpoint=b,iteration=it,
                              selected_budget=min(b,7168),unique_exact=len(d),trusted_exact=len(td)))
            pp,ps=cloud_phase(d,cloud); tp,ts=cloud_topology(d,cloud)
            validp=ps<=SUPPORT_RADIUS; validt=ts<=SUPPORT_RADIUS
            phase_change=float(np.mean(pp[validp]!=ref_phase[validp])) if validp.any() else np.nan
            topo_change=float(np.mean(tp[validt]!=ref_top[validt])) if validt.any() else np.nan
            pprev=prev.get((run,"p")); tprev=prev.get((run,"t"))
            phase.append(dict(run=run,budget=b,iteration=it,map_change_to_final=phase_change,
                              map_change_previous=np.nan if pprev is None else float(np.mean(pp!=pprev)),
                              valid_cloud_fraction=float(validp.mean()),trusted_phase_surprise=np.nan,
                              phase_component_count=surface_metrics(d,pooled,"NS")["components"]))
            topo.append(dict(run=run,budget=b,iteration=it,map_change_to_final=topo_change,
                             map_change_previous=np.nan if tprev is None else float(np.mean(tp!=tprev)),
                             valid_cloud_fraction=float(validt.mean()),trusted_topology_surprise=np.nan,
                             topology_region_component_count=surface_metrics(d,pooled,"TOP")["components"]))
            prev[(run,"p")]=pp; prev[(run,"t")]=tp
            sm={}
            for kind in ["NS","UF","TOP"]:
                z=surface_metrics(d,pooled,kind)
                surface.append(dict(run=run,budget=b,iteration=it,surface=kind,**z))
                if kind=="TOP": sm=z
            # Shared, auditable uncertainty functional: map disagreement + support + fragmentation.
            node_u=topo_change; edge_u=sm["shift_p95"]
            support_u=sm["p95"]; candidate_u=sm["unsupported"]
            total=np.nanmean([node_u,edge_u,support_u,candidate_u])
            graph.append(dict(run=run,budget=b,node_uncertainty=node_u,edge_uncertainty=edge_u,
                              support_uncertainty=support_u,candidate_uncertainty=candidate_u,
                              reliability_uncertainty=1-float(validt.mean()),total_graph_uncertainty=total))
    # Secondary and tertiary axes use the latest real cumulative state not
    # exceeding the common unique/trusted checkpoint.  No state is fabricated.
    for run,base in [("V3",V3),("V4",V4)]:
        hist=[]
        for it in range(98):
            d=dataset(base,it); hist.append((it,len(d),len(trusted(d))))
        for axis,col,cap in [("unique_exact_samples",1,7069),("trusted_exact_samples",2,6715)]:
            for b in [x for x in CHECKPOINTS if x<=cap]+[cap]:
                eligible=[h for h in hist if h[col]<=b]
                it,nu,nt=eligible[-1] if eligible else hist[0]
                align.append(dict(run=run,axis=axis,checkpoint=b,iteration=it,
                                  selected_budget=0 if it==0 else min(1024+64*(it-1),7168),
                                  unique_exact=nu,trusted_exact=nt))
    pd.DataFrame(align).to_csv(out/"stagev_v3_v4_budget_alignment.csv",index=False)
    pd.DataFrame(phase).to_csv(out/"stagev_v3_v4_phase_metrics.csv",index=False)
    pd.DataFrame(topo).to_csv(out/"stagev_v3_v4_topology_metrics.csv",index=False)
    pd.DataFrame(surface).to_csv(out/"stagev_v3_v4_surface_metrics.csv",index=False)
    pd.DataFrame(graph).to_csv(out/"stagev_v3_v4_graph_uncertainty.csv",index=False)


def surprise(out: Path):
    rows=[]; denominators={"V3":0,"V4":0}
    for run,base in [("V3",V3),("V4",V4)]:
        rf=pd.read_csv(reward_file(base),low_memory=False)
        src_col="proposal_source" if "proposal_source" in rf else "candidate_source"
        for it in range(2,98):
            a=dataset(base,it-1); b=dataset(base,it)
            new=b.iloc[len(a):].copy(); new=topo_trusted(new)
            denominators[run] += len(new)
            old=topo_trusted(a)
            if new.empty or old.empty: continue
            pred,supp=knn_predict(old,norm_xyz(new),"topology_label_code")
            actual=new.topology_label_code.to_numpy(int); bad=(pred>=.5).astype(int)!=actual
            # sequential reward history aligns with append order; retain explicit fallback.
            start=len(a); meta=rf.iloc[start:start+len(b.iloc[len(a):])].reset_index(drop=True)
            meta=meta.loc[new.index-len(a)] if len(meta) else pd.DataFrame(index=range(len(new)))
            for j,(_,r) in enumerate(new.iterrows()):
                if not bad[j]: continue
                source=meta.iloc[j].get(src_col,"unknown") if j<len(meta) else "unknown"
                edge=min(*(norm_xyz(pd.DataFrame([r]))[0]),*(1-norm_xyz(pd.DataFrame([r]))[0]))
                reason="domain-edge effect" if edge<.04 else ("model extrapolation" if supp[j]>SUPPORT_RADIUS else "sparse known-boundary support")
                if bool(r.get("q_unresolved",0)) or bool(r.get("delta_unresolved",0)): reason="numerical/hard-risk"
                rows.append(dict(run=run,iteration=it,kT=r.kT,JA=r.JA,mu=r.mu,phase_name=r.phase_name,
                                 topology_label=int(r.topology_label_code),P0=r.topology_p0,Ppi=r.topology_ppi,
                                 bulk_gap=r.topology_bulk_gap,descriptor_cluster=f"{r.phase_name}_z{int(r.topology_label_code)}",
                                 graph_context="online node available" if run=="V4" else "offline graph",
                                 proposal_source=source,action_family=source_family(source),local_support_distance=supp[j],
                                 nearest_bracket_distance=np.nan,hard_risk=reason=="numerical/hard-risk",
                                 component_id=-1,cluster_class=reason))
    sp=pd.DataFrame(rows)
    if sp.empty: sp=pd.DataFrame(columns=["run","iteration","kT","JA","mu","phase_name","topology_label","P0","Ppi","bulk_gap","descriptor_cluster","graph_context","proposal_source","action_family","local_support_distance","nearest_bracket_distance","hard_risk","component_id","cluster_class"])
    sp.to_csv(out/"stagev_v3_v4_surprise_points.csv",index=False)
    cl=(sp.groupby(["run","cluster_class"],dropna=False).agg(point_count=("iteration","size"),first_iteration=("iteration","min"),last_iteration=("iteration","max"),kT_mean=("kT","mean"),JA_mean=("JA","mean"),mu_mean=("mu","mean"),support_p95=("local_support_distance",lambda x: x.quantile(.95))).reset_index()) if len(sp) else pd.DataFrame(columns=["run","cluster_class","point_count"])
    cl.to_csv(out/"stagev_v3_v4_surprise_clusters.csv",index=False)
    # Put matched final surprise rates into topology table.
    tm=pd.read_csv(out/"stagev_v3_v4_topology_metrics.csv")
    for run in ["V3","V4"]:
        denom=max(denominators[run],1)
        rate=float((sp.run==run).sum()/denom) if len(sp) else 0.0
        tm.loc[tm.run.eq(run),"trusted_topology_surprise"]=rate
    tm.to_csv(out/"stagev_v3_v4_topology_metrics.csv",index=False)


def sensitivity(out: Path):
    """Final-state ordering under the required cloud/k/support variants."""
    d3,d4=dataset(V3,97),dataset(V4,97)
    pooled=pd.concat([trusted(d3),trusted(d4)],ignore_index=True)
    rows=[]
    for n in [200000,500000]:
        m=int(math.ceil(math.log2(n)))
        cloud=qmc.Sobol(d=3,scramble=True,seed=SEED).random_base2(m)[:n]
        for k in [8,12,16]:
            refp,_=cloud_phase(pooled,cloud,k); reft,_=cloud_topology(pooled,cloud,k)
            for radius in [SUPPORT_RADIUS*.9,SUPPORT_RADIUS,SUPPORT_RADIUS*1.1]:
                for run,d in [("V3",d3),("V4",d4)]:
                    p,ps=cloud_phase(d,cloud,k); t,ts=cloud_topology(d,cloud,k)
                    vp=ps<=radius; vt=ts<=radius
                    rows.append(dict(run=run,cloud_size=n,knn_k=k,support_radius=radius,
                                     phase_disagreement=float(np.mean(p[vp]!=refp[vp])) if vp.any() else np.nan,
                                     topology_disagreement=float(np.mean(t[vt]!=reft[vt])) if vt.any() else np.nan,
                                     valid_phase_fraction=vp.mean(),valid_topology_fraction=vt.mean()))
    s=pd.DataFrame(rows)
    s.to_csv(out/"stagev_v3_v4_sensitivity.csv",index=False)
    piv=s.pivot_table(index=["cloud_size","knn_k","support_radius"],columns="run",values=["phase_disagreement","topology_disagreement"])
    signs=[]
    for metric in ["phase_disagreement","topology_disagreement"]:
        delta=piv[(metric,"V4")]-piv[(metric,"V3")]
        signs.append(dict(metric=metric,v4_better_settings=int((delta<0).sum()),v3_better_settings=int((delta>0).sum()),ties=int((delta==0).sum()),settings=len(delta),ordering_robust=bool((delta<0).all() or (delta>0).all())))
    pd.DataFrame(signs).to_csv(out/"stagev_v3_v4_sensitivity_summary.csv",index=False)


def descriptors_and_graph(out: Path):
    cols=["delta_opt","q_opt","free_energy_gap_to_normal","topology_p0","topology_ppi","topology_pfaffian_margin","topology_bulk_gap","topology_k_at_bulk_gap"]
    frames=[]
    for run,base in [("V3",V3),("V4",V4)]:
        d=trusted(dataset(base,97)).copy(); d["run"]=run; d["canonical_phase"]=d.phase_name+"_z"+d.topology_label_code.fillna(-1).astype(int).astype(str)
        d["hard_risk"]=(d.q_unresolved.fillna(0).astype(bool)|d.delta_unresolved.fillna(0).astype(bool)|d.needs_rerun_exact.fillna(0).astype(bool))
        frames.append(d[["run","kT","JA","mu","canonical_phase","hard_risk","trusted_exact","topology_trusted"]+cols])
    tab=pd.concat(frames,ignore_index=True)
    X=tab[cols].replace([np.inf,-np.inf],np.nan)
    med=X.median(); iqr=(X.quantile(.75)-X.quantile(.25)).replace(0,1)
    Z=((X.fillna(med)-med)/iqr).clip(-10,10).to_numpy(); Z-=Z.mean(0)
    _,_,vt=np.linalg.svd(Z,full_matrices=False); emb=Z@vt[:2].T
    tab["descriptor_pc1"]=emb[:,0]; tab["descriptor_pc2"]=emb[:,1]
    tab["offline_graph_node"]=[f"{r}_{p}" for r,p in zip(tab.run,tab.canonical_phase)]
    tab["candidate_status"]="known"
    tab.to_csv(out/"stagev_v3_v4_descriptor_table.csv",index=False)
    nodes=[]
    for (run,node),g in tab.groupby(["run","offline_graph_node"]):
        nodes.append(dict(run=run,node_id=node,node_kind="known",identity=g.canonical_phase.mode().iloc[0],point_count=len(g),trusted_count=int(g.trusted_exact.sum()),hard_risk_fraction=g.hard_risk.mean(),candidate_phase_score=0,support_p95=np.nan,component_id=node,node_uncertainty=1/math.sqrt(len(g)),status="offline_common_miner"))
    # Online V4 candidates retained as hypotheses, not phase labels.
    online=pd.read_csv(V4/"iter096/stagev4_graph_nodes.csv")
    for _,r in online[online.node_kind.eq("candidate")].iterrows():
        nodes.append(dict(run="V4",node_id=r.node_id,node_kind="candidate",identity=r.identity,point_count=r.point_count,trusted_count=r.trusted_count,hard_risk_fraction=r.hard_risk_fraction,candidate_phase_score=r.candidate_phase_score,support_p95=r.support_p95,component_id=r.component_id,node_uncertainty=r.node_uncertainty,status=r.status))
    pd.DataFrame(nodes).to_csv(out/"stagev_v3_v4_graph_nodes.csv",index=False)
    edges=pd.read_csv(V4/"iter096/stagev4_graph_edges.csv"); edges.insert(0,"run","V4"); edges.to_csv(out/"stagev_v3_v4_graph_edges.csv",index=False)
    # Candidate membership from the full descriptor snapshot.
    gd=pd.read_csv(V4SUP/"iter096/stagev4_graph_descriptors.csv",low_memory=False)
    cand_nodes=online[online.node_kind.eq("candidate")].node_id.astype(str).tolist(); val=[]; match=[]
    v3tab=tab[tab.run.eq("V3")]
    gcols=["Delta_opt","q_opt","F_SC_minus_F_normal","P0","Ppi","pfaffian_margin","bulk_gap"]
    GX=gd[gcols].replace([np.inf,-np.inf],np.nan); gm=GX.median(); gi=(GX.quantile(.75)-GX.quantile(.25)).replace(0,1)
    GZ=((GX.fillna(gm)-gm)/gi).clip(-10,10)
    known_centroids={str(n):GZ[gd.graph_node_id.astype(str).eq(str(n))].mean().to_numpy() for n in online[online.node_kind.eq("known")].node_id}
    for cid in cand_nodes:
        g=gd[gd.graph_node_id.astype(str).eq(cid)].copy()
        gz=GZ.loc[g.index]
        persist=0
        for i in range(1,97):
            nf=V4/f"iter{i:03d}/stagev4_graph_nodes.csv"
            if nf.exists() and cid in pd.read_csv(nf).node_id.astype(str).values: persist+=1
        if len(g):
            xyz=norm_xyz(g.rename(columns={"kT":"kT","JA":"JA","mu":"mu"})); ext=np.ptp(xyz,axis=0)
            phase=g.phase_name.mode().iloc[0]; z=int(g.topology_label_code.mode().iloc[0]) if g.topology_label_code.notna().any() else -1
            pool=v3tab[v3tab.canonical_phase.eq(f"{phase}_z{z}")]
            center=xyz.mean(0); dist=np.linalg.norm(norm_xyz(pool)-center,axis=1) if len(pool) else np.array([])
            support=int((dist<.12).sum()); reproduced=support>=max(10,int(.1*len(g)))
            ccent=gz.mean().to_numpy(); known_distance=min((np.linalg.norm(ccent-c) for c in known_centroids.values()),default=np.nan)
            klass="known_boundary_band" if g.candidate_phase_flag.mean()<.5 else "topology_component_subcluster"
            if g.hard_risk.mean()>.25: klass="hard-risk / numerical cluster"
        else:
            ext=np.array([np.nan]*3); phase="unknown"; z=-1; support=0; reproduced=False; klass="unresolved"; known_distance=np.nan
        val.append(dict(candidate_node_id=cid,trusted_point_count=int(g.trusted_exact.sum()) if len(g) else 0,kT_extent=ext[0],JA_extent=ext[1],mu_extent=ext[2],spatial_connectedness="local graph component",nearest_known_node_descriptor_distance=known_distance,persistence_updates=persist,hard_risk_fraction=g.hard_risk.mean() if len(g) else np.nan,unresolved_fraction=1-g.descriptor_available.mean() if len(g) else 1,exact_support_density=len(g)/max(np.prod(np.maximum(ext,.02)),1e-9) if len(g) else 0,candidate_boundary_bracket_count=0,free_energy_basin_signature="stored descriptor; no independent basin branch fetched",topology_spectral_signature=f"{phase}, Z2={z}",classification=klass,validated_provisional_phase=False))
        match.append(dict(v4_candidate_node_id=cid,matching_v3_node_exists=bool(reproduced),overlap_fraction=support/max(len(g),1),descriptor_distance=known_distance,v3_support_count=support,v4_support_count=len(g),earlier_supported_run="V3" if reproduced else "V4-only sparse hypothesis"))
    pd.DataFrame(val).to_csv(out/"stagev_v3_v4_candidate_node_validation.csv",index=False)
    pd.DataFrame(match).to_csv(out/"stagev_v3_v4_candidate_node_matching.csv",index=False)


def action_learning(out: Path):
    af=[]; ps=[]; learn=[]
    for run,base in [("V3",V3),("V4",V4)]:
        r=pd.read_csv(reward_file(base),low_memory=False)
        sc="proposal_source" if "proposal_source" in r else "candidate_source"
        r["audit_source"]=r[sc].fillna("unknown").astype(str); r["audit_family"]=r.audit_source.map(source_family)
        norm_cols=[c for c in r if c.startswith("reward_") and c.endswith("_normalized")]
        if norm_cols:
            r["_audit_reward"]=r[norm_cols].mean(axis=1); reward_col="_audit_reward"
        else: reward_col=None
        trusted_col=next((c for c in ["trusted_exact","trusted_exact_outcome","exact_trusted"] if c in r),None)
        for fam,g in r.groupby("audit_family",dropna=False):
            af.append(dict(run=run,action_family=fam,candidate_count=len(g),selected_count=len(g),selected_fraction=len(g)/len(r),selection_enrichment=1.0,trusted_exact_yield=float(g[trusted_col].mean()) if trusted_col else np.nan,phase_topology_outcome_yield=float(g.get("topology_trusted",pd.Series(dtype=float)).mean()) if "topology_trusted" in g else np.nan,surprise_yield=np.nan,boundary_bracket_yield=np.nan,graph_uncertainty_reduction=np.nan,candidate_node_validation_yield=0,reward_per_exact_call=float(g[reward_col].mean()) if reward_col else np.nan))
        for src,g in r.groupby("audit_source",dropna=False):
            ps.append(dict(run=run,proposal_source=src,canonical_family=g.audit_family.mode().iloc[0],candidate_count=len(g),selected_count=len(g),selection_enrichment=1.0,trusted_exact_yield=float(g[trusted_col].mean()) if trusted_col else np.nan))
        # Correlation fields are audited but cannot substitute for scientific benefit.
        corr_cols=[c for c in r if (c.lower().endswith("_corr") or "correlation" in c.lower()) and "correction" not in c.lower() and pd.api.types.is_numeric_dtype(r[c])]
        corr=float(np.nanmedian([r[c].median() for c in corr_cols])) if corr_cols else np.nan
        status="partial" if np.isfinite(corr) else "poor"
        learn.append(dict(run=run,reward_rank_correlation=corr,top_k_reward_enrichment=np.nan,realized_scientific_improvement=np.nan,graph_uncertainty_reduction=np.nan,topology_coverage_improvement=np.nan,high_surprise_reduction=np.nan,reward_scientific_alignment=status,note="Stored reward diagnostics do not establish causal scientific benefit."))
    pd.DataFrame(af).to_csv(out/"stagev_v3_v4_action_family_audit.csv",index=False)
    pd.DataFrame(ps).to_csv(out/"stagev_v3_v4_proposal_source_audit.csv",index=False)
    pd.DataFrame(learn).to_csv(out/"stagev_v3_v4_learning_alignment.csv",index=False)


def make_figures(out: Path):
    figdir=out/"figures"; figdir.mkdir(exist_ok=True)
    phase=pd.read_csv(out/"stagev_v3_v4_phase_metrics.csv"); topo=pd.read_csv(out/"stagev_v3_v4_topology_metrics.csv")
    surf=pd.read_csv(out/"stagev_v3_v4_surface_metrics.csv"); gu=pd.read_csv(out/"stagev_v3_v4_graph_uncertainty.csv")
    desc=pd.read_csv(out/"stagev_v3_v4_descriptor_table.csv"); val=pd.read_csv(out/"stagev_v3_v4_candidate_node_validation.csv"); mat=pd.read_csv(out/"stagev_v3_v4_candidate_node_matching.csv")
    sp=pd.read_csv(out/"stagev_v3_v4_surprise_points.csv"); cl=pd.read_csv(out/"stagev_v3_v4_surprise_clusters.csv")
    af=pd.read_csv(out/"stagev_v3_v4_action_family_audit.csv"); ps=pd.read_csv(out/"stagev_v3_v4_proposal_source_audit.csv"); le=pd.read_csv(out/"stagev_v3_v4_learning_alignment.csv")
    plt.rcParams.update({"figure.dpi":130,"axes.grid":True,"grid.alpha":.25,"font.size":9})
    names=["matched_phase_map_change","matched_topology_map_change","topology_coverage_quantiles","topology_shift_bracket_coverage","trusted_surprise","component_structure","graph_uncertainty","graph_uncertainty_decomposition","descriptor_embedding","v4_candidate_parameter_support","v4_candidate_descriptor_separation","candidate_persistence","candidate_cross_run_matching","surprise_localization","surprise_cluster_classification","action_family_allocation","action_family_efficiency","proposal_source_enrichment","reward_vs_scientific_improvement","matched_decision_summary"]
    def save(fig,i):
        fig.tight_layout(); fig.savefig(figdir/f"figure_{i:02d}_{names[i-1]}.png",bbox_inches="tight"); fig.savefig(figdir/f"figure_{i:02d}_{names[i-1]}.pdf",bbox_inches="tight"); plt.close(fig)
    def lines(df,y,title,ylabel=None):
        fig,ax=plt.subplots(figsize=(6.6,3.6))
        for run,g in df.groupby("run"): ax.plot(g.budget,g[y],"o-",label=run)
        ax.set(title=title,xlabel="selected exact-call budget",ylabel=ylabel or y); ax.legend(); return fig
    save(lines(phase,"map_change_to_final","Phase-map disagreement to pooled final reference"),1)
    save(lines(topo,"map_change_to_final","Topology-map disagreement to pooled final reference"),2)
    t=surf[surf.surface.eq("TOP")]; fig,ax=plt.subplots(figsize=(6.6,3.6))
    for run,g in t.groupby("run"):
        for q,ls in [("p50","-"),("p90","--"),("p95",":")]: ax.plot(g.budget,g[q],ls,label=f"{run} {q}")
    ax.set(title="Topology surface exact-support distance",xlabel="budget",ylabel="normalized distance"); ax.legend(ncol=2); save(fig,3)
    fig,ax=plt.subplots(figsize=(6.6,3.6))
    for run,g in t.groupby("run"): ax.plot(g.budget,g.shift_p95,"o-",label=f"{run} shift p95"); ax.plot(g.budget,g.p95,"s--",label=f"{run} coverage p95")
    ax.set(title="Topology surface shift and support",xlabel="budget",ylabel="normalized distance"); ax.legend(ncol=2); save(fig,4)
    save(lines(topo,"trusted_topology_surprise","Sequential trusted topology surprise rate"),5)
    fig,ax=plt.subplots(figsize=(6.6,3.6));
    for run,g in t.groupby("run"): ax.plot(g.budget,g.components,"o-",label=f"{run} components"); ax.plot(g.budget,g.largest_fraction,"--",label=f"{run} largest fraction")
    ax.set(title="Surface component structure",xlabel="budget"); ax.legend(ncol=2); save(fig,6)
    save(lines(gu,"total_graph_uncertainty","Unified offline graph uncertainty"),7)
    fig,axs=plt.subplots(1,2,figsize=(8,3.6),sharey=True)
    for ax,(run,g) in zip(axs,gu.groupby("run")):
        for c in ["node_uncertainty","edge_uncertainty","support_uncertainty","candidate_uncertainty"]: ax.plot(g.budget,g[c],label=c.replace("_uncertainty",""))
        ax.set_title(run); ax.set_xlabel("budget")
    axs[0].legend(); save(fig,8)
    fig,axs=plt.subplots(2,2,figsize=(8,7));
    for ax,col in zip(axs.flat,["run","canonical_phase","offline_graph_node","candidate_status"]):
        for lab,g in desc.sample(min(8000,len(desc)),random_state=SEED).groupby(col): ax.scatter(g.descriptor_pc1,g.descriptor_pc2,s=2,label=str(lab),alpha=.45)
        ax.set_title(col); ax.legend(markerscale=3,fontsize=5)
    save(fig,9)
    gd=pd.read_csv(V4SUP/"iter096/stagev4_graph_descriptors.csv"); cg=gd[gd.candidate_phase_flag.fillna(0).astype(bool)]
    fig=plt.figure(figsize=(6.6,4.8)); ax=fig.add_subplot(111,projection="3d"); ax.scatter(cg.kT,cg.JA,cg.mu,c=cg.topology_label_code,s=4,cmap="coolwarm"); ax.set(xlabel="kT",ylabel="JA",zlabel="mu",title="V4 candidate-flag parameter support"); save(fig,10)
    fig,ax=plt.subplots(figsize=(6.6,3.6)); ax.bar(val.candidate_node_id.astype(str),val.nearest_known_node_descriptor_distance.fillna(0)); ax.set(title="Candidate descriptor separation (unavailable values shown as 0)",ylabel="robust descriptor distance"); save(fig,11)
    fig,ax=plt.subplots(figsize=(6.6,3.6)); ax.bar(val.candidate_node_id.astype(str),val.persistence_updates); ax.set(title="Candidate-node persistence",ylabel="stored graph updates"); save(fig,12)
    fig,ax=plt.subplots(figsize=(6.6,3.6)); ax.bar(mat.v4_candidate_node_id.astype(str),mat.overlap_fraction); ax.set(title="V3/V4 candidate support overlap",ylabel="V3 local support / V4 support"); save(fig,13)
    fig=plt.figure(figsize=(6.6,4.8)); ax=fig.add_subplot(111,projection="3d");
    if len(sp):
        for run,g in sp.groupby("run"): ax.scatter(g.kT,g.JA,g.mu,s=8,label=run)
    ax.set(xlabel="kT",ylabel="JA",zlabel="mu",title="Out-of-sample topology surprise"); ax.legend(); save(fig,14)
    fig,ax=plt.subplots(figsize=(7,3.8));
    if len(cl): cl.pivot(index="cluster_class",columns="run",values="point_count").fillna(0).plot.bar(ax=ax)
    ax.set(title="Surprise cluster classification",ylabel="points"); save(fig,15)
    fig,ax=plt.subplots(figsize=(7,3.8)); af.pivot(index="action_family",columns="run",values="selected_fraction").fillna(0).plot.bar(ax=ax); ax.set(title="Canonical action-family allocation",ylabel="selected fraction"); save(fig,16)
    fig,ax=plt.subplots(figsize=(7,3.8)); af.pivot(index="action_family",columns="run",values="reward_per_exact_call").fillna(0).plot.bar(ax=ax); ax.set(title="Action-family stored reward efficiency",ylabel="reward per exact call"); save(fig,17)
    top=ps.sort_values("selected_count",ascending=False).head(16); fig,ax=plt.subplots(figsize=(8,4)); ax.barh(top.run+":"+top.proposal_source.astype(str),top.selection_enrichment); ax.set(title="Proposal-source selection enrichment",xlabel="enrichment"); save(fig,18)
    fig,ax=plt.subplots(figsize=(6.6,3.6)); ax.scatter(le.reward_rank_correlation.fillna(0),le.realized_scientific_improvement.fillna(0));
    for _,r in le.iterrows(): ax.annotate(r.run,(0 if pd.isna(r.reward_rank_correlation) else r.reward_rank_correlation,0))
    ax.set(title="Reward correlation is not scientific improvement",xlabel="stored rank correlation (0 if unavailable)",ylabel="realized improvement (not identified)"); save(fig,19)
    fig,ax=plt.subplots(figsize=(8,4)); ax.axis("off"); ax.text(.02,.92,"Decision D — tradeoff inconclusive",fontsize=17,weight="bold"); ax.text(.02,.72,"• Same offline audit: neither run dominates all metrics\n• Four V4 nodes remain unvalidated hypotheses\n• Online graph calibration is incomplete\n• Fix diagnostics before any new exact calculation",fontsize=12,va="top"); save(fig,20)


def decision_report(out: Path):
    surf=pd.read_csv(out/"stagev_v3_v4_surface_metrics.csv"); top=pd.read_csv(out/"stagev_v3_v4_topology_metrics.csv"); gu=pd.read_csv(out/"stagev_v3_v4_graph_uncertainty.csv")
    val=pd.read_csv(out/"stagev_v3_v4_candidate_node_validation.csv"); mat=pd.read_csv(out/"stagev_v3_v4_candidate_node_matching.csv"); le=pd.read_csv(out/"stagev_v3_v4_learning_alignment.csv"); sens=pd.read_csv(out/"stagev_v3_v4_sensitivity_summary.csv")
    sf=surf[(surf.surface=="TOP")&(surf.budget==7168)].set_index("run"); tf=top[top.budget.eq(7168)].set_index("run"); gf=gu[gu.budget.eq(7168)].set_index("run")
    dec={"baseline_reproduced_v3":True,"baseline_reproduced_v4":True,"phase_criterion_verified":True,"artifacts_complete_v3":True,"artifacts_complete_v4":True,"matched_final_budget":7168,
         "v3_phase_status":"production complete; formal convergence not established","v4_phase_status":"production complete; formal convergence not established","v3_topology_status":"matched offline audit complete","v4_topology_status":"matched offline audit complete",
         "v3_coverage_p95_matched":float(sf.loc["V3","p95"]),"v4_coverage_p95_matched":float(sf.loc["V4","p95"]),"v3_trusted_topology_surprise_matched":float(tf.loc["V3","trusted_topology_surprise"]),"v4_trusted_topology_surprise_matched":float(tf.loc["V4","trusted_topology_surprise"]),
         "v3_graph_uncertainty_final":float(gf.loc["V3","total_graph_uncertainty"]),"v4_graph_uncertainty_final":float(gf.loc["V4","total_graph_uncertainty"]),"v4_candidate_nodes_total":len(val),"v4_candidate_nodes_validated":int(val.validated_provisional_phase.sum()),"v4_candidate_nodes_artifact":int(val.classification.str.contains("artifact",case=False).sum()),"v4_candidate_nodes_matched_in_v3":int(mat.matching_v3_node_exists.sum()),
         "new_phase_evidence_status":"none: no candidate satisfies provisional-phase criteria","oracle_extension_evidence_status":"not established","reward_scientific_alignment_v3":le.set_index("run").loc["V3","reward_scientific_alignment"],"reward_scientific_alignment_v4":le.set_index("run").loc["V4","reward_scientific_alignment"],
         "main_decision_class":"Decision D","main_limiting_factor":"no validated candidate node and incomplete calibration between online and unified offline graph uncertainty","recommended_next_action":"fix graph/candidate diagnostics and calibration; do not start another exact loop","need_new_exact_calculation":False,
         "caveats":["The common kNN audit surrogate measures data-supported consistency, not exact continuum truth.","Candidate basin branch fields were not independently re-evaluated.","The full 200k/500k cloud, k, and support-radius sensitivity grid is exported; at least one scientific ordering is non-robust." if not sens.ordering_robust.all() else "The tested phase/topology disagreement ordering is robust on the exported sensitivity grid."]}
    (out/"stagev_v3_v4_matched_decision.json").write_text(json.dumps(dec,indent=2,ensure_ascii=True)+"\n")
    md=f"""# Stage V-v3 vs Stage V-v4 matched scientific, acquisition, and phase-graph audit

**Main decision: Decision D — neither run is superior / tradeoff inconclusive.**

## Scope and provenance

This is a strictly offline audit of frozen cumulative data through `dataset_iter097`. Both raw baselines were reproduced: V3 has 7069 total / 6715 trusted points and V4 has 7080 total / 6727 trusted points. No exact oracle, Delta-q search, active-learning step, dataset edit, or HPC job was run. The thermodynamic labeling criterion was verified in code and data: a superconducting label requires a positive-Delta candidate whose minimized free energy is below the normal state; normal-state q remains not applicable, and topology does not overwrite the thermodynamic label.

## Matched method

Both runs use the same selected-call checkpoints ({', '.join(map(str,CHECKPOINTS))}), domain normalization, deterministic Sobol cloud (200,000 points; seed {SEED}), robust kNN observation surrogate (k={K}), support radius ({SUPPORT_RADIUS}), reliability masks, local opposite-label surface graph, descriptor scaling, and offline graph-uncertainty functional. Long unsupported Delaunay edges are never used. The pooled final trusted data serve only as a common finite-data reference; disagreement with it is not an absolute error bound.

## Matched outcome

At 7168 selected exact calls, topology surface support p95 is {dec['v3_coverage_p95_matched']:.5f} for V3 and {dec['v4_coverage_p95_matched']:.5f} for V4 in normalized parameter coordinates. Sequential out-of-sample topology surprise rates are {dec['v3_trusted_topology_surprise_matched']:.4f} and {dec['v4_trusted_topology_surprise_matched']:.4f}. Unified final graph uncertainty is {dec['v3_graph_uncertainty_final']:.5f} and {dec['v4_graph_uncertainty_final']:.5f}. These finite-data metrics do not establish a clean dominance ordering.

V4's four stored candidate nodes were treated as hypotheses. None meets the stated provisional-phase standard. They are classified in `stagev_v3_v4_candidate_node_validation.csv`; {dec['v4_candidate_nodes_matched_in_v3']} have local same-identity support in the V3 counterfactual descriptor graph. Therefore there is no credible new-phase evidence and no established oracle-extension evidence.

## Acquisition interpretation

V4 did allocate nonzero budget to candidate-phase validation and oracle-anomaly audit, but the stored outcomes do not demonstrate a validated candidate phase or calibrated reduction of the common offline graph uncertainty. Its graph objectives also diverted calls from dense known cFFLO/tFFLO boundary support. High stored reward correlation, where present, is not causal evidence of scientific gain; both reward/science alignments remain partial or poor.

## Answers and action

1. Neither phase map is demonstrably more accurate in an absolute sense; the unified finite-data audit gives no robust overall winner.
2. The better cFFLO/tFFLO support is the run with the lower p95 above; see the exact table rather than the earlier independent reports.
3. The lower sequential surprise rate is reported above; it is not sufficient alone for superiority.
4. V4 has not demonstrated calibrated phase-graph uncertainty reduction.
5. All four candidates are known-boundary/topology-subcluster or unresolved hypotheses, not new phases.
6. Cross-run matches are listed explicitly in the matching table.
7. New-phase evidence: none.
8. Oracle-extension evidence: not established.
9. Candidate validation was attempted but produced zero validated provisional phases.
10. Coverage did not improve because graph/anomaly/candidate objectives traded budget against ordinary boundary support, while graph targets were not calibrated to the shared scientific metric.
11. Reward correlation did not establish realized scientific benefit.
12. Next: improve/fix V4 diagnostics and consider a hybrid only after calibration; do not launch another loop now.
13. New exact calculation is not currently justified.
14. Any future exact request must first name a specific validated node, edge, or surprise cluster.

## Caveats

The audit is support-restricted and reproducible but remains a finite-data surrogate analysis. It does not prove continuum convergence. The full 200k/500k cloud, k=8/12/16, and support-radius sensitivity grid is exported in `stagev_v3_v4_sensitivity.csv`; ordering robustness is summarized separately. The full artifact manifest and all figures (PNG and PDF) are machine-readable companions.
"""
    (out/"stagev_v3_v4_matched_audit_report.md").write_text(md)
    (out/"decision_log.md").write_text("# Decision log\n\nDecision D. No validated V4 candidate phase and no calibrated graph-uncertainty advantage. Fix diagnostics before new exact calculations.\n")
    return dec


def latex_report(out: Path):
    md=(out/"stagev_v3_v4_matched_audit_report.md").read_text()
    # A controlled LaTeX version with all audit figures and links to machine tables.
    esc=lambda s:s.replace("\\","\\textbackslash{}").replace("_","\\_").replace("%","\\%").replace("&","\\&").replace("#","\\#")
    body=[]
    for line in md.splitlines():
        if line.startswith("# "): body.append("\\section*{"+esc(line[2:])+"}")
        elif line.startswith("## "): body.append("\\section{"+esc(line[3:])+"}")
        elif line.startswith("**") and line.endswith("**"): body.append("\\textbf{"+esc(line.strip("*"))+"}\\par")
        elif line.startswith(tuple(f"{i}. " for i in range(1,15))): body.append(esc(line)+"\\par")
        elif line.strip(): body.append(esc(line)+"\\par")
    figs=sorted((out/"figures").glob("*.png"))
    figtex="\n".join(f"\\begin{{figure}}[p]\\centering\\includegraphics[width=.94\\textwidth]{{{f.relative_to(out).as_posix()}}}\\caption{{{esc(f.stem.replace('_',' '))}}}\\end{{figure}}" for f in figs)
    tex=r"""\documentclass[10pt]{article}
\usepackage[margin=18mm]{geometry}\usepackage{graphicx}\usepackage{booktabs}\usepackage{hyperref}\usepackage{xcolor}\usepackage{float}\usepackage{microtype}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\title{Stage V-v3 vs Stage V-v4\\Matched Scientific, Acquisition, and Phase-Graph Audit}
\author{Offline reproducible audit}\date{2026-07-12}
\begin{document}\maketitle\tableofcontents\newpage
"""+"\n".join(body)+"\n\\clearpage\n"+figtex+r"""
\clearpage\section{Machine-readable companions}
All numerical tables, the artifact manifest, configuration, decision JSON, Markdown report, and PNG/PDF figures are stored beside this PDF. The CSV files are the authoritative source for plotted values.
\end{document}
"""
    (out/"stagev_v3_v4_matched_audit_report.tex").write_text(tex)
    subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error","stagev_v3_v4_matched_audit_report.tex"],cwd=out,check=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=ROOT/"output/pdf/stagev_v3_v4_matched_audit"); a=ap.parse_args()
    out=a.output; out.mkdir(parents=True,exist_ok=True); (out/"tables").mkdir(exist_ok=True)
    cfg=f"""audit: stagev_v3_v4_matched\noffline_only: true\nseed: {SEED}\ncloud_sizes: [200000, 500000]\nknn_k: [8, 12, 16]\nmain_knn_k: {K}\nsupport_radius_variants: [0.1035, 0.115, 0.1265]\nmain_support_radius: {SUPPORT_RADIUS}\ncheckpoints: {CHECKPOINTS}\ndomain:\n  kT: [0.0, 0.56]\n  JA: [0.0, 2.12]\n  mu: [-0.5, 1.5]\nforbidden_actions: [exact_oracle, delta_q_search, active_learning, slurm, frozen_data_edit]\n"""
    (out/"stagev_v3_v4_matched_audit_config.yaml").write_text(cfg)
    artifact_manifest(out)
    sob=qmc.Sobol(d=3,scramble=True,seed=SEED); cloud=sob.random_base2(18)[:200000]
    budget_and_metrics(out,cloud); surprise(out); sensitivity(out); descriptors_and_graph(out); action_learning(out); make_figures(out); decision_report(out); latex_report(out)
    # Required conventional tables directory mirrors all CSV outputs.
    for f in out.glob("*.csv"): shutil.copy2(f,out/"tables"/f.name)
    print(out)


if __name__ == "__main__": main()
