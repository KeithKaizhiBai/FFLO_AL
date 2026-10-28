"""Native 164-mm method derivatives; reuses the existing supplement engine adapter."""
from pathlib import Path
import json
import sys
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from figures.build_publication_supplement_figures import _contract, SPECS, configure_matplotlib, patch_svg
from figures.figure_engine.builders.figure_build import preflight
from figures.figure_engine.validators.figure_lint import lint_figure

OUT=ROOT/"figures/supplementary_publication_methods_v7"
ARCH=ROOT/"downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive"


def line(ax,y,title,description,c):
    ax.text(.03,y,title,fontsize=9.5,weight="bold",va="top",color=c["fflo"])
    ax.text(.40,y,description,fontsize=9,va="top",linespacing=1.25)
    ax.plot([.03,.97],[y-.12,y-.12],color=c["axis"],lw=.5)


def draw(key,ax,c):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    if key.startswith("S1"):
        ax.text(.03,.96,"Exact BdG evaluation: checks and returned information",size=11,weight="bold",va="top")
        line(ax,.81,"1  Minimize",r"Compare $F_{\rm impl}(\Delta,q)$ minima and normal reference",c)
        line(ax,.64,"2  Test numerical resolution","Check momentum-window enclosure\nand near-zero pairing amplitude",c)
        line(ax,.47,"3  Separate status","Reliable label: Normal / uniform SC / FFLO\nKeep delicate and rerun flags separately",c)
        line(ax,.30,"4  Test topology",r"For applicable FFLO states, evaluate bulk gap;"+"\n"+r"only resolved gapped states receive $\nu=0$ or $1$",c)
        ax.text(.03,.08,"The surrogate schedules calls; it does not assign physical labels.",size=9)
    elif key.startswith("S2"):
        ax.text(.03,.96,"Regression ensemble and classifier: stage-specific inputs",size=11,weight="bold",va="top")
        line(ax,.81,"Stages I–III",r"Inputs: $k_BT/t,\ J_A/t$"+"\nTwo hidden layers: 64 + 64 ReLU units",c)
        line(ax,.64,"Stage IV",r"Inputs: $k_BT/t,\ J_A/t,\ \mu/t$"+"\nTwo hidden layers: 96 + 96 ReLU units",c)
        line(ax,.47,"Both tasks","Five independent members per task; retrain each round\nRegression: MSE; classification: cross entropy",c)
        line(ax,.30,"Outputs",r"Regression: $\Delta,q,\eta,I_c^+,I_c^-$ and dispersion"+"\nClassification: phase probabilities",c)
        ax.text(.03,.08,"Acquisition is a scheduler only; exact BdG results supply the labels.",size=9)
    elif key.startswith("S3"):
        ax.text(.03,.96,"Production acquisition: two-dimensional full profile",size=11,weight="bold",va="top")
        line(ax,.81,"1  Score candidates",r"$A_0=A_{\rm phase}+0.4E_q+w_{\rm ext}E_{\rm ext}$"+"\nPhase ambiguity, numerical risk, exploration",c)
        line(ax,.64,"2  Restrict the pool",r"$A_{\rm pool}=A_0 P_{\rm interior}$"+"\nExact-coordinate masks and scheduled quantiles",c)
        line(ax,.47,"3  Select a microbatch",r"$p(x)\propto[A_{\rm pool}R_{\rm obs}R_{\rm batch}]^{\gamma}$"+"\nUpdate batch repulsion after every selected point",c)
        line(ax,.30,"4  Return to BdG","Evaluate selected coordinates; retain numerical status\nMerge observations and retrain",c)
        ax.text(.03,.08,"Stage III uses topology scores; Stage IV uses channel quotas (see text).",size=9)
    else:
        ax.text(.03,.96,"Stage-I sampling: locations are not executed-attempt counts",size=11,weight="bold",va="top")
        for y,value,label,color in [(.69,77361,"Candidate universe",c["cool"]),(.43,6880,"Final exact dataset",c["selected"])]:
            ax.text(.03,y+.14,label,size=9.5,weight="bold")
            ax.barh(y,value/77361*.72,left=.03,height=.08,color=color,edgecolor=c["axis"])
            ax.text(.79,y,f"{value:,}",size=10,va="center")
        ax.text(.03,.23,"Selected records: 8,192; unique selected coordinates: 7,305.",size=9)
        ax.text(.03,.14,r"Retained-location fraction: 8.89%; grid/retained ratio: 11.24$\times$.",size=9)
        ax.text(.03,.05,"Location counts do not establish a runtime gain or matched-baseline result.",size=9)


def main():
    records=[]
    for key,original in SPECS.items():
        folder=OUT/key;folder.mkdir(parents=True,exist_ok=True)
        spec=dict(original);spec["figure_id"]=spec["figure_id"].replace("V1","V7")
        if not key.startswith("S4"):
            spec["source"]=ARCH/"code_archive/stage_iii/ml_phase"/{"S1_reliability_flow":"exact_oracle.py","S2_surrogate_architecture":"models.py","S3_acquisition_decomposition":"acquisition.py"}[key]
        path=_contract(folder,key,spec)
        contract=yaml.safe_load(path.read_text());contract["layout"]["canvas"]["print_mm"]=[164,103]
        contract["data_sources"] += [str(ARCH/"data_archive/stage_iv/active_phase_topology_3d_t_ja_mu_from_scratch_v1/run_config.json")]
        contract["text_budget"]["minimum_font_size_pt"]=8.0
        path.write_text(yaml.safe_dump(contract,sort_keys=False))
        ctx=preflight(path,record_path=folder/"engine_preflight.json");c=configure_matplotlib(ctx)
        fig=plt.figure(figsize=(164/25.4,103/25.4),dpi=150,facecolor=c["white"])
        ax=fig.add_axes([.02,.02,.96,.96]);draw(key,ax,c);fig.canvas.draw()
        renderer=fig.canvas.get_renderer();geo=json.loads((folder/"geometry_manifest.json").read_text())
        for i,t in enumerate(fig.findobj(Text)):
            if not t.get_visible() or not t.get_text().strip():continue
            b=t.get_window_extent(renderer);box={"x":b.x0/fig.bbox.width*1600,"y":(fig.bbox.height-b.y1)/fig.bbox.height*1000,"width":b.width/fig.bbox.width*1600,"height":b.height/fig.bbox.height*1000}
            geo["objects"].append({"id":f"text-{i}","type":"text","role":"body","parent":"canvas","bbox":box,"font_size_pt":t.get_fontsize(),"font_family":"DejaVu Sans","allow_overlap_with":["scientific_content"]})
        (folder/"geometry_manifest.json").write_text(json.dumps(geo,indent=2))
        for ext in ("svg","pdf","png"):fig.savefig(folder/f"{key}.{ext}",format=ext,dpi=250)
        plt.close(fig);patch_svg(folder/f"{key}.svg",spec["figure_id"],path.name)
        Image.open(folder/f"{key}.png").convert("L").save(folder/f"{key}_grayscale.png")
        result=lint_figure(contract_path=path,svg_path=folder/f"{key}.svg",geometry_path=folder/"geometry_manifest.json",pdf_path=folder/f"{key}.pdf",png_path=folder/f"{key}.png",output_dir=folder)
        if result["status"]!="pass":raise RuntimeError(f"Engine rejected {key}: inspect {folder}")
        records.append({"key":key,"sha256":hashlib.sha256((folder/f"{key}.pdf").read_bytes()).hexdigest(),"native_width_mm":164,"min_base_font_pt":8,"status":"HUMAN_WAIT"})
    (OUT/"figure_manifest.json").write_text(json.dumps({"physics_recomputed":False,"figures":records},indent=2))
    print(json.dumps(records,indent=2))


if __name__=="__main__":main()
