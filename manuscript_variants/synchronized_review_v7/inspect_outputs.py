"""Read-only scientific inputs; document QA, comparison previews, and tables."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import zipfile
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"output/pdf/manuscript_synchronized_review_v7"
VAR=Path(__file__).resolve().parent


def table(name, rows):
    p=OUT/"tables"/name;p.parent.mkdir(exist_ok=True)
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def brace(text,start):
    depth=0
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start+1:i],i+1
    raise ValueError("Unbalanced LaTeX")


def markdown_companion(text):
    # Preserve TeX math verbatim; convert structural markup without PDF glyph loss.
    for command, prefix, suffix in [("section","## ","\n"),("subsection","### ","\n"),("paragraph","**","**"),("textbf","**","**"),("textit","*","*"),("path","`","`"),("texttt","`","`")]:
        pattern=re.compile(r"\\"+command+r"\{")
        while m:=pattern.search(text):
            body,end=brace(text,m.end()-1);text=text[:m.start()]+prefix+body+suffix+text[end:]
    text=re.sub(r"\\label\{[^}]+\}","",text)
    text=re.sub(r"\\(?:eqref|ref)\{([^}]+)\}",lambda m:"["+m[1]+"]",text)
    text=re.sub(r"\\cite\{([^}]+)\}",lambda m:"["+m[1]+"]",text)
    text=text.replace(r"\begin{equation}","\n$$\n").replace(r"\end{equation}","\n$$\n")
    text=text.replace(r"\begin{align}","\n$$\n\\begin{aligned}").replace(r"\end{align}","\\end{aligned}\n$$\n")
    text=text.replace(r"\notag","").replace(r"\[","\n$$\n").replace(r"\]","\n$$\n")
    text=text.replace(r"\begin{enumerate}","\n").replace(r"\end{enumerate}","\n").replace(r"\item ","\n1. ")
    # Complex table and figure TeX is retained as source blocks; the rendered
    # documents and source-matched PNGs accompany these blocks.
    text=re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}",lambda m:"\n```latex\n"+m[0]+"\n```\n",text,flags=re.S)
    return re.sub(r"\\par\b", "\n\n", text.replace("~"," ")).replace(r"\LayoutAudit","")


def inspect():
    checks=json.loads((OUT/"source/built/validation.json").read_text())
    shutil.copy2(OUT/"source/built/input_hashes.json",OUT/"source_manifest.json")
    rows=[]; fonts=[]
    for job,data in checks.items():
        for i,form in enumerate(data["embedded_forms"],1):
            rows.append({"document":job,"figure":i,"page":form["page"],"width_mm":form["rendered_width_mm"],"height_mm":form["rendered_height_mm"]})
    for n in range(1,5):
        p=PdfReader(OUT/f"source/figures/Fig{n}_candidate.pdf").pages[0]
        sizes=[]
        def visitor(t,cm,tm,font,size):
            if len(t.strip())>1:sizes.append((float(size)*math.hypot(cm[0],cm[1]),t.strip()))
        p.extract_text(visitor_text=visitor)
        for size,text in sorted(sizes)[:8]:
            fonts.append({"figure":n,"source_text_run_pt":size,"main_placed_pt":size*checks["manuscript_double_column"]["embedded_forms"][n-1]["matrix"][0],"text":text,"interpretation":"text-run diagnostic; subscripts/transforms require visual review"})
    table("figure_placements.csv",rows);table("small_text_runs.csv",fonts)
    counts=[]
    for part in ("introduction","results","discussion"):
        result=subprocess.run(["texcount",f"sections/{part}.tex"],cwd=OUT/"source",capture_output=True,text=True,check=True).stdout
        counts.append({key:int(re.search(pattern+ r": (\d+)",result)[1]) for key,pattern in [("text","Words in text"),("headers","Words in headers"),("captions",r"Words outside text \(captions, etc.\)"),("inline","Number of math inlines")]})
    length=[{"component":"body_and_section_headings","equivalent_words":sum(c["text"]+c["headers"] for c in counts),"method":"TeXcount; abstract/title/authors/bibliography excluded"},
            {"component":"captions","equivalent_words":sum(c["captions"] for c in counts),"method":"TeXcount caption text"},
            {"component":"inline_math_allowance","equivalent_words":sum(c["inline"] for c in counts),"method":"one token per inline math expression; approximate"},
            {"component":"displayed_equations","equivalent_words":32,"method":"2 single-column rows x 16"}]
    for i,f in enumerate(checks["manuscript_double_column"]["embedded_forms"],1):
        length.append({"component":f"figure_{i}","equivalent_words":600*f["rendered_height_mm"]/f["rendered_width_mm"]+40,"method":"double-column figure: 600/aspect + 40"})
    length.append({"component":"main_tables","equivalent_words":0,"method":"none"})
    length.append({"component":"total_estimate","equivalent_words":sum(x["equivalent_words"] for x in length),"method":"estimate, not publisher certification"})
    table("length_estimate.csv",length)
    sm=(OUT/"source/sections/supplement.tex").read_text()
    for n,match in enumerate(re.finditer(r"\\begin\{tabular\}",sm),1):
        _,end=brace(sm,match.end());body=sm[end:sm.index(r"\end{tabular}",end)]
        body=re.sub(r"\\(?:toprule|midrule|bottomrule)","",body)
        rows=[[cell.strip() for cell in row.split('&')] for row in body.split(r"\\") if row.strip()]
        with (OUT/"tables"/f"sm_table_{n}.csv").open("w",newline="") as f:csv.writer(f).writerows(rows)
    # Match saved bin totals rather than derive a new scientific result.
    archived=ROOT/"figures/fig4_strict_data_v2/canonical/fig4_selected_fixed_mu_slices.csv"
    shutil.copy2(archived,OUT/"tables/fig4_bin_counts.csv")
    for name in ("AUTHOR_REVIEW.md","policy_notes.md"):
        shutil.copy2(VAR/name,OUT/name)
    figs=OUT/"figures";figs.mkdir(exist_ok=True)
    manifest=json.loads((OUT/"input_manifest.json").read_text())
    for item in manifest["figures"].values():shutil.copy2(ROOT/item["source_png"],figs/f"Fig{item['number']}.png")
    for p in (ROOT/"figures/supplementary_publication_methods_v7").glob("S*/*.png"):
        if "grayscale" not in p.name:shutil.copy2(p,figs/p.name)
    shutil.copy2(ROOT/"figures/sm_stageiv_numerical_diagnostics/sm_stageiv_numerical_diagnostics.png",figs/"S5_stageiv_diagnostics.png")
    for job in checks:
        parts=["supplement"] if job=="supplement" else ["abstract","introduction","results","discussion"]
        text="\n\n".join((OUT/"source/sections"/(part+".tex")).read_text() for part in parts)
        md="# "+job.replace("_"," ")+"\n\nSource-synchronized companion; mathematical expressions retain LaTeX notation. Cross-reference keys resolve in the accompanying source and PDF. Tables/figure blocks preserve source content.\n\n"+markdown_companion(text)
        if job!="supplement":md+="\n\n## Bibliographic source\n\n```bibtex\n"+(OUT/"source/references.bib").read_text()+"\n```\n"
        (OUT/(job+".md")).write_text(md)
    preview=OUT/"previews";preview.mkdir(exist_ok=True)
    prefixes={"manuscript_double_column":"fflo_final_main","manuscript_single_column_review":"fflo_final_review","supplement":"fflo_final_sm"}
    for job,prefix in prefixes.items():
        for page in range(1,checks[job]["pages"]+1):
            files=list(Path("/tmp").glob(f"{prefix}-{page}.png"))+list(Path("/tmp").glob(f"{prefix}-{page:02d}.png"))
            if not files: raise RuntimeError(f"Missing rendered page {job}/{page}")
            shutil.copy2(files[0],preview/f"{job}_{page:02d}.png")
    subprocess.run(["pdftoppm","-scale-to","1300","-singlefile","-png",str(OUT/"source/figures/Fig1_candidate.pdf"),str(preview/"approved_fig1")],check=True)
    f=checks["manuscript_double_column"]["embedded_forms"][0]
    full=Image.open(preview/f"manuscript_double_column_{f['page']:02d}.png").convert("RGB")
    page=PdfReader(OUT/"manuscript_double_column.pdf").pages[f["page"]-1]
    sx=full.width/float(page.mediabox.width); sy=full.height/float(page.mediabox.height)
    x,y=f["matrix"][4:6];w=f["rendered_width_mm"]*72/25.4;h=f["rendered_height_mm"]*72/25.4
    crop=full.crop((round(x*sx),round((float(page.mediabox.height)-y-h)*sy),round((x+w)*sx),round((float(page.mediabox.height)-y)*sy)))
    a=Image.open(preview/"approved_fig1.png").convert("RGB");a.thumbnail((950,650));crop.thumbnail((950,650))
    canvas=Image.new("RGB",(1940,710),"white");draw=ImageDraw.Draw(canvas)
    draw.text((20,15),"Approved engine-v4 source",fill="black");draw.text((990,15),"Actual main PDF, page 2 (cropped for QA only)",fill="black")
    canvas.paste(a,(20,50));canvas.paste(crop,(990,50));canvas.save(preview/"fig1_source_vs_embedded.png")
    with zipfile.ZipFile(OUT/"latex_source_package.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in (OUT/"source").rglob("*"):
            if p.is_file() and not any(part in {"built","built_logs","__pycache__"} for part in p.relative_to(OUT/"source").parts):z.write(p,p.relative_to(OUT))
    print(json.dumps({"length":length[-1],"pages":{j:d['pages'] for j,d in checks.items()}},indent=2))


if __name__=="__main__":inspect()
