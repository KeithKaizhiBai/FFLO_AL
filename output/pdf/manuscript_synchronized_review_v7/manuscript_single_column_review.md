# manuscript single column review

Source-synchronized companion; mathematical expressions retain LaTeX notation. Cross-reference keys resolve in the accompanying source and PDF. Tables/figure blocks preserve source content.

Finite-momentum superconducting phase diagrams require repeated minimization over pairing amplitude and momentum, followed by spectral tests wherever topology is meaningful.  We combine an exact Bogoliubov--de Gennes oracle with active learning that schedules, but never replaces, these calculations.  The method reconstructs the two-dimensional Normal, uniform-superconducting, and FFLO regions with a sparse allocation of exact calculations.  Restricting topology to reliable gapped FFLO states reveals connected conventional and topological FFLO support separated by a numerically reconstructed gap-closing region.  Extending this hierarchy to $(k_BT/t,J_A/t,\mu/t)$ exposes systematic thermodynamic and topology-aware structure across a finite sampled domain.  The latter evidence is reported as discrete support and narrow-$\mu$ bins, not as a converged continuous topological surface or complete three-dimensional atlas.


## Introduction

Superconductivity with nonzero center-of-mass pairing momentum provides a route beyond a spatially uniform condensate. Fulde and Ferrell and, independently, Larkin and Ovchinnikov established the finite-momentum pairing and spatially modulated mechanisms now grouped under the FFLO setting [FuldeFerrell1964,LarkinOvchinnikov1965]. Subsequent reviews have developed the thermodynamic, material, and ultracold-gas consequences of Fermi-surface mismatch and competing pairing momenta [CasalbuoniNardulli2004,MatsudaShimahara2007,KinnunenEtAl2018]. The central phase-selection problem is therefore intrinsically thermodynamic: competing minima must be compared over both pairing amplitude and momentum.


The search for FFLO structure spans several physical platforms. Heavy-fermion and organic superconductors have displayed field-induced phases and spectroscopic, calorimetric, or magnetic signatures discussed in relation to FFLO order [RadovanEtAl2003,BianchiEtAl2003,KenzelmannEtAl2008,LortzEtAl2007,MayaffreEtAl2014]; these representative studies motivate the problem without implying that every identification is universal or uncontested. Population-imbalanced ultracold Fermi gases provide a complementary tunable setting for pairing under mismatch, phase separation, and finite-momentum phases [LiaoEtAl2010,ZwierleinEtAl2006,PartridgeEtAl2006,SheehyRadzihovsky2007,RadzihovskySheehy2010].

Altermagnets add a distinct route to this competition. Their compensated magnetic order can nevertheless generate symmetry-structured momentum-dependent spin splitting [SmejkalSinovaJungwirth2022,SmejkalEtAl2022,Mazin2022,SmejkalEtAl2020]. Model studies have connected this splitting to zero-field finite-momentum or other unconventional pairing states [ChakrabortyBlackSchaffer2024,HongParkKim2025,BoseVadnaisParamekanti2024], and neighboring altermagnetic constructions have been proposed for topological superconductivity and Majorana modes [ZhuEtAl2023,GhorashiHughesCano2024]. These precedents motivate an altermagnetic finite-$q$ problem but do not determine the phase structure of the one-dimensional model studied here.

Topology adds a second, logically separate requirement. Finite-momentum pairing can coexist with topological superfluidity in spin-orbit-coupled models [ZhangYi2013,QuEtAl2013,CaoEtAl2014]. More generally, bulk symmetries and invariants organize topological superconducting phases [ReadGreen2000,SatoAndo2017,QiZhang2011,SchnyderEtAl2008,RyuEtAl2010]. Majorana-platform studies connect those bulk distinctions to boundary physics [LutchynSauDasSarma2010,OregRefaelVonOppen2010,Alicea2012,Beenakker2013]. In a gapped one-dimensional Bogoliubov--de Gennes (BdG) system in symmetry class D, the relevant $\mathbb Z_2$ distinction can be expressed through a Pfaffian invariant [Kitaev2001,BudichArdonne2013]; its use is meaningful only after bulk-gap applicability is established. A topology-resolved FFLO map must therefore identify a reliable thermodynamic minimum before classifying trusted gapped FFLO states.

This hierarchy is computationally demanding because every accepted parameter point requires minimization of $F(\Delta,q)$, comparison with the normal reference, adaptive checks of shallow or edge minima, and, where applicable, full-zone spectral and Pfaffian diagnostics. Active learning and Bayesian experimental design formalize the selection of informative sequential evaluations [CohnAtlasLadner1994,MacKay1992,ShahriariEtAl2016]; phase-diagram applications demonstrate boundary-focused sampling [TerayamaEtAl2019,DaiBrussGlotzer2020,LookmanEtAl2019], while adaptive sampling has also reduced expensive electronic-structure, atomistic, and chemical search budgets [TranUlissi2018,PodryabinkinShapeev2017,JinnouchiEtAl2019,VandermauseEtAl2020,BatraEtAl2021,NoeEtAl2019]. For a physical phase diagram, however, an acquisition model should decide where to calculate next, not replace the exact calculation that defines the phase.

Here we combine an exact BdG oracle with active learning to resolve the thermodynamic and topology-aware structure of an altermagnetic finite-momentum superconductor.  The oracle supplies every physical label, while the surrogate redirects subsequent evaluations toward uncertain or rapidly varying regions.  This strategy reconstructs the two-dimensional thermodynamic phases from a sparse subset of the prescribed candidate locations, resolves connected cFFLO and tFFLO support within the reliable gapped FFLO sector, and exposes systematic structure across a finite three-dimensional domain.  The three-dimensional evidence remains a discrete point cloud and narrow-$\mu$ bins rather than a converged continuous topological surface.

The detailed model, algorithms, counting conventions, and numerical diagnostics are provided in the Supplemental Material [SupplementalMaterial].


## Results




```latex
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/Fig1_candidate.pdf}
\caption{**Physical hierarchy and adaptive scheduling.** Schematic illustration, not computed phase data: the altermagnetic and spin-orbit BdG model generates a normal-referenced $F_{\rm impl}(\Delta,q)$ landscape. Its reliable minimum determines Normal, uniform-SC, or FFLO order; only a gapped FFLO minimum proceeds to the Pfaffian test. The surrogate cycle proposes subsequent exact calls and does not determine either label.}
\end{figure*}
```


*Model and physical hierarchy.---*
The single-particle sector is
$h(k)=t\cos k\,\sigma_0+\lambda_y\sin k\,\sigma_y+[\lambda_z\sin k+J_A\cos k]\sigma_z$.
The odd-in-$k$ spin-orbit fields and the altermagnetic $J_A\cos k$ splitting produce a momentum-dependent spin texture, while pairing connects states shifted by $\pm q/2$ in the Nambu Hamiltonian.  A finite-momentum state therefore cannot be found by optimizing the order-parameter amplitude alone: for every external coordinate $x$ the exact oracle determines

$$
(\Delta_{\rm opt},q_{\rm opt})=\operatorname*{arg\,min}_{\Delta,q}F_{\rm impl}(\Delta,q),
$$

where $F_{\rm impl}$ is the normal-referenced occupation-weighted spectral objective used by the implemented numerical calculation.  Its reliable minimum gives Normal order when the optimum is unpaired, uniform superconductivity when $\Delta_{\rm opt}>0$ and $q_{\rm opt}$ is zero within resolution, and FFLO order when both the amplitude and finite momentum are resolved.



Topology is a second, logically separate test.  It is evaluated only for a thermodynamically accepted, numerically reliable FFLO minimum whose full Brillouin-zone spectrum is gapped.  Under that applicability condition,

$$
(-1)^\nu=\operatorname{sgn}[P(0)P(\pi)]
$$

distinguishes cFFLO ($\nu=0$) from tFFLO ($\nu=1$); a change of index requires a bulk gap closing and reopening, not a change in basin depth or the sign of $q$.  Figure [fig:framework] summarizes this hierarchy.  Regression and classification ensembles see accepted exact observations and rank the next microbatch, but every physical label remains an output of the BdG objective and spectral tests.





```latex
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/Fig2_candidate.pdf}
\caption{**Two-dimensional thermodynamic reconstruction.** **(a)** Sparse exact initialization. **(b)** Exact-call allocation across the Stage-I discovery and Stage-II reliability continuation. **(c)** The 7,434-coordinate Stage-II endpoint, which contains the 6,880-coordinate Stage-I endpoint plus 554 additional exact coordinates; open symbols denote numerically flagged evaluations, not a phase. **(d)** Numerical histories; the location-count comparison uses Stage I, not the Stage-II continuation.}
\end{figure*}
```


*Two-dimensional thermodynamic organization.---*
The exact observations in Fig. [fig:thermo] organize into a curved outer envelope separating superconducting support below from Normal support above.  At low temperature the superconducting sector reaches comparatively large $J_A/t$; as temperature rises, the largest sampled splitting that still supports pairing moves downward.  Most of the paired region is finite-$q$ FFLO.  Uniform superconductivity instead forms a narrow strip close to $J_A=0$, so the sampled state changes rapidly from zero- to finite-momentum pairing once the altermagnetic splitting is introduced.  This thin inner structure and the descending outer envelope are precisely the regions where small changes of the optimized minimum can change a label, motivating concentrated exact minimization rather than a smooth surrogate boundary.



The search begins with a sparse random design over the full $(k_BT/t,J_A/t)$ domain.  Subsequent calls accumulate along both thermodynamic structures while a continuing exploration channel samples their interiors and the remaining domain.  In the primary Stage-I discovery, 6,880 unique exact coordinates are retained from 77,361 prescribed candidate locations, corresponding to an 8.89\% location fraction and an 11.24-fold reduction relative to evaluating every grid location.  The Stage-II reliability continuation adds exact evaluations without changing any shared phase label and supplies the 7,434-coordinate endpoint shown in panel (c).  The efficiency ratio is therefore a Stage-I location count, not a wall-time acceleration or a comparison with matched random sampling. Selected requests and retained coordinates are distinguished in Supplemental Material, Table [sm-tab:counts]; neither is a complete execution ledger.





```latex
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/Fig3_candidate.pdf}
\caption{**Bulk topology inside the thermodynamic FFLO sector.** **(a)** Trusted gapped FFLO observations resolved into cFFLO and tFFLO by the full-zone gap and Pfaffian invariant. **(b)** A support-restricted contour and local opposite-$\mathbb Z_2$ brackets locate a gap-closing region but not an exact boundary. **(c,d)** Numerical histories and neighborhood sensitivity (Supplemental Material, Sec. [sm-sec:topology]).}
\end{figure*}
```


*Topology-resolved FFLO structure.---*
Thermodynamic FFLO order alone does not specify the bulk topological sector.  Applying the full-zone gap guard and Pfaffian invariant resolves the trusted FFLO observations into two gapped supports (Fig. [fig:topology]).  The cFFLO sector occupies lower $J_A/t$ and extends to higher temperature, whereas the sampled tFFLO sector is shifted toward stronger altermagnetic splitting and lower temperature.  Their meeting region follows a curved locus where the bulk gap closes before reopening with the opposite Pfaffian product.  Thus the topology-resolved map adds a bulk-state distinction that neither the position of the $F_{\rm impl}$ minimum nor the sign of $q_{\rm opt}$ can provide.



The exact labels are discrete, so the plotted transition line is a support-restricted reconstruction rather than an independently calculated phase boundary.  Nearby observations with opposite $\mathbb Z_2$ indices bracket the transition region, and changing the reconstruction neighborhood produces a finite displacement of the guide.  We therefore use the contour to organize the sampled close-and-reopen structure, not to claim an exact line or a stable third phase.  The retained sensitivity also leaves open narrow nodal alternatives that the present sampling does not exhaust.  This separation between exact bulk labels and a derived visual guide is carried into the higher-dimensional extension.





```latex
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/Fig4_candidate.pdf}
\caption{**Finite-domain three-dimensional exploration.** **(a)** Exact thermodynamic observations in $(k_BT/t,J_A/t,\mu/t)$. **(b)** Discrete trusted FFLO topology support. **(c)** Four narrow-$\mu$ bins (centers $-0.45,0.15,0.85,1.45$, half-width 0.025) without interpolation; see Supplemental Material, Sec. [sm-sec:3d]. Slice populations reflect adaptive sampling and are not phase-volume estimates; the four cuts do not define a continuous topological surface.}
\end{figure*}
```


*Chemical-potential-dependent higher-dimensional structure.---*
The four narrow-$\mu$ bins of exact observations in Fig. [fig:3d] show that the two-dimensional organization does not remain rigid as the third coordinate changes.  At $\mu/t=-0.45$ and $0.15$, sampled cFFLO support lies mainly at lower $J_A/t$, while tFFLO is concentrated toward higher splitting and lower temperature; in the latter cut the trusted tFFLO support is especially confined to low temperature.  At $\mu/t=0.85$, uniform-SC support extends over a broader $J_A/t$ range and trusted tFFLO spans a wider part of the sampled slice.  By $\mu/t=1.45$, the observed paired support is restricted to low splitting and moderate-to-low temperature, and the narrow slice contains no trusted tFFLO observation.  This last statement is about sampled support, not proof that tFFLO is absent elsewhere near that chemical potential.



The three-dimensional point cloud establishes the finite sampled domain and displays how exact thermodynamic and topology labels populate it; the narrow slices provide readable local cross-sections without imposing continuity between them.  Selective evaluation is especially useful in this enlarged candidate space because it can retain direct BdG evaluation while directing calls toward changing structures.  Nevertheless, nonuniform sampling prevents phase counts from being interpreted as thermodynamic volumes, and four cuts cannot determine a continuous boundary or topology surface.  Figure [fig:3d] therefore demonstrates scalable higher-dimensional exploration and chemical-potential-dependent reorganization, not a complete three-dimensional atlas or a fully resolved new mechanism.


## Discussion

The physical result is a hierarchy of structure rather than a single phase map.  The two-dimensional exact data show a narrow uniform-SC strip, an extended finite-$q$ region, and a curved Normal--superconducting envelope; the spectral analysis then resolves the FFLO sector into gapped conventional and topological bulk supports.  Varying chemical potential reorganizes the coordinate support of those states and can remove trusted tFFLO observations from an individual sampled slice.  Topology therefore supplies information that a thermodynamic FFLO label alone cannot encode.

Methodologically, reliability-aware active learning differs from a black-box phase classifier because physical labels are obtained from the BdG calculation at every layer.  Numerical reliability is stored separately from phase identity, topology is masked outside reliable gapped FFLO states, and uncertainty affects where the next calculation is performed rather than what phase is declared.  This organization makes the expensive minimization and spectral tests selectively deployable without turning the surrogate into a physics solver.

The location-count economy does not establish wall-time acceleration, because no matched exhaustive timing baseline is archived, and it does not establish boundary enrichment, because the candidate-wide boundary denominator is unavailable.  The stopping settings diagnose a particular numerical workflow rather than universal physical scales.  Likewise, the three-dimensional evidence is discrete and nonuniformly sampled, so it supports higher-dimensional exploration but neither phase volumes nor a closed continuous atlas.

The broader significance is a route for coupling direct BdG numerical objectives to adaptive exploration while preserving applicability masks and failure states.  Exploratory learned-acquisition studies have not established superiority over the production acquisition.  A future physical world model would need spectral descriptors, calibrated uncertainty, and acquisition utility tied to physical validity; these ingredients remain an outlook rather than a validated result here.

**Data availability.** The numerical data and source configurations are preserved in the project archive. No public repository or persistent identifier has been established for this review version.


## Bibliographic source

```bibtex
@article{LarkinOvchinnikov1965,
  author = {A. I. Larkin and Yu. N. Ovchinnikov},
  title = {{Nonuniform State of Superconductors}},
  journal = {Soviet Physics JETP},
  volume = {20},
  pages = {762--769},
  year = {1965},
  url = {https://jetp.ras.ru/}
}

@article{FuldeFerrell1964,
  author = {Peter Fulde and Richard A. Ferrell},
  title = {{Superconductivity in a Strong Spin-Exchange Field}},
  journal = {Physical Review},
  volume = {135},
  pages = {A550--A563},
  year = {1964},
  doi = {10.1103/physrev.135.a550},
  url = {https://link.aps.org/doi/10.1103/PhysRev.135.A550}
}

@article{CasalbuoniNardulli2004,
  author = {Roberto Casalbuoni and Giuseppe Nardulli},
  title = {{Inhomogeneous superconductivity in condensed matter and QCD}},
  journal = {Reviews of Modern Physics},
  volume = {76},
  pages = {263--320},
  year = {2004},
  doi = {10.1103/revmodphys.76.263},
  url = {https://link.aps.org/doi/10.1103/RevModPhys.76.263}
}

@article{MatsudaShimahara2007,
  author = {Yuji Matsuda and Hiroshi Shimahara},
  title = {{Fulde–Ferrell–Larkin–Ovchinnikov State in Heavy Fermion Superconductors}},
  journal = {Journal of the Physical Society of Japan},
  volume = {76},
  pages = {051005},
  year = {2007},
  doi = {10.1143/jpsj.76.051005},
  url = {https://journals.jps.jp/doi/10.1143/JPSJ.76.051005}
}

@article{KinnunenEtAl2018,
  author = {Jami J Kinnunen and Jildou E Baarsma and Jani-Petri Martikainen and Päivi Törmä},
  title = {{The Fulde–Ferrell–Larkin–Ovchinnikov state for ultracold fermions in lattice and harmonic potentials: a review}},
  journal = {Reports on Progress in Physics},
  volume = {81},
  pages = {046401},
  year = {2018},
  doi = {10.1088/1361-6633/aaa4ad},
  url = {https://iopscience.iop.org/article/10.1088/1361-6633/aaa4ad}
}

@article{RadovanEtAl2003,
  author = {H. A. Radovan and N. A. Fortune and T. P. Murphy and S. T. Hannahs and E. C. Palm and S. W. Tozer and D. Hall},
  title = {{Magnetic enhancement of superconductivity from electron spin domains}},
  journal = {Nature},
  volume = {425},
  pages = {51--55},
  year = {2003},
  doi = {10.1038/nature01842},
  url = {https://www.nature.com/articles/nature01842}
}

@article{BianchiEtAl2003,
  author = {A. Bianchi and R. Movshovich and C. Capan and P. G. Pagliuso and J. L. Sarrao},
  title = {{Possible Fulde-Ferrell-Larkin-Ovchinnikov Superconducting State in CeCoIn5}},
  journal = {Physical Review Letters},
  volume = {91},
  pages = {187004},
  year = {2003},
  doi = {10.1103/physrevlett.91.187004},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.91.187004}
}

@article{KenzelmannEtAl2008,
  author = {M. Kenzelmann and Th. Strässle and C. Niedermayer and M. Sigrist and B. Padmanabhan and M. Zolliker and A. D. Bianchi and R. Movshovich and E. D. Bauer and J. L. Sarrao and J. D. Thompson},
  title = {{Coupled Superconducting and Magnetic Order in CeCoIn5}},
  journal = {Science},
  volume = {321},
  pages = {1652--1654},
  year = {2008},
  doi = {10.1126/science.1161818},
  url = {https://www.science.org/doi/10.1126/science.1161818}
}

@article{LortzEtAl2007,
  author = {R. Lortz and Y. Wang and A. Demuer and P. H. M. Böttger and B. Bergk and G. Zwicknagl and Y. Nakazawa and J. Wosnitza},
  title = {{Calorimetric Evidence for a Fulde-Ferrell-Larkin-Ovchinnikov Superconducting State in the Layered Organic Superconductorκ−(BEDT−TTF)2Cu(NCS)2}},
  journal = {Physical Review Letters},
  volume = {99},
  pages = {187002},
  year = {2007},
  doi = {10.1103/physrevlett.99.187002},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.99.187002}
}

@article{MayaffreEtAl2014,
  author = {H. Mayaffre and S. Krämer and M. Horvatić and C. Berthier and K. Miyagawa and K. Kanoda and V. F. Mitrović},
  title = {{Evidence of Andreev bound states as a hallmark of the FFLO phase in κ-(BEDT-TTF)2Cu(NCS)2}},
  journal = {Nature Physics},
  volume = {10},
  pages = {928--932},
  year = {2014},
  doi = {10.1038/nphys3121},
  url = {https://www.nature.com/articles/nphys3121}
}

@article{LiaoEtAl2010,
  author = {Yean-an Liao and Ann Sophie C. Rittner and Tobias Paprotta and Wenhui Li and Guthrie B. Partridge and Randall G. Hulet and Stefan K. Baur and Erich J. Mueller},
  title = {{Spin-imbalance in a one-dimensional Fermi gas}},
  journal = {Nature},
  volume = {467},
  pages = {567--569},
  year = {2010},
  doi = {10.1038/nature09393},
  url = {https://www.nature.com/articles/nature09393}
}

@article{ZwierleinEtAl2006,
  author = {Martin W. Zwierlein and André Schirotzek and Christian H. Schunck and Wolfgang Ketterle},
  title = {{Fermionic Superfluidity with Imbalanced Spin Populations}},
  journal = {Science},
  volume = {311},
  pages = {492--496},
  year = {2006},
  doi = {10.1126/science.1122318},
  url = {https://www.science.org/doi/10.1126/science.1122318}
}

@article{PartridgeEtAl2006,
  author = {Guthrie B. Partridge and Wenhui Li and Ramsey I. Kamar and Yean-an Liao and Randall G. Hulet},
  title = {{Pairing and Phase Separation in a Polarized Fermi Gas}},
  journal = {Science},
  volume = {311},
  pages = {503--505},
  year = {2006},
  doi = {10.1126/science.1122876},
  url = {https://www.science.org/doi/10.1126/science.1122876}
}

@article{SheehyRadzihovsky2007,
  author = {Daniel E. Sheehy and Leo Radzihovsky},
  title = {{BEC–BCS crossover, phase transitions and phase separation in polarized resonantly-paired superfluids}},
  journal = {Annals of Physics},
  volume = {322},
  pages = {1790--1924},
  year = {2007},
  doi = {10.1016/j.aop.2006.09.009},
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0003491606002041}
}

@article{RadzihovskySheehy2010,
  author = {Leo Radzihovsky and Daniel E Sheehy},
  title = {{Imbalanced Feshbach-resonant Fermi gases}},
  journal = {Reports on Progress in Physics},
  volume = {73},
  pages = {076501},
  year = {2010},
  doi = {10.1088/0034-4885/73/7/076501},
  url = {https://iopscience.iop.org/article/10.1088/0034-4885/73/7/076501}
}

@article{SmejkalSinovaJungwirth2022,
  author = {Libor Šmejkal and Jairo Sinova and Tomas Jungwirth},
  title = {{Emerging Research Landscape of Altermagnetism}},
  journal = {Physical Review X},
  volume = {12},
  pages = {040501},
  year = {2022},
  doi = {10.1103/physrevx.12.040501},
  url = {https://link.aps.org/doi/10.1103/PhysRevX.12.040501}
}

@article{SmejkalEtAl2022,
  author = {Libor Šmejkal and Jairo Sinova and Tomas Jungwirth},
  title = {{Beyond Conventional Ferromagnetism and Antiferromagnetism: A Phase with Nonrelativistic Spin and Crystal Rotation Symmetry}},
  journal = {Physical Review X},
  volume = {12},
  pages = {031042},
  year = {2022},
  doi = {10.1103/physrevx.12.031042},
  url = {https://link.aps.org/doi/10.1103/PhysRevX.12.031042}
}

@article{Mazin2022,
  author = {Igor Mazin},
  title = {{Editorial: Altermagnetism—A New Punch Line of Fundamental Magnetism}},
  journal = {Physical Review X},
  volume = {12},
  pages = {040002},
  year = {2022},
  doi = {10.1103/physrevx.12.040002},
  url = {https://link.aps.org/doi/10.1103/PhysRevX.12.040002}
}

@article{SmejkalEtAl2020,
  author = {Libor Šmejkal and Rafael González-Hernández and T. Jungwirth and J. Sinova},
  title = {{Crystal time-reversal symmetry breaking and spontaneous Hall effect in collinear antiferromagnets}},
  journal = {Science Advances},
  volume = {6},
  pages = {eaaz8809},
  year = {2020},
  doi = {10.1126/sciadv.aaz8809},
  url = {https://www.science.org/doi/10.1126/sciadv.aaz8809}
}

@article{ChakrabortyBlackSchaffer2024,
  author = {Debmalya Chakraborty and Annica M. Black-Schaffer},
  title = {{Zero-field finite-momentum and field-induced superconductivity in altermagnets}},
  journal = {Physical Review B},
  volume = {110},
  pages = {L060508},
  year = {2024},
  doi = {10.1103/physrevb.110.l060508},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.110.L060508}
}

@article{HongParkKim2025,
  author = {SeungBeom Hong and Moon Jip Park and Kyoung-Min Kim},
  title = {{Unconventional p-wave and Finite-Momentum Superconductivity Induced by Altermagnetism through the Formation of Bogoliubov Fermi Surface}},
  journal = {Physical Review B},
  volume = {111},
  pages = {054501},
  year = {2025},
  doi = {10.1103/physrevb.111.054501},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.111.054501}
}

@article{ZhuEtAl2023,
  author = {Di Zhu and Zheng-Yang Zhuang and Zhigang Wu and Zhongbo Yan},
  title = {{Topological superconductivity in two-dimensional altermagnetic metals}},
  journal = {Physical Review B},
  volume = {108},
  pages = {184505},
  year = {2023},
  doi = {10.1103/physrevb.108.184505},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.108.184505}
}

@article{GhorashiHughesCano2024,
  author = {Sayed Ali Akbar Ghorashi and Taylor L. Hughes and Jennifer Cano},
  title = {{Altermagnetic Routes to Majorana Modes in Zero Net Magnetization}},
  journal = {Physical Review Letters},
  volume = {133},
  pages = {106601},
  year = {2024},
  doi = {10.1103/physrevlett.133.106601},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.133.106601}
}

@article{BoseVadnaisParamekanti2024,
  author = {Anjishnu Bose and Samuel Vadnais and Arun Paramekanti},
  title = {{Altermagnetism and Superconductivity in a Multiorbital t-J Model}},
  journal = {Physical Review B},
  volume = {110},
  pages = {205120},
  year = {2024},
  doi = {10.1103/physrevb.110.205120},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.110.205120}
}

@article{ZhangYi2013,
  author = {Wei Zhang and Wei Yi},
  title = {{Topological Fulde–Ferrell–Larkin–Ovchinnikov states in spin–orbit-coupled Fermi gases}},
  journal = {Nature Communications},
  volume = {4},
  pages = {2711},
  year = {2013},
  doi = {10.1038/ncomms3711},
  url = {https://www.nature.com/articles/ncomms3711}
}

@article{QuEtAl2013,
  author = {Chunlei Qu and Zhen Zheng and Ming Gong and Yong Xu and Li Mao and Xubo Zou and Guangcan Guo and Chuanwei Zhang},
  title = {{Topological superfluids with finite-momentum pairing and Majorana fermions}},
  journal = {Nature Communications},
  volume = {4},
  pages = {2710},
  year = {2013},
  doi = {10.1038/ncomms3710},
  url = {https://www.nature.com/articles/ncomms3710}
}

@article{CaoEtAl2014,
  author = {Ye Cao and Shu-Hao Zou and Xia-Ji Liu and Su Yi and Gui-Lu Long and Hui Hu},
  title = {{Gapless Topological Fulde-Ferrell Superfluidity in Spin-Orbit Coupled Fermi Gases}},
  journal = {Physical Review Letters},
  volume = {113},
  pages = {115302},
  year = {2014},
  doi = {10.1103/physrevlett.113.115302},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.113.115302}
}

@article{Kitaev2001,
  author = {A Yu Kitaev},
  title = {{Unpaired Majorana fermions in quantum wires}},
  journal = {Physics-Uspekhi},
  volume = {44},
  pages = {131--136},
  year = {2001},
  doi = {10.1070/1063-7869/44/10s/s29},
  url = {https://www.mathnet.ru/eng/ufn5648}
}

@article{BudichArdonne2013,
  author = {Jan Carl Budich and Eddy Ardonne},
  title = {{Equivalent topological invariants for one-dimensional Majorana wires in symmetry class D}},
  journal = {Physical Review B},
  volume = {88},
  pages = {075419},
  year = {2013},
  doi = {10.1103/physrevb.88.075419},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.88.075419}
}

@article{ReadGreen2000,
  author = {N. Read and Dmitry Green},
  title = {{Paired states of fermions in two dimensions with breaking of parity and time-reversal symmetries and the fractional quantum Hall effect}},
  journal = {Physical Review B},
  volume = {61},
  pages = {10267--10297},
  year = {2000},
  doi = {10.1103/physrevb.61.10267},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.61.10267}
}

@article{LutchynSauDasSarma2010,
  author = {Roman M. Lutchyn and Jay D. Sau and S. Das Sarma},
  title = {{Majorana Fermions and a Topological Phase Transition in Semiconductor-Superconductor Heterostructures}},
  journal = {Physical Review Letters},
  volume = {105},
  pages = {077001},
  year = {2010},
  doi = {10.1103/physrevlett.105.077001},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.105.077001}
}

@article{OregRefaelVonOppen2010,
  author = {Yuval Oreg and Gil Refael and Felix von Oppen},
  title = {{Helical Liquids and Majorana Bound States in Quantum Wires}},
  journal = {Physical Review Letters},
  volume = {105},
  pages = {177002},
  year = {2010},
  doi = {10.1103/physrevlett.105.177002},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.105.177002}
}

@article{Alicea2012,
  author = {Jason Alicea},
  title = {{New directions in the pursuit of Majorana fermions in solid state systems}},
  journal = {Reports on Progress in Physics},
  volume = {75},
  pages = {076501},
  year = {2012},
  doi = {10.1088/0034-4885/75/7/076501},
  url = {https://iopscience.iop.org/article/10.1088/0034-4885/75/7/076501}
}

@article{Beenakker2013,
  author = {C.W.J. Beenakker},
  title = {{Search for Majorana Fermions in Superconductors}},
  journal = {Annual Review of Condensed Matter Physics},
  volume = {4},
  pages = {113--136},
  year = {2013},
  doi = {10.1146/annurev-conmatphys-030212-184337},
  url = {https://www.annualreviews.org/doi/10.1146/annurev-conmatphys-030212-184337}
}

@article{SatoAndo2017,
  author = {Masatoshi Sato and Yoichi Ando},
  title = {{Topological superconductors: a review}},
  journal = {Reports on Progress in Physics},
  volume = {80},
  pages = {076501},
  year = {2017},
  doi = {10.1088/1361-6633/aa6ac7},
  url = {https://iopscience.iop.org/article/10.1088/1361-6633/aa6ac7}
}

@article{QiZhang2011,
  author = {Xiao-Liang Qi and Shou-Cheng Zhang},
  title = {{Topological insulators and superconductors}},
  journal = {Reviews of Modern Physics},
  volume = {83},
  pages = {1057--1110},
  year = {2011},
  doi = {10.1103/revmodphys.83.1057},
  url = {https://link.aps.org/doi/10.1103/RevModPhys.83.1057}
}

@article{SchnyderEtAl2008,
  author = {Andreas P. Schnyder and Shinsei Ryu and Akira Furusaki and Andreas W. W. Ludwig},
  title = {{Classification of topological insulators and superconductors in three spatial dimensions}},
  journal = {Physical Review B},
  volume = {78},
  pages = {195125},
  year = {2008},
  doi = {10.1103/physrevb.78.195125},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.78.195125}
}

@article{RyuEtAl2010,
  author = {Shinsei Ryu and Andreas P Schnyder and Akira Furusaki and Andreas W W Ludwig},
  title = {{Topological insulators and superconductors: tenfold way and dimensional hierarchy}},
  journal = {New Journal of Physics},
  volume = {12},
  pages = {065010},
  year = {2010},
  doi = {10.1088/1367-2630/12/6/065010},
  url = {https://iopscience.iop.org/article/10.1088/1367-2630/12/6/065010}
}

@article{TerayamaEtAl2019,
  author = {Kei Terayama and Ryo Tamura and Yoshitaro Nose and Hidenori Hiramatsu and Hideo Hosono and Yasushi Okuno and Koji Tsuda},
  title = {{Efficient construction method for phase diagrams using uncertainty sampling}},
  journal = {Physical Review Materials},
  volume = {3},
  pages = {033802},
  year = {2019},
  doi = {10.1103/physrevmaterials.3.033802},
  url = {https://link.aps.org/doi/10.1103/PhysRevMaterials.3.033802}
}

@article{DaiBrussGlotzer2020,
  author = {Chengyu Dai and Isaac R. Bruss and Sharon C. Glotzer},
  title = {{Efficient Phase Diagram Sampling by Active Learning}},
  journal = {The Journal of Physical Chemistry B},
  volume = {124},
  pages = {1275--1284},
  year = {2020},
  doi = {10.1021/acs.jpcb.9b09202},
  url = {https://pubs.acs.org/doi/10.1021/acs.jpcb.9b09202}
}

@article{LookmanEtAl2019,
  author = {Turab Lookman and Prasanna V. Balachandran and Dezhen Xue and Ruihao Yuan},
  title = {{Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design}},
  journal = {npj Computational Materials},
  volume = {5},
  pages = {21},
  year = {2019},
  doi = {10.1038/s41524-019-0153-8},
  url = {https://www.nature.com/articles/s41524-019-0153-8}
}

@article{CohnAtlasLadner1994,
  author = {David Cohn and Les Atlas and Richard Ladner},
  title = {{Improving generalization with active learning}},
  journal = {Machine Learning},
  volume = {15},
  pages = {201--221},
  year = {1994},
  doi = {10.1007/bf00993277},
  url = {http://link.springer.com/10.1007/BF00993277}
}

@article{MacKay1992,
  author = {David J. C. MacKay},
  title = {{Information-Based Objective Functions for Active Data Selection}},
  journal = {Neural Computation},
  volume = {4},
  pages = {590--604},
  year = {1992},
  doi = {10.1162/neco.1992.4.4.590},
  url = {https://direct.mit.edu/neco/article/4/4/590-604/5648}
}

@article{ShahriariEtAl2016,
  author = {Bobak Shahriari and Kevin Swersky and Ziyu Wang and Ryan P. Adams and Nando de Freitas},
  title = {{Taking the Human Out of the Loop: A Review of Bayesian Optimization}},
  journal = {Proceedings of the IEEE},
  volume = {104},
  pages = {148--175},
  year = {2016},
  doi = {10.1109/jproc.2015.2494218},
  url = {https://ieeexplore.ieee.org/document/7352306/}
}

@article{TranUlissi2018,
  author = {Kevin Tran and Zachary W. Ulissi},
  title = {{Active learning across intermetallics to guide discovery of electrocatalysts for CO2 reduction and H2 evolution}},
  journal = {Nature Catalysis},
  volume = {1},
  pages = {696--703},
  year = {2018},
  doi = {10.1038/s41929-018-0142-1},
  url = {https://www.nature.com/articles/s41929-018-0142-1}
}

@article{PodryabinkinShapeev2017,
  author = {Evgeny V. Podryabinkin and Alexander V. Shapeev},
  title = {{Active learning of linearly parametrized interatomic potentials}},
  journal = {Computational Materials Science},
  volume = {140},
  pages = {171--180},
  year = {2017},
  doi = {10.1016/j.commatsci.2017.08.031},
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0927025617304536}
}

@article{JinnouchiEtAl2019,
  author = {Ryosuke Jinnouchi and Ferenc Karsai and Georg Kresse},
  title = {{On-the-fly machine learning force field generation: Application to melting points}},
  journal = {Physical Review B},
  volume = {100},
  pages = {014105},
  year = {2019},
  doi = {10.1103/physrevb.100.014105},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.100.014105}
}

@article{VandermauseEtAl2020,
  author = {Jonathan Vandermause and Steven B. Torrisi and Simon Batzner and Yu Xie and Lixin Sun and Alexie M. Kolpak and Boris Kozinsky},
  title = {{On-the-fly active learning of interpretable Bayesian force fields for atomistic rare events}},
  journal = {npj Computational Materials},
  volume = {6},
  pages = {20},
  year = {2020},
  doi = {10.1038/s41524-020-0283-z},
  url = {https://www.nature.com/articles/s41524-020-0283-z}
}

@article{BatraEtAl2021,
  author = {Rohit Batra and Le Song and Rampi Ramprasad},
  title = {{Emerging materials intelligence ecosystems propelled by machine learning}},
  journal = {Nature Reviews Materials},
  volume = {6},
  pages = {655--678},
  year = {2021},
  doi = {10.1038/s41578-020-00255-y},
  url = {https://www.nature.com/articles/s41578-020-00255-y}
}

@article{NoeEtAl2019,
  author = {Frank Noé and Simon Olsson and Jonas Köhler and Hao Wu},
  title = {{Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning}},
  journal = {Science},
  volume = {365},
  pages = {eaaw1147},
  year = {2019},
  doi = {10.1126/science.aaw1147},
  url = {https://www.science.org/doi/10.1126/science.aaw1147}
}

@misc{SupplementalMaterial, note={See Supplemental Material at [URL will be inserted by publisher] for the model, stage-specific algorithms, numerical diagnostics, and selection accounting.}}

```
