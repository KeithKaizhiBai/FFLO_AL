import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from run_stage_vi0_2 import LEVELS
def test_v2_removes_kgap_and_deltaf_uf_from_core():
 assert LEVELS['level3']==['log_gap'];assert 'deltaF_UF' not in sum(LEVELS.values(),[]);assert 'k_gap' not in LEVELS['level3']
def test_topology_is_separate_level():
 assert LEVELS['level4']==['p0','ppi','pf_margin']
