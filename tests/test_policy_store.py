import shutil
from pathlib import Path

import pytest

from crop_insurance_agent.rag import PolicyNotFoundError, PolicyStore
from tests.conftest import hash_embedder

RWBCIS = "sample_rwbcis_paddy_thanjavur"
PMFBY = "sample_pmfby_mid_season_adversity"


@pytest.fixture
def store(tmp_path: Path) -> PolicyStore:
    return PolicyStore(tmp_path / "chroma", embedder=hash_embedder)


@pytest.fixture
def pdfs(policies_dir: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(policies_dir, tmp_path / "policies"))


def test_sync_indexes_once_and_tracks_changes(store: PolicyStore, pdfs: Path):
    assert sorted(store.sync(pdfs)) == [PMFBY, RWBCIS]
    assert store.policy_ids() == [PMFBY, RWBCIS]
    assert store.sync(pdfs) == []  # unchanged files are not re-embedded

    shutil.copy(pdfs / f"{PMFBY}.pdf", pdfs / f"{RWBCIS}.pdf")
    assert store.sync(pdfs) == [RWBCIS]
    assert all("Mid-season" in c.text or "Scope" in c.text for c in store.search(RWBCIS, "x", 10))

    (pdfs / f"{PMFBY}.pdf").unlink()
    store.sync(pdfs)
    assert store.policy_ids() == [RWBCIS]


def test_index_persists_across_instances(tmp_path: Path, pdfs: Path):
    PolicyStore(tmp_path / "chroma", embedder=hash_embedder).sync(pdfs)
    reopened = PolicyStore(tmp_path / "chroma", embedder=hash_embedder)
    assert reopened.policy_ids() == [PMFBY, RWBCIS]
    assert reopened.sync(pdfs) == []


def test_search_is_scoped_to_one_policy(store: PolicyStore, pdfs: Path):
    store.sync(pdfs)
    hits = store.search(RWBCIS, "growing degree days high temperature", k=2)
    assert hits and {c.policy_id for c in hits} == {RWBCIS}
    assert any("growing degree days" in c.text for c in hits)
    assert all(c.page in (1, 2) for c in hits)


def test_unknown_policy(store: PolicyStore, pdfs: Path):
    store.sync(pdfs)
    with pytest.raises(PolicyNotFoundError):
        store.search("nope", "rain")
