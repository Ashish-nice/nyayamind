"""Create and deploy the merged NyayaMind Vespa application.

One Vespa application ("legalknowledgebase") holds every NyayaMindDB
collection as its own schema:

    centralacts             Central Acts sections            (data/CA)
    scjudgments             Supreme Court judgments          (data/SC)
    hcjudgments             High Court judgments             (data/HC)
    sa_<state>              State Acts sections, one per state CSV in data/SA

Each schema has the paper's HNSW index (euclidean, M=16, efConstruction=200)
on a 768-d arctic-embed vector and BM25-enabled text fields, plus two rank
profiles used by the retrieval scripts: ``semantic`` and ``bm25``.

Laptop notes: vectors are stored as bfloat16, Vespa data lives in a named
Docker volume (fast, survives container restarts), and the container memory
is capped (``--memory``, default 6G) so Docker Desktop's WSL2 VM - which by
default gets half of a 16 GB machine - is not exhausted.

Usage:
    python code/rag-setup/vespa/new_setup_merged.py            # generate + deploy
    python code/rag-setup/vespa/new_setup_merged.py --package-only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from vespa.package import (
    HNSW,
    ApplicationPackage,
    Document,
    Field,
    FieldSet,
    RankProfile,
    Schema,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

VECTOR_TYPE = f"tensor<bfloat16>(x[{c.EMBED_DIM}])"
QUERY_INPUT = ("query(q_emb)", f"tensor<float>(x[{c.EMBED_DIM}])")


def embedding_field() -> Field:
    return Field(
        name="embedding",
        type=VECTOR_TYPE,
        indexing=["attribute", "index"],
        attribute=["paged"],
        ann=HNSW(
            distance_metric="euclidean",
            max_links_per_node=16,
            neighbors_to_explore_at_insert=200,
        ),
    )


def rank_profiles(bm25_expression: str) -> list[RankProfile]:
    return [
        RankProfile(
            name="semantic",
            inputs=[QUERY_INPUT],
            first_phase="closeness(field, embedding)",
        ),
        RankProfile(name="bm25", first_phase=bm25_expression),
    ]


def text_field(name: str, bm25: bool = True, attribute: bool = False) -> Field:
    indexing = ["summary", "index"] + (["attribute"] if attribute else [])
    return Field(
        name=name,
        type="string",
        indexing=indexing,
        index="enable-bm25" if bm25 else None,
    )


def meta_field(name: str, type_: str = "string") -> Field:
    return Field(name=name, type=type_, indexing=["summary", "attribute"])


def central_acts_schema() -> Schema:
    return Schema(
        name=c.CA_SCHEMA,
        document=Document(
            fields=[
                meta_field("id"),
                text_field("name_of_statute"),
                text_field("section_text"),
                meta_field("section_number"),
                text_field("section_title"),
                embedding_field(),
            ]
        ),
        fieldsets=[FieldSet(name="default", fields=["section_text", "section_title"])],
        rank_profiles=rank_profiles("bm25(section_text) + bm25(section_title)"),
    )


def supreme_court_schema() -> Schema:
    return Schema(
        name=c.SC_SCHEMA,
        document=Document(
            fields=[
                Field(name="id", type="string", indexing=["summary"]),
                text_field("content"),
                meta_field("case_name"),
                meta_field("diary_number"),
                meta_field("judgment_type"),
                meta_field("case_number"),
                meta_field("petitioner"),
                meta_field("respondent"),
                meta_field("petitioner_advocate"),
                meta_field("respondent_advocate"),
                meta_field("bench"),
                meta_field("judgment_by"),
                meta_field("judgment_date"),
                meta_field("temp_link"),
                meta_field("citations"),
                meta_field("chunk_index", "int"),
                embedding_field(),
            ]
        ),
        # case_name is attribute-only, so it cannot share a fieldset with the
        # BM25 text index (mixed match settings); BM25 runs on content.
        fieldsets=[FieldSet(name="default", fields=["content"])],
        rank_profiles=rank_profiles("bm25(content)"),
    )


def state_act_schema(state_name: str) -> Schema:
    return Schema(
        name=c.state_schema(state_name),
        document=Document(
            fields=[
                Field(name="id", type="string", indexing=["summary", "index"]),
                meta_field("state_name"),
                text_field("name_of_statute"),
                meta_field("section_number"),
                text_field("section_title"),
                meta_field("original_doc_id", "int"),
                text_field("section_text"),
                embedding_field(),
            ]
        ),
        fieldsets=[
            FieldSet(
                name="default",
                fields=["section_text", "name_of_statute", "section_title"],
            )
        ],
        rank_profiles=rank_profiles("bm25(section_text) + bm25(section_title)"),
    )


def high_court_schema() -> Schema:
    return Schema(
        name=c.HC_SCHEMA,
        document=Document(
            fields=[
                meta_field("id"),
                text_field("case_name"),
                meta_field("court_name"),
                meta_field("judgment_date"),
                meta_field("citations"),
                meta_field("bench"),
                meta_field("chunk_index", "int"),
                Field(name="text", type="string", indexing=["index", "summary"], index="enable-bm25"),
                embedding_field(),
            ]
        ),
        fieldsets=[FieldSet(name="default", fields=["text", "case_name"])],
        rank_profiles=rank_profiles("bm25(text) + bm25(case_name)"),
    )


def create_merged_app() -> ApplicationPackage:
    states = c.available_states()
    missing = sorted(set(c.STATE_LIST) - set(states))
    if missing:
        print(f"Skipping {len(missing)} states without a CSV in {c.SA_DIR}: {missing}")
    schemas = [
        central_acts_schema(),
        supreme_court_schema(),
        high_court_schema(),
        *(state_act_schema(s) for s in states),
    ]
    print(f"Schemas ({len(schemas)}): {[s.name for s in schemas]}")
    return ApplicationPackage(name=c.APP_NAME, schema=schemas)


def patch_services(services_path: Path, memory_limit: float) -> None:
    """Relax feed-block limits so a nearly-full laptop disk does not block feeding."""
    xml = services_path.read_text(encoding="utf-8")
    if "<resource-limits>" not in xml:
        patch = f"""
        <tuning>
            <resource-limits>
                <disk>0.98</disk>
                <memory>{memory_limit}</memory>
            </resource-limits>
        </tuning>
        """
        xml = xml.replace("</content>", f"{patch}\n</content>")
        services_path.write_text(xml, encoding="utf-8")
        print("Patched services.xml with resource limits.")


def write_package(app_dir: Path, memory_limit: float) -> None:
    if app_dir.exists():
        shutil.rmtree(app_dir)
    create_merged_app().to_files(str(app_dir))
    patch_services(app_dir / "services.xml", memory_limit)
    print(f"Application package written to: {app_dir}")


def deploy(app_dir: Path, container_memory: str, volume: str):
    from vespa.deployment import VespaDocker

    vespa_docker = VespaDocker(
        port=c.VESPA_PORT,
        cfgsrv_port=c.VESPA_CFG_PORT,
        container_memory=container_memory,
        # A named volume (not a Windows bind mount): Vespa needs a Linux
        # filesystem with mmap/locking, and the data survives restarts.
        volumes=[f"{volume}:/opt/vespa/var"],
    )
    app = vespa_docker.deploy_from_disk(
        application_name=c.APP_NAME,
        application_root=app_dir,
        max_wait_configserver=300,
        max_wait_application=600,
    )
    print(f"Vespa application deployed. Endpoint: {app.url}:{c.VESPA_PORT}")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app-dir", type=Path, default=c.VESPA_DIR / "vespa_app_source")
    parser.add_argument("--memory", default="6G", help="Docker container memory cap (default 6G).")
    parser.add_argument("--volume", default="nyayamind_vespa_var", help="Docker volume for Vespa data.")
    parser.add_argument("--memory-limit", type=float, default=0.9, help="Vespa feed-block memory fraction.")
    parser.add_argument("--package-only", action="store_true", help="Only write the application package.")
    args = parser.parse_args()

    write_package(args.app_dir, args.memory_limit)
    if not args.package_only:
        deploy(args.app_dir, args.memory, args.volume)


if __name__ == "__main__":
    main()
