import os
import shutil
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    FieldSet,
    FirstPhaseRanking
)
from vespa.deployment import VespaDocker

def create_merged_app():
    # 1. Central Acts (CA) Schema
    ca_schema = Schema(
        name="centralacts",
        document=Document(
            fields=[
                Field(name="id", type="string", indexing=["summary", "attribute"]),
                Field(name="name_of_statute", type="string", indexing=["summary", "index"], index="enable-bm25"),
                Field(name="section_text", type="string", indexing=["summary", "index"], index="enable-bm25"),
                Field(name="section_number", type="string", indexing=["summary", "attribute"]),
                Field(name="section_title", type="string", indexing=["summary", "index"], index="enable-bm25"),
                Field(
                    name="embedding",
                    type="tensor<float>(x[768])",
                    indexing=["attribute", "index"],
                    attribute=["paged"],
                    ann=HNSW(
                        distance_metric="euclidean",
                        max_links_per_node=16,
                        neighbors_to_explore_at_insert=200
                    )
                )
            ]
        ),
        fieldsets=[
            FieldSet(name="default", fields=["section_text", "section_title"])
        ],
        rank_profiles=[
            RankProfile(
                name="semantic",
                inputs=[("query(q_emb)", "tensor<float>(x[768])")],
                first_phase="closeness(field, embedding)"
            ),
            RankProfile(
                name="bm25",
                first_phase="bm25(section_text) + bm25(section_title)"
            )
        ]
    )

    # 2. Supreme Court (SC) Schema
    sc_schema = Schema(
        name="scjudgments",
        document=Document(
            fields=[
                Field(name="id", type="string", indexing=["summary"]),
                Field(name="content", type="string", indexing=["index", "summary"], index="enable-bm25"),
                Field(name="case_name", type="string", indexing=["summary", "attribute"]),
                Field(name="diary_number", type="string", indexing=["summary", "attribute"]),
                Field(name="judgment_type", type="string", indexing=["summary", "attribute"]),
                Field(name="case_number", type="string", indexing=["summary", "attribute"]),
                Field(name="petitioner", type="string", indexing=["summary", "attribute"]),
                Field(name="respondent", type="string", indexing=["summary", "attribute"]),
                Field(name="petitioner_advocate", type="string", indexing=["summary", "attribute"]),
                Field(name="respondent_advocate", type="string", indexing=["summary", "attribute"]),
                Field(name="bench", type="string", indexing=["summary", "attribute"]),
                Field(name="judgment_by", type="string", indexing=["summary", "attribute"]),
                Field(name="judgment_date", type="string", indexing=["summary", "attribute"]),
                Field(name="temp_link", type="string", indexing=["summary", "attribute"]),
                Field(name="citations", type="string", indexing=["summary", "attribute"]),
                Field(
                    name="embedding",
                    type="tensor<float>(x[768])",
                    indexing=["attribute", "index"],
                    attribute=["paged"],
                    ann=HNSW(
                        distance_metric="euclidean",
                        max_links_per_node=16,
                        neighbors_to_explore_at_insert=200
                    )
                )
            ]
        ),
        fieldsets=[
            FieldSet(name="default", fields=["content", "case_name"])],
        rank_profiles=[
            RankProfile(
                name="semantic",
                inputs=[("query(q_emb)","tensor<float>(x[768])")],
                first_phase=FirstPhaseRanking(expression="closeness(field, embedding)")
            ),
            RankProfile(
                name="bm25",
                first_phase=FirstPhaseRanking(expression="bm25(content)")
            )
        ]
    )

    state_list = ["Lakshadweep", "Madhya Pradesh", "Tamil Nadu", "Chandigarh", "Bihar",
    "Uttarakhand", "Rajasthan", "Jharkhand", "Jammu and Kashmir", "Odisha",
    "Puducherry", "Punjab", "Andhra Pradesh", "West Bengal", "Tripura",
    "Ladakh", "Chhattisgarh", "Maharashtra", "Telangana", "Himachal Pradesh",
    "Gujarat", "Manipur", "Karnataka", "Arunachal Pradesh",
    "Andaman and Nicobar Islands", "Assam", "Delhi", "Goa", "Meghalaya",
    "Dadra and Nagar Haveli and Daman and Diu", "Kerala", "Haryana",
    "Uttar Pradesh"]

    sa_schemas = []

    for stateName in state_list:
        file_path = f'/home/shubham/tanmay/lexensemble/code/state_acts_csv/{stateName}.csv'
        #ADD STATE ACTS FILEPATHS HERE

        # Only create schema if the file exists
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping `{stateName}` — CSV not found at {file_path}")
            continue

        # Sanitize state name for Schema name (lowercase, no spaces)
        # e.g., "Tamil Nadu" -> "sa_tamil_nadu"
        clean_name = stateName.lower().replace(" ", "_")
        schema_name = f"sa_{clean_name}"

        new_sa_schema = Schema(
            name=schema_name,
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["summary", "index"]),
                    Field(name="state_name", type="string", indexing=["summary", "attribute"]),
                    Field(name="name_of_statute", type="string", indexing=["summary", "index", "attribute"]),
                    Field(name="section_number", type="string", indexing=["summary", "attribute"]),
                    Field(name="section_title", type="string", indexing=["summary", "index"], index="enable-bm25"),
                    Field(name="original_doc_id", type="int", indexing=["summary", "attribute"]),
                    Field(name="section_text", type="string", indexing=["summary", "index"], index="enable-bm25"),
                    Field(
                        name="embedding",
                        type="tensor<float>(x[768])",
                        indexing=["attribute", "index"],
                        attribute=["paged"],
                        ann=HNSW(distance_metric="euclidean", max_links_per_node=16, neighbors_to_explore_at_insert=200)
                    ),
                ]
            ),
            fieldsets=[
                FieldSet(name="default", fields=["section_text", "name_of_statute", "section_title"])
            ],
            rank_profiles=[
                RankProfile(
                    name="semantic",
                    inputs=[("query(q_emb)", "tensor<float>(x[768])")],
                    first_phase="closeness(field, embedding)"
                ),
                RankProfile(
                    name="bm25",
                    first_phase="bm25(section_text) + bm25(section_title)"
                )
            ]
        )
        sa_schemas.append(new_sa_schema)


    # 4. High Courts (HC) Schemas

    collection_map = {"/home/shubham/tanmay/lexensemble/code/hc_cases_judgements.csv": "hcjudgments",
    }

    # Extract unique collection names from the map values
    hc_collection_names = set(collection_map.values())

    hc_schemas = []

    for hc_name in hc_collection_names:
        schema_name = hc_name.lower()

        new_hc_schema = Schema(
            name=schema_name,
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["summary", "attribute"]),
                    Field(name="case_name", type="string", indexing=["summary", "attribute", "index"], index="enable-bm25"),
                    Field(name="judgment_date", type="string", indexing=["summary", "attribute"]),
                    Field(name="citations", type="string", indexing=["summary", "attribute"]),
                    Field(name="bench", type="string", indexing=["summary", "attribute"]),
                    Field(
                        name="text",
                        type="string",
                        indexing=["index", "summary"],
                        index="enable-bm25"
                    ),
                    Field(
                        name="embedding",
                        type="tensor<float>(x[768])",
                        indexing=["attribute", "index"],
                        attribute=["paged"],
                        ann=HNSW(
                            distance_metric="euclidean",
                            max_links_per_node=16,
                            neighbors_to_explore_at_insert=200
                        )
                    )
                ]
            ),
            fieldsets=[
                FieldSet(name="default", fields=["text", "case_name"])
            ],
            rank_profiles=[
                RankProfile(
                    name="semantic",
                    inputs=[("query(q_emb)", "tensor<float>(x[768])")],
                    first_phase="closeness(field, embedding)"
                ),
                RankProfile(
                    name="bm25",
                    first_phase="bm25(text) + bm25(case_name)"
                )
            ]
        )
        hc_schemas.append(new_hc_schema)

    # Combine ALL schemas: Central Acts + Supreme Court + (Dynamic State Acts) + (Dynamic High Courts)
    all_schemas = [ca_schema, sc_schema] + sa_schemas + hc_schemas
    print("ALL SCHEMAS:", all_schemas)

    app_package = ApplicationPackage(
        name="legalknowledgebase",
        schema=all_schemas
    )

    return app_package

def deploy():
    cwd = os.getcwd()
    app_source_dir = os.path.join(cwd, "vespa_app_source")
    data_persistence_dir = os.path.join(cwd, "vespa_data")

    if os.path.exists(app_source_dir):
        shutil.rmtree(app_source_dir)

    if not os.path.exists(data_persistence_dir):
        os.makedirs(data_persistence_dir)

    try:
        os.chmod(data_persistence_dir, 0o777) 
    except Exception as e:
        print(f"Warning: Could not set permissions: {e}")


    print("Generating application package...")
    app_package = create_merged_app()
    app_package.to_files(app_source_dir)
    print(f"Application package written to: {app_source_dir}")

    # --- XML PATCHING START (CORRECTED) ---
    # Fixes: INVALID_APPLICATION_PACKAGE and NO_SPACE
    services_path = os.path.join(app_source_dir, "services.xml")
    
    with open(services_path, "r") as f:
        xml_content = f.read()

    # We must inject <tuning> block containing resource-limits
    if "<resource-limits>" not in xml_content:
        patch_str = """
        <tuning>
            <resource-limits>
                <disk>0.98</disk>
                <memory>0.98</memory>
            </resource-limits>
        </tuning>
        """
        # Insert before the closing content tag
        xml_content = xml_content.replace("</content>", f"{patch_str}\n</content>")
        
        with open(services_path, "w") as f:
            f.write(xml_content)
        print("Patched services.xml: Increased disk limit to 98% (inside <tuning> tag).")
    # --- XML PATCHING END ---

    vespa_docker = VespaDocker(
        port=8085,
        volumes={data_persistence_dir: {'bind': '/opt/vespa/var', 'mode': 'rw'}},
    )

    print(f"Deploying with data persistence at: {data_persistence_dir}")

    app = vespa_docker.deploy_from_disk(
        application_name="legalknowledgebase",
        application_root=app_source_dir
    )

    print("Vespa application deployed successfully.")
    print(f"Endpoint: {app.url}")
    return app

if __name__ == "__main__":
    deploy()
