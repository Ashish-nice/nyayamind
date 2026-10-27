from vespa.application import Vespa

VESPA_URL = "http://localhost:8083"
app = Vespa(url=VESPA_URL)

# All schemas from new_setup_merged.py
schemas = [
    # Central Acts
    "centralacts",
    # Supreme Court
    "scjudgments",
    # High Court (unified)
    "hcjudgments",
    # State Acts (33 per-state schemas)
    "sa_lakshadweep",
    "sa_madhya_pradesh",
    "sa_tamil_nadu",
    "sa_chandigarh",
    "sa_bihar",
    "sa_uttarakhand",
    "sa_rajasthan",
    "sa_jharkhand",
    "sa_jammu_and_kashmir",
    "sa_odisha",
    "sa_puducherry",
    "sa_punjab",
    "sa_andhra_pradesh",
    "sa_west_bengal",
    "sa_tripura",
    "sa_ladakh",
    "sa_chhattisgarh",
    "sa_maharashtra",
    "sa_telangana",
    "sa_himachal_pradesh",
    "sa_gujarat",
    "sa_manipur",
    "sa_karnataka",
    "sa_arunachal_pradesh",
    "sa_andaman_and_nicobar_islands",
    "sa_assam",
    "sa_delhi",
    "sa_goa",
    "sa_meghalaya",
    "sa_dadra_and_nagar_haveli_and_daman_and_diu",
    "sa_kerala",
    "sa_haryana",
    "sa_uttar_pradesh",
]

print(f"{'Schema':<50} {'Documents':>10}")
print("=" * 62)

total = 0
for schema in schemas:
    try:
        response = app.query(body={
            "yql": f"select * from sources {schema} where true",
            "hits": 0
        })
        count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
        print(f"{schema:<50} {count:>10}")
        total += count
    except Exception as e:
        print(f"{schema:<50} {'ERROR':>10}  ({e})")

print("=" * 62)
print(f"{'TOTAL':<50} {total:>10}")
