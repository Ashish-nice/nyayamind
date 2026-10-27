from pymilvus import Collection, connections

connections.connect(host="localhost", port="19530")
collection = Collection("SC_Judgments_DB")
collection.load()

all_values = set()  # Use a set to store only unique values
batch_size = 1000
last_id = 0

while True:
    expr = f"id > {last_id}"
    res = collection.query(
        expr=expr,
        limit=batch_size,
        output_fields=["id"],
        order_by="id"
    )

    if not res:
        break

    current_ids = [r["id"] for r in res]
    all_values.update(current_ids)
    last_id = max(current_ids)

    if len(res) < batch_size:
        break

print(f"Total unique values: {len(all_values)}")