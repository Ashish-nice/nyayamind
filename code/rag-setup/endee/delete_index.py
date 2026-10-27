from endee import Endee

client = Endee(token="shubham:NXdEaTCJQEmqnPTKicXNVDFRRvuryuAR")
client.set_base_url("http://172.27.21.35:8080/api/v1")
index_name = "HC_Judgments_DB"
client.delete_index(index_name)

print(f"Index '{index_name}' deleted successfully")
