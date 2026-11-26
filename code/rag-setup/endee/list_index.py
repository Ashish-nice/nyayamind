import os
from endee import Endee

client = Endee(token=os.environ["ENDEE_TOKEN"])  # set ENDEE_TOKEN in .env / environment
client.set_base_url("http://172.27.21.35:8080/api/v1")
print(client.list_indexes())