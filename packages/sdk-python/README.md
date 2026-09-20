# ragfabric-sdk

Python client for a RagFabric server, talking HTTP only.

```python
from ragfabric_sdk import Client

client = Client("http://localhost:8000", token="a-jwt-or-api-key")
answer = client.ask("What is the refund policy?")
print(answer.answer, answer.citations)
```
