import requests
import json

r = requests.get('http://127.0.0.1:8000/api/demo/role_resolver_bugfix')
data = r.json()
print("Scenario:", data.get("scenario"))
for idx, step in enumerate(data["steps"]):
    spk_id = step.get("speaker_id")
    role = step.get("speaker_role")
    conf = step.get("role_confidence")
    print(f"Step {idx+1}: {step.get('text')} -> ID: {spk_id}, Role: {role} ({conf:.0%})")

last = data["steps"][-1]
print("\nFinal Persistent Speaker Roles:")
for spk, info in last["speaker_roles"].items():
    print(f"  {spk} -> {info['role'].upper()} ({info['confidence']:.0%}, {info['status']})")
    print(f"    Evidence logs: {info.get('evidence_logs', [])}")
