from pathlib import Path
import re

path = Path.home() / "robot-project/week3/camera/robot_cloud_brain_v6_threaded.py"
text = path.read_text()

# Convert DepthAI output queues to latest-frame behavior.
# Goal: no backlog of stale frames.
patterns = [
    (
        r'getOutputQueue\(([^)]*name\s*=\s*["\']rgb["\'][^)]*)\)',
        r'getOutputQueue(\1, maxSize=1, blocking=False)'
    ),
    (
        r'getOutputQueue\(([^)]*["\']rgb["\'][^)]*)\)',
        r'getOutputQueue(\1, maxSize=1, blocking=False)'
    ),
]

for pattern, repl in patterns:
    text = re.sub(pattern, repl, text)

# Avoid duplicated maxSize/blocking if patch was partially applied.
text = text.replace(", maxSize=1, blocking=False, maxSize=1, blocking=False", ", maxSize=1, blocking=False")

# Also patch common variable names manually if existing calls use positional queue names.
text = text.replace(
    'device.getOutputQueue(name="rgb")',
    'device.getOutputQueue(name="rgb", maxSize=1, blocking=False)'
)
text = text.replace(
    "device.getOutputQueue('rgb')",
    "device.getOutputQueue('rgb', maxSize=1, blocking=False)"
)
text = text.replace(
    'device.getOutputQueue("rgb")',
    'device.getOutputQueue("rgb", maxSize=1, blocking=False)'
)

path.write_text(text)
print("Patched DepthAI rgb output queue to maxSize=1, blocking=False where found.")
