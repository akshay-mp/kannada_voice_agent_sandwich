
from elevenlabs.realtime.scribe import CommitStrategy
print("CommitStrategy members:")
for member in CommitStrategy:
    print(f"{member.name}: {member.value}")
