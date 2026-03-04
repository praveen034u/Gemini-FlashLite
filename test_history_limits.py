"""
Test script to verify hybrid history limits implementation
"""
from main import apply_history_limits, estimate_tokens, LLM_CONFIG
from langchain_core.messages import HumanMessage, AIMessage

# Create sample messages
messages = []
for i in range(30):
    messages.append(HumanMessage(content=f"User question {i}: This is a test message with some content"))
    messages.append(AIMessage(content=f"AI response {i}: This is a longer response with more detailed information about the topic"))

print(f"✓ Created {len(messages)} test messages")

# Get limits from config
history_limits = LLM_CONFIG.get("history_limits", {})
max_history_messages = history_limits.get("max_history_messages", 20)
max_history_tokens = history_limits.get("max_history_tokens", 6000)
max_history_age_days = history_limits.get("max_history_age_days", 7)

print(f"\n✓ Limits from config:")
print(f"  - Max messages: {max_history_messages}")
print(f"  - Max tokens: {max_history_tokens}")
print(f"  - Max age (days): {max_history_age_days}")

# Apply limits
filtered = apply_history_limits(messages, max_history_messages, max_history_tokens, max_history_age_days)

print(f"\n✓ Filtering results:")
print(f"  - Original: {len(messages)} messages")
print(f"  - Filtered: {len(filtered)} messages")

# Calculate total tokens in filtered messages
total_tokens = sum(estimate_tokens(msg.content) for msg in filtered)
print(f"  - Total tokens: ~{total_tokens}")

# Verify limits are respected
assert len(filtered) <= max_history_messages, "Message count limit exceeded!"
assert total_tokens <= max_history_tokens, "Token limit exceeded!"

print(f"\n✓ All hybrid history limits are working correctly!")
