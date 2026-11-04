"""Manual live check for Chunk B2 (uses the REAL Groq API). Not part of pytest."""

from src.llm.extraction import ConversationTurn, ExtractionContext, extract_symptoms

DAY1 = ExtractionContext(
    conversation=[
        ConversationTurn(role="assistant", text="How are you feeling today?"),
        ConversationTurn(role="patient", text="I have a fever and feel weak."),
        ConversationTurn(role="assistant", text="When did your fever start?"),
    ],
    last_question="When did your fever start?",
    last_question_symptom="fever",
    last_question_field="duration",
)

DAY1_TEMP = DAY1.model_copy(update={
    "conversation": DAY1.conversation + [
        ConversationTurn(role="patient", text="Yesterday."),
        ConversationTurn(role="assistant", text="Have you measured your temperature?"),
    ],
    "last_question": "Have you measured your temperature?",
    "last_question_field": "details",
})

DAY2 = ExtractionContext(
    previous_symptoms=[
        {"name": "fever", "status": "ongoing", "latest_duration": "since yesterday"},
        {"name": "weakness", "status": "ongoing"},
    ],
    last_question="Yesterday you mentioned a fever and weakness. How are you feeling today?",
)

CASES = [
    ("Day 1 short answer", "Yesterday.", DAY1),
    ("Day 1 temperature", "Yes, it was 101 F.", DAY1_TEMP),
    ("Day 2 resolution", "My fever is gone, but I still feel weak.", DAY2),
    ("Day 2 ambiguous", "Same as before, I guess.", DAY2),
]

for title, text, ctx in CASES:
    print(f"\n=== {title}: \"{text}\"")
    result = extract_symptoms(text, ctx)
    if result is None:
        print("  FAILED: no result")
        continue
    for s in result.symptoms:
        print("  ", s.model_dump(exclude_none=True))
    if not result.symptoms:
        print("   (no symptoms)")
    if result.ungrounded_symptoms:
        print("   ignored (not in new statement):", result.ungrounded_symptoms)