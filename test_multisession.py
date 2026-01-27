"""
Test script for multi-session data generation.
"""

import json
import logging

from persona_gym.data_generators import MultiSessionGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Test persona
PERSONA = """
A 32-year-old software engineer at a mid-size tech company who has been feeling
stagnant in their career. They have 8 years of experience in backend development
but are curious about transitioning to machine learning. They're methodical,
prefer structured learning, and are budget-conscious. They live in Seattle with
their partner and enjoy hiking on weekends.
"""


def main():
    print("=" * 60)
    print("Multi-Session Generator Test")
    print("=" * 60)

    generator = MultiSessionGenerator(
        persona=PERSONA.strip(),
        num_sessions=2,
        num_preferences=5,
        num_to_evolve=2,
    )

    # Generate multi-session data
    result = generator.generate_multi_session()

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    print(f"\nPersona ID: {result.persona_id}")
    print(f"Total Sessions: {len(result.sessions)}")
    print(f"Total Preferences: {len(result.timeline.preferences)}")
    print(f"Active Preferences: {len(result.timeline.get_active_preferences())}")

    # Print life events
    print("\n--- Life Events ---")
    for event in result.life_events:
        print(f"  [{event.session_id}] {event.date}: {event.event}")

    # Print preference timeline
    print("\n--- Preference Timeline ---")
    for pref_id, pref in result.timeline.preferences.items():
        status = "ACTIVE" if pref.is_active else f"SUPERSEDED by {pref.superseded_by}"
        print(f"  {pref_id}: [{pref.category}] {pref.fact[:60]}... ({status})")

    # Print session summaries
    print("\n--- Sessions ---")
    for session in result.sessions:
        print(f"\n  Session {session.session_id}:")
        print(f"    Life Event: {session.life_event.event[:50]}...")
        print(f"    Conversation Turns: {len(session.conversation)}")
        print(f"    Active Prefs: {session.active_preference_ids}")
        if session.evolved_preference_ids:
            print(f"    Evolved: {session.evolved_preference_ids}")

        # Print first 2 turns of conversation
        print("    Sample conversation:")
        for turn in session.conversation[:4]:
            content = turn['content'][:80].replace('\n', ' ')
            print(f"      [{turn['role']}]: {content}...")

    # Save to file
    output_path = "outputs/test_multisession_output.json"
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"\n✅ Full output saved to: {output_path}")


if __name__ == "__main__":
    main()
