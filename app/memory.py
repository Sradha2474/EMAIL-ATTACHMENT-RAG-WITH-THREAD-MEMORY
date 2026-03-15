import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionMemory:
    session_id : str
    thread_id  : Optional[str] = None
    turns      : list = field(default_factory=list)
    entities   : dict = field(default_factory=dict)

    def add_turn(self, user_text: str, rewrite: str, answer: str):
        self.turns.append({
            "user"    : user_text,
            "rewrite" : rewrite,
            "answer"  : answer,
        })
        if len(self.turns) > 5:
            self.turns.pop(0)

    def rewrite_query(self, query: str) -> str:
        """Resolve pronouns and short follow-ups using conversation history."""
        if not self.turns:
            return query

        q_lower    = query.lower().strip()
        last_answer = self.turns[-1]["answer"]
        last_user   = self.turns[-1]["user"]

        pronouns   = {
            "it", "that", "this", "they", "them", "those",
            "he", "she", "her", "his", "its"
        }
        words      = q_lower.split()
        first_word = words[0] if words else ""

        # Resolve pronoun at start of question
        if first_word in pronouns or q_lower.startswith("what about"):
            context = last_answer[:200].replace("\n", " ")
            return f"[Referring to: {context}] {query}"

        # Very short follow-up — attach previous question for context
        if len(words) < 5:
            return f"{last_user} — specifically: {query}"

        return query

    def update_entities(self, text: str):
        """Extract and remember key facts from answers."""
        for amount in re.findall(r"\$[\d,]+(?:\.\d+)?", text):
            self.entities[amount] = "amount"
        for date in re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
            self.entities[date] = "date"
        for fname in re.findall(r"\b[\w-]+\.(?:pdf|doc|txt|xls|xlsx)\b", text, re.I):
            self.entities[fname] = "filename"