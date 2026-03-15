#remembers the conversation 
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
        """Resolve pronouns like 'it', 'that', 'this' using last answer."""
        if not self.turns:
            return query
        trigger_words = {"it", "that", "this", "they", "them", "those", "he", "she"}
        first_word    = query.strip().lower().split()[0] if query.strip() else ""
        if first_word in trigger_words:
            context = self.turns[-1]["answer"][:150].replace("\n", " ")
            return f"[Prior context: {context}] {query}"
        return query

    def update_entities(self, text: str):
        for amount in re.findall(r"\$[\d,]+(?:\.\d+)?", text):
            self.entities[amount] = "amount"
        for date in re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
            self.entities[date] = "date"
        for fname in re.findall(r"\b[\w-]+\.(?:pdf|doc|txt|xls)\b", text, re.I):
            self.entities[fname] = "filename"