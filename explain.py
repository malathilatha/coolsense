"""
Gemma turns a detected fault into (1) a message for the resident / shop owner in
their language and (2) a job card for the technician.

Rule: the fault and every number come from code + the models. Gemma only words
them, using the knowledge base below, so it cannot invent a fault or a part.
If Ollama is not running, a plain English template is used instead.
"""
import json
import time

SEPARATE_LOW_VS_BLOCKED = ("Low refrigerant and a blocked liquid line look alike (both starve the indoor coil). "
                           "Measure subcooling at the condenser outlet: LOW subcooling = low refrigerant, "
                           "HIGH subcooling = restriction.")

KB = {
    "Low refrigerant (gas leak)": {
        "urgency": "high",
        "resident_can_do": "Keep the AC on a normal setting; turning it to the coldest setting will not help. "
                           "Ask for the leak to be found and fixed, not only a gas top-up, or it will happen again.",
        "causes": ["Refrigerant leak at a flare joint, coil or valve", "Previous top-up without leak repair"],
        "parts": ["Electronic leak detector", "Nitrogen + pressure test kit", "Brazing kit", "Filter-drier",
                  "Vacuum pump + micron gauge", "R410A cylinder + charging scale"],
        "checks": ["Superheat and subcooling", "Leak search on joints and coils", SEPARATE_LOW_VS_BLOCKED,
                   "Charge by weight after vacuum"],
    },
    "Blocked liquid line / filter-drier": {
        "urgency": "high",
        "resident_can_do": "Nothing to do yourself. Book a technician soon; the AC is working harder than it should.",
        "causes": ["Clogged filter-drier", "Kinked or crushed liquid line", "Debris at the expansion valve"],
        "parts": ["Filter-drier (match line size)", "Brazing kit", "Nitrogen", "Vacuum pump"],
        "checks": ["Temperature drop across the filter-drier", SEPARATE_LOW_VS_BLOCKED],
    },
    "Restricted suction line": {
        "urgency": "high",
        "resident_can_do": "Nothing to do yourself. Book a technician; ice on the pipes is a warning sign.",
        "causes": ["Kinked or crushed suction line", "Ice blocking the line", "Partly closed service valve"],
        "parts": ["Brazing kit", "Pipe bender / replacement copper", "Insulation"],
        "checks": ["Inspect suction line for kinks and frost", "Pressure drop between evaporator outlet and compressor"],
    },
    "Dirty outdoor (condenser) coil": {
        "urgency": "medium",
        "resident_can_do": "Keep 50 cm clear around the outdoor unit; no boxes, plants or covers. Dust and sand build up fast in summer.",
        "causes": ["Dust and sand on the outdoor coil", "Blocked airflow around the unit", "Weak outdoor fan"],
        "parts": ["Coil cleaner", "Low-pressure water spray", "Fin comb", "Outdoor fan capacitor (check rating)"],
        "checks": ["Inspect and wash condenser coil", "Check outdoor fan speed and amps", "Recheck high-side pressure after cleaning"],
    },
    "Low indoor airflow (dirty filter / indoor coil)": {
        "urgency": "medium",
        "resident_can_do": "Clean or replace the air filter, and make sure furniture or curtains are not blocking vents.",
        "causes": ["Dirty filter", "Dirty indoor coil", "Weak blower or blocked duct"],
        "parts": ["Replacement filters", "Indoor coil cleaner", "Blower capacitor / belt"],
        "checks": ["Filter condition", "Indoor coil for dirt or ice", "Blower amps and airflow"],
    },
    "Too much refrigerant (overcharge)": {
        "urgency": "medium",
        "resident_can_do": "Nothing to do yourself. This often follows a recent gas top-up; mention it to the technician.",
        "causes": ["Over-filled during a previous top-up"],
        "parts": ["Refrigerant recovery machine + cylinder", "Charging scale"],
        "checks": ["Subcooling against the nameplate target", "Recover to correct charge by weight"],
    },
}

# Unicode ranges used to check Gemma really answered in the requested language.
SCRIPTS = {"Arabic": [(0x0600, 0x06FF)], "Urdu": [(0x0600, 0x06FF)], "Hindi": [(0x0900, 0x097F)],
           "Tamil": [(0x0B80, 0x0BFF)], "Malayalam": [(0x0D00, 0x0D7F)]}


def right_script(text: str, language: str) -> bool:
    """True if most letters are in the requested script (English/Tagalog: Latin)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if language not in SCRIPTS:
        ok = sum(c.isascii() for c in letters)
    else:
        ok = sum(any(lo <= ord(c) <= hi for lo, hi in SCRIPTS[language]) for c in letters)
    return ok / len(letters) >= 0.6


SCHEMA = {
    "type": "object",
    "properties": {"resident_message": {"type": "string"}, "technician_summary": {"type": "string"}},
    "required": ["resident_message", "technician_summary"],
}


def job_card(fault: str, confidence: float, facts: list[str]) -> dict:
    """Built by code from the knowledge base: safe to show even with the LLM off."""
    kb = KB[fault]
    return {"fault": fault, "confidence": round(confidence, 2), "urgency": kb["urgency"],
            "evidence": facts, "likely_causes": kb["causes"], "bring": kb["parts"], "checks": kb["checks"]}


def explain(fault: str, confidence: float, facts: list[str], language: str = "English",
            model: str = "gemma3:4b") -> dict:
    card = job_card(fault, confidence, facts)
    prompt = (
        "You are the assistant inside an air-conditioning unit in the UAE.\n"
        f"The fault detector found: {fault} (confidence {confidence:.0%}).\n"
        "Measured facts (use only these, invent nothing):\n- " + "\n- ".join(facts or ["readings are far from this unit's healthy level"]) + "\n"
        f"What the person can safely do: {KB[fault]['resident_can_do']}\n"
        f"Urgency: {KB[fault]['urgency']}\n\n"
        "Return JSON with:\n"
        f"1. resident_message: 2-3 short friendly sentences in {language} for a non-technical person. "
        "Say what is wrong in plain words, what they can safely do, and that a technician has the details. "
        "Never tell them to handle refrigerant or open the unit.\n"
        "2. technician_summary: one English sentence with the key evidence."
    )
    t = time.time()
    out, source = None, None
    try:
        import ollama
        for attempt in range(2):          # small models sometimes drift into another language
            p = prompt if attempt == 0 else prompt + (
                f"\nIMPORTANT: resident_message MUST be written in {language} script only. "
                "Do not use any other language.")
            r = ollama.chat(model=model, messages=[{"role": "user", "content": p}],
                            format=SCHEMA, options={"temperature": 0.2})
            cand = json.loads(r["message"]["content"])
            if right_script(cand.get("resident_message", ""), language):
                out, source = cand, model
                break
        if out is None:
            raise ValueError(f"{model} did not answer in {language}")
    except Exception as e:  # noqa: BLE001  (Ollama off, model missing, bad JSON, wrong language)
        out = {"resident_message": f"Your AC has a problem: {fault.lower()}. {KB[fault]['resident_can_do']} "
                                   "A technician has been sent the details.",
               "technician_summary": f"{fault}: " + "; ".join(facts[:3])}
        source = f"English template ({type(e).__name__}: {e})"
    out.update(job_card=card, llm=source, seconds=round(time.time() - t, 1))
    return out


if __name__ == "__main__":
    print(json.dumps(explain("Low refrigerant (gas leak)", 0.93,
                             ["suction (low-side) pressure is lower than this unit's healthy level (11.6x its normal variation)",
                              "compressor discharge temperature is higher than this unit's healthy level (11.9x its normal variation)"],
                             language="Arabic"), ensure_ascii=False, indent=2))
