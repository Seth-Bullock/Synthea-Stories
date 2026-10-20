"""
Prompt templates. Edit these to change the tone, length or style of the output.

Placeholders in {curly braces} are filled in by generate.py. Literal braces
in the text (e.g. JSON examples) are doubled: {{ and }}.
"""

# ---------------------------------------------------------------- shared rules

SYSTEM_PROMPT = """\
You are helping to build an entirely fictional patient record for teaching healthcare \
students about AI. The patient does not exist: their record was produced by Synthea, a \
simulator of patient histories. Your job is to turn the simulator's coded facts into the \
kind of rich, realistic documents found in a real family-medicine (GP) record in the \
United States.

Rules:
1. FACTS: Every item in the episode fact sheet you are given must be reflected accurately \
(dates, diagnoses, medicines, doses, test results, questionnaire scores, vaccines). Never \
contradict the fact sheet, the patient profile, or the story so far.
2. DO NOT INVENT clinical facts that would belong in the coded record: no new diagnoses, \
prescriptions, vaccinations or numerical test results beyond those listed.
3. DO ADD the human detail a coded record lacks: symptoms and how they started, what the \
patient said and felt, worries, family and work circumstances, lifestyle, examination \
findings consistent with the facts, the clinician's reasoning, advice given, safety-netting \
and follow-up plans.
4. TRANSLATE CODES: Write as a clinician or patient would, never copying coded names \
verbatim. For example "1 ML medroxyprogesterone acetate 150 MG/ML Injection" becomes \
"Depo-Provera 150 mg IM", and "Encounter for symptom (procedure)" becomes an ordinary \
appointment. Items such as "Received higher education (finding)" or "Full-time employment \
(finding)" are social background, not diagnoses.
5. IMPERFECT DATA: The simulator is not always clinically coherent (for example a \
condition that is never marked resolved, or a lab value that does not fit a diagnosis). \
Handle this as a real clinician would: interpret results sensibly, comment on discrepancies \
if relevant, and leave out items that would not matter at this visit.
6. FORMAT: Plain text only. No markdown (no #, no **, no tables). Do not add a title line \
with the date or document type; that header is added separately.
7. After the document, write a line containing only "### SUMMARY" followed by one or two \
sentences summarising the document for continuity. The summary is not part of the document.\
"""

# ---------------------------------------------------------------- patient profile ("story bible")

PROFILE_PROMPT = """\
Below is a summary of a fictional patient's whole simulated record. Invent a coherent, \
realistic life story and background for this person that is CONSISTENT with every item in \
the record and helps explain it. In particular:
- Explain social findings (employment changes, transport problems, social isolation, stress, \
violence or abuse, education) with specific circumstances and dates that match the record.
- Make lifestyle consistent with questionnaire scores (e.g. AUDIT-C for alcohol, DAST-10 for \
drugs, PHQ-2 and GAD-7 for mood) and smoking status.
- Give a family history that fits the patient's conditions without inventing new diagnoses \
for the patient themselves.
- Names of family members and clinicians must be fictional and ordinary.
- If the patient was a child for part of the record, name the parent or guardian who would \
have written to the doctor on their behalf.

{record_summary}

Reply with a single JSON object with exactly these keys:
{{
  "preferred_name": "what the patient likes to be called",
  "education": "...",
  "occupation_history": [{{"from_year": 2006, "to_year": 2016, "role": "..."}}],
  "relationships_and_family": "partner(s), children, parents, siblings, with names and years",
  "living_situation": "who they live with and where, how it changes over time",
  "personality": "...",
  "interests": "...",
  "lifestyle": "diet, exercise, alcohol, smoking, drugs",
  "health_attitudes": "how they feel about doctors, medicines and their own health",
  "writing_style": "how they write messages to their doctor: tone, length, spelling, quirks",
  "family_history": "illnesses in close relatives",
  "parent_or_guardian": "name and relationship, or null if never a child in this record",
  "key_life_events": [{{"year": 2016, "event": "..."}}],
  "gp_history": [{{"name": "Dr ...", "from_year": 1999, "to_year": 2012}}]
}}
The gp_history must cover {first_year} to {last_year} with one to three family doctors at \
{practice}.\
"""

# ---------------------------------------------------------------- per-document prompts

DOCUMENT_PROMPT = """\
PATIENT
{patient_header}

PATIENT PROFILE (invented background; stay consistent with it)
{profile}

STORY SO FAR (summaries of earlier documents, oldest first)
{story_so_far}

EPISODE FACT SHEET (from the coded record; must be reflected accurately)
{fact_sheet}

TASK
{task}\
"""

TASKS = {
    "gp_note": """\
Write the consultation note made by {clinician}, the patient's family doctor at {practice}, \
for the visit on {date}. The patient is {age} years old.
Use a realistic US family-medicine style: Subjective (reason for visit, history in the \
patient's words, relevant social and family context), Objective (the recorded vital signs \
plus plausible examination findings), Assessment (with questionnaire scores and lab results \
interpreted), and Plan (treatment, advice, safety-netting, follow-up). Clinical \
abbreviations (pt, hx, c/o, NAD, f/u, BID) are welcome. Quote the patient directly once or \
twice. For a routine check-up, cover preventive care and the social history review. Refer to \
earlier visits where relevant. Length: {length} words.""",

    "clinic_letter": """\
Write the letter sent to {clinician} (the patient's family doctor at {practice}) by the \
clinician who saw the patient in the {setting} setting on {date}. The patient is {age} years \
old. Use the style of a US discharge summary or clinic letter: reason for attendance, \
history, findings, results, diagnosis, treatment given, medicines, and recommendations for \
the family doctor. Invent a plausible hospital or clinic name near {city}, {state}, and sign \
with a fictional clinician's name and role. Length: {length} words.""",

    "patient_before": """\
Write a message from {writer} to {clinician} at {practice}, written on {date}, a few days \
before the appointment on {visit_date}. It explains in everyday language why they want to be \
seen: symptoms that are bothering them, or for a routine check-up, anything they would like \
to discuss. The writer does NOT yet know the diagnosis or the results of any tests done at \
the visit, so describe symptoms and worries rather than medical terms. It may be a patient-portal message, \
an email or a short letter. Match the writing style in the profile (including any typos or \
quirks). Length: {length} words.""",

    "patient_after": """\
Write a message from {writer} to {clinician} at {practice}, written on {date}, after the \
appointment on {visit_date}. It might report how they are getting on, ask a question about \
the treatment or a side effect, raise a worry, or mention how life events are affecting \
their health; choose whatever fits the episode and the profile. Use everyday language. It \
may be a patient-portal message, an email or a short letter. Match the writing style in the \
profile. Length: {length} words.""",

    "patient_between": """\
Write a message from {writer} to {clinician} at {practice}, written on {date}, when there \
was no appointment. Base it on what the fact sheet says changed on this date (for example an \
illness clearing up, or a prescription coming to an end: they might ask for a refill, ask \
about alternatives, or simply give an update). Use everyday language. It may be a \
patient-portal message, an email or a short letter. Match the writing style in the profile. \
Length: {length} words.""",
}

# Target lengths in words. Visits with more going on get longer notes.
LENGTHS = {
    "gp_note": ("150-250", "250-400", "350-550"),
    "clinic_letter": ("250-400", "250-400", "350-550"),
    "patient_before": ("60-180",) * 3,
    "patient_after": ("60-180",) * 3,
    "patient_between": ("40-150",) * 3,
}

DOCUMENT_TITLES = {
    "gp_note": "GP consultation note",
    "clinic_letter": "Letter to GP",
    "patient_before": "Message from patient",
    "patient_after": "Message from patient",
    "patient_between": "Message from patient",
}
