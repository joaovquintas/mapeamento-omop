NORMALIZER_PROMPT = """

<OBJECTIVE>
Normalize and translate medical device terms from Portuguese to English, following SNOMED CT writing conventions.

IMPORTANT: You will NOT receive additional context about the devices. The provided term may contain common Portuguese abbreviations that must be expanded into the correct full English name.

You must analyze the provided term and:
1. Identify and expand common Portuguese abbreviations into the correct full name
2. Validate whether the term represents a mappable medical device for SNOMED CT
3. Translate the expanded term into English following SNOMED CT conventions
4. Normalize the term according to medical device naming standards

INVALID terms are:
- Overly generic terms (e.g., "machine", "chair")
- Terms that do not represent specific devices
- Ambiguous terms without sufficient context
- Terms that cannot be mapped to SNOMED CT
</OBJECTIVE>

<INSTRUCTIONS>
1. EXPANSION OF COMMON PORTUGUESE ABBREVIATIONS:
    - The term may contain common abbreviations used in Brazilian Portuguese
    - You MUST expand these abbreviations to the full name before translating
    - Common abbreviations and their expansions:
        * FAV → "Fístula Arteriovenosa" / "Arteriovenous Fistula" 
        * SVD → "Sonda Vesical de Demora" / "Foley Catheter" or "Indwelling Catheter" 
        * SNE → "Sonda Nasoenteral" / "Nasoenteral tube" 
        * TQT → "Tubo de taqueostromia" / "tracheostomy tube" 
        * TOT → "Tubo Orotraqueal" / "endotracheal tube" 
        * DRENO → "Dreno" / "Drain" 
        * PICC → "Cateter Central de Inserção Periférica" / "Peripherally Inserted Central Catheter" 
        * AVP → "Acesso Venoso Periférico" / "Peripheral Venous Access" 
        * SNG → "Sonda Nasogástrica" / "Nasogastric tube"

2. DEVICE VALIDATION:
    - is_valid_term=true: the term represents a specific and mappable medical device
        * Valid examples: "cateter", "acesso periferico venoso", "colchao pneumatico"
    - is_valid_term=false: generic term or does not represent a specific device
        * Invalid examples: "maquina", "poltrona", "cadeira", "fraldas"

3. TRANSLATION AND NORMALIZATION:
    - FIRST: expand any abbreviation into the full Portuguese name
    - THEN: translate the expanded term from Portuguese to English
    - Follow SNOMED CT writing conventions for medical devices:
        * Use standard English medical terminology
        * Maintain the structure: [device type] [location/anatomy] when applicable
        * Examples:
            - "cateter VJD" → expand → "jugular venous catheter" → "Central Venous Catheter"
            - "dreno torax" → expand → "thoracic drain" → "Thoracic Drain"

4. SNOMED CT CONVENTIONS:
   - Use standard technical medical terms in English
   - Maintain logical order: device
   - Avoid non-standard abbreviations in the final result
   - Use formal medical terminology
   - The final result (normalized_en) must ALWAYS be in English and contain the fully expanded name

5. NORMALIZED_EN:
   - Normalized English version of the term (with abbreviations expanded)
   - Must follow SNOMED CT conventions
   - Must contain the full device name, never abbreviations
   - If is_valid_term=false, it must be null
</INSTRUCTIONS>

<OUTPUT_FORMAT>
Return ONLY a valid JSON with the following structure:

{
    "is_valid_term": boolean,
    "normalized_en": string | null
}

IMPORTANT:
- If is_valid_term=false, normalized_en must be null
- Return ONLY the JSON, without additional text
- Use null (not "null" as a string) for missing values
</OUTPUT_FORMAT>

<EXAMPLES>
Example 1: Valid device - full or abbreviated name
Input: "cateter VJD"
Output:
{
    "is_valid_term": true,
    "normalized_en": "Central Venous Catheter"
}

Example 2: Valid device - abbreviation
Input: "FAV"
Output:
{
    "is_valid_term": true,
    "normalized_en": "Arteriovenous Fistula"
}

Example 3: Invalid device - generic term
Input: "Machine"
Output:
{
    "is_valid_term": false,
    "normalized_en": null
}

</EXAMPLES>

<INPUT>
Portuguese term: {term_original}
</INPUT>

"""

RERANKER_PROMPT = """
<OBJECTIVE>
Select the best SNOMED CT candidate for a normalized medical device term by analyzing the clinical and semantic correspondence between the original Portuguese term, the normalized English term, and the retrieved SNOMED CT candidates.

You must:
1. Analyze whether each SNOMED CT candidate represents the same medical device as the original term
2. Use the normalized English term (normalized_en) as the main reference
3. Prioritize clinical/semantic equivalence over lexical similarity
4. Select the most appropriate SNOMED CT candidate
5. Provide a clear justification explaining the selection
</OBJECTIVE>

<INSTRUCTIONS>
1. SEMANTIC ANALYSIS:
   - Compare the original Portuguese term (term_original), the normalized English term (normalized_en), and each candidate concept_name
   - Evaluate correspondence for:
     * Device type, e.g., catheter, tube, drain, mattress, fistula, access device
     * Device function or clinical use, e.g., urinary drainage, venous access, enteral feeding, airway support
     * Anatomical location or access route when applicable, e.g., venous, urinary, nasogastric, tracheostomy, thoracic
     * Specificity level, e.g., generic catheter vs central venous catheter vs PICC
     * Whether the candidate is truly a device concept and not a procedure, finding, body structure, disorder, or generic concept

2. SCORE USAGE:
   - semantic_score and lexical_score are supporting signals only
   - Do NOT select a candidate only because it has the highest lexical_score
   - Do NOT select a candidate only because it has the highest semantic_score if the concept_name does not match the device meaning
   - Prefer the candidate with the best clinical meaning match, even if its score is slightly lower
   - Use lexical_score mainly as a tie-breaker between semantically equivalent candidates

3. BEST CANDIDATE SELECTION:
   - Prioritize exact or near-exact device equivalence
   - If multiple candidates are acceptable, choose the candidate that is:
     * semantically correct
     * not overly broad
     * not more specific than the input term unless the normalized_en supports that specificity
     * aligned with SNOMED CT device naming conventions
   - Avoid selecting:
     * procedures involving the device
     * clinical findings
     * anatomical structures
     * disorders
     * overly generic objects
     * unrelated devices with similar words

4. SPECIAL CASES:
   - If no candidate adequately represents the device, return selected_concept_id=null
   - If candidates are too generic or too specific compared with the input term, explain this in reasoning
   - If the original Portuguese term is ambiguous but normalized_en resolves the ambiguity, follow normalized_en
   - If both semantic and lexical scores are low and the concept_name does not clearly match, return null

5. REASONING:
   - Explain why the selected candidate is the best match
   - Mention the matching device type, function, anatomical route/location, or specificity when applicable
   - Mention if the scores support the selection, but do not make the scores the main reason
</INSTRUCTIONS>

<OUTPUT_FORMAT>
Return ONLY a valid JSON with the following structure:

{
    "selected_concept_id": integer | string | null,
    "selected_concept_code": string | null,
    "selected_concept_name": string | null,
    "reasoning": string
}

IMPORTANT:
- If no candidate is appropriate, return null for selected_concept_id, selected_concept_code, and selected_concept_name
- The reasoning field is mandatory
- Return ONLY the JSON, without markdown, comments, or additional text
- Use null, not "null" as a string
</OUTPUT_FORMAT>

<EXAMPLES>
Example 1: Exact medical device match

Input:
term_original: "SVD"
normalized_en: "Foley Catheter"
candidates:
[
  {
    "concept_id": 111,
    "concept_code": "111",
    "concept_name": "Foley catheter",
    "semantic_score": 0.91,
    "lexical_score": 0.88
  },
  {
    "concept_id": 112,
    "concept_code": "112",
    "concept_name": "Urinary catheterization",
    "semantic_score": 0.89,
    "lexical_score": 0.82
  },
  {
    "concept_id": 113,
    "concept_code": "113",
    "concept_name": "Urinary catheter",
    "semantic_score": 0.86,
    "lexical_score": 0.80
  }
]

Output:
{
    "selected_concept_id": 111,
    "selected_concept_code": "111",
    "selected_concept_name": "Foley catheter",
    "reasoning": "The candidate 'Foley catheter' is the best match because the original term 'SVD' was normalized to 'Foley Catheter'. It represents the same urinary drainage device. 'Urinary catheterization' is a procedure rather than a device, and 'Urinary catheter' is less specific."
}

Example 2: Avoid procedure when device is requested

Input:
term_original: "PICC"
normalized_en: "Peripherally Inserted Central Catheter"
candidates:
[
  {
    "concept_id": 221,
    "concept_code": "221",
    "concept_name": "Insertion of peripherally inserted central catheter",
    "semantic_score": 0.94,
    "lexical_score": 0.92
  },
  {
    "concept_id": 222,
    "concept_code": "222",
    "concept_name": "Peripherally inserted central venous catheter",
    "semantic_score": 0.90,
    "lexical_score": 0.89
  }
]

Output:
{
    "selected_concept_id": 222,
    "selected_concept_code": "222",
    "selected_concept_name": "Peripherally inserted central venous catheter",
    "reasoning": "The selected candidate represents the device itself, matching the normalized term 'Peripherally Inserted Central Catheter'. The first candidate has slightly higher scores but describes the insertion procedure, not the device."
}

Example 3: No adequate candidate

Input:
term_original: "poltrona"
normalized_en: null
candidates:
[
  {
    "concept_id": 331,
    "concept_code": "331",
    "concept_name": "Chair",
    "semantic_score": 0.70,
    "lexical_score": 0.75
  }
]

Output:
{
    "selected_concept_id": null,
    "selected_concept_code": null,
    "selected_concept_name": null,
    "reasoning": "No candidate was selected because the original term does not represent a specific mappable medical device. The available candidate is a generic object rather than a clinically specific SNOMED CT device concept."
}
</EXAMPLES>

<INPUT>
Original Portuguese term: {term_original}
Normalized English term: {normalized_en}
SNOMED CT Candidates:
{candidates_json}
</INPUT>
"""