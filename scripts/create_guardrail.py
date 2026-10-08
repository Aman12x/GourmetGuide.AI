"""Create the GourmetGuide Bedrock Guardrail and publish version 1.

    python scripts/create_guardrail.py

Then set BEDROCK_GUARDRAIL_ID and BEDROCK_GUARDRAIL_VERSION (env or Streamlit secrets).
Needs bedrock:CreateGuardrail and bedrock:CreateGuardrailVersion in us-east-1.
"""

import boto3

BLOCKED = "Sorry, I can only help with finding dishes on our menu."

GUARDRAIL = {
    "name": "gourmetguide-guardrail",
    "description": "Content safety, prompt-attack defence and no medical or allergy-safety guarantees.",
    "contentPolicyConfig": {
        "filtersConfig": [
            {"type": t, "inputStrength": "HIGH", "outputStrength": "HIGH"}
            for t in ("HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT")
        ]
        + [{"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"}]
    },
    "topicPolicyConfig": {
        "topicsConfig": [
            {
                "name": "MedicalAndAllergyGuarantees",
                "definition": "Medical, nutritional-therapy or clinical advice, or guarantees that a dish is safe "
                "for someone with a severe allergy, intolerance or medical condition.",
                "examples": [
                    "Is this dish safe for my child's anaphylactic peanut allergy?",
                    "What should I eat to manage my diabetes?",
                    "Can you guarantee there is no cross-contamination with gluten?",
                ],
                "type": "DENY",
            }
        ]
    },
    "sensitiveInformationPolicyConfig": {
        "piiEntitiesConfig": [
            {"type": t, "action": "BLOCK"}
            for t in ("CREDIT_DEBIT_CARD_NUMBER", "US_SOCIAL_SECURITY_NUMBER", "PASSWORD")
        ]
    },
    "blockedInputMessaging": BLOCKED,
    "blockedOutputsMessaging": BLOCKED,
}


def main(region="us-east-1"):
    client = boto3.client("bedrock", region_name=region)
    created = client.create_guardrail(**GUARDRAIL)
    version = client.create_guardrail_version(guardrailIdentifier=created["guardrailId"], description="v1")
    print(f"BEDROCK_GUARDRAIL_ID={created['guardrailId']}")
    print(f"BEDROCK_GUARDRAIL_VERSION={version['version']}")


if __name__ == "__main__":
    main()
