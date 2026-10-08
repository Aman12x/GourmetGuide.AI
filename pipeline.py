"""One chat turn: (describe image) -> plan -> filtered retrieval -> rank and summarize.

Model calls per turn: 1 for small talk, 2 for a recommendation, +1 when a photo is attached.
"""

from dataclasses import dataclass, field

from bedrock import detect_image_format, history_messages, user_message
from retrieval import Filters, search

MAX_HISTORY_TURNS = 5
MAX_PICKS = 3


class UnsupportedImage(ValueError):
    pass


@dataclass
class Turn:
    text: str
    recommendations: list = field(default_factory=list)  # [(metadata, summary)]
    filters: Filters | None = None


DESCRIBE_SYSTEM = (
    "You describe food photos for a menu search engine. Name the dish (or describe it if unsure), "
    "its likely cuisine and key visible ingredients in one short paragraph of search-friendly terms. "
    "Ignore plates, cutlery and decoration."
)


def describe_image(llm, image_bytes):
    fmt = detect_image_format(image_bytes)
    if fmt is None:
        raise UnsupportedImage("Please upload a PNG, JPEG, GIF or WebP image.")
    message = user_message("Describe this dish for search.", image_bytes, fmt)
    return llm.text(DESCRIBE_SYSTEM, [message])


PLAN_SYSTEM = """You are GourmetGuide, a friendly food concierge for a fixed menu of dishes from local restaurants.
Decide what the user wants from the latest message, using the conversation for context
(for example "something cheaper" refines the previous request).

- intent "recommend": the user wants dish suggestions. Write search_query as the dish, flavour and
  ingredient terms to search for, and set only the filters the user actually stated or implied.
- intent "chat": greetings, questions about you, or anything that is not a dish request. Put a short,
  friendly reply in "reply" that steers back to food preferences. Never invent dishes.

Filter rules: max_price in dollars; vegetarian true for vegetarian or vegan requests, false only when
the user asks for meat; cuisines only from the allowed list (several are fine, e.g. Italian and Pizza
for pizza); exclude_allergens for any allergy or intolerance the user mentions (e.g. "nut allergy" -> nuts)."""


def plan_schema(cuisines):
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["recommend", "chat"]},
            "reply": {"type": "string", "description": "Reply to send when intent is chat."},
            "search_query": {"type": "string"},
            "max_price": {"type": ["number", "null"]},
            "vegetarian": {"type": ["boolean", "null"]},
            "cuisines": {"type": "array", "items": {"type": "string", "enum": cuisines}},
            "exclude_allergens": {"type": "array", "items": {"type": "string"}},
            "min_rating": {"type": ["number", "null"]},
        },
        "required": ["intent"],
    }


def plan_turn(llm, user_text, image_description, history, cuisines):
    text = user_text.strip()
    if image_description:
        text = f"{text}\n\nI uploaded a photo of a dish I like. It shows: {image_description}".strip()
    messages = history_messages(history[-MAX_HISTORY_TURNS:]) + [user_message(text)]
    plan = llm.structured(
        PLAN_SYSTEM,
        messages,
        "plan_turn",
        "Classify the user's message and extract the menu search query and hard filters.",
        plan_schema(cuisines),
    )
    if not plan.get("search_query"):
        plan["search_query"] = text
    return plan


RANK_SYSTEM = """You recommend dishes from a menu. You get the user's request and candidate dishes that
already satisfy their hard constraints. Pick up to 3 that best fit, best first, using only the given ids.
For each, write a two-sentence summary saying why it suits the request, using only facts in its details.
Never claim a dish is free of an allergen unless the request's allergens were filtered out, and never add
dishes, prices or ingredients that are not in the details. Also write a one-sentence intro."""

RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "intro": {"type": "string"},
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "summary": {"type": "string"}},
                "required": ["id", "summary"],
            },
        },
    },
    "required": ["intro", "picks"],
}


def dish_context(doc):
    m = doc.metadata
    warnings = m.get("dietary_warnings", "").strip() or "none listed"
    return (
        f"id: {m['id']}\nname: {m['menu_item_name']} ({m['restaurant_name']}, {m['cuisine']})\n"
        f"price: ${m['price']} | rating: {m['average_rating']} | {m['vegetarian']} | warnings: {warnings}\n"
        f"details: {doc.page_content}"
    )


def rank_and_summarize(llm, docs, request):
    candidates = "\n\n".join(dish_context(d) for d in docs)
    message = user_message(f"Request: {request}\n\nCandidates:\n\n{candidates}")
    result = llm.structured(RANK_SYSTEM, [message], "recommend_dishes", "Return the chosen dishes.", RANK_SCHEMA)
    by_id = {d.metadata["id"]: d for d in docs}
    picks, seen = [], set()
    for pick in result.get("picks", []):
        doc = by_id.get(pick.get("id"))
        if doc is None or pick["id"] in seen:
            continue  # drop ids the model invented or repeated
        seen.add(pick["id"])
        picks.append((doc.metadata, pick.get("summary", "")))
        if len(picks) == MAX_PICKS:
            break
    return result.get("intro", ""), picks


def handle_turn(llm, db, cuisines, user_text, image_bytes=None, history=()):
    history = list(history)
    image_description = describe_image(llm, image_bytes) if image_bytes else ""
    plan = plan_turn(llm, user_text, image_description, history, cuisines)

    if plan.get("intent") != "recommend":
        return Turn(text=plan.get("reply") or "Tell me a cuisine, dish or budget and I'll find something.")

    filters = Filters.from_plan(plan)
    docs = search(db, plan["search_query"], filters)
    if not docs:
        constraint = filters.describe() or "that request"
        return Turn(text=f"Nothing on the menu matches {constraint}. Want me to relax one of those?", filters=filters)

    request = f"{user_text} {image_description}".strip()
    if filters.describe():
        request += f" (constraints: {filters.describe()})"
    intro, picks = rank_and_summarize(llm, docs, request)
    if not picks:
        return Turn(text="I couldn't find a good match on the menu. Could you tell me a bit more?", filters=filters)
    return Turn(text=intro or "Here are some dishes you might like:", recommendations=picks, filters=filters)
