"""Hybrid retrieval: hard metadata filters (price, diet, cuisine, allergens, rating) applied
over FAISS vector search, so constraints are enforced in code instead of hoped for in a prompt."""

from dataclasses import dataclass, field

# Each allergen maps to the substrings that flag it in a dish's warnings or ingredients.
# Matching is deliberately broad: excluding a safe dish is better than serving an unsafe one.
ALLERGEN_TERMS = {
    "dairy": ["dairy", "milk", "cheese", "cream", "butter", "paneer", "yogurt", "curd", "ghee", "mozzarella", "parmesan"],
    "gluten": ["gluten", "wheat", "flour", "bread", "bun", "pasta", "noodle", "naan", "tortilla", "seitan", "barley"],
    "nuts": ["nut", "peanut", "cashew", "almond", "walnut", "pistachio", "pecan", "hazelnut"],
    "peanuts": ["peanut"],
    "eggs": ["egg", "mayonnaise", "mayo"],
    "soy": ["soy", "tofu", "edamame", "miso"],
    "fish": ["fish", "salmon", "tuna", "anchov"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "shellfish"],
    "sesame": ["sesame", "tahini"],
    "mustard": ["mustard"],
}

_ALLERGEN_ALIASES = {"milk": "dairy", "lactose": "dairy", "egg": "eggs", "nut": "nuts", "tree nuts": "nuts", "peanut": "peanuts", "wheat": "gluten"}


@dataclass
class Filters:
    max_price: float | None = None
    vegetarian: bool | None = None
    cuisines: list[str] = field(default_factory=list)
    exclude_allergens: list[str] = field(default_factory=list)
    min_rating: float | None = None

    @classmethod
    def from_plan(cls, plan):
        return cls(
            max_price=plan.get("max_price"),
            vegetarian=plan.get("vegetarian"),
            cuisines=list(plan.get("cuisines") or []),
            exclude_allergens=list(plan.get("exclude_allergens") or []),
            min_rating=plan.get("min_rating"),
        )

    def describe(self):
        parts = []
        if self.max_price is not None:
            parts.append(f"up to ${self.max_price:g}")
        if self.vegetarian is True:
            parts.append("vegetarian")
        elif self.vegetarian is False:
            parts.append("non-vegetarian")
        if self.cuisines:
            parts.append(" or ".join(self.cuisines))
        if self.exclude_allergens:
            parts.append("without " + ", ".join(self.exclude_allergens))
        if self.min_rating is not None:
            parts.append(f"rated {self.min_rating:g}+")
        return ", ".join(parts)


def allergen_terms(allergen):
    key = allergen.strip().lower()
    key = _ALLERGEN_ALIASES.get(key, key)
    return ALLERGEN_TERMS.get(key, [key])


def contains_allergen(metadata, allergen):
    haystack = f"{metadata.get('dietary_warnings', '')} {metadata.get('ingredients', '')}".lower()
    return any(term in haystack for term in allergen_terms(allergen))


def matches(metadata, filters: Filters):
    if filters.max_price is not None and float(metadata["price"]) > filters.max_price:
        return False
    if filters.vegetarian is not None:
        is_veg = metadata["vegetarian"].strip().lower() == "vegetarian"
        if is_veg != filters.vegetarian:
            return False
    if filters.cuisines:
        cuisine = metadata["cuisine"].lower()
        if not any(c.lower() in cuisine for c in filters.cuisines):
            return False
    if any(contains_allergen(metadata, a) for a in filters.exclude_allergens):
        return False
    if filters.min_rating is not None and float(metadata["average_rating"]) < filters.min_rating:
        return False
    return True


def search(db, query, filters: Filters, k=5):
    """Vector search over only the dishes that satisfy every hard filter.

    fetch_k covers the whole index so a strict filter never starves the result list."""
    return db.similarity_search(
        query,
        k=k,
        filter=lambda metadata: matches(metadata, filters),
        fetch_k=db.index.ntotal,
    )
