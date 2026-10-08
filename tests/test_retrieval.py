from retrieval import Filters, contains_allergen, matches, search


def test_price_vegetarian_and_allergen_filters_hold_for_every_result(db):
    filters = Filters(max_price=10, vegetarian=True, exclude_allergens=["dairy"])
    docs = search(db, "spicy curry", filters, k=5)
    assert docs, "expected at least one match"
    for doc in docs:
        m = doc.metadata
        assert m["price"] <= 10
        assert m["vegetarian"] == "Vegetarian"
        assert not contains_allergen(m, "dairy")


def test_strict_filter_still_fills_k_from_whole_index(db, dishes):
    # Every dish satisfying the filter is reachable, not just those in the default top 20.
    filters = Filters(cuisines=["Japanese"])
    expected = {d["id"] for d in dishes if d["cuisine"] == "Japanese"}
    docs = search(db, "burger fries cheese", filters, k=5)
    assert {d.metadata["id"] for d in docs} == expected


def test_impossible_filter_returns_nothing(db):
    assert search(db, "anything", Filters(max_price=1), k=5) == []


def test_cuisine_match_includes_regional_variants(dishes):
    south_indian = next(d for d in dishes if d["cuisine"] == "South Indian")
    assert matches(south_indian, Filters(cuisines=["Indian"]))


def test_allergens_checked_against_ingredients_not_just_warnings():
    dish = {"dietary_warnings": "", "ingredients": "rice, paneer, spices"}
    assert contains_allergen(dish, "dairy")
    assert contains_allergen({"dietary_warnings": "Contains peanuts", "ingredients": ""}, "nuts")


def test_allergen_aliases():
    dish = {"dietary_warnings": "Contains eggs", "ingredients": ""}
    assert contains_allergen(dish, "egg")
    assert contains_allergen({"dietary_warnings": "Contains dairy", "ingredients": ""}, "milk")


def test_filters_describe():
    f = Filters(max_price=12, vegetarian=True, cuisines=["Italian"], exclude_allergens=["nuts"])
    assert f.describe() == "up to $12, vegetarian, Italian, without nuts"
