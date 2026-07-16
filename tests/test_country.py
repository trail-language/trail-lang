from trail.country import to_iso3


def test_alpha2_to_iso3():
    assert to_iso3("US") == "USA"
    assert to_iso3("gb") == "GBR"  # case-insensitive
    assert to_iso3("CA") == "CAN"


def test_alpha3_passthrough_and_validation():
    assert to_iso3("USA") == "USA"
    assert to_iso3("deu") == "DEU"
    assert to_iso3("ZZZ") is None  # 3 letters but not a real code


def test_unknown_and_empty():
    assert to_iso3(None) is None
    assert to_iso3("") is None
    assert to_iso3("XX") is None
