from app.domain.value_objects.section_path import SectionPath


def test_section_path_child_and_parent():
    root = SectionPath.empty()
    a = root.child("Bayesian Methods")
    b = a.child("Results")
    assert b.as_list() == ["Bayesian Methods", "Results"]
    assert b.parent().as_list() == ["Bayesian Methods"]
    assert b.as_string() == "Bayesian Methods > Results"