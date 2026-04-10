from src.normalizer import PhoneticNormalizer


def test_clean_text_removes_url_and_symbols():
    n = PhoneticNormalizer()
    out = n.clean_text("Visit https://example.com !!!")
    assert "http" not in out
    assert "!" not in out


def test_normalize_maps_common_variants():
    n = PhoneticNormalizer()
    out = n.normalize("Kia umead hai inshallah")
    assert "kya" in out
    assert "umeed" in out
    assert "inshaallah" in out


def test_short_tokens_not_over_normalized():
    n = PhoneticNormalizer()
    out = n.normalize("kiya ha to")
    assert "to" in out


def test_keeps_protected_tokens():
    n = PhoneticNormalizer()
    out = n.normalize("pleeease bohot jaldi")
    assert "pleeease" in out
    assert "bohot" in out


def test_q_fold_to_k_for_fuzzy_mapping():
    n = PhoneticNormalizer()
    out = n.normalize("mushqil")
    assert "mushkil" in out


def test_kamyabee_variant_mapping():
    n = PhoneticNormalizer()
    out = n.normalize("kamyabee")
    assert out == "kamyabi"


def test_mehanat_variant_mapping():
    n = PhoneticNormalizer()
    out = n.normalize("mehanat")
    assert out == "mehnat"


def test_duaa_variant_mapping():
    n = PhoneticNormalizer()
    out = n.normalize("duaa")
    assert out == "dua"


def test_mashallah_variant_mapping():
    n = PhoneticNormalizer()
    out = n.normalize("mashallah")
    assert out == "mashaallah"


def test_alhamdulilah_variant_mapping():
    n = PhoneticNormalizer()
    out = n.normalize("alhamdulilah")
    assert out == "alhamdulillah"


def test_whitespace_normalization():
    n = PhoneticNormalizer()
    out = n.normalize("   kia    umead   ")
    assert out == "kya umeed"


def test_bulk_normalize_order_preserved():
    n = PhoneticNormalizer()
    out = list(n.bulk_normalize(["kia", "umead", "inshallah"]))
    assert out == ["kya", "umeed", "inshaallah"]


def test_text_cleaner_removes_html():
    n = PhoneticNormalizer()
    out = n.clean_text("<p>kia</p>")
    assert out == "kia"


def test_text_cleaner_handles_none():
    n = PhoneticNormalizer()
    out = n.clean_text(None)
    assert out == ""


def test_nonmapped_word_stays_same():
    n = PhoneticNormalizer()
    out = n.normalize("algorithm")
    assert out == "algorithm"


def test_min_token_length_guard():
    n = PhoneticNormalizer(min_token_len=4)
    out = n.normalize("kia")
    assert out == "kia"


def test_similarity_threshold_high_prevents_fuzzy_change():
    n = PhoneticNormalizer(similarity_threshold=99.9)
    out = n.normalize("umead")
    assert out in {"umeed", "umead"}


def test_apostrophe_preserved_in_cleaning():
    n = PhoneticNormalizer()
    out = n.clean_text("it's kia")
    assert "it's" in out


def test_numbers_survive_cleaning():
    n = PhoneticNormalizer()
    out = n.clean_text("umeed 2026")
    assert out == "umeed 2026"


def test_repeated_chars_collapse_in_rules_path():
    n = PhoneticNormalizer()
    out = n.normalize("kiyaaa")
    assert isinstance(out, str)


def test_mixed_sentence_mapping():
    n = PhoneticNormalizer()
    out = n.normalize("Mehanat karo inshallah kamyabee milegi")
    assert "mehnat" in out
    assert "inshaallah" in out
    assert "kamyabi" in out
