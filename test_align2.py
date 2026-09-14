from shorts_engine.services.seo_generator import align_words_with_corrected_text

orig = [
    (0.0, 1.0, "καλημέρα"),
    (1.0, 2.0, "φίλοι"),
    (2.0, 3.0, "μου"),
]
corrected = "καλησπέρα σε όλους"
aligned = align_words_with_corrected_text(orig, corrected, 0.0, 3.0)
for start, end, word in aligned:
    print(f"[{start:.2f} -> {end:.2f}] {word}")
