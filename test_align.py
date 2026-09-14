from shorts_engine.services.seo_generator import align_words_with_corrected_text

orig = [
    (0.0, 0.5, "χαίρο"),
    (0.5, 1.0, "με"),
    (1.0, 1.5, "που"),
    (1.5, 2.0, "σας"),
    (2.0, 2.5, "βλέπω"),
]
corrected = "χαίρομαι που σας βλέπω"
aligned = align_words_with_corrected_text(orig, corrected, 0.0, 2.5)
for start, end, word in aligned:
    print(f"[{start:.2f} -> {end:.2f}] {word}")
