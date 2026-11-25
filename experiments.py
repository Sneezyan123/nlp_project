"""Model comparison and error analysis for the interim presentation.

Run after download.ps1 (needs data/corpus.zip and data/oct*.csv.gz):
    python experiments.py
Writes everything to results/experiments/.
Entity masking needs spaCy + en_core_web_sm; masked texts are cached.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

import analysis

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = analysis.LABELS
SEED = 42
# Publisher boilerplate found among the strongest E1 features: photo credits, wire datelines,
# "You can now listen to Fox News articles!", outlet names, timestamps and any token with digits.
ARTEFACTS = re.compile(
    r"(?:new)?you can now listen to (?:fox news )?articles!?|story highlights|all rights reserved|"
    r"\S*(?:credit|published|updated|newyou|\d)\S*|"
    r"\b(?:getty|images?|photos?|picture|caption|landov|ap|afp|reuters|associated press|huffpost|newsmax|"
    r"examiner|daily caller|caller|wsj|wall street journal|bbc|cbs|nbc|abc|msnbc|fox|fox news|slatest|"
    r"politicalticker|screen_name|click|download|follow|unlimited|upfor|pic|com|listen)\b", re.I)
MARKERS = {"fox_listen_plug": r"listen to (?:fox news )?articles|newyou", "credit": r"credit", "getty": r"getty",
           "caption": r"caption", "year_2020_2022": r"\b202[0-2]\b"}


def clean(texts):
    return [" ".join(ARTEFACTS.sub(" ", t).split()) for t in texts]


MASK_TYPES = {"PERSON", "ORG", "NORP", "GPE", "LOC", "FAC", "EVENT"}


def mask_entities(texts, cache):
    if cache.exists():
        return pd.read_csv(cache, keep_default_na=False).masked.tolist()
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    masked = []
    for doc in nlp.pipe(texts, batch_size=64, n_process=2):
        parts, last = [], 0
        for ent in doc.ents:
            if ent.label_ in MASK_TYPES:
                parts += [doc.text[last:ent.start_char], f" ent{ent.label_.lower()} "]
                last = ent.end_char
        parts.append(doc.text[last:])
        masked.append("".join(parts))
    pd.DataFrame({"masked": masked}).to_csv(cache, index=False)
    return masked


def tfidf(ngrams=(1, 1)):
    return TfidfVectorizer(min_df=3, max_features=40000 if ngrams == (1, 1) else 120000,
                           sublinear_tf=True, stop_words="english", ngram_range=ngrams)


CONFIGS = {
    "E1 TF-IDF unigrams + LogReg (current)": dict(text="text", vec=lambda: tfidf(), clf=lambda: LogisticRegression(max_iter=400, class_weight="balanced", random_state=SEED)),
    "E2 TF-IDF 1-2grams + LogReg": dict(text="text", vec=lambda: tfidf((1, 2)), clf=lambda: LogisticRegression(max_iter=600, class_weight="balanced", random_state=SEED)),
    "E3 TF-IDF 1-2grams + LinearSVC": dict(text="text", vec=lambda: tfidf((1, 2)), clf=lambda: LinearSVC(class_weight="balanced", C=0.5, random_state=SEED)),
    "E4 TF-IDF 1-2grams + ComplementNB": dict(text="text", vec=lambda: tfidf((1, 2)), clf=lambda: ComplementNB(alpha=0.3)),
    "E5 Entities masked + unigrams + LogReg": dict(text="masked", vec=lambda: tfidf(), clf=lambda: LogisticRegression(max_iter=400, class_weight="balanced", random_state=SEED)),
    "E6 Artefacts removed + unigrams + LogReg": dict(text="clean", vec=lambda: tfidf(), clf=lambda: LogisticRegression(max_iter=400, class_weight="balanced", random_state=SEED)),
    "E7 Artefacts removed + 1-2grams + LinearSVC": dict(text="clean", vec=lambda: tfidf((1, 2)), clf=lambda: LinearSVC(class_weight="balanced", C=0.5, random_state=SEED)),
}


def fit(cfg, train):
    vec, clf = cfg["vec"](), cfg["clf"]()
    clf.fit(vec.fit_transform(train[cfg["text"]]), train.label)
    return vec, clf


def bootstrap_f1(y, p, n=1000):
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    vals = [f1_score(y[i], p[i], average="macro") for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5]).round(3).tolist()


def mcnemar(y, a, b):
    y, a, b = map(np.asarray, (y, a, b))
    only_a, only_b = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": p}


def top_features(vec, clf, k=25):
    names = np.array(vec.get_feature_names_out())
    rows = []
    for i, label in enumerate(clf.classes_):
        for rank, j in enumerate(np.argsort(clf.coef_[i])[::-1][:k], 1):
            rows.append({"class": label, "rank": rank, "feature": names[j], "weight": round(float(clf.coef_[i, j]), 3)})
    return pd.DataFrame(rows)


def premask_news():
    news, _ = analysis.news_data()
    mask_entities(news.reset_index(drop=True).text.tolist(), OUT / "news_masked.csv.gz")


def main():
    corpus = analysis.training_data().reset_index(drop=True)
    corpus["masked"] = mask_entities(corpus.text.tolist(), OUT / "corpus_masked.csv.gz")
    # File names such as "Left Data/Left Data/1000_0 (2).txt": 1000 is the AllSides story id,
    # shared by the left, center and right articles about the same event.
    corpus["clean"] = clean(corpus.text)
    marker_share = {k: corpus.groupby("label").text.apply(lambda t: t.str.contains(p, case=False, regex=True).mean()).round(3).to_dict()
                    for k, p in MARKERS.items()}
    corpus["story"] = corpus.file.str.extract(r"/(\d+)_\d+", expand=False)
    train, test = train_test_split(corpus, test_size=.2, random_state=SEED, stratify=corpus.label)

    # 1. Model comparison on the same held-out split + 5-fold CV
    rows, preds, models = [], {}, {}
    majority = [train.label.mode()[0]] * len(test)
    rows.append({"model": "E0 Majority class (Left)", "accuracy": accuracy_score(test.label, majority),
                 "macro_f1": f1_score(test.label, majority, average="macro")})
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    group_folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(corpus, corpus.label, corpus.story))
    test_stories = set(test.story)
    story_overlap = {"test_articles_with_story_in_train": float(test.story.isin(set(train.story)).mean()),
                     "test_articles_with_same_label_and_story_in_train": float(test.set_index(["story", "label"]).index.isin(train.set_index(["story", "label"]).index).mean())}
    for name, cfg in CONFIGS.items():
        vec, clf = fit(cfg, train)
        p = clf.predict(vec.transform(test[cfg["text"]]))
        preds[name], models[name] = p, (vec, clf)
        cv = []
        for tr, te in folds.split(corpus, corpus.label):
            v, c = fit(cfg, corpus.iloc[tr])
            cv.append(f1_score(corpus.label.iloc[te], c.predict(v.transform(corpus[cfg["text"]].iloc[te])), average="macro"))
        gcv = []
        for tr, te in group_folds:
            v, c = fit(cfg, corpus.iloc[tr])
            gcv.append(f1_score(corpus.label.iloc[te], c.predict(v.transform(corpus[cfg["text"]].iloc[te])), average="macro"))
        per_class = f1_score(test.label, p, average=None, labels=LABELS)
        rows.append({"model": name, "accuracy": accuracy_score(test.label, p), "macro_f1": f1_score(test.label, p, average="macro"),
                     "macro_f1_95ci": bootstrap_f1(test.label, p), "cv5_macro_f1_mean": np.mean(cv), "cv5_macro_f1_std": np.std(cv),
                     "story_cv5_macro_f1_mean": np.mean(gcv), "story_cv5_macro_f1_std": np.std(gcv),
                     **{f"f1_{l}": v for l, v in zip(LABELS, per_class)},
                     "confusion": confusion_matrix(test.label, p, labels=LABELS).tolist()})
        print(name, round(rows[-1]["macro_f1"], 3), round(np.mean(cv), 3), round(np.mean(gcv), 3), flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)

    base = "E1 TF-IDF unigrams + LogReg (current)"
    best = max(CONFIGS, key=lambda n: comparison.set_index("model").macro_f1[n])
    tests = {f"{base} vs {n}": mcnemar(test.label, preds[base], preds[n]) for n in CONFIGS if n != base}

    # 2. Error analysis of the current model
    vec, clf = models[base]
    X_test = vec.transform(test.text)
    proba = clf.predict_proba(X_test)
    ea = test[["file", "label"]].copy()
    ea["predicted"] = preds[base]
    ea["confidence"] = proba.max(1)
    ea["words"] = test.text.str.split().str.len().values
    for i, l in enumerate(clf.classes_):
        ea[f"p_{l}"] = proba[:, i]
    ea["correct"] = ea.label == ea.predicted
    recall_by_marker = {}
    for label, marker in [("Right", "fox_listen_plug"), ("Left", "credit"), ("Center", "getty")]:
        part = ea[ea.label == label]
        has = test.loc[part.index, "text"].str.contains(MARKERS[marker], case=False, regex=True)
        recall_by_marker[f"{label} recall, {marker}"] = {"with": [int(has.sum()), round(float(part[has].correct.mean()), 3)],
                                                         "without": [int((~has).sum()), round(float(part[~has].correct.mean()), 3)]}

    by_conf = ea.groupby(pd.cut(ea.confidence, [0, .4, .5, .6, .7, .8, 1.0]), observed=True).correct.agg(["size", "mean"])
    by_len = ea.groupby(pd.cut(ea.words, [39, 100, 200, 349, 350]), observed=True).correct.agg(["size", "mean"])

    # near-duplicates between train and test (same story, different wording)
    sims = (X_test @ vec.transform(train.text).T).max(axis=1).toarray().ravel()
    ea["max_sim_train"] = sims
    nearest_label = train.label.values[np.asarray((X_test @ vec.transform(train.text).T).argmax(axis=1)).ravel()]
    ea["nearest_train_label"] = nearest_label
    by_sim = ea.groupby(pd.cut(ea.max_sim_train, [0, .3, .5, .7, .9, 1.01]), observed=True).correct.agg(["size", "mean"])
    nn_acc = float((ea.nearest_train_label == ea.label).mean())

    feats = top_features(vec, clf)
    feats.to_csv(OUT / "top_features_E1.csv", index=False)
    top_features(*models["E5 Entities masked + unigrams + LogReg"]).to_csv(OUT / "top_features_E5_masked.csv", index=False)
    top_features(*models["E6 Artefacts removed + unigrams + LogReg"]).to_csv(OUT / "top_features_E6_clean.csv", index=False)

    errors = ea[~ea.correct].sort_values("confidence", ascending=False)
    errors = errors.assign(snippet=test.loc[errors.index, "text"].str.split().str[:30].str.join(" "))
    errors.head(40).to_csv(OUT / "confident_errors.csv", index=False)
    ea.to_csv(OUT / "test_error_analysis.csv", index=False)

    # 3. Transfer: does the outlet ranking survive another model?  Prior shift?
    news, _ = analysis.news_data()
    news = news.reset_index(drop=True)
    news["masked"] = mask_entities(news.text.tolist(), OUT / "news_masked.csv.gz")
    news = news[~news.id.isin(corpus.id)].reset_index(drop=True)
    news["clean"] = clean(news.text)
    marker_share["news_all_the_news"] = {k: round(float(news.text.str.contains(p, case=False, regex=True).mean()), 3) for k, p in MARKERS.items()}
    panel = {}
    news_means = {}
    for name in [base, "E2 TF-IDF 1-2grams + LogReg", "E5 Entities masked + unigrams + LogReg", "E6 Artefacts removed + unigrams + LogReg"]:
        v, c = models[name]
        pr = c.predict_proba(v.transform(news[CONFIGS[name]["text"]]))
        cls = list(c.classes_)
        news["score"] = pr[:, cls.index("Right")] - pr[:, cls.index("Left")]
        panel[name] = news.groupby(["outlet", "year"]).score.mean()
        news_means[name] = dict(zip(cls, pr.mean(0).round(3)))
    panel = pd.DataFrame(panel)
    panel.round(3).to_csv(OUT / "outlet_scores_by_model.csv")
    stability = {n: spearmanr(panel[base], panel[n]).statistic for n in panel if n != base}
    shares = dict(zip(clf.classes_, proba.mean(0).round(3)))

    summary = {
        "n_train": len(train), "n_test": len(test), "best_model": best, "story_overlap": story_overlap,
        "n_stories": int(corpus.story.nunique()),
        "mcnemar": tests,
        "recall_by_marker": recall_by_marker,
        "accuracy_by_confidence": {str(k): v for k, v in by_conf.round(3).to_dict("index").items()},
        "accuracy_by_length": {str(k): v for k, v in by_len.round(3).to_dict("index").items()},
        "accuracy_by_max_similarity_to_train": {str(k): v for k, v in by_sim.round(3).to_dict("index").items()},
        "nearest_neighbour_label_accuracy": nn_acc,
        "mean_probabilities_test": shares, "true_shares_test": test.label.value_counts(normalize=True).round(3).to_dict(),
        "mean_probabilities_news": news_means, "artefact_marker_share": marker_share,
        "outlet_rank_stability_spearman": stability,
        "errors_by_pair": ea[~ea.correct].groupby(["label", "predicted"]).size().to_dict(),
    }
    summary["errors_by_pair"] = {f"{a}->{b}": int(n) for (a, b), n in summary["errors_by_pair"].items()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(comparison.drop(columns="confusion").round(3).to_string())
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    import sys
    premask_news() if "--premask-news" in sys.argv else main()
