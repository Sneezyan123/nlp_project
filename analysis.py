from pathlib import Path
import csv
import gzip
import hashlib
import json
import re
import zipfile

import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "results"
OUT.mkdir(exist_ok=True)
LABELS = ["Left", "Center", "Right"]
OUTLETS = ["CNN", "CNBC", "Politico", "The Hill", "The New York Times", "Vox", "Fox News"]
POLITICS = re.compile(r"\b(trump|clinton|democrats?|democratic party|republicans?|gop|white house|congress|senate|presidential)\b", re.I)
BRANDING = re.compile(r"https?://\S+|\b(?:cnn|cnbc|fox news|politico|the hill|new york times|vox|reuters|associated press|npr|breitbart|hide caption|toggle caption)\b", re.I)


def words(text):
    return " ".join(text.lower().split())


def features(text):
    return " ".join(BRANDING.sub(" ", text).split()[:350])


def fingerprint(text):
    return hashlib.sha256(words(text).encode()).hexdigest()


def training_data():
    rows = []
    with zipfile.ZipFile(DATA / "corpus.zip") as archive:
        for file in archive.namelist():
            label = file.split(" ")[0]
            if label in LABELS and file.endswith(".txt"):
                text = archive.read(file).decode("utf-8-sig", "replace")
                body = features(text)
                if len(body.split()) >= 40:
                    rows.append((fingerprint(body), file, label, body))
    frame = pd.DataFrame(rows, columns=["id", "file", "label", "text"])
    consistent = frame.groupby("id").label.transform("nunique") == 1
    return frame[consistent].drop_duplicates("id")


def news_data():
    csv.field_size_limit(20_000_000)
    rows = []
    raw = 0
    for year in range(2016, 2020):
        with gzip.open(DATA / f"oct{year}.csv.gz", "rt", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                raw += 1
                if row["publication"] not in OUTLETS or not row["date"].startswith(f"{year}-10"):
                    continue
                if not POLITICS.search(row["title"]):
                    continue
                body = features(row["article"])
                if len(body.split()) >= 40:
                    rows.append((year, row["publication"], row["date"][:10], row["title"], row["url"], body, fingerprint(body)))
    frame = pd.DataFrame(rows, columns=["year","outlet","date","title","url","text","id"])
    return frame.drop_duplicates(["outlet","id"]).copy(), raw


def state_analysis(news):
    votes = pd.read_csv(DATA / "votes2016.csv")
    # Prefer the full compound name in e.g. West Virginia.
    pattern = re.compile(r"\b(?:" + "|".join(sorted(votes.state, key=len, reverse=True)) + r")\b", re.I)
    states = []
    for row in news[news.year.eq(2016)].itertuples():
        names = {name.title() for name in pattern.findall(row.title + " " + row.text)}
        if len(names) == 1:
            state = names.pop()
            if state not in {"Washington","New York","Georgia","District Of Columbia"}:
                states.append((state, row.score))
    frame = pd.DataFrame(states, columns=["state","score"]).groupby("state").score.agg(["size","mean"])
    frame = votes.merge(frame, on="state")
    frame["r_share"] = frame.trump / (frame.trump + frame.clinton)
    frame.to_csv(OUT / "state_votes.csv", index=False)
    results = {}
    for cutoff in [5, 10, 20]:
        selected = frame[frame["size"] >= cutoff]
        rho = spearmanr(selected["mean"], selected.r_share).statistic if len(selected) > 2 else None
        results[cutoff] = {"states":len(selected), "articles":int(selected["size"].sum()), "rho":rho}
    return results


def narrative_counts(news):
    topics = {
        "election": r"\b(?:elect(?:ion|ions|oral)|campaigns?|polls?|polling|ballots?|voters?|voting)\b",
        "impeachment": r"\bimpeach(?:ment|ed|ing)?\b",
        "immigration": r"\b(?:immigra\w*|migrants?|border|asylum)\b",
    }
    flags = {topic: news.title.str.contains(pattern, case=False, regex=True)
             for topic, pattern in topics.items()}
    news.assign(**flags).groupby(["outlet","year"])[list(topics)].mean().to_csv(OUT / "topic_shares.csv")


def main():
    corpus = training_data()
    train, test = train_test_split(corpus, test_size=.2, random_state=42, stratify=corpus.label)
    tfidf = TfidfVectorizer(min_df=3, max_features=40000, sublinear_tf=True, stop_words="english")
    model = LogisticRegression(max_iter=400, class_weight="balanced", random_state=42)
    model.fit(tfidf.fit_transform(train.text), train.label)
    predictions = model.predict(tfidf.transform(test.text))
    baseline = [train.label.mode()[0]] * len(test)
    test = test.assign(predicted=predictions)
    test[["file","label","predicted"]].to_csv(OUT / "test_predictions.csv", index=False)
    news, raw = news_data()
    news = news[~news.id.isin(corpus.id)].copy()
    probabilities = model.predict_proba(tfidf.transform(news.text))
    for i, label in enumerate(model.classes_):
        news[label] = probabilities[:,i]
    news["score"] = news.Right - news.Left
    news.drop(columns="text").to_csv(OUT / "articles.csv.gz", index=False)
    panel = news.groupby(["outlet","year"]).agg(n=("score","size"),score=("score","mean"),left=("Left","mean"),center=("Center","mean"),right=("Right","mean"))
    narrative_counts(news)
    panel.to_csv(OUT / "outlet_october.csv")
    result = dict(raw_news=raw, n_news=len(news), counts=corpus.label.value_counts().to_dict(), train=len(train), test=len(test),
                  accuracy=accuracy_score(test.label,predictions), macro_f1=f1_score(test.label,predictions,average="macro"),
                  base_accuracy=accuracy_score(test.label,baseline), base_f1=f1_score(test.label,baseline,average="macro"),
                  per_class=classification_report(test.label,predictions,output_dict=True), matrix=confusion_matrix(test.label,predictions,labels=LABELS).tolist(),
                  states=state_analysis(news))
    (OUT / "metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(panel[["n","score"]].unstack().round(2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
