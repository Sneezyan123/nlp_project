"""Build the interim PDF from the saved results. Run from any working directory."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
DEST = ROOT / "output/pdf/polarization.pdf"
DEST.parent.mkdir(parents=True, exist_ok=True)
m = json.loads((OUT / "metrics.json").read_text())
panel = pd.read_csv(OUT / "outlet_october.csv").set_index(["outlet","year"])
topics = pd.read_csv(OUT / "topic_shares.csv").set_index(["outlet","year"])
states = pd.read_csv(OUT / "state_votes.csv")
with (ROOT / "team.csv").open(encoding="utf-8-sig") as stream:
    team = list(csv.DictReader(stream))
font, bold = "Helvetica", "Helvetica-Bold"
for family, filename in [("Body","arial.ttf"),("Strong","arialbd.ttf")]:
    path = Path("C:/Windows/Fonts") / filename
    if path.exists():
        pdfmetrics.registerFont(TTFont(family,str(path)))
        if family == "Body": font = family
        else: bold = family
c = Canvas(str(DEST), pagesize=(960,540))
c.setTitle("Evolution of Political Polarization in the United States")
ink, muted, teal = "#182F39", "#455D65", "#006E70"
paper, rule = "#FAF9F4", "#C2D0CB"


def block(x, top, value, size=19, width=850, face=None, color=ink):
    p = Paragraph(escape(str(value)), ParagraphStyle("text",fontName=face or font,fontSize=size, leading=size*1.3,textColor=colors.HexColor(color)))
    _, h = p.wrap(width,540)
    assert top+h < 530, (value,top,h)
    p.drawOn(c,x,540-top-h)
    return h


def page(section, title):
    c.setFillColor(colors.HexColor(paper))
    c.rect(0,0,960,540,fill=1,stroke=0)
    block(45,25,section,13,face=bold,color=teal)
    block(45,55,title,32,face="Times-Bold")
    c.setStrokeColor(colors.HexColor(rule))
    c.line(45,430,915,430)
    block(879,505,str(c.getPageNumber()),12,width=35,color=muted)


def item(top, label, body):
    block(45,top,label,18,width=198,face=bold,color=teal)
    block(259,top,body,19,width=645)


def table(top, positions, widths, headers, rows, step=38, size=17):
    for x,w,h in zip(positions,widths,headers):
        block(x,top,h,14,w,bold,teal)
    for j,row in enumerate(rows):
        y=top+32+j*step
        for i,(x,w,val) in enumerate(zip(positions,widths,row)):
            block(x,y,val,size,w,bold if i==0 else font)
        c.setStrokeColor(colors.HexColor(rule))
        c.line(45,540-y-step+5,915,540-y-step+5)


def note(value, top=450):
    block(45,top,value,16,855,color=muted)


def finish():
    c.showPage()


page("NLP semester project   /   Interim defence","Evolution of Political Polarization in the United States")
block(45,161,"How does the language of US political news change across outlets and elections?",37,785,"Times-Roman")
block(46,310,"NLP Analysis of News Coverage and Media Narratives",22,850,bold,teal)
block(46,366,"Four October samples, 2016–2019   /   Presidential vote by state, 2016",18)
block(46,465,", ".join(r["name"] for r in team),15,color=muted)
finish()

page("Problem understanding","What can news language tell us about polarization?")
item(140,"Motivation","Two outlets can cover the same election while giving different issues and voices prominence. Comparing their language makes those editorial choices measurable.")
item(235,"Inputs and outputs","Inputs are English online articles, dates, outlets and state vote totals. Outputs are a three-class language score, outlet-by-October trends, headline topic shares and a state-level association.")
item(340,"Scope","Six US outlets have observations in all four Octobers. Fox News appears in one. TV broadcasts and individual media exposure fall outside this sample.")
note("Defined: article and state units. Open question: how much of each outlet difference survives matching on the same event?",462)
finish()

page("Data understanding","Two news corpora serve different purposes")
table(136,[45,264,431],[208,155,480],["Source","Records used","Role and preparation"],[
["Labelled Corpus","17,018 / 17,362","AllSides editors selected left, center and right examples. Remove short texts, repeated inputs and conflicting labels."],
["All the News 2.0",f"{m['n_news']:,} / {m['raw_news']:,}","Four October files. Select seven outlets and headlines with political keywords, then deduplicate within outlet."],
["Federal Elections 2016","51 jurisdictions","Extract Trump and Clinton popular votes by state and DC from the Federal Election Commission, printed page 6."]],step=87,size=17)
note("Training labels after cleaning: 7,665 Left, 3,954 Center and 5,399 Right. The labelled archive lacks structured dates and publisher IDs. No synthetic articles are used.",438)
finish()

page("Data understanding","Headline topics changed across the four Octobers")
rows=[]
for year in range(2016,2020):
    row=topics.loc["CNN",year]
    rows.append([year,int(panel.loc[("CNN",year),"n"]),*[f"{row[k]:.1%}" for k in ["election","impeachment","immigration"]]])
table(133,[45,200,370,565,745],[125,150,185,170,160],["CNN sample","Articles","Election terms","Impeachment","Immigration"],rows)
block(45,344,"In October 2016, roughly one in five selected CNN headlines used an election term. In October 2019, impeachment appeared in roughly one in six.",19,850)
note("Overlapping keyword groups count explicit wording. For example, “polls” counts toward election coverage and “asylum” toward immigration. Paraphrases can be missed.",435)
finish()

page("Data understanding   /   Two articles from 1 October 2019","How two outlets framed the impeachment inquiry")
item(145,"CNN","“The Trump impeachment effort will be different from all the others.” The headline invites a historical comparison. The article score was −0.25.")
item(248,"The Hill","“Red-state Democrats worry impeachment may spin out of control.” The headline foregrounds electoral risk inside the Democratic Party. Its article score was +0.01.")
block(45,370,"For The Hill, the model assigned 0.33 Left, 0.34 Center and 0.33 Right. All three readings were close. That uncertainty makes a claim of impartiality unwarranted.",19)
note("Illustrative pair on the same date and topic. Scores use article openings. Original links and exact headlines are in results/articles.csv.gz.",465)
finish()

page("Model training","Learning the three editorial labels")
item(134,"Representation","Use the first 350 words of each article. Strip URLs, selected outlet names and caption markers. TF-IDF weights unigrams by how distinctive they are across the training texts.")
item(231,"Training","Fit class-balanced logistic regression on 13,614 articles. Keep 3,404 for one stratified test (seed 42). The vocabulary is fitted on training articles alone.")
item(329,"Application","For each historical article, subtract P(Left) from P(Right). Average that score across one outlet and October. The scale runs from −1 to +1.")
note("The interpretable linear model is inexpensive to train. Topic, quotations and publisher style can still drive its predictions.",456)
finish()

page("Automatic evaluation","The model recovers editorial labels with macro-F1 of 0.673")
table(131,[45,435,670],[375,210,220],["Method","Accuracy","Macro-F1"],[
["Always predict Left",f"{m['base_accuracy']:.3f}",f"{m['base_f1']:.3f}"],
["TF-IDF + logistic regression",f"{m['accuracy']:.3f}",f"{m['macro_f1']:.3f}"]])
block(45,271,"Held-out articles by editorial label and model prediction",19,850,bold)
table(311,[45,365,530,710],[280,140,150,165],["Editorial label","Predict Left","Predict Center","Predict Right"],[[k,*v] for k,v in zip(["Left","Center","Right"],m["matrix"])],step=30,size=16)
note("Macro-F1 weights the three classes equally. Center is hardest (F1 0.614). Of 791 Center articles, 203 were classified Left and 89 Right.",455)
finish()

page("Automatic evaluation   /   Transfer to historical news","Outlet scores vary by year and by coverage")
years=list(range(2016,2020))
order=["CNN","CNBC","Politico","The Hill","The New York Times","Vox","Fox News"]
rows=[]
for outlet in order:
    vals=[]
    for year in years:
        if (outlet,year) in panel.index:
            v=panel.loc[outlet,year]
            vals.append(f"{v.score:+.2f}  ({int(v.n):,})")
        else: vals.append("—")
    rows.append([outlet,*vals])
table(125,[45,305,465,621,773],[248,150,150,145,140],["Mean score (n)",*["Oct "+str(y) for y in years]],rows,step=35,size=16)
note("CNN scored further left than The Hill in all four samples. Every observed mean is negative, even Fox News in its only available October. The scale needs external validation before assigning party labels.",438)
finish()

page("Voting and news coverage","The state association changes sign with the coverage cutoff")
selected=states[states["size"]>=10]
left,bottom,w,h=80,140,420,205
xmin,xmax=selected["mean"].min()-.025,selected["mean"].max()+.025
ymin,ymax=.20,.80
c.setStrokeColor(colors.HexColor(rule))
for tick in [.3,.4,.5,.6,.7]:
    y=bottom+(tick-ymin)*h/(ymax-ymin)
    c.line(left,y,left+w,y)
    block(45,540-y-8,f"{tick:.0%}",11,35,color=muted)
for tick in [-.30,-.25,-.20,-.15,-.10,-.05,0]:
    if xmin<=tick<=xmax:
        x=left+(tick-xmin)*w/(xmax-xmin)
        block(x-15,408,f"{tick:.2f}",11,50,color=muted)
for row in selected.itertuples():
    x=left+(row.mean-xmin)*w/(xmax-xmin)
    y=bottom+(row.r_share-ymin)*h/(ymax-ymin)
    c.setFillColor(colors.HexColor(teal))
    c.circle(x,y,4,stroke=0,fill=1)
block(53,128,"Trump share of two-party vote",15,450,bold)
block(117,429,"Mean article score for the state mentioned",14,405)
block(560,138,"Spearman rank correlation",20,345,bold)
for i,(key,r) in enumerate(m["states"].items()):
    block(560,190+43*i,f"≥{key} articles    {r['states']} states    ρ = {r['rho']:+.2f}",17,350)
block(560,331,"Each dot is one state with at least ten eligible October 2016 articles. States receive equal weight.",16,345)
note("Articles mentioning exactly one jurisdiction provide a proxy for coverage about a place. Audience exposure, local readership and voting causality remain unmeasured.",469)
finish()

page("Interim status","What remains before estimating partisan alignment?")
table(132,[45,222,443,681],[166,210,225,231],["Component","Completed","Current issue","Next test"],[
["Problem","Define outlet and state comparisons.","Party support needs a candidate-specific label.","Specify support, criticism, quotation and neutral reporting."],
["Data","Clean both corpora and join official votes.","Fox has one usable October. Topic coverage varies.","Add matched events and audit archive coverage."],
["Training","Fit and apply a three-class baseline.","Names, topics and style can identify sources.","Mask entities and compare with a topic-only model."],
["Evaluation","Measure held-out F1 and cutoff sensitivity.","Historical articles lack independent labels.","Blind double-code 150 historical items. Report agreement and F1."]],step=74,size=15)
note("A further election cycle with geographically attributable audience data would allow a stronger test of the voting relationship.",472)
finish()

page("Data and reproducibility","Sources and evidence")
for i,(label,url,detail) in enumerate([
("Labelled Corpus – Political Bias","https://www.kaggle.com/datasets/surajkarakulath/labelled-corpus-political-bias-hugging-face","Article text and editor-selected AllSides perspectives, distributed by Suraj Karakulath."),
("All the News 2.0","https://github.com/Cerpint4xt/All_the_News_2_0_Component_One","Component One corpus. Monthly archives used here cover October 2016, 2017, 2018 and 2019."),
("Federal Elections 2016","https://www.fec.gov/resources/cms-content/documents/federalelections2016.pdf","Federal Election Commission. State popular votes from the table on printed page 6.")]):
    y=132+i*85
    block(45,y,label,21,850,bold,teal)
    c.linkURL(url,(45,540-y-29,420,540-y),relative=0)
    block(45,y+32,detail,17)
block(45,420,"Reproduction: analysis.py writes article predictions, outlet aggregates, topic shares, state comparisons and test metrics. README.md documents the sources and analytical choices.",18)
finish()

page("Team contributions","Team members and contributions to date")
if team:
    table(145,[45,320],[250,580],["Team member","Completed contribution"],[[r["name"],r["contribution"]] for r in team],step=min(78,290//max(1,len(team))))
else:
    table(145,[45,320],[250,580],["Team member","Completed contribution"],[["",""]],step=58)
finish()
c.save()
