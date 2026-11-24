$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force data | Out-Null
function Get-Archive($url, $file) {
    curl.exe -fLsS --retry 3 -o "$file.part" $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $url" }
    Move-Item -Force -LiteralPath "$file.part" -Destination $file
}
Get-Archive "https://api.kaggle.com/v1/datasets/download/surajkarakulath/labelled-corpus-political-bias-hugging-face" "data/corpus.zip"
foreach ($year in 2016..2019) {
    Get-Archive "https://github.com/Cerpint4xt/All_the_News_2_0_Component_One/releases/download/data_$year/news_data_${year}_10.csv.gz" "data/oct$year.csv.gz"
}
Get-Archive "https://www.fec.gov/resources/cms-content/documents/federalelections2016.pdf" "data/fec2016.pdf"
