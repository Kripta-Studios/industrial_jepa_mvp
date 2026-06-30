# Industrial Predictive Quality Sales Materials

This directory contains sales material for EVOCON Solutions' `Industrial Predictive Quality` MVP.

The strongest current sales route is a narrow 4-6 week pilot:

* visual defect heatmaps and inspection ranking;
* machine/process risk scoring from sensors, test logs, SCADA or process context;
* temporal deviation views showing when a process starts drifting;
* a go/no-go report comparing the MVP against simple and strong baselines.

## Use This For Audi

Primary files:

* `audi_industrial_predictive_quality_mvp_short_en.html`: short 8-slide meeting deck for Audi. Use this in the actual meeting.
* `audi_industrial_predictive_quality_mvp_short_en.pdf`: PDF export of the short meeting deck.
* `audi_industrial_predictive_quality_eli5_en.html`: non-technical English deck for Audi/manufacturing meetings.
* `audi_industrial_predictive_quality_eli5_en.pdf`: PDF export of the same deck.
* `audi_industrial_predictive_quality_briefing_en.tex`: plain-English briefing source.
* `audi_industrial_predictive_quality_briefing_en.pdf`: briefing compiled to PDF.

Recommended use:

* Show `audi_industrial_predictive_quality_mvp_short_en.pdf` in the first Audi meeting.
* Use `audi_industrial_predictive_quality_eli5_en.pdf` only as preparation or follow-up material.
* Use `audi_industrial_predictive_quality_briefing_en.pdf` as speaker notes, not as the first file to send.

Supporting animation:

* `../assets/industrial_predictive_quality_algorithm_demo.html`
* `../assets/industrial_predictive_quality_algorithm_demo.png`

For a first Audi meeting, send the PDF deck and keep the briefing as internal speaker notes.

## Spain Outreach Files

* `spain_industrial_outreach_targets.csv`: prioritized account list with channels, contacts/routes, what to sell and first action.
* `spain_industrial_outreach_playbook.md`: exact emails, LinkedIn messages, phone script, Audi route and cluster-specific talk tracks.

The outreach material does not claim that any company is a customer, partner or interested party. Public emails and forms should be verified once more before sending.

## Existing Galicia Decks

* `mvp_galicia.html`: Spanish scrolling dossier.
* `mvp_galicia_reunion.html`: Spanish meeting deck.
* `mvp_galicia_en.html`: English scrolling dossier.
* `mvp_galicia_reunion_en.html`: English meeting deck.

Existing generated PDFs:

* `mvp_galicia.pdf`
* `mvp_galicia_reunion.pdf`
* `mvp_galicia_en.pdf`
* `mvp_galicia_reunion_en.pdf`
* `Industrial Predictive Quality MVP.pdf`
* `MVP de Calidad Predictiva Industrial.pdf`

## What To Say

Use:

> EVOCON helps factories turn existing images, sensors and process logs into earlier warnings, defect heatmaps and a ranked list of what to inspect first.

Avoid:

* production-ready claims;
* guarantees of zero defects;
* claiming an autonomous factory system;
* leading with model names instead of business value.

## Regeneration Commands

Compile briefing:

```powershell
cd presentation
lualatex -interaction=nonstopmode -halt-on-error audi_industrial_predictive_quality_briefing_en.tex
cd ..
```

Export the HTML deck to PDF with Edge/Chrome headless:

```powershell
# from repo root
$html = (Resolve-Path presentation\audi_industrial_predictive_quality_eli5_en.html).Path
$pdf = (Resolve-Path presentation).Path + "\audi_industrial_predictive_quality_eli5_en.pdf"
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu --print-to-pdf="$pdf" "file:///$($html.Replace('\','/'))"
```

The broader Galicia PDFs can still be generated from:

```powershell
python scripts/58_build_sales_presentations.py --pdf
```
