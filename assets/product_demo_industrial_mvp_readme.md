# Product Demo Industrial MVP Asset

Files:

- `product_demo_industrial_mvp.html`
- `product_demo_industrial_mvp.svg`
- `product_demo_industrial_mvp.png`

## What The Image Shows

This is a commercial MVP interface mockup for EVOCON Solutions Industrial Predictive Quality.

The visual style follows `presentation/industrial_predictive_quality_mvp_light_en.html`: light Apple-like background, white translucent cards, soft borders, blue/green/amber accent colors and minimal B2B product UI.

Panels:

- **Visual Inspection**: synthetic industrial part, original view and anomaly heatmap overlay.
- **Inspection Priority**: ranked review queue with risk bars and operator actions.
- **Process Deviation**: synthetic sensor curves for vibration, spindle current and temperature, with the deviation onset marked at Cycle 140.
- **Local Pilot Workflow**: camera/sensors to feature extraction, anomaly/risk models and prioritized review.

## Synthetic Data

All values and signals are illustrative:

- `Anomaly Score: 0.94`
- inspection part IDs
- sensor curves
- temporal surprise
- persistence
- risk trend

The asset intentionally includes the note:

```text
Illustrative MVP interface · Synthetic demonstration data
```

No client logos, certifications, production deployment claims or guaranteed accuracy claims are included.

## Regenerate The PNG

The HTML is the source layout. It contains a responsive 16:9 stage:

```css
.stage {
  width: min(100vw, calc(100vh * 16 / 9));
  aspect-ratio: 16 / 9;
}
```

This means the full interface scales down when opened in a smaller browser window instead of being cut off.

The SVG file is extracted from the inline SVG in the HTML.

From the repository root:

```powershell
python - <<'PY'
from pathlib import Path
html = Path('assets/product_demo_industrial_mvp.html').read_text(encoding='utf-8')
start = html.index('<svg id="product-demo-svg"')
end = html.index('</svg>', start) + len('</svg>')
svg = html[start:end].replace(' id="product-demo-svg"', '', 1)
Path('assets/product_demo_industrial_mvp.svg').write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg + '\n', encoding='utf-8')
PY
```

Export PNG with Microsoft Edge or another Chromium browser:

```powershell
$profile = (Resolve-Path assets).Path + '\edge-profile'
New-Item -ItemType Directory -Force $profile | Out-Null
$html = (Resolve-Path assets/product_demo_industrial_mvp.html).Path.Replace('\','/')
$png = (Resolve-Path assets).Path + '\product_demo_industrial_mvp.png'
& 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe' `
  --headless=new `
  --disable-gpu `
  --hide-scrollbars `
  --disable-crash-reporter `
  --disable-features=Crashpad `
  --user-data-dir="$profile" `
  --force-device-scale-factor=1 `
  --window-size=1920,1080 `
  --screenshot="$png" `
  "file:///$html"
```

## Editing

To change copy, scores or colors, edit `product_demo_industrial_mvp.html`.

Useful anchors:

- Top bar: search for `EVOCON Solutions`.
- Visual score: search for `Anomaly Score`.
- Ranking rows: search for `Part A-1842`.
- Sensor chart labels: search for `Process Deviation`.
- System flow: search for `Local Pilot Workflow`.
- Color palette: edit SVG gradients and classes in the `<defs><style>` block.

After editing HTML, regenerate the SVG and PNG using the commands above.
