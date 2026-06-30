# Industrial Predictive Quality Algorithm Demo

This asset is a standalone HTML/CSS/JS animation for the EVOCON Solutions MVP sales presentation.

## Files

- `industrial_predictive_quality_algorithm_demo.html`: interactive looping demo.
- `industrial_predictive_quality_algorithm_demo.png`: static opening preview.
- `industrial_predictive_quality_algorithm_demo_mid.png`: generated preview frame for visual QA.
- `industrial_predictive_quality_algorithm_demo_1366.png`: laptop-size visual QA preview.

## What the Demo Shows

- A synthetic industrial part scan with a localized heatmap.
- A feature cloud where observations drift away from the normal cluster.
- A risk signal that rises from green to amber/red and returns after action.
- A process timeline where the deviation starts before the final issue.
- Operator-facing actions: review first, inspect where/when, confirm the fix.

All values and visuals are synthetic demonstration data. The demo does not claim production deployment, guaranteed accuracy, or customer integration.

## How to Use

Open the HTML directly:

```powershell
start assets\industrial_predictive_quality_algorithm_demo.html
```

The animation loops automatically. For a sales call, show it after the static overview diagram to explain how the product works over time.

The page auto-fits to laptop-height viewports and includes 100%, 90% and 80% zoom controls in the lower-right corner. Use 90% or 80% if a browser toolbar or projector mode reduces the visible height.

## How to Edit

- Main colors are defined in the `:root` CSS variables.
- The animation duration is controlled by `--cycle`.
- The risk score values are updated in the small script at the bottom of the HTML.
- The part and heatmap are SVG elements inside the left panel.
- The feature cloud points are SVG circles in the center panel.
