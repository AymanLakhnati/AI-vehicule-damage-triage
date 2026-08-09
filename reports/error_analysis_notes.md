# Error Analysis Notes

## Overview
This note summarizes the qualitative failure patterns observed in the saved false-positive and false-negative images for each CarDD class. The focus is on the most consistent visual cues, ambiguous cases, and model confusion sources.

## `crack`
- False positives: often triggered by narrow, linear defects that resemble cracks but are actually rust creases, paint chips, or reflections along seams.
- False negatives: missed when the crack is subtle, small, or occurs on a highly reflective / glossy surface. The model appears to struggle with low-contrast crack lines embedded in damaged but cluttered regions.
- Pattern: the class is sensitive to thin line-like structures and can be confused by seam edges and surface texture.

## `dent`
- False positives: predicted dent for large curved body damage or paint scuffs where there is significant deformation of the panel but no true dent annotation.
- False negatives: missed when the dent is shallow, partly occluded, or localized on a small section of the body panel. The model misses some subtle shape changes.
- Pattern: the model relies on broad shape irregularities and can over-predict on strong curvature or surface warping.

## `scratch`
- False positives: appears when there are paint scrapes, chipped paint, or rubbed-off coatings. These visually resemble scratches, especially when they follow the panel contour.
- False negatives: missed for faint or partially occluded scratches, especially on dark or low-contrast surfaces.
- Pattern: the model is good at detecting obvious surface damage, but cannot always separate a scratch from a scrape or paint-removal artifact.

## `lamp_broken`
- False positives: often on lamps that are dirty, foggy, or bordered by adjacent collision damage. Cracked lens patterns and deformation near headlights also confuse the model.
- False negatives: missed when the broken lamp is partly occluded, framed at an angle, or when the damage is limited to the lamp housing rather than a clear shattered lens.
- Pattern: model confusion is driven by lamp-related context and mirror damage around the headlight region.

## `glass_shatter`
- False positives: very few, but when present they appear on glass-like surfaces with texture, glare, or reflections that mimic broken glass patterns.
- False negatives: the model misses partly shattered windshields or windows where the damage is only visible in a small region.
- Pattern: the class is sensitive to bright, fractured-looking textures but struggles with partial or subtle shatter evidence.

## `tire_flat`
- False positives: predicted flat on tires with strong perspective distortion or when the tire sidewall is compressed by camera angle. The model can mix up low pressure with normal wheel shape under foreshortening.
- False negatives: missed on flats that are only partially visible or when the tire is obscured by shadow/ground contrast. Low-confidence outputs often occur on low-contrast tire edges.
- Pattern: tire flat detection depends heavily on clear sidewall / rim visibility, and it is fragile under occlusion and perspective variation.

## Recommendations
- Add more targeted examples for subtle cracks and shallow dents to improve sensitivity in low-contrast cases.
- Use higher-resolution crops or local patch attention around expected damage regions to better distinguish scratch versus scrape.
- Consider class-specific augmentations for glass shatter and tire flat to improve robustness under glare and perspective distortion.
- Review annotation consistency for boundary cases (e.g., paint scrapes vs scratches or dent-like panel warping) to reduce noise in training labels.

## Notes on `crack`
- This class is especially challenging because the network can mistake seam lines, rusted edges, and paint chips for actual cracks.
- Improving crack recall may require more explicit localization or edge-aware features, since false negatives often involve thin damage lines that do not dominate the scene.
