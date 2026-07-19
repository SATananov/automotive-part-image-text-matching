# Step 010.1 — Manual Visual Review

- Reviewed candidates: **50**
- Approved: **35**
- Rejected: **15**
- Pending: **0**
- Expected validation readiness after application: **REPLACEMENT_IMAGES_REQUIRED**

## Category result

| Category | Approved | Rejected | Replacement needed |
|---|---:|---:|---:|
| air_filter | 4 | 1 | 1 |
| alternator | 3 | 2 | 2 |
| brake_disc | 4 | 1 | 1 |
| brake_pad | 2 | 3 | 3 |
| coil_spring | 1 | 4 | 4 |
| headlight | 5 | 0 | 0 |
| oil_filter | 5 | 0 | 0 |
| shock_absorber | 5 | 0 | 0 |
| starter | 1 | 4 | 4 |
| taillight | 5 | 0 | 0 |

## Rejected candidates

### commons_air_filter_97257871

- Category: `air_filter`
- Reason: Wrong subject/category.
- Review note: Wheel-loader photograph; no automotive engine air filter is visible.
- Source: https://commons.wikimedia.org/wiki/File:IR_HDR_with_Polarizing_filter_(3054081120).jpg

### commons_alternator_10775454

- Category: `alternator`
- Reason: Non-automotive alternator component.
- Review note: Historical armature-winding illustration; not a car alternator or usable automotive-part photograph.
- Source: https://commons.wikimedia.org/wiki/File:Alternator_armature_winding_(Rankin_Kennedy,_Electrical_Installations,_Vol_III,_1903).jpg

### commons_alternator_15826343

- Category: `alternator`
- Reason: Non-automotive generator equipment.
- Review note: Alternator from a stationary generating set, not an automotive alternator.
- Source: https://commons.wikimedia.org/wiki/File:Alternator_from_Allen_generating_set,_ex-Malvern.jpg

### commons_brake_disc_41431532

- Category: `brake_disc`
- Reason: Non-car vehicle part.
- Review note: Motorcycle front brake disc; the external collection is restricted to car and light-vehicle parts.
- Source: https://commons.wikimedia.org/wiki/File:BMW_S1000R_front_Brembo_monobloc.jpg

### commons_brake_pad_187668

- Category: `brake_pad`
- Reason: Wrong part category.
- Review note: Clutch disc, not a brake pad.
- Source: https://commons.wikimedia.org/wiki/File:Clutchdisc.jpg

### commons_brake_pad_32291489

- Category: `brake_pad`
- Reason: Non-car vehicle part.
- Review note: Motorcycle or scooter brake pads, not passenger-car brake pads.
- Source: https://commons.wikimedia.org/wiki/File:Brake_pad.jpg

### commons_brake_pad_70439455

- Category: `brake_pad`
- Reason: Target part not clearly visible.
- Review note: Image primarily shows the brake disc and caliper; the brake pads are obscured.
- Source: https://commons.wikimedia.org/wiki/File:Alarm_bells!_(25681917701).jpg

### commons_coil_spring_12176711

- Category: `coil_spring`
- Reason: Non-automotive rail component.
- Review note: Passenger-rail-coach suspension spring, not a car suspension part.
- Source: https://commons.wikimedia.org/wiki/File:ICF_passenger_coach_coil_spring_suspension.jpg

### commons_coil_spring_43715061

- Category: `coil_spring`
- Reason: Non-automotive rail component.
- Review note: Railway bogie spring, not a car suspension part.
- Source: https://commons.wikimedia.org/wiki/File:W%C3%B3zek_kolejowy_04.JPG

### commons_coil_spring_8517029

- Category: `coil_spring`
- Reason: Non-automotive rail component.
- Review note: Locomotive suspension spring, not a car suspension part.
- Source: https://commons.wikimedia.org/wiki/File:Lokomotiva_162.036,_odpru%C5%BEen%C3%AD.jpg

### commons_coil_spring_8673688

- Category: `coil_spring`
- Reason: Non-automotive rail component.
- Review note: Locomotive suspension spring, not a car suspension part.
- Source: https://commons.wikimedia.org/wiki/File:Lokomotiva_163,_odpru%C5%BEen%C3%AD.jpg

### commons_starter_25996075

- Category: `starter`
- Reason: Wrong part category.
- Review note: Car-battery cross-section, not a starter motor.
- Source: https://commons.wikimedia.org/wiki/File:Car_battery_cross-section.jpeg

### commons_starter_46356726

- Category: `starter`
- Reason: Wrong part category.
- Review note: Automotive battery, not a starter motor.
- Source: https://commons.wikimedia.org/wiki/File:Powerstart_Automotive_Battery.jpeg

### commons_starter_47878235

- Category: `starter`
- Reason: Wrong part category.
- Review note: Battery charger or maintainer, not a starter motor.
- Source: https://commons.wikimedia.org/wiki/File:Car_Battery_Charger.jpg

### commons_starter_5571964

- Category: `starter`
- Reason: Wrong part category.
- Review note: Battery charger, not a starter motor.
- Source: https://commons.wikimedia.org/wiki/File:Car-battery-recharger.jpg

## Replacement behavior

The collector now excludes `rejected` rows from the five-candidate target.
Existing rejected files and metadata remain preserved for audit and are never downloaded again.
A replacement candidate is added as `pending` until another manual review pass.
