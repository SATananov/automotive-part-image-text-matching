# Dataset Design

## Objective

The dataset is designed for a multimodal classification task.
Each sample contains an automotive part image and a short text description.

The model must predict whether the image and description match.

## Classes

### MATCH

The image and description represent the same automotive part category.

Example:

Image category: brake_disc
Description: Front brake disc

### PARTIAL_MATCH

The description represents another part category from the same automotive system.

Example:

Image category: brake_disc
Description: Front brake pad set

Both parts belong to the braking family, but they are different components.

### MISMATCH

The description represents a part from a different automotive system.

Example:

Image category: brake_disc
Description: Engine air filter

## Part families

| Family | Category 1 | Category 2 |
|---|---|---|
| Electrical | starter | alternator |
| Braking | brake_disc | brake_pad |
| Suspension | shock_absorber | coil_spring |
| Lighting | headlight | taillight |
| Filtration | oil_filter | air_filter |

## Sample construction

Each image will be paired with:

- one matching description;
- one description from the same family but a different category;
- one description from a different family.

This produces an equal number of samples for the three classes.

## Initial dataset target

- 10 categories;
- 20 physical parts per category;
- at least one image per physical part;
- three descriptions per image.

With 200 images, the dataset will contain 600 image-text samples.

## Data splitting

The dataset will be divided by `part_group_id`, not by individual rows.

This prevents photographs or descriptions of the same physical part from appearing in both training and evaluation data.
