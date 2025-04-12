from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

types_set = set()
for obj in OBJ_CATEGORIES.keys():
    for key in OBJ_CATEGORIES[obj].keys():
        for k in OBJ_CATEGORIES[obj][key].types:
            types_set.add(k)

interesting_types = (
    "drink",
    "meat",
    "bread_food",
    "fruit",
    "condiment",
    "packaged_food",
    "vegetable",
    "container",
)
object_sets = {k: set() for k in interesting_types}
for obj in OBJ_CATEGORIES.keys():
    for key in OBJ_CATEGORIES[obj].keys():
        for k in OBJ_CATEGORIES[obj][key].types:
            if k in interesting_types:
                object_sets[k].add(obj)
for k in interesting_types:
    print(f"{k}:\n{object_sets[k]}\n\n")
