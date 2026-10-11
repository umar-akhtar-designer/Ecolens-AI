"""Conservative mapping from recognised object labels to material categories.

IMPORTANT / HONESTY NOTE
------------------------
The default model is an **ImageNet object classifier**, not a dedicated
waste-material classifier. It predicts *objects* (e.g. "water bottle"), not
material composition. This module only translates a curated subset of object
labels into a *material hint* category. It never asserts chemical composition,
contamination, purity or recyclability.

Any label not present in :data:`MATERIAL_CLASS_MAP` is reported as a general
object with **no** material claim (``None``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Material category slugs must match ``data/material_knowledge.json`` entries.
PAPER = "paper_cardboard"
PLASTIC = "plastic"
GLASS = "glass"
METAL = "metal"
TEXTILE = "textile"
E_WASTE = "e_waste"
ORGANIC = "organic"
MIXED = "mixed_uncertain"

# Multiple ImageNet labels can map to the same slug; the value carries a short
# human-readable reason describing *why* the hint is offered.
MATERIAL_CLASS_MAP: dict[str, tuple[str, str]] = {
    # --- Paper / cardboard -------------------------------------------------
    "binder": (PAPER, "Binding/paper stationery object."),
    "book jacket": (PAPER, "Paper-based printed cover."),
    "comic book": (PAPER, "Printed paper publication."),
    "envelope": (PAPER, "Paper envelope."),
    "packet": (PAPER, "Paper/laminate packet (may be multi-material)."),
    "paper towel": (PAPER, "Paper tissue product."),
    "toilet tissue": (PAPER, "Paper tissue product."),
    "carton": (PAPER, "Carton (often paperboard with a liner)."),
    # --- Plastic -----------------------------------------------------------
    "pop bottle": (PLASTIC, "Carbonated-drink bottle, usually PET."),
    "water bottle": (PLASTIC, "Water bottle, commonly PET."),
    "plastic bag": (PLASTIC, "Plastic film bag."),
    "pill bottle": (PLASTIC, "Small plastic container."),
    "bottlecap": (PLASTIC, "Bottle cap (plastic or metal)."),
    "bucket": (PLASTIC, "Bucket, typically plastic (may be metal)."),
    # --- Glass -------------------------------------------------------------
    "beer bottle": (GLASS, "Glass beverage bottle."),
    "beer glass": (GLASS, "Glass drinking vessel."),
    "goblet": (GLASS, "Glass stemware."),
    "wine bottle": (GLASS, "Glass wine bottle."),
    "vase": (GLASS, "Decorative vessel, often glass or ceramic."),
    "pitcher": (GLASS, "Pouring vessel, often glass or ceramic."),
    "water jug": (GLASS, "Jug, often glass."),
    "whiskey jug": (GLASS, "Jug, often glass or ceramic."),
    # --- Metal -------------------------------------------------------------
    "milk can": (METAL, "Metal container."),
    "frying pan": (METAL, "Cookware, usually metal."),
    "can opener": (METAL, "Metal tool."),
    "ladle": (METAL, "Metal utensil."),
    "spatula": (METAL, "Utensil, often metal."),
    "steel drum": (METAL, "Steel container."),
    "iron": (METAL, "Metal appliance (clothes iron)."),
    "nail": (METAL, "Metal fastener."),
    "hammer": (METAL, "Metal-headed tool."),
    "screwdriver": (METAL, "Metal hand tool."),
    "padlock": (METAL, "Metal lock."),
    "chain": (METAL, "Metal chain."),
    "buckle": (METAL, "Metal buckle."),
    # --- Textile -----------------------------------------------------------
    "bath towel": (TEXTILE, "Woven textile item."),
    "jersey": (TEXTILE, "Knit garment."),
    "sweatshirt": (TEXTILE, "Knit garment."),
    "cardigan": (TEXTILE, "Knit garment."),
    "suit": (TEXTILE, "Tailored garment."),
    "jean": (TEXTILE, "Denim garment."),
    "miniskirt": (TEXTILE, "Garment."),
    "kimono": (TEXTILE, "Garment."),
    "fur coat": (TEXTILE, "Outerwear garment (animal or synthetic)."),
    "trench coat": (TEXTILE, "Outerwear garment."),
    "lab coat": (TEXTILE, "Workwear garment."),
    "gown": (TEXTILE, "Garment."),
    "abaya": (TEXTILE, "Garment."),
    "academic gown": (TEXTILE, "Garment."),
    "poncho": (TEXTILE, "Garment."),
    "feather boa": (TEXTILE, "Textile accessory."),
    "scarf": (TEXTILE, "Textile accessory."),
    "diaper": (TEXTILE, "Hygiene textile (disposable or cloth)."),
    "handkerchief": (TEXTILE, "Small textile item."),
    "pillow": (TEXTILE, "Textile/filling item."),
    "quilt": (TEXTILE, "Textile bedding item."),
    "sock": (TEXTILE, "Knit garment."),
    "mitten": (TEXTILE, "Knit garment."),
    "running shoe": (TEXTILE, "Footwear (mixed materials)."),
    "sandal": (TEXTILE, "Footwear (mixed materials)."),
    "sleeping bag": (TEXTILE, "Textile/synthetic outdoor gear."),
    # --- Electronic waste --------------------------------------------------
    # NB: ImageNet's "notebook" (n03832673) is a notebook *computer*, not paper.
    "notebook": (E_WASTE, "Portable computer hardware."),
    "cellular telephone": (E_WASTE, "Consumer electronic device."),
    "dial telephone": (E_WASTE, "Telephone device."),
    "pay-phone": (E_WASTE, "Telephone device."),
    "desktop computer": (E_WASTE, "Computer hardware."),
    "hand-held computer": (E_WASTE, "Handheld computer hardware."),
    "laptop": (E_WASTE, "Portable computer hardware."),
    "computer keyboard": (E_WASTE, "Computer peripheral."),
    "typewriter keyboard": (E_WASTE, "Keyboard peripheral."),
    "monitor": (E_WASTE, "Display device."),
    "mouse": (E_WASTE, "Computer peripheral."),
    "printer": (E_WASTE, "Computer peripheral."),
    "remote control": (E_WASTE, "Consumer electronic controller."),
    "hard disc": (E_WASTE, "Computer storage device."),
    "cassette": (E_WASTE, "Legacy media/electronic item."),
    "cassette player": (E_WASTE, "Consumer electronic device."),
    "radio": (E_WASTE, "Consumer electronic device."),
    "loudspeaker": (E_WASTE, "Audio device."),
    "microphone": (E_WASTE, "Audio device."),
    "television": (E_WASTE, "Display/appliance."),
    "projector": (E_WASTE, "Electronic display device."),
    "modem": (E_WASTE, "Networking device."),
    "iPod": (E_WASTE, "Portable electronic device (contains a battery)."),
    "digital watch": (E_WASTE, "Wearable electronic (contains a battery)."),
    "stopwatch": (E_WASTE, "Small electronic (may contain a battery)."),
    "dishwasher": (E_WASTE, "Large home appliance."),
    "refrigerator": (E_WASTE, "Large home appliance."),
    "washer": (E_WASTE, "Large home appliance."),
    "microwave": (E_WASTE, "Home appliance."),
    "toaster": (E_WASTE, "Small home appliance."),
    "electric fan": (E_WASTE, "Electrical appliance."),
    "power drill": (E_WASTE, "Power tool (may contain a battery)."),
    "waffle iron": (E_WASTE, "Small home appliance."),
    # --- Organic -----------------------------------------------------------
    "banana": (ORGANIC, "Food waste (fruit)."),
    "apple": (ORGANIC, "Food waste (fruit)."),
    "granny smith": (ORGANIC, "Food waste (fruit)."),
    "orange": (ORGANIC, "Food waste (fruit)."),
    "lemon": (ORGANIC, "Food waste (fruit)."),
    "fig": (ORGANIC, "Food waste (fruit)."),
    "pineapple": (ORGANIC, "Food waste (fruit)."),
    "pomegranate": (ORGANIC, "Food waste (fruit)."),
    "strawberry": (ORGANIC, "Food waste (fruit)."),
    "jackfruit": (ORGANIC, "Food waste (fruit)."),
    "custard apple": (ORGANIC, "Food waste (fruit)."),
    "broccoli": (ORGANIC, "Food waste (vegetable)."),
    "cauliflower": (ORGANIC, "Food waste (vegetable)."),
    "zucchini": (ORGANIC, "Food waste (vegetable)."),
    "cucumber": (ORGANIC, "Food waste (vegetable)."),
    "artichoke": (ORGANIC, "Food waste (vegetable)."),
    "bell pepper": (ORGANIC, "Food waste (vegetable)."),
    "head cabbage": (ORGANIC, "Food waste (vegetable)."),
    "mushroom": (ORGANIC, "Food waste (fungus)."),
    "corn": (ORGANIC, "Food waste (vegetable)."),
    "acorn squash": (ORGANIC, "Food waste (vegetable)."),
    "butternut squash": (ORGANIC, "Food waste (vegetable)."),
    "spaghetti squash": (ORGANIC, "Food waste (vegetable)."),
    "bagel": (ORGANIC, "Food waste (bread)."),
    "pretzel": (ORGANIC, "Food waste (bread)."),
    "cheeseburger": (ORGANIC, "Food waste (prepared food)."),
    "hotdog": (ORGANIC, "Food waste (prepared food)."),
    "pizza": (ORGANIC, "Food waste (prepared food)."),
    "burrito": (ORGANIC, "Food waste (prepared food)."),
}

# Waste/container objects. Recognising one of these tells you the *container's*
# class, NOT what is inside or around it. We never infer contents from it.
CONTAINER_LABELS: frozenset[str] = frozenset(
    {
        "ashcan",
        "garbage truck",
        "trash can",
        "dumpster",
        "wastebin",
        "recycle bin",
        "bucket",
        "dustbin",
        "can",
        "packet",
    }
)

# Labels whose prediction should trigger an explicit safety caution, even before
# material confirmation. Matched on the top recognised object name.
HAZARD_CLASS_MAP: dict[str, str] = {
    "iPod": "This object likely contains a lithium battery. Do not shred or "
    "puncture; take it to an e-waste or battery collection point.",
    "digital watch": "Wearable electronics contain a small battery; use a "
    "battery/e-waste drop-off rather than general waste.",
    "hard disc": "Storage devices may hold personal data. Consider secure erasure before disposal.",
    "beer glass": "Broken or shattered glass is a cut hazard; wrap safely before disposal.",
    "goblet": "Glass can shatter; handle and wrap broken pieces safely.",
    "wine bottle": "Broken glass is a cut hazard; handle carefully.",
}


@dataclass(frozen=True)
class MaterialHint:
    """A non-authoritative material suggestion derived from an object label."""

    category_slug: str
    reason: str
    matched_label: str
    is_direct_material_classifier: bool = False
    basis: str = "object_label_inference"

    @property
    def evidence(self) -> str:
        """One-line, honest description of how this hint was produced."""
        return (
            f"Inferred from the recognised object '{self.matched_label}' "
            f"({self.reason}) — not a material measurement."
        )


def map_label_to_material(label: str) -> MaterialHint | None:
    """Return a material hint for a recognised label, or ``None``.

    ``None`` means "no reliable material claim" — not "unknown material".
    """
    if not label:
        return None
    entry = MATERIAL_CLASS_MAP.get(label.strip().lower())
    if entry is None:
        return None
    slug, reason = entry
    return MaterialHint(category_slug=slug, reason=reason, matched_label=label)


def hazard_note_for_label(label: str) -> str | None:
    if not label:
        return None
    return HAZARD_CLASS_MAP.get(label.strip())


def is_container_label(label: str) -> bool:
    """True when the recognised object is a waste bin, truck, or similar.

    Used to guard the UI against implying that a bin's *contents* are known —
    the model only reports the container's own class.
    """
    if not label:
        return False
    return label.strip().lower() in CONTAINER_LABELS
